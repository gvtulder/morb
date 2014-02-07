from morb.base import Parameters

import theano
import theano.tensor as T
from theano.tensor.nnet import conv

from morb.misc import tensordot # better tensordot implementation that can be GPU accelerated
# tensordot = T.tensordot # use theano implementation

class FixedBiasParameters(Parameters):
    # Bias fixed at -1, which is useful for some energy functions (like Gaussian with fixed variance, Beta)
    def __init__(self, rbm, units, name=None, energy_multiplier=1, value=-1):
        super(FixedBiasParameters, self).__init__(rbm, [units], name=name, energy_multiplier = energy_multiplier)
        self.variables = []
        self.u = units
        
        self.terms[self.u] = lambda vmap, pmap: T.constant(value, theano.config.floatX) # T.constant is necessary so scan doesn't choke on it
        
    def energy_term(self, vmap, pmap):
        s = vmap[self.u]
        return self.energy_multiplier * T.sum(s, axis=range(1, s.ndim)) # NO minus sign! bias is -1 so this is canceled.
        # sum over all but the minibatch dimension.
        
        
class ProdParameters(Parameters):
    def __init__(self, rbm, units_list, W, name=None, energy_multiplier=1):
        super(ProdParameters, self).__init__(rbm, units_list, name=name, energy_multiplier = energy_multiplier)
        assert len(units_list) == 2
        self.var = W
        self.variables = [self.var]
        self.vu = units_list[0]
        self.hu = units_list[1]
        
        self.terms[self.vu] = lambda vmap, pmap: T.dot(vmap[self.hu], pmap[W].T)
        self.terms[self.hu] = lambda vmap, pmap: T.dot(vmap[self.vu], pmap[W])
        
        self.energy_gradients[self.var] = lambda vmap, pmap: vmap[self.vu].dimshuffle(0, 1, 'x') * vmap[self.hu].dimshuffle(0, 'x', 1)
        self.energy_gradient_sums[self.var] = lambda vmap, pmap: T.dot(vmap[self.vu].T, vmap[self.hu])

    def weights_for(self, units, pmap):
        assert units in [self.vu, self.hu]
        if self.vu == units:
            # (minibatches, hiddens, visible)
            return pmap[self.var].dimshuffle('x', 1, 0)
        else:
            # (minibatches, visible, hiddens)
            return pmap[self.var].dimshuffle('x', 0, 1)
                
    def energy_term(self, vmap, pmap):
        return - self.energy_multiplier * T.sum(self.terms[self.hu](vmap, pmap) * vmap[self.hu], axis=1)
        # return - T.sum(T.dot(vmap[self.vu], self.var) * vmap[self.hu])
        # T.sum sums over the hiddens dimension.
        
    
class BiasParameters(Parameters):
    def __init__(self, rbm, units, b, name=None, energy_multiplier=1):
        super(BiasParameters, self).__init__(rbm, [units], name=name, energy_multiplier = energy_multiplier)
        self.var = b
        self.variables = [self.var]
        self.u = units
        
        self.terms[self.u] = lambda vmap, pmap: pmap[self.var]
        
        self.energy_gradients[self.var] = lambda vmap, pmap: vmap[self.u]
        
    def energy_term(self, vmap, pmap):
        return - self.energy_multiplier * T.dot(vmap[self.u], pmap[self.var])
        # bias is NOT TRANSPOSED because it's a vector, and apparently vectors are COLUMN vectors by default.


class QuadraticBiasParameters(BiasParameters):
    def __init__(self, rbm, units, b, name=None, energy_multiplier=1):
        super(QuadraticBiasParameters, self).__init__(rbm, units, b, name, energy_multiplier)
        self.energy_gradients[self.var] = lambda vmap, pmap: vmap[self.u] - pmap[self.var].dimshuffle('x', 0)
        
    def energy_term(self, vmap, pmap):
        return self.energy_multiplier * (- T.dot(vmap[self.u], pmap[self.var]) + (pmap[self.var] ** 2) / 2).dimshuffle('x', 0)
        # bias is NOT TRANSPOSED because it's a vector, and apparently vectors are COLUMN vectors by default.


class AdvancedProdParameters(Parameters):
    def __init__(self, rbm, units_list, dimensions_list, W, name=None, energy_multiplier=1):
        super(AdvancedProdParameters, self).__init__(rbm, units_list, name=name, energy_multiplier = energy_multiplier)
        assert len(units_list) == 2
        self.var = W
        self.variables = [self.var]
        self.vu = units_list[0]
        self.hu = units_list[1]
        self.vd = dimensions_list[0]
        self.hd = dimensions_list[1]
        self.vard = self.vd + self.hd
        
        # there are vd visible dimensions and hd hidden dimensions, meaning that the weight matrix has
        # vd + hd = Wd dimensions.
        # the hiddens and visibles have hd+1 and vd+1 dimensions respectively, because the first dimension
        # is reserved for minibatches!
        self.terms[self.vu] = lambda vmap, pmap: tensordot(vmap[self.hu], pmap[W], axes=(range(1,self.hd+1),range(self.vd, self.vard)))
        self.terms[self.hu] = lambda vmap, pmap: tensordot(vmap[self.vu], pmap[W], axes=(range(1,self.vd+1),range(0, self.vd)))
        
        def gradient(vmap, pmap):
            v_indices = range(0, self.vd + 1) + (['x'] * self.hd)
            h_indices = [0] + (['x'] * self.vd) + range(1, self.hd + 1)
            v_reshaped = vmap[self.vu].dimshuffle(v_indices)
            h_reshaped = vmap[self.hu].dimshuffle(h_indices)
            return v_reshaped * h_reshaped
        
        self.energy_gradients[self.var] = gradient
        self.energy_gradient_sums[self.var] = lambda vmap, pmap: tensordot(vmap[self.vu], vmap[self.hu], axes=([0],[0]))
        # only sums out the minibatch dimension.
                
    def energy_term(self, vmap, pmap):
        # v_part = tensordot(vmap[self.vu], self.var, axes=(range(1, self.vd+1), range(0, self.vd)))
        v_part = self.terms[self.hu](vmap, pmap)
        neg_energy = tensordot(v_part, vmap[self.hu], axes=(range(1, self.hd+1), range(1, self.hd+1)))
        # we do not sum over the first dimension, which is reserved for minibatches!
        return - self.energy_multiplier * neg_energy # don't forget to flip the sign!

    def weights_for(self, units, pmap):
        assert units in [self.vu, self.hu]
        if self.vu == units:
            # (minibatches, maps, map dims, visible)
            print (['x'] + range(self.vd, self.hd+self.vd) + range(0, self.vd))
            return pmap[self.var].dimshuffle(['x'] + range(self.vd, self.hd+self.vd) + range(0, self.vd))
        else:
            # (minibatches, hidden, visible)
            raise Exception("AdvancedProdWeights.weights_for(hidden) not implemented")
            return pmap[self.var].dimshuffle('x', 0, 'x', 'x', 1)


class AdvancedBiasParameters(Parameters):
    def __init__(self, rbm, units, dimensions, b, name=None, energy_multiplier=1):
        super(AdvancedBiasParameters, self).__init__(rbm, [units], name=name, energy_multiplier = energy_multiplier)
        self.var = b
        self.variables = [self.var]
        self.u = units
        self.ud = dimensions
        
        self.terms[self.u] = lambda vmap, pmap: pmap[self.var]
        
        self.energy_gradients[self.var] = lambda vmap, pmap: vmap[self.u]
        
    def energy_term(self, vmap, pmap):
        return - self.energy_multiplier * tensordot(vmap[self.u], pmap[self.var], axes=(range(1, self.ud+1), range(0, self.ud)))
        

class SharedBiasParameters(Parameters):
    """
    like AdvancedBiasParameters, but a given number of trailing dimensions are 'shared'.
    """
    def __init__(self, rbm, units, dimensions, shared_dimensions, b, name=None, energy_multiplier=1, divide_by_number_of_nodes=False):
        super(SharedBiasParameters, self).__init__(rbm, [units], name=name, energy_multiplier = energy_multiplier)
        self.var = b
        self.variables = [self.var]
        self.u = units
        self.ud = dimensions
        self.sd = shared_dimensions
        self.nd = self.ud - self.sd
        
        self.terms[self.u] = lambda vmap, pmap: T.shape_padright(pmap[self.var], self.sd)
        
        if divide_by_number_of_nodes:
          self.energy_gradients[self.var] = lambda vmap, pmap: T.mean(vmap[self.u], axis=self._shared_axes(vmap))
        else:
          print "SharedBiasParameters: divide_by_number_of_nodes==False"
          self.energy_gradients[self.var] = lambda vmap, pmap: T.sum(vmap[self.u], axis=self._shared_axes(vmap))
        
    def _shared_axes(self, vmap):
        d = vmap[self.u].ndim
        return range(d - self.sd, d)
            
    def energy_term(self, vmap, pmap):
        # b_padded = T.shape_padright(self.var, self.sd)
        # return - T.sum(tensordot(vmap[self.u], b_padded, axes=(range(1, self.ud+1), range(0, self.ud))), axis=0)
        # this does not work because tensordot cannot handle broadcastable dimensions.
        # instead, the dimensions of b_padded which are broadcastable should be summed out afterwards.
        # this comes down to the same thing. so:
        t = tensordot(vmap[self.u], pmap[self.var], axes=(range(1, self.nd+1), range(0, self.nd)))
        # now sum t over its trailing shared dimensions, which mimics broadcast + tensordot behaviour.
        axes = range(t.ndim - self.sd, t.ndim)
        return - self.energy_multiplier * T.sum(t, axis=axes)


class SharedQuadraticBiasParameters(SharedBiasParameters):
    def __init__(self, rbm, units, dimensions, shared_dimensions, b, name=None, energy_multiplier=1):
        super(SharedQuadraticBiasParameters, self).__init__(rbm, units, dimensions, shared_dimensions, b, name, energy_multiplier)

        self.terms[self.u] = lambda vmap, pmap: T.shape_padright(pmap[self.var], self.sd)
        self.energy_gradients[self.var] = lambda vmap, pmap: T.mean(vmap[self.u], axis=self._shared_axes(vmap)) - pmap[self.var].dimshuffle('x', 0)
            
    def energy_term(self, vmap, pmap):
        # b_padded = T.shape_padright(self.var, self.sd)
        # return - T.sum(tensordot(vmap[self.u], b_padded, axes=(range(1, self.ud+1), range(0, self.ud))), axis=0)
        # this does not work because tensordot cannot handle broadcastable dimensions.
        # instead, the dimensions of b_padded which are broadcastable should be summed out afterwards.
        # this comes down to the same thing. so:
        t = tensordot(vmap[self.u], pmap[self.var], axes=(range(1, self.nd+1), range(0, self.nd)))
        # now sum t over its trailing shared dimensions, which mimics broadcast + tensordot behaviour.
        axes = range(t.ndim - self.sd, t.ndim)
        number_of_shared_units = 1
        u_shape = vmap[self.u].shape
        for a in self._shared_axes(vmap):
          number_of_shared_units *= u_shape[a]
        number_of_shared_units = T.cast(number_of_shared_units, dtype=theano.config.floatX)
        return self.energy_multiplier * (- T.sum(t, axis=axes) + T.sum(number_of_shared_units * (pmap[self.var] * pmap[self.var]) / 2.0))

               
class SharedProdParameters(Parameters):
    def __init__(self, rbm, units_list, dimensions, shared_dimensions, W, name=None, energy_multiplier=1, pooling_operator=T.mean):
        super(SharedProdParameters, self).__init__(rbm, units_list, name=name, energy_multiplier = energy_multiplier)
        assert len(units_list) == 2
        self.var = W
        self.variables = [self.var]
        self.vu = units_list[0]
        self.hu = units_list[1]

        self.hud = dimensions
        self.hsd = shared_dimensions
        self.hnd = self.hud - self.hsd

        self.pooling_operator = pooling_operator
        if pooling_operator != T.sum:
          raise "Pooling operator is not sum, are you sure that will work?"

        def from_hu(m, vmap):
          return self.pooling_operator(m, axis=self._shared_axes(vmap))

        def to_hu(m):
          return T.shape_padright(m, self.hsd)
        
        self.terms[self.vu] = lambda vmap, pmap: tensordot(from_hu(vmap[self.hu], vmap), pmap[W], \
                                                           ( range(1, self.hnd+1), range(1, self.hnd+1) ))
        #                                  T.dot(from_hu(vmap[self.hu], vmap), W.T)
        self.terms[self.hu] = lambda vmap, pmap: to_hu(tensordot(vmap[self.vu], pmap[W], ( (1,), (0,) )))
        #                                  to_hu(T.dot(vmap[self.vu], W))
        
        self.energy_gradients[self.var] = lambda vmap, pmap: vmap[self.vu].dimshuffle([0,1] + ['x'] * self.hnd) * from_hu(vmap[self.hu], vmap).dimshuffle([0,'x'] + range(1, self.hnd+1))
        if self.hnd == 1:
            self.energy_gradient_sums[self.var] = lambda vmap, pmap: T.dot(vmap[self.vu].T, from_hu(vmap[self.hu], vmap))
        
    def _shared_axes(self, vmap):
        d = vmap[self.hu].ndim
        return range(d - self.hsd, d)

    def weights_for(self, units, pmap):
        assert units in [self.vu, self.hu]
        if self.vu == units:
            # (minibatches, maps, map dims, visible)
            d = ['x'] + range(1, self.hnd+1) + ['x'] * self.hsd + [0]
            return pmap[self.var].dimshuffle(d)
        else:
            # (minibatches, hidden, visible)
            raise Exception("SharedProdWeights.weights_for(hidden) not implemented")
            return pmap[self.var].dimshuffle('x', 0, 'x', 'x', 1)
                
    def energy_term(self, vmap, pmap):
        return - self.energy_multiplier * T.sum(vmap[self.vu] * self.terms[self.vu](vmap, pmap), axis=1)
        
    
class Convolutional2DParameters(Parameters):
    def __init__(self, rbm, units_list, W, shape_info=None, name=None, energy_multiplier=1):
        # use the shape_info parameter to provide a dict with keys:
        # hidden_maps, visible_maps, filter_height, filter_width, visible_height, visible_width, mb_size
        
        super(Convolutional2DParameters, self).__init__(rbm, units_list, name=name, energy_multiplier = energy_multiplier)
        assert len(units_list) == 2
        self.var = W # (hidden_maps, visible_maps, filter_height, filter_width)
        self.variables = [self.var]
        self.vu = units_list[0] # (mb_size, visible_maps, visible_height, visible_width)
        self.hu = units_list[1] # (mb_size, hidden_maps, hidden_height, hidden_width)
        self.shape_info = shape_info

        # conv input is (output_maps, input_maps, filter height [numrows], filter width [numcolumns])
        # conv input is (mb_size, input_maps, input height [numrows], input width [numcolumns])
        # conv output is (mb_size, output_maps, output height [numrows], output width [numcolumns])
        
        def term_vu(vmap, pmap):
            # input = hiddens, output = visibles so we need to swap dimensions
            W_shuffled = pmap[self.var].dimshuffle(1, 0, 2, 3)
            if self.filter_shape is not None:
                shuffled_filter_shape = [self.filter_shape[k] for k in (1, 0, 2, 3)]
            else:
                shuffled_filter_shape = None
            # (this requires a flipped convolution; conv2d does that)
            return conv.conv2d(vmap[self.hu], W_shuffled, border_mode='full', \
                               image_shape=self.hidden_shape, filter_shape=shuffled_filter_shape)
            
        def term_hu(vmap, pmap):
            # input = visibles, output = hiddens, flip filters
            # (flip because conv2d flips the kernel a second time)
            W_flipped = pmap[self.var][:, :, ::-1, ::-1]
            return conv.conv2d(vmap[self.vu], W_flipped, border_mode='valid', \
                               image_shape=self.visible_shape, filter_shape=self.filter_shape)
        
        self.terms[self.vu] = term_vu
        self.terms[self.hu] = term_hu
        
        def gradient(vmap, pmap):
            raise NotImplementedError # TODO
        
        def gradient_sum(vmap, pmap):
            if self.visible_shape is not None:
                i_shape = [self.visible_shape[k] for k in [1, 0, 2, 3]]
            else:
                i_shape = None
        
            if self.hidden_shape is not None:
                f_shape = [self.hidden_shape[k] for k in [1, 0, 2, 3]]
            else:
                f_shape = None
            
            v_shuffled = vmap[self.vu].dimshuffle(1, 0, 2, 3)
            h_shuffled = vmap[self.hu].dimshuffle(1, 0, 2, 3)
            # (flip because conv2d flips the kernel a second time)
            h_shuffled = h_shuffled[:, :, ::-1, ::-1]
            
            c = conv.conv2d(v_shuffled, h_shuffled, border_mode='valid', image_shape=i_shape, filter_shape=f_shape)
            # must use the mean over all hidden nodes
            # ( = the size of the feature maps )
            # (see, e.g., Lee et al., 2012:
            #  "Unsupervised Learning of Hierarchical Representations
            #   with Convolutional Deep Belief Networks")
            #
            # (2013.08.02: I now think this is not correct.)
#           number_of_hiddens = ((self.shape_info['visible_height']-self.shape_info['filter_height']+1) \
#               * (self.shape_info['visible_width']-self.shape_info['filter_width']+1))
#           return c.dimshuffle(1, 0, 2, 3) / number_of_hiddens

            return c.dimshuffle(1, 0, 2, 3)
            
        self.energy_gradients[self.var] = gradient
        self.energy_gradient_sums[self.var] = gradient_sum
    
    @property    
    def filter_shape(self):
        keys = ['hidden_maps', 'visible_maps', 'filter_height', 'filter_width']
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            return tuple(self.shape_info[k] for k in keys)
        else:
            return None

    @property            
    def visible_shape(self):
        keys = ['mb_size', 'visible_maps', 'visible_height', 'visible_width']                
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            return tuple(self.shape_info[k] for k in keys)
        else:
            return None

    @property            
    def hidden_shape(self):
        keys = ['mb_size', 'hidden_maps', 'visible_height', 'visible_width']
        if self.shape_info is not None and all(k in self.shape_info for k in keys):
            hidden_height = self.shape_info['visible_height'] - self.shape_info['filter_height'] + 1
            hidden_width = self.shape_info['visible_width'] - self.shape_info['filter_width'] + 1
            return (self.shape_info['mb_size'], self.shape_info['hidden_maps'], hidden_height, hidden_width)
        else:
            return None
        
    def energy_term(self, vmap, pmap):
        return - self.energy_multiplier * T.sum(self.terms[self.hu](vmap, pmap) * vmap[self.hu], axis=[1,2,3])
        # sum over all but the minibatch axis
        
        
        
        
# TODO: 1D convolution + optimisation




class ThirdOrderParameters(Parameters):
    def __init__(self, rbm, units_list, W, name=None, energy_multiplier=1):
        super(ThirdOrderParameters, self).__init__(rbm, units_list, name=name, energy_multiplier = energy_multiplier)
        assert len(units_list) == 3
        self.var = W
        self.variables = [self.var]
        self.u0 = units_list[0]
        self.u1 = units_list[1]
        self.u2 = units_list[2]
        
        def term_u0(vmap, pmap):
            p = tensordot(vmap[self.u1], pmap[W], axes=([1],[1])) # (mb, u0, u2)
            return T.sum(p * vmap[self.u2].dimshuffle(0, 'x', 1), axis=2) # (mb, u0)
            # cannot use two tensordots here because of the minibatch dimension.
            
        def term_u1(vmap, pmap):
            p = tensordot(vmap[self.u0], pmap[W], axes=([1],[0])) # (mb, u1, u2)
            return T.sum(p * vmap[self.u2].dimshuffle(0, 'x', 1), axis=2) # (mb, u1)
            
        def term_u2(vmap, pmap):
            p = tensordot(vmap[self.u0], pmap[W], axes=([1],[0])) # (mb, u1, u2)
            return T.sum(p * vmap[self.u1].dimshuffle(0, 1, 'x'), axis=1) # (mb, u2)
            
        self.terms[self.u0] = term_u0
        self.terms[self.u1] = term_u1
        self.terms[self.u2] = term_u2
                
        def gradient(vmap, pmap):
            p = vmap[self.u0].dimshuffle(0, 1, 'x') * vmap[self.u1].dimshuffle(0, 'x', 1) # (mb, u0, u1)
            p2 = p.dimshuffle(0, 1, 2, 'x') * vmap[self.u2].dimshuffle(0, 'x', 'x', 1) # (mb, u0, u1, u2)
            return p2
            
        self.energy_gradients[self.var] = gradient
        
    def energy_term(self, vmap, pmap):
        return - self.energy_multiplier * T.sum(self.terms[self.u1](vmap, pmap) * vmap[self.u1], axis=1)
        # sum is over the u1 dimension, not the minibatch dimension!




class ThirdOrderFactoredParameters(Parameters):
    """
    Factored 3rd order parameters, connecting three Units instances. Each factored
    parameter matrix has dimensions (units_size, num_factors).
    """
    def __init__(self, rbm, units_list, variables, name=None, energy_multiplier=1):
        super(ThirdOrderFactoredParameters, self).__init__(rbm, units_list, name=name, energy_multiplier = energy_multiplier)
        assert len(units_list) == 3
        assert len(variables) == 3
        self.variables = variables
        self.var0 = variables[0]
        self.var1 = variables[1]
        self.var2 = variables[2]
        self.u0 = units_list[0]
        self.u1 = units_list[1]
        self.u2 = units_list[2]
        self.prod0 = lambda vmap, pmap: T.dot(vmap[self.u0], self.var0) # (mb, f)
        self.prod1 = lambda vmap, pmap: T.dot(vmap[self.u1], self.var1) # (mb, f)
        self.prod2 = lambda vmap, pmap: T.dot(vmap[self.u2], self.var2) # (mb, f)
        self.terms[self.u0] = lambda vmap, pmap: T.dot(self.prod1(vmap, pmap) * self.prod2(vmap, pmap), pmap[self.var0].T) # (mb, u0)
        self.terms[self.u1] = lambda vmap, pmap: T.dot(self.prod0(vmap, pmap) * self.prod2(vmap, pmap), pmap[self.var1].T) # (mb, u1)
        self.terms[self.u2] = lambda vmap, pmap: T.dot(self.prod0(vmap, pmap) * self.prod1(vmap, pmap), pmap[self.var2].T) # (mb, u2)
        
        # if the same parameter variable is used multiple times, the energy gradients should be added.
        # so we need a little bit of trickery here to make this work.
        energy_gradient_sums_list = [
            lambda vmap, pmap: T.dot(vmap[self.u0].T, self.prod1(vmap, pmap) * self.prod2(vmap, pmap)), # (u0, f)
            lambda vmap, pmap: T.dot(vmap[self.u1].T, self.prod0(vmap, pmap) * self.prod2(vmap, pmap)), # (u1, f)
            lambda vmap, pmap: T.dot(vmap[self.u2].T, self.prod0(vmap, pmap) * self.prod1(vmap, pmap)), # (u2, f)
        ] # the T.dot also sums out the minibatch dimension
        
        energy_gradient_sums_dict = {}
        for var, grad in zip(self.variables, energy_gradient_sums_list):
            if var not in energy_gradient_sums_dict:
                energy_gradient_sums_dict[var] = []
            energy_gradient_sums_dict[var].append(grad)
            
        for var, grad_list in energy_gradient_sums_dict.items():
            def tmp(): # create a closure, otherwise grad_list will always
                # refer to the one of the last iteration!
                # TODO: this is nasty, is there a cleaner way?
                g = grad_list
                self.energy_gradient_sums[var] = lambda vmap, pmap: sum(f(vmap, pmap) for f in g)
            tmp()
            
        # TODO: do the same for the gradient without summing!
    
    def energy_term(self, vmap, pmap):
        return - self.energy_multiplier * T.sum(self.terms[self.u1](vmap, pmap) * vmap[self.u1], axis=1)
        # sum is over the u1 dimension, not the minibatch dimension!
        



class TransformedParameters(Parameters):
    """
    Transform parameter variables, adapt gradients accordingly
    """
    def __init__(self, params, transforms, transform_gradients, name=None, energy_multiplier=1):
        """
        params: a Parameters instance for which variables should be transformed
        transforms: a dict mapping variables to their transforms
        gradients: a dict mapping variables to the gradient of their transforms
        
        IMPORTANT: the original Parameters instance should not be used afterwards
        as it will be removed from the RBM.
        
        ALSO IMPORTANT: because of the way the chain rule is applied, the old
        Parameters instance is expected to be linear in the variables.
        
        Example usage:
            rbm = RBM(...)
            h = Units(...)
            v = Units(...)
            var_W = theano.shared(...)
            W = ProdParameters(rbm, [u, v], var_W, name='W')
            W_tf = TransformedParameters(W, { var_W: T.exp(var_W) }, { var_W: T.exp(var_W) }, name='W_tf')
        """
        self.encapsulated_params = params
        self.transforms = transforms
        self.transform_gradients = transform_gradients

        # remove the old instance, this one will replace it
        params.rbm.remove_parameters(params)
        # TODO: it's a bit nasty that the old instance is first added to the RBM and then removed again.
        # maybe there is a way to prevent this? For example, giving the old parameters a 'dummy' RBM
        # like in the factor implementation. But then this dummy has to be initialised first...
        
        # initialise
        super(TransformedParameters, self).__init__(params.rbm, params.units_list, name, energy_multiplier = energy_multiplier)
        
        self.variables = params.variables
        for u, l in params.terms.items(): # in the terms, replace the vars by their transforms
            self.terms[u] = lambda vmap: theano.clone(l(vmap), transforms)
            
        for v, l in params.energy_gradients.items():
            self.energy_gradients[v] = lambda vmap: l(vmap) * transform_gradients[v] # chain rule
            
        for v, l in params.energy_gradient_sums.items():
            self.energy_gradient_sums[v] = lambda vmap: l(vmap) * transform_gradients[v] # chain rule
            
    def energy_term(self, vmap):
        old = self.encapsulated_params.energy_term(vmap)
        return self.energy_multiplier * theano.clone(old, self.transforms)
        
        
