from collections import OrderedDict

import theano
from theano import tensor as T
import numpy as np

from morb import parameters

def label_prediction(rbm, vmap, visible_units, label_units, hidden_units, name='func', mb_size=32, mode=None):
    """ Calculate p(y|v), the probability of the labels given the visible state.

    $
        p\left(y\left|v\right.\right) = \frac{
            \exp\left( b_y + \sum_j \mathrm{softplus} \left( c_j + U_{jy} + \sum_i W_{ji} x_i \right) \right)
        }{
            \sum_{y^*} \exp\left( b_y + \sum_j \mathrm{softplus} \left( c_j + U_{jy^*} + \sum_i W_{ji} x_i \right) \right)
        }
    $

    Based on:
    [1] Larochelle, H., Mandel, M., Pascanu, R., & Bengio, Y. (2012).
        Learning Algorithms for the Classification Restricted Boltzmann Machine.
        Journal of Machine Learning Research, 13, 643-669.

    :param rbm:
        the RBM
    :type rbm:
        morb.base.RBM
    :param vmap:
        dictionary with RBM value formulas
    :param visible_units:
        the visible units (v)
    :type visible_units:
        list of morb.base.Units
    :param label_units:
        the label units (y)
    :type label_units:
        list of morb.base.Units
    :param hidden_units:
        the hidden units (h)
    :type visible_units:
        list of morb.base.Units
    """
    # TODO, some time, context
    
    probability_map = []

    # Larochelle, 2012
    for y in label_units:
        all_params_y = rbm.params_affecting(y)
        
        # bias for labels
        by = [ param for param in all_params_y if param.affects_only(y) ]
        assert len(by)==1
        assert isinstance(by[0], parameters.BiasParameters)
            
        # (minibatches, labels)
        by_weights_for_v = T.shape_padleft(by[0].var, 1)

        # collect all components
        label_activation = by_weights_for_v

        for h in hidden_units:
            h_act_given_v = h.activation(vmap, skip_units=label_units)
            
            # weights U connecting labels to hidden
            U = [ param for param in all_params_y if param.affects(h) ]
            
            assert len(U)==1
            assert U[0].weights_for
            
            # sum over hiddens
            a = T.nnet.softplus(U[0].weights_for(y) + T.shape_padright(h_act_given_v, 1))
            # sum over hiddens
            a = T.sum(a, axis=range(1,a.ndim - 1))
            # result: (minibatches, labels)
            label_activation += a
        
        label_activation = T.exp(label_activation)
        
        # normalise over labels
        label_activation = label_activation / T.sum(label_activation, axis=1, keepdims=True)
        
        # (minibatches, labels)
        probability_map.append(label_activation)



    # initialise data sets
    data_sets = OrderedDict()
    for u in visible_units:
        shape = (1,) * vmap[u].ndim
        data_sets[u] = theano.shared(value = np.zeros(shape, dtype=theano.config.floatX),
                                      name="dataset for '%s'"  % u.name)

    index = T.lscalar() # index to a minibatch
    
    # construct givens for the compiled theano function - mapping variables to data
    givens = dict((vmap[u], data_sets[u][index*mb_size:(index+1)*mb_size]) for u in visible_units)

    TF = theano.function([index], probability_map, givens = givens, name = name, mode = mode)

#   theano.printing.debugprint(hidden_units[0].activation(vmap))
#   vmap[hidden_units[0]] = hidden_units[0].activation(vmap)
#   theano.printing.debugprint(label_units[0].activation(vmap))
#   theano.printing.debugprint(TF)

    def func(dmap):
        # dmap is a dict that maps unit types on their respective datasets (numeric).
        units_list = dmap.keys()
        data_sizes = [int(np.ceil(dmap[u].shape[0] / float(mb_size))) for u in units_list]
        
        if data_sizes.count(data_sizes[0]) != len(data_sizes): # check if all data sizes are equal
            raise RuntimeError("The sizes of the supplied datasets for the different input units are not equal.")

        data_cast = [dmap[u].astype(theano.config.floatX) for u in units_list]
        
        for i, u in enumerate(units_list):
            data_sets[u].set_value(data_cast[i], borrow=True)
            
        for batch_index in xrange(min(data_sizes)):
            yield TF(batch_index)
            
    return func


