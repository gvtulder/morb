import theano
from theano import tensor as T

from morb import parameters

def label_prediction(rbm, vmap, visible_units, label_units, hidden_units):
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
    
    probability_map = {}

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
        probability_map = { y: theano.function([vmap[u] for u in visible_units], label_activation) }

    return probability_map

