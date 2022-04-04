from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform, Initializer, Constant
import numpy as np


class RBFLayer(Layer):


    def __init__(self, output_dim, initializer=None, betas=None, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape[1])
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=self.init_betas,
                                     # initializer=Constant(value=2.79469),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        print(x.shape)
        C = K.expand_dims(self.centers)
        print(C.shape)
        H = K.transpose(C-K.transpose(x))
        print(H.shape)
        l2=K.sum(H**2, axis=1)
        print(l2.shape)
        print(self.betas.shape)

        ret=K.exp(-self.betas * l2)
        print(ret.shape)

        return ret

        # C = self.centers[np.newaxis, :, :]
        # X = x[:, np.newaxis, :]
        #
        # diffnorm = K.sum((C-X)**2, axis=-1)
        # ret = K.exp( - self.betas * diffnorm)
        # return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


