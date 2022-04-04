from keras.initializers import Initializer
import tensorflow as tf


import numpy as np

def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

class InitBeta(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self,X,k, max_iter=100):

        self.X = X
        self.max_iter = max_iter
        self.k=k

    def input_fn(self):
        return tf.train.limit_epochs(
            tf.convert_to_tensor(self.X, dtype=tf.float32), num_epochs=4)


    def __call__(self, shape, dtype=None):


        km = tf.compat.v1.estimator.experimental.KMeans(num_clusters=self.k, relative_tolerance=0.001)
        km.train(self.input_fn)
        self.centers=km.cluster_centers()
        dMax = np.max([get_distance(c1, c2) for c1 in self.centers for c2 in self.centers])
        self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        return self.std_list
