from keras.initializers import Initializer
import tensorflow as tf





class InitCentersKMeans(Initializer):
    """ Initializer for initialization of centers of RBF network
        by clustering the given data set.
    # Arguments
        X: matrix, dataset
    """

    def __init__(self, X, max_iter=100,k=10):
        self.X = X
        self.max_iter = max_iter
        self.k=k
    def input_fn(self):
        return tf.train.limit_epochs(
            tf.convert_to_tensor(self.X, dtype=tf.float32), num_epochs=4)

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]

        n_centers = shape[0]

        km = tf.compat.v1.estimator.experimental.KMeans(num_clusters=n_centers, relative_tolerance=0.001)
        km.train(self.input_fn)


        return km.cluster_centers()
