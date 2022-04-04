import numpy as np
from RLS import rls
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import RMSprop

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
# from classification import geometric_mean_score
from multiprocessing import Pool
from keras.models import Sequential
from keras.datasets import mnist
import statsmodels.api as sm
import numpy as np
import padasip as pa
import keras
from sklearn.cluster import KMeans
from keras.utils import np_utils
def geometric_mean(yreal,ypred):
    matrix = tf.math.confusion_matrix(
        yreal, ypred, num_classes=10, weights=None, dtype=tf.dtypes.int32,
        name=None
    )
    matrix=matrix.eval(session=tf.compat.v1.Session())
    gacc=1
    i=0
    for row in matrix:
        s=np.sum(row)
        acc=row[i]/s
        gacc*=acc
        i+=1
    # print(matrix)
    print(np.power(gacc,0.1))
    return np.power(gacc,0.1)
RBF_list = []
def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

class RBF:

    def __init__(self, X, y,y_real,xt,yt, num_of_classes,
                 k, std_from_clusters=True):
        self.X = X
        self.y = y
        self.xt=xt
        self.yt=yt
        self.yreal=y_real
        print(self.X.shape)
        print(self.y.shape)

        self.number_of_classes = num_of_classes
        self.k = k
        self.std_from_clusters = std_from_clusters

    # def convert_to_one_hot(self, x, num_of_classes):
    #     arr = np.zeros((len(x), num_of_classes))
    #     for i in range(len(x)):
    #         c = int(x[i])
    #         arr[i][c] = 1
    #     return arr
    #
    # def rbf(self, x, c, s):
    #     distance = get_distance(x, c)
    #     return 1 / np.exp(-distance / s ** 2)
    #
    # def cal(self,x):
    #
    #     rbf = [self.rbf(x, c, s) for (c, s) in zip(self.centroids, self.std_list)]
    #     RBF_list.append(rbf)

    def rbf_list(self, X):
        RBF_list = []
        for x in X:
            rbf=[]
            for i in range(0,self.k):
               rbf.append(np.math.exp(-np.dot((x - self.centroids[i]), np.transpose(x - self.centroids[i])) / np.math.pow(2*self.std_list[0], 2)))
            # rbf.append(1)
            RBF_list.append(rbf)

        return np.array(RBF_list)


    def input_fn(self):

        return tf.train.limit_epochs(
            tf.convert_to_tensor(self.X, dtype=tf.float32), num_epochs=4)

    def fit(self):


        # km=tf.compat.v1.estimator.experimental.KMeans(num_clusters=self.k,relative_tolerance=0.001)
        # km.train(self.input_fn)
        # self.centroids=km.cluster_centers()

        km=KMeans(n_clusters=self.k, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=5, algorithm='auto')
        km.fit(self.X)
        self.centroids=km.cluster_centers_
        dMax = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
        self.std_list = np.repeat(dMax / np.sqrt(2 * self.k), self.k)
        # rls(self.X,self.y,self.k,self.centroids,self.std_list[0],self.yreal)
        # print("start computing RBF_ list")
        #
        RBF_X = self.rbf_list(self.X)
        #
        # print("start computing w")

        #self.w = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.y

        # mod = sm.RecursiveLS(RBF_X, self.y)
        # res = mod.fit()
        # self.pred_ty=mod.predict()

        # self.pred_ty=f.predict(RBF_X)
        # print(self.pred_ty.shape)
        # print(self.yreal.shape)
        # print(y.shape)

        #tensorflow least square
        self.w = tf.linalg.lstsq(
           RBF_X, self.y.astype('float64'), fast=False, name=None)
        RBF_list_tst = self.rbf_list(self.xt)

        # rls(RBF_X,self.y,self.k,self.centroids,self.std_list[0],self.yreal,self.w,RBF_list_tst,self.yt)
        self.pred_ty = RBF_X @ self.w
        print(self.pred_ty.shape)

        self.pred_ty=keras.activations.softmax(self.pred_ty, axis=-1)
        self.pred_ty=keras.backend.argmax(self.pred_ty, axis=1)
        print("accuracy:")
        acc=geometric_mean(self.yreal,self.pred_ty)
        print("test accuracy:")
        self.pred_ty = RBF_list_tst @ self.w
        self.pred_ty = keras.activations.softmax(self.pred_ty, axis=-1)
        self.pred_ty = keras.backend.argmax(self.pred_ty, axis=1)
        tacc=geometric_mean(self.yt, self.pred_ty)


        # print(self.pred_ty.shape)
        # matrix=tf.math.confusion_matrix(
        #     self.yreal, self.pred_ty, num_classes=10, weights=None, dtype=tf.dtypes.int32,
        #     name=None
        # )
        # with tf.Session() as sess:  print(matrix.eval())
        # print("Geometric Mean:",geometric_mean_score(self.yreal, self.pred_ty))
        return acc,tacc

    def predict(self,xt,yt):

        RBF_list_tst = self.rbf_list(xt)

        self.pred_ty =RBF_list_tst@self.w
        self.pred_ty = keras.activations.softmax(self.pred_ty, axis=-1)
        self.pred_ty = keras.backend.argmax(self.pred_ty, axis=1)
        print(self.pred_ty.shape)
        matrix = tf.math.confusion_matrix(
            yt, self.pred_ty, num_classes=10, weights=None, dtype=tf.dtypes.int32,
            name=None
        )
        with tf.Session() as sess:  print(matrix.eval())

        # self.pred_ty = np.array([np.argmax(x) for x in self.pred_ty])
        #
        # diff = self.pred_ty -yt
        #
        # print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff))
        # print("Geometric Mean:", geometric_mean_score(yt, self.pred_ty))





