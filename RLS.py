import keras
from numba import jit, cuda
import numpy as np
from keras import backend as K
import tensorflow as tf

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

# @jit(target="cuda")
def rls(X_train,Y_train,K_cent,cent,sigma,yreal,w,xt,yt):

    shape = X_train.shape
    row = shape[0]
    column = K_cent
    G = np.empty((1, column), dtype=float)

    num_hd = K_cent
    num_out = 10

    sig = 0.0001
    P = sig ** (-1) * np.eye(num_hd)
    landa = 1
    W = np.zeros((num_out, num_hd), dtype=float)
    # print(w.shape)
    # n=w.eval(session=tf.compat.v1.Session())
    # W=np.transpose(n)
    # print(W)
    print("Start training with RLS:")
    for epoch in range(0, 100):
        print("epoch: ",epoch)
        # np.random.shuffle(X_train)
        # print(X_train)
        for iter_i in range(0, row):


            gi=X_train[iter_i]
            gi=gi.reshape(num_hd,1)
            pai = np.dot(P, gi)

            kk = pai / (landa + np.dot(np.transpose(gi), pai))
            pred_ty=np.transpose(np.dot(W, gi))

            e = Y_train[iter_i] - pred_ty

            w_delta = np.dot(kk, e)

            wtemp = W + np.transpose(w_delta)
            W = wtemp
            P = landa ** (-1) * (P - np.dot(np.dot(kk, np.transpose(gi)), P))


        print("w shape",W.shape)
        pred_ty = X_train @ np.transpose(W)
        # print(pred_ty.shape)np.transpose(np.dot(W, gi)
        pred_ty = tf.convert_to_tensor(pred_ty)
        pred_ty = keras.activations.softmax(pred_ty, axis=-1)
        pred_ty = keras.backend.argmax(pred_ty, axis=1)
        print("ACCURACY")
        # matrix = tf.math.confusion_matrix(
        #     yreal, pred_ty, num_classes=10, weights=None, dtype=tf.dtypes.int32,
        #     name=None
        # )
        # with tf.Session() as sess:
        #     print(matrix.eval())
        geometric_mean(yreal,pred_ty)


        pred_ty = xt @ np.transpose(W)
        pred_ty = tf.convert_to_tensor(pred_ty)
        pred_ty = keras.activations.softmax(pred_ty, axis=-1)
        pred_ty = keras.backend.argmax(pred_ty, axis=1)
        geometric_mean(yt, pred_ty)
        # matrix = tf.math.confusion_matrix(
        #     yt, pred_ty, num_classes=10, weights=None, dtype=tf.dtypes.int32,
        #     name=None
        # )
        # with tf.Session() as sess:
        #     print(matrix.eval())


    pred_ty = X_train @ np.transpose(W)
        # print(pred_ty.shape)np.transpose(np.dot(W, gi)
    pred_ty = tf.convert_to_tensor(pred_ty)
    pred_ty = keras.activations.softmax(pred_ty, axis=-1)
    pred_ty = keras.backend.argmax(pred_ty, axis=1)
    print(pred_ty.shape)
        # matrix = tf.math.confusion_matrix(
        #     yreal, pred_ty, num_classes=10, weights=None, dtype=tf.dtypes.int32,
        #     name=None
        # )
        # with tf.Session() as sess:
        #     print(matrix.eval())
    geometric_mean(yreal,pred_ty)


    pred_ty = xt @ np.transpose(W)
    pred_ty = tf.convert_to_tensor(pred_ty)
    pred_ty = keras.activations.softmax(pred_ty, axis=-1)
    pred_ty = keras.backend.argmax(pred_ty, axis=1)
    geometric_mean(yt, pred_ty)
    # pred_ty = X_train @np.transpose(W)
    #     # print(pred_ty.shape)
    # pred_ty=tf.convert_to_tensor(pred_ty)
    # pred_ty = keras.activations.softmax(pred_ty, axis=-1)
    # pred_ty = keras.backend.argmax(pred_ty, axis=1)
    # print(pred_ty.shape)
    # matrix = tf.math.confusion_matrix(
    #         yreal, pred_ty, num_classes=10, weights=None, dtype=tf.dtypes.int32,
    #         name=None
    #     )
    # with tf.Session() as sess:
    #     print(matrix.eval())




        # prediction = np.empty((row,10))
        # for iter in range(0, row):
        #     gi=g[iter]
        #     out=np.dot(W, gi)
        #     pred=[]
        #     for j in out:
        #         pred.append(j[0])
        #     # print(pred)
        #     pred=np.exp(pred) / np.sum(np.exp(pred))
        #     # print(pred)
        #     prediction[iter] = pred
        #     for j in range(0, 10):
        #         if prediction[iter][j] > threshold:
        #             prediction[iter][j] = 1
        #         else:
        #             prediction[iter][j] = 0
        # print(prediction.shape)
        # pred_y = keras.backend.argmax(prediction, axis=1)
        # matrix = tf.math.confusion_matrix(
        #         yreal, pred_y, num_classes=10, weights=None, dtype=tf.dtypes.int32,
        #         name=None
        #     )
        # with tf.Session() as sess:  print(matrix.eval())