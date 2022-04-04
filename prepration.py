
import statistics
import StratifiedKFold as s
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import keras
from keras.datasets import mnist, cifar10
from keras.datasets import fashion_mnist
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import version1 as model
# import Lenet5
import Mymodel


# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
n_split=10
strkfold=s.StratifiedKFold(n_split)
strkfold.generate_fold(x_train,y_train)
print(x_train.shape)
accuracy=[]
for i in range(0,n_split):
    (x_t, y_t), (x_val, y_val)=strkfold.pop(i)

    np.save("cifar_updateted2/str"+str(i)+"/trainx.npy", x_t)
    np.save("cifar_updateted2/str" + str(i) + "/trainy.npy", y_t)
    np.save("cifar_updateted2/str" + str(i) + "/valx.npy", x_val)
    np.save("cifar_updateted2/str" + str(i) + "/valy.npy", y_val)
    x_t = x_t.astype('float32') / 255
    x_val = x_val.astype('float32') / 255


    x_train_mean = np.mean(x_t, axis=0)
    x_t -= x_train_mean
    x_val -= x_train_mean

#
    # Convert class vectors to binary class matrices.
    y_t = keras.utils.to_categorical(y_t, 10)
    y_val = keras.utils.to_categorical(y_val, 10)
    input_shape = x_t.shape[1:]

    accuracy.append(model.Create_Model(input_shape,i,x_t,y_t,x_val,y_val))
    # accuracy.append(Lenet5.Create_Model(input_shape, i, x_t, y_t, x_val, y_val))

    print(accuracy)


print(accuracy)
print("Average: "+ str(statistics.mean(accuracy)))
print("Std: "+ str(statistics.stdev(accuracy)))















