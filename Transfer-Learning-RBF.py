import matplotlib
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from RBFLayer import RBF
from keras.datasets import mnist, cifar10
from keras.datasets import fashion_mnist
import keras
from keras.models import load_model
# from classification import geometric_mean_score


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
        # print(acc)
        gacc*=acc
        i+=1
    # print(matrix)
    print(np.power(gacc,0.1))

# (x_t, y_t), (x_val, y_val) =cifar10.load_data()
i=9
x_t = np.load("cifar_updateted2/str" + str(i) + "/trainx.npy")
y_t = np.load("cifar_updateted2/str" + str(i) + "/trainy.npy")
x_val = np.load("cifar_updateted2/str" + str(i) + "/valx.npy")
y_val = np.load("cifar_updateted2/str" + str(i) + "/valy.npy")
n_split=10
input_shape = x_t.shape[1:]
print(input_shape)
print("training data y: " + str(y_t.shape))
print("training data x: " + str(x_t.shape))
x_t = x_t.astype('float32') / 255
x_val = x_val.astype('float32') / 255
#
#
# x_t = x_t.reshape(-1, 28, 28, 1)
# x_val = x_val.reshape(-1, 28, 28, 1)
print("training data y: " + str(y_t.shape))
print("training data x: " + str(x_t.shape))
x_train_mean = np.mean(x_t, axis=0)
x_t -= x_train_mean
x_val -= x_train_mean
y_train=y_t
y_validation=y_val
y_t = keras.utils.to_categorical(y_t, 10)
y_val = keras.utils.to_categorical(y_val, 10)
input_shape = x_t.shape[1:]

model = load_model("Lenet_cifar10/run1/run9/mnist10_LeNet_model.033.h5")

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='mean_squared_error',
    metrics=['accuracy'])
import tensorflow as tf
model.summary()
y_pred=model.predict(x_t)
pred_ty=keras.backend.argmax(y_pred, axis=1)
print(pred_ty.shape)
geometric_mean(y_train, pred_ty)
# matrix=tf.math.confusion_matrix(
#             y_train, pred_ty, num_classes=10, weights=None, dtype=tf.dtypes.int32,
#             name=None
#         )
# with tf.Session() as sess:  print(matrix.eval())
y_pred=model.predict(x_val)
pred_ty=keras.backend.argmax(y_pred, axis=1)
print(pred_ty.shape)
geometric_mean(y_validation, pred_ty)


#extract the output of convolutional layer for both training and validation data
layer_name = 'flatten_5'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(x_t)
intermediate_output_test=intermediate_layer_model.predict(x_val)
print(intermediate_output.shape)

#train the RBFN 
RBF_CLASSIFIER = RBF(intermediate_output, y_t, y_train, intermediate_output_test, y_validation, num_of_classes=10,
                     k=10, std_from_clusters=False)

acc, tac = RBF_CLASSIFIER.fit()



