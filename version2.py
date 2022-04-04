
import os
from keras.models import Sequential
import numpy as np


import keras
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

def lenet_backend(input_shape):
    print(input_shape)
    inputs = Input(shape=input_shape)
    c1 = Conv2D(filters=32, kernel_size=3, strides=1, activation="relu")(inputs)
    s2 = MaxPooling2D(pool_size=2)(c1)
    c3 = Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")(s2)
    s4 = MaxPooling2D(pool_size=2)(c3)
    c5 = Dense(256, activation="relu")(Flatten()(s4))

    return inputs, c5
def lr_schedule(epoch):


    lr = 1e-3
    if epoch > 120:
        lr *= 0.5e-3
    elif epoch > 100:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def get_callbacks(n_run):
    model_type = "LeNet"
    save_dir = os.path.join(os.getcwd(), 'Lenet_cifar10/run1/run' + str(n_run))
    model_name = "mnist10_%s_model.{epoch:03d}.h5" % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True)

    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        factor=np.sqrt(0.1),
        cooldown=0,
        patience=5,
        min_lr=0.5e-6)

    return [checkpoint,es,lr_reducer]

def Create_Model(input_shape,n_run,x_train,y_train,x_val,y_val):
    print("training data y: "+str(y_train.shape))
    print("training data x: "+str(x_train.shape))
    inputs, c5 = lenet_backend(input_shape=input_shape)
    # # f6 = Dense(84, activation="relu")(c5)
    # # f6 = Dense(128, activation="relu")(c5)
    # # f7 = Dense(10, activation="softmax")(f6)
    f7=Dense(10, activation="softmax")(c5)
    model = Model(inputs=inputs, outputs=f7)
    strides = 1,


    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    # model.compile(
    #     optimizer=keras.optimizers.SGD(),
    #     loss='mse',
    #     metrics=['accuracy'])

    model.summary()

    # In[ ]:

    callbacks = get_callbacks(n_run)


    datagen=ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.01,  # Randomly zoom image
        width_shift_range=0.03,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.03,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False
    )

    datagen.fit(x_train,  augment=True)
    batch_size = 32
    print("training data y: "+str(y_train.shape))
    print("training data x: "+str(x_train.shape))
    print("val data y: "+str(y_val.shape))

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,
              verbose=2,
              validation_data=(x_val, y_val),
              callbacks=callbacks)
    score = model.evaluate(x_val, y_val, verbose=0)
    return callbacks[0].best


