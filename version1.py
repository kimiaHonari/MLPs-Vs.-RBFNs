from keras.models import Sequential
from keras import models, layers
import keras
import os
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from RBFLayer import RBF

def lr_schedule(epoch):


    lr = 1e-2
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-3
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def get_callbacks(n_run):
    model_type = "LeNet"
    save_dir = os.path.join(os.getcwd(), 'My_Lenet5_mnist/total/run' + str(n_run))
    model_name = "%s_model.{epoch:03d}.h5" % model_type
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
    lr_reducer = keras.callbacks.ReduceLROnPlateau(
        factor=np.sqrt(0.1),
        cooldown=0,
        patience=5,
        min_lr=0.5e-6)

    return [checkpoint,es, lr_reducer]

def Create_Model(input_shape,n_run,x_train,y_train,x_val,y_val):
    # Instantiate an empty model
    model = Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1),padding="same", activation='relu', input_shape = (32,32,1)))

    # S2 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # C3 Convolutional Layer
    model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))

    # S4 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # C5 Fully Connected Convolutional Layer
    model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    # Flatten the CNN output so that we can connect it with fully connected layers
    model.add(layers.Flatten())

    # FC6 Fully Connected Layer
    model.add(layers.Dense(84, activation='relu'))

    # Output Layer with softmax activation
    model.add(layers.Dense(10, activation='softmax'))

    # # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    # Compile the model
    # model.compile(
    #     optimizer=keras.optimizers.SGD(),
    #     loss='mse',
    #     metrics=['accuracy'])

    model.summary()
    callbacks = get_callbacks(n_run)
    datagen = ImageDataGenerator(
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

    datagen.fit(x_train)
    batch_size = 32

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,
              verbose=2,
              validation_data=(x_val, y_val),
              callbacks=callbacks)
    score = model.evaluate(x_val, y_val, verbose=0)

    return callbacks[0].best