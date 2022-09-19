#!/usr/bin/env python3

import os
import sys
import pickle
import xmippLib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

# from protocol_deep_classify.py import readImageStep


def dataRead(listImage, listImageForClass, numClass):

    n = np.random.randint(0, numClass)
    clTest = int(n)
    start = 0

    for i in range(n):
        start = start + listImageForClass[i]
    end = start + listImageForClass[n]

    im = np.random.randint(start, end-1)
    imTest = listImage[im]

    # clTest  = clTest.reshape(dim, dim, 1)
    return clTest, imTest

def __data_generation(batch_size, dim):
    'Generates data containing batch_size samples'
    # Initialization
    # clBatch = np.zeros((batch_size,dim,dim,1),dtype=np.float64)
    # clBatch = np.empty(batch_size, dtype=np.int64)
    # clBatch=[]
    # imBatch = np.zeros((batch_size, dim, dim, 1), dtype=np.float64)
    imBatch = []
    clBatch = []

    # Generate data
    for i in range(batch_size):
        clGen, imGen = dataRead(imageInput, listCount, numClass)

        imGen = imGen.reshape(dim, dim, 1)

        imBatch.append(imGen)
        clBatch.append(clGen)

    BatchIm = np.asarray(imBatch).astype('float64')
    BatchCl = np.asarray(clBatch).astype('int64')

    return BatchIm, BatchCl
    # yield (imBatch, clBatch)


def constructModel(Dim):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(Dim, Dim, 1)),  # vectorize the image
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        # on each iteration i am going to cut 20% of the connections so that is does not do overfitting
        tf.keras.layers.Dense(5, activation='softmax')  # output of 10 neurons
    ])
    # model = keras.Sequential()
    #
    # model.add(keras.Input(shape=(Dim, Dim, 1)))
    #
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(Dim, Dim, 1), padding='same'))
    # #    model.add(BatchNormalization())
    # #    model.add(Conv3D(32, (5,5,5), activation='relu', padding='same'))
    # #    model.add(BatchNormalization())
    # #    model.add(Dropout(0.2))
    # #    model.add(Conv3D(32, (11,11,11), activation='relu', padding='same'))
    # #    model.add(BatchNormalization())
    # #    model.add(Conv3D(32, (5,5,5), activation='relu', padding='same'))
    # #    model.add(BatchNormalization())
    # #    model.add(Dropout(0.2))
    # #    model.add(Conv3D(32, (3,3,3), activation='relu', padding='same'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dropout(0.2))
    # #    model.add(Dropout(0.2))
    # model.add(layers.Dense(5, activation='sigmoid'))

    return model


if __name__=="__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    checkIf_tf_keras_installed()

    classArrays = sys.argv[1]
    imageArrays = sys.argv[2]
    imageForClass = sys.argv[3]
    numClass = int(sys.argv[4])
    dim = int(sys.argv[5])


    classInput = np.load(classArrays)
    imageInput = np.load(imageArrays)

    with open(imageForClass, "rb") as fp:
        listCount = pickle.load(fp)

    print("arg 3", listCount)


    batch_size = 256
    model = constructModel(dim)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    print("batchsize = ", batch_size)
    im_train, cl_train = __data_generation(batch_size, dim)

    print("modelprueba")

    model.fit(im_train, cl_train, epochs=10)

    print("modelprueba 2")
    #model.fit_generator(generator=data, epochs=5, steps_per_epoch=5)
    # model.evaluate(im_test, cl_test, verbose=2)









        # Definicion del callback para Tensorboard.
        # logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        # model.fit(x_train, y_train, epochs=5)
        # Versión con salida a Tensorboard
        # model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])  # allows to se the evolution of the training (tensorboard),
        # copy and paste these two lines whenever I need to use ir.

        # model.evaluate(x_test, y_test, verbose=2)



        # checkpoint
        # filepath = "/home/erney/data/test_allboxes_dense_512_2-12A_window13_paral_noise/training_32_torvalds_checkpoint_5000.h5"
        # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
        #
        # callbacks_list = [checkpoint]
        #
        # # fit
        # model.fit_generator(generator=pdbM, epochs=5000, steps_per_epoch=50, callbacks=callbacks_list, verbose=1,
        #                     use_multiprocessing = True, workers = 4)
        # model.save(
        #     '/home/erney/data/test_allboxes_dense_512_2-12A_window13_paral_noise/training_32_torvalds_final_5000.h5')

#
#
#     class DataGenerator(keras.utils.Sequence):
#         'Generates data for Keras'
#         def __init__(self, listClass, listImage, batch_size, dim, totalIm):
#             'Initialization'
#             self.dim = dim
#             self.batch_size = batch_size
#             self.listClass = listClass
#             self.listImage = listImage
#             self.totalIm = totalIm
#
#
#         def __len__(self):
#             'Denotes the number of batches per epoch'
#             return int(np.floor((len(self.totalIm)) / self.batch_size))
#
#         def __getitem__(self, index):
#             'Generate one batch of data'
#             clMat, imMat = self.__data_generation()
#
#             return clMat, imMat
#
#         def __data_generation(self):
#             'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#             # Initialization
#             clBatch = np.zeros((self.batch_size,self.dim,self.dim,1),dtype=np.float64)
#             imBatch = np.zeros((self.batch_size, self.dim, self.dim, 1), dtype=np.float64)
#
#
#             # Generate data
#             for i in range(self.batch_size):
#
#                 clGen, imGen = dataRead(listClass, listImage, refNum)
#
#                 clMat = xmippLib.Image(clGen).getData()
#                 imMat = xmippLib.Image(imGen).getData()
#
#                 clMat = (clMat - np.mean(clMat)) / np.std(clMat)
#                 imMat = (imMat - np.mean(imMat)) / np.std(imMat)
#
#                 clBatch[i,] = clMat
#                 imBatch[i,] = imMat
#
#             return clMat, imMat
#
#
#     # def constructModel(Xdim):
#     #     inputLayer = Input(shape=(Xdim,Xdim,1), name="input")
#     #
#     #     #Network model
#     #     L = Conv2D(16, (int(Xdim/3), int(Xdim/3)), activation="relu") (inputLayer) #33 filter size before
#     #     L = BatchNormalization()(L)
#     #     L = MaxPooling2D()(L)
#     #     L = Conv2D(32, (int(Xdim/10), int(Xdim/10)), activation="relu") (L) #11 filter size before
#     #     L = BatchNormalization()(L)
#     #     L = MaxPooling2D()(L)
#     #     L = Conv2D(64, (int(Xdim/20), int(Xdim/20)), activation="relu") (L) #5 filter size before
#     #     L = BatchNormalization()(L)
#     #     L = MaxPooling2D()(L)
#     #     L = Dropout(0.2)(L)
#     #     L = Flatten() (L)
#     #     L = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
#     #     L = BatchNormalization()(L)
#     #
#     #     if numOut>2:
#     #         L = Dense(numOut, name="output", activation="softmax") (L)
#     #     elif numOut==2:
#     #         L = Dense(1, name="output", activation="sigmoid") (L)
#     #     return Model(inputLayer, L)
#
#
#
# # norm_images = []
# # for i in images:
# #   i = np.array(i)
# #   i = cv2.resize(i, (256,256))
# #   i = i/np.max(i)
# #   norm_images.append(i)
#
#
#
#     print(listClass)
#     print(listImage)
#     print(refNum)
#     print(dim)
#     print(totalIm)
