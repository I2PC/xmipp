import numpy as np
import os
import sys
import xmippLib
from time import time

if __name__=="__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    checkIf_tf_keras_installed()
    listClass = sys.argv[1]
    listImage = sys.argv[2]
    refNum = sys.argv[3]

    def dataRead(listClass, listImage, refNum):

        n = np.random.randint(0, refNum - 1)
        print("random class = ", n)
        clTest = listClass[n]
        imTest = np.random.choice(listImage[n])

        return clTest, imTest
        print(clTest, imTest)


    class DataGenerator(keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, list_IDs_zeros, list_IDs_ones, labels, batch_size, dim, shuffle, pathsExp):
            'Initialization'
            self.dim = dim
            self.batch_size = batch_size
            self.labels = labels
            self.list_IDs_zeros = list_IDs_zeros
            self.list_IDs_ones = list_IDs_ones
            self.shuffle = shuffle
            self.pathsExp = pathsExp
            self.on_epoch_end()

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor((len(self.labels)) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes_zeros = self.indexes_zeros[index*self.batch_size:(index+1)*self.batch_size]
            indexes_ones = self.indexes_ones[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            #list_IDs_temp = [self.list_IDs[k] for k in indexes]
            list_IDs_temp = []
            for i in range(int(self.batch_size//2)):
                list_IDs_temp.append(indexes_zeros[i])
            for i in range(int(self.batch_size//2)):
                list_IDs_temp.append(indexes_ones[i])

            # Generate data
            Xexp, y = self.__data_generation(list_IDs_temp)

            return Xexp, y

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes_zeros = self.list_IDs_zeros
            self.indexes_ones = self.list_IDs_ones
            if self.shuffle == True:
                np.random.shuffle(self.indexes_zeros)
                np.random.shuffle(self.indexes_ones)

        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            clBatch = np.zeros((self.batch_size,self.dim,self.dim,1),dtype=np.float64)
            imBatch = np.zeros((self.batch_size, self.dim, self.dim, 1), dtype=np.float64)


            # Generate data
            for i in range (self.batch_size):

                clGen, imGen = dataRead(listClass, listImage, refNum)

                clMat = xmippLib.Image(clGen).getData()
                imMat = xmippLib.Image(imGen).getData()

                clMat = (clMat - np.mean(clMat)) / np.std(clMat)
                imMat = (imMat - np.mean(imMat)) / np.std(imMat)

                clBatch[i,] = clMat
                imBatch[i,] = imMat

            return clMat, imMat


    def constructModel(Xdim, numOut):
        inputLayer = Input(shape=(Xdim,Xdim,1), name="input")

        #Network model
        L = Conv2D(16, (int(Xdim/3), int(Xdim/3)), activation="relu") (inputLayer) #33 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Conv2D(32, (int(Xdim/10), int(Xdim/10)), activation="relu") (L) #11 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Conv2D(64, (int(Xdim/20), int(Xdim/20)), activation="relu") (L) #5 filter size before
        L = BatchNormalization()(L)
        L = MaxPooling2D()(L)
        L = Dropout(0.2)(L)
        L = Flatten() (L)
        L = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(L)
        L = BatchNormalization()(L)

        if numOut>2:
            L = Dense(numOut, name="output", activation="softmax") (L)
        elif numOut==2:
            L = Dense(1, name="output", activation="sigmoid") (L)
        return Model(inputLayer, L)





