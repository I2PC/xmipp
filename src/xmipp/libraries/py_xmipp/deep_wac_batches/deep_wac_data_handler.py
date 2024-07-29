
##---------------- DEPRECATED LIBRARY --------------


## TODO: check if this works

import math
import numpy as np
import xmippLib
from operator import itemgetter
import tensorflow as tf

# Objects from this class are iterables ready to be handled by the keras' fit/predict method
# each iteration contains a batch from the original data it has been fed with

class DataHandler(tf.data.Dataset):
        
        #Initialization
        def __init__(self, fnImgs, labels, batchSize, dim):
            self.fnImgs = fnImgs
            if labels is not None:
                self.labels = labels
            self.batchSize = batchSize
            if self.batch_size > len(self.fnImgs):
                self.batch_size = len(self.fnImgs)
            self.dim = dim
            #TODO: Check whether on_epoch_end is necessary here or not
            #self.on_epoch_end()
            #self.readInMemory = readInMemory
            #self.shifts = shifts (Deleted in get_labels and when invoking generator)

            # Abre la imagen, la carga en memoria de verdad guardandola con sus dimensiones y luego la normaliza
            # Read all data in memory
            #TODO: Supposedly, this class could allow to charge all data en memory or do it dynamically rn it gets overwritten (CHECK)
            """
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.labels), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.labels)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)
            """
        # Obtains the number of batches per epoch
        def __len__(self):
            numBatches = int(np.floor((len(self.fnImgs)) / self.batchSize))
            return numBatches

        #TODO: review how this is written
        # Returns a batch of data every time the object is iterated
        def __getitem__(self, index):

            # Generate indexes of the batch
            indexes = self.indexes[index * self.batchSize:(index + 1) * self.batchSize]
            # Find list of IDs
            listTempIDs = []
            for i in range(int(self.batchSize)):
                listTempIDs.append(indexes[i])
            # Generate data
            Xexp, y = self.__data_generation(listTempIDs)

            return Xexp, y

        """
        # TODO: understand why this is used and how (why shuffle?)
        def on_epoch_end(self):
            #Updates indexes after each epoch
            self.indexes = [i for i in range(len(self.labels))]
            np.random.shuffle(self.indexes)
        """

        # Returns the constructed data batch
        def __data_generation(self, listTempIDs, labels):

            # Functions to handle the data
            def get_image(fn_image):
                img = np.reshape(xmippLib.Image(fn_image).getData(), (self.dim, self.dim, 1))
                return (img - np.mean(img)) / np.std(img)

            # obtaining the file names for the selected temporal indices
            fnImgTemp = list(itemgetter(*listTempIDs)(self.fnImgs))

            # TODO: Shall I change the variable name into something similar to the previous?
            # executes get_image for every temporally selected image file and saves all the real images in a list
            Iexp = list(map(get_image, fnImgTemp))

            if labels is not None:
                return Iexp, labels
            else:
                return Iexp