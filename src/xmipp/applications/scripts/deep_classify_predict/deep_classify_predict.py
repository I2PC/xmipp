import sys
import pickle
import xmippLib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import math


def __data_generation(imInput, dim, batch_size, counter):
    'Generates data containing batch_size samples'
    # Initialization
    imBatch = []

    # Generate data
    for j in range(batch_size):

        imGen = imInput[counter+j]
        imGen = imGen.reshape(dim, dim, 1)
        imBatch.append(imGen)

    BatchIm = np.asarray(imBatch).astype('float64')

    return BatchIm


if __name__=="__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    checkIf_tf_keras_installed()

    model_train = sys.argv[1]
    imageArrays = sys.argv[2]
    dim = int(sys.argv[5])

    imageInput = np.load(imageArrays)

    batch_size = 500
    count = 0
    num_batch = len(imageInput)/batch_size
    math.ceil(num_batch)
    num_batch = int(num_batch)

    im_pr = []
    cl_pr = []

    for i in range(num_batch):
        # if

        im_pr = __data_generation(imageInput, dim, batch_size, count)

        count += batch_size

        model = load_model(model_train)
        cl_pr = model.predict_classes(im_pr)

    for i in range(len(im_pr)):
        print("X=%s, Predicted=%s" % (im_pr[i], cl_pr[i]))

