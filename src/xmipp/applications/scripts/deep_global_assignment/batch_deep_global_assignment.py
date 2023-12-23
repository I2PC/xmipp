#!/usr/bin/env python3

import gc
import math
import numpy as np
from operator import itemgetter
import os
import sys
import xmippLib
from time import time
from scipy.ndimage import shift, rotate, affine_transform

if __name__ == "__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    from xmippPyModules.deepGlobalAssignment import Redundancy

    checkIf_tf_keras_installed()
    fnXmdExp = sys.argv[1]
    fnModel = sys.argv[2]
    maxShift = float(sys.argv[3])
    numParticlesTraining = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    gpuId = sys.argv[6]
    numModels = int(sys.argv[7])
    learning_rate = float(sys.argv[8])
    symmetry = sys.argv[9]
    SNR = float(sys.argv[10])
    modelSize = int(sys.argv[11])
    angPrec = float(sys.argv[12])

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
        Activation, GlobalAveragePooling2D, Add
    import keras
    from keras.models import load_model
    import tensorflow as tf

    class DataGenerator(keras.utils.all_utils.Sequence):
        """Generates data for fnImgs"""

        def __init__(self, fnImgs, labels, maxShift, batch_size, dim, shifts, maxSigmaN, maxAngle, readInMemory=True):
            """Initialization"""
            self.fnImgs = fnImgs
            self.labels = labels
            self.maxShift = maxShift
            self.batch_size = batch_size
            self.dim = dim
            self.readInMemory = readInMemory
            self.on_epoch_end()
            self.shifts = shifts
            self.maxSigmaN = maxSigmaN
            self.maxAngle = maxAngle

            # Read all data in memory
            if self.readInMemory:
                self.Xexp = np.zeros((len(self.labels), self.dim, self.dim, 1), dtype=np.float64)
                for i in range(len(self.labels)):
                    Iexp = np.reshape(xmippLib.Image(self.fnImgs[i]).getData(), (self.dim, self.dim, 1))
                    self.Xexp[i,] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

        def __len__(self):
            """Denotes the number of batches per epoch"""
            num_batches = int(np.floor((len(self.labels)) / self.batch_size))
            return num_batches

        def __getitem__(self, index):
            """Generate one batch of data"""
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            # Find list of IDs
            list_IDs_temp = []
            for i in range(int(self.batch_size)):
                list_IDs_temp.append(indexes[i])
            # Generate data
            Xexp, y = self.__data_generation(list_IDs_temp)

            return Xexp, y

        def on_epoch_end(self):
            """Updates indexes after each epoch"""
            self.indexes = [i for i in range(len(self.labels))]
            np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            """Generates data containing batch_size samples"""
            yvalues = np.array(itemgetter(*list_IDs_temp)(self.labels))
            yshifts = np.array(itemgetter(*list_IDs_temp)(self.shifts))

            # Functions to handle the data
            def get_image(fn_image):
                img = np.reshape(xmippLib.Image(fn_image).getData(), (self.dim, self.dim, 1))
                return (img - np.mean(img)) / np.std(img)

            def euler_to_rotation6d(angles, psi_rotation):
                mat =  xmippLib.Euler_angles2matrix(angles[0],angles[1],angles[2] + psi_rotation)
                return np.reshape(mat[0:2,:],(6))

            if self.readInMemory:
                Iexp = list(itemgetter(*list_IDs_temp)(self.Xexp))
            else:
                fnIexp = list(itemgetter(*list_IDs_temp)(self.fnImgs))
                Iexp = list(map(get_image, fnIexp))

            def shift_then_rotate_image(img, shiftx, shifty, angle):
                imgShifted=shift(img, (shiftx, shifty, 0), order=1, mode='wrap')
                imgRotated=rotate(imgShifted, angle, order=1, mode='reflect', reshape=False)
                sigmaN=np.random.uniform(0, self.maxSigmaN)
                imgRotated+=np.random.normal(0,sigmaN,imgRotated.shape)
                return imgRotated

            rX = self.maxShift * np.random.uniform(-1, 1, size=self.batch_size)
            rY = self.maxShift * np.random.uniform(-1, 1, size=self.batch_size)
            rAngle = self.maxAngle * np.random.uniform(-1, 1, size=self.batch_size)
            Xexp = np.array(list((map(shift_then_rotate_image, Iexp, rX-yshifts[:,0], rY-yshifts[:,1], rAngle))))
            y = np.array(list((map(euler_to_rotation6d, yvalues, rAngle))))

            return Xexp, y

    def conv_block(tensor, filters):
        # Convolutional block of RESNET
        x = Conv2D(filters, (3, 3), padding='same', strides=(2, 2))(tensor)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x_res = Conv2D(filters, (1, 1), strides=(2, 2))(tensor)
        x = Add()([x, x_res])
        x = Activation('relu')(x)
        return x

    def constructModel(Xdim, modelSize):
        """RESNET architecture"""
        inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")
        x = conv_block(inputLayer, filters=64)
        x = conv_block(x, filters=128)
        if modelSize>=1:
            x = conv_block(x, filters=256)
        if modelSize>=2:
            x = conv_block(x, filters=512)
        if modelSize>=3:
            x = conv_block(x, filters=1024)
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, name="output", activation="linear")(x)
        return Model(inputLayer, x)

    def get_labels(fnImages):
        """Returns dimensions, images, angles and shifts values from images files"""
        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnImages)
        mdExp = xmippLib.MetaData(fnImages)
        fnImg = mdExp.getColumnValues(xmippLib.MDL_IMAGE)
        shiftX = mdExp.getColumnValues(xmippLib.MDL_SHIFT_X)
        shiftY = mdExp.getColumnValues(xmippLib.MDL_SHIFT_Y)
        rots = mdExp.getColumnValues(xmippLib.MDL_ANGLE_ROT)
        tilts = mdExp.getColumnValues(xmippLib.MDL_ANGLE_TILT)
        psis = mdExp.getColumnValues(xmippLib.MDL_ANGLE_PSI)

        label = []
        for r, t, p in zip(rots, tilts, psis):
            label.append(np.array((r,t,p)))
        img_shift = [np.array((sX,sY)) for sX, sY in zip(shiftX, shiftY)]

        return Xdim, fnImg, label, img_shift

    tfAt = tf.cast(tf.transpose(Redundancy().Apinv),tf.float32)
    SL = xmippLib.SymList()
    listSymmetryMatrices = [tf.convert_to_tensor(np.kron(np.eye(2),np.transpose(np.array(R))), dtype=tf.float32)
                            for R in SL.getSymmetryMatrices(symmetry)]
    def custom_lossAngles(y_true, y_pred):
        y_6d = tf.matmul(y_pred, tfAt)
        # Take care of symmetry
        num_rows = tf.shape(y_6d)[0]
        min_errors = tf.fill([num_rows], float('inf'))
        rotated_versions = tf.TensorArray(dtype=tf.float32, size=num_rows)
        for i, symmetry_matrix in enumerate(listSymmetryMatrices):
            transformed = tf.matmul(y_6d, symmetry_matrix)
            errors = tf.reduce_mean(tf.abs(y_true - transformed), axis=1)

            # Update minimum errors and rotated versions
            for j in tf.range(num_rows):
                if errors[j] < min_errors[j]:
                    min_errors = tf.tensor_scatter_nd_update(min_errors, [[j]], [errors[j]])
                    rotated_versions = rotated_versions.write(j, transformed[j])
        y_6d = rotated_versions.stack()

        e1_true = y_true[:, :3] # First 3 components
        e2_true = y_true[:, 3:] # Last 3 components
        e3_true = tf.linalg.cross(e1_true, e2_true)

        e1_pred = y_6d[:, :3] # First 3 components
        e2_pred = y_6d[:, 3:] # Last 3 components

        # Gram-Schmidt orthogonalization
#        e1_pred = tf.clip_by_value(e1_pred, -1.0, 1.0)
        e1_pred = tf.nn.l2_normalize(e1_pred, axis=-1)  # Normalize e1
        projection = tf.reduce_sum(e2_pred * e1_pred, axis=-1, keepdims=True)
        e2_pred = e2_pred - projection * e1_pred
#        e2_pred = tf.clip_by_value(e2_pred, -1.0, 1.0)
        e2_pred = tf.nn.l2_normalize(e2_pred, axis=-1)
        e3_pred = tf.linalg.cross(e1_pred, e2_pred)
#        e3_pred = tf.clip_by_value(e3_pred, -1.0, 1.0)
        e3_pred = tf.nn.l2_normalize(e3_pred, axis=-1)

        epsilon = 1e-7
        angle1 = tf.acos(tf.clip_by_value(tf.reduce_sum(e1_true * e1_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
        angle2 = tf.acos(tf.clip_by_value(tf.reduce_sum(e2_true * e2_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
        angle3 = tf.acos(tf.clip_by_value(tf.reduce_sum(e3_true * e3_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))

        return tf.reduce_mean(tf.abs(angle1)+tf.abs(angle2)+tf.abs(angle3))/3.0*(180.0 / np.pi)
        #return tf.reduce_mean(tf.abs(angle1)+tf.abs(angle2))/2.0*(180.0 / np.pi)

    def custom_lossVectors(y_true, y_pred):
        y_6d = tf.matmul(y_pred, tfAt)
        esym = tf.stack([tf.reduce_mean(tf.abs(y_true-tf.matmul(y_6d, tensor)),axis=1)
                          for tensor in listSymmetryMatrices],axis=1)
        return tf.reduce_mean(tf.reduce_min(esym, axis=1))

    Xdims, fnImgs, labels, shifts = get_labels(fnXmdExp)

    numEpochs = math.ceil(numParticlesTraining/len(fnImgs))
    subEpochs=100
    stepEpochs = math.ceil(numEpochs/subEpochs)

    def cleanMemory():
        global model
        print("Cleaning memory")
        fnModelTmp = fnModel + "_tmp.h5"
        model.save(fnModelTmp)
        tf.keras.backend.clear_session()
        gc.collect()
        model =  keras.models.load_model(fnModelTmp, custom_objects={'custom_lossAngles': custom_lossAngles})
        os.remove(fnModelTmp)

        from pympler import muppy, summary
        all_objects = muppy.get_objects()
        sum1 = summary.summarize(all_objects)
        summary.print_(sum1)

        return model

    def train(maxAngle, maxShift, N, numEpochs, save_best_model):
        global model
        global training_generator
        if not hasattr(train,"Ntrains"):
            train.Ntrains=0
        # train.Ntrains+=1
        # if train.Ntrains==10:
        #     model = cleanMemory()
        #     train.Ntrains=0
        training_generator.maxSigmaN = N
        training_generator.maxShift = maxShift
        training_generator.maxAngle = maxAngle
        history = model.fit(training_generator, epochs=numEpochs, callbacks=[save_best_model])
        return history.history['loss'][-1]


    if SNR > 0:
        finalN = math.sqrt(1.0 / SNR)
    else:
        finalN = 0
    training_generator = DataGenerator(fnImgs, labels, 0.0, batch_size, Xdims, shifts, 0.0, 0.0)
    for index in range(numModels):

        fnModelIndex =  fnModel + str(index) + ".h5"
        save_best_model = ModelCheckpoint(fnModelIndex, monitor='loss', save_best_only=True)
        if os.path.exists(fnModel):
            model = load_model(fnModelIndex)
        else:
            model = constructModel(Xdims, modelSize)
        adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.summary()
#        model.compile(loss=custom_lossVectors, optimizer=adam_opt)
        model.compile(loss=custom_lossAngles, optimizer=adam_opt)

        # Train without any perturbation
        imax=subEpochs
        for i in range(subEpochs):
            print("Iteration %d no perturbation"%i)
            loss=train(0.0, 0.0, 0.0, stepEpochs, save_best_model)
            if loss<angPrec:
                imax = subEpochs
                break

        # Train now with geometric perturbations
        angStep=180.0/subEpochs
        shiftStep=maxShift/subEpochs
        for i in range(subEpochs):
            print("Iteration %d"%i)
            for j in range(imax):
                print("Training %d with maxAngle=%f, maxShift=%f"%(j,(i+1)*angStep, (i+1)*shiftStep))
                loss=train((i+1)*angStep, (i+1)*shiftStep, 0.0, stepEpochs, save_best_model)
                if loss<angPrec:
                    break

        # Estabilize with geometric perturbations
        for i in range(subEpochs):
            print("Iteration %d estabilize geometric"%i)
            loss=train(180, maxShift, 0.0, stepEpochs, save_best_model)
            if loss<angPrec:
                break

        if finalN>0:
            Nstep=finalN/subEpochs
            # Now with noise
            for i in range(subEpochs):
                print("Iteration %d noise" % i)
                for j in range(imax):
                    print("Training %d with noise %f" % (j,(i + 1) * Nstep))
                    loss=train(180, maxShift, (i+1)*Nstep, stepEpochs, save_best_model)
                    if loss<angPrec:
                        break
