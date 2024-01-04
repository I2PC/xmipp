#!/usr/bin/env python3

import math
import numpy as np
import os
import sys
import xmippLib
from time import time
from xmipp_base import *

SHIFT_MODE = 0
FULL_MODE = 1

class ScriptDeepGlobalAssignment(XmippScript):

    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Train a deep global assignment model')
        ## params
        self.addParamsLine(' --isim <metadata>            : xmd file with the list of simulated images,'
                           'they must have a 3D alignment')
        self.addParamsLine('[--iexp <metadata="">]        : xmd file with the list of experimental images,'
                           'they must have a 3D alignment, the same angles and shifts as the simulated')
        self.addParamsLine(' --oroot <root>               : Rootname for the models')
        self.addParamsLine('[--maxEpochs <N=3000>]        : Max. number of epochs')
        self.addParamsLine('[--batchSize <N=8>]           : Batch size')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--Nmodels <N=5>]             : Number of models')
        self.addParamsLine('[--learningRate <lr=0.0001>]  : Learning rate')
        self.addParamsLine('[--sym <s=c1>]                : Symmetry')
        self.addParamsLine('[--modelSize <s=0>]           : Model size (0=277k, 1=1M, 2=5M, 3=19M parameters)')
        self.addParamsLine('[--precision <s=0.07>]        : Alignment precision measured in percentage of '
                           'the image size')

    def run(self):
        fnXmdSim = self.getParam("--isim")
        fnXmdExp = self.getParam("--iexp")
        fnModel = self.getParam("--oroot")
        maxEpochs = int(self.getParam("--maxEpochs"))
        batch_size = int(self.getParam("--batchSize"))
        gpuId = self.getParam("--gpu")
        numModels = int(self.getParam("--Nmodels"))
        learning_rate = float(self.getParam("--learningRate"))
        symmetry = self.getParam("--sym")
        modelSize = int(self.getParam("--modelSize"))
        precision = float(self.getParam("--precision"))

        import tensorflow as tf
        import keras
        from keras.callbacks import ModelCheckpoint
        from keras.models import Model
        from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
            Activation, GlobalAveragePooling2D, Add, Layer, Dropout, Lambda, Resizing, Flatten, Reshape, \
            Conv2DTranspose
        from tensorflow.keras.losses import mean_absolute_error
        from tensorflow.keras import backend as K
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
        from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
        checkIf_tf_keras_installed()

        class DataGenerator():
            """Generates data for fnImgs"""

            def __init__(self, fnImgsSim, fnImgsExp, angles, shifts, batch_size, dim):
                """Initialization"""
                self.fnImgsSim = fnImgsSim
                self.fnImgsExp = fnImgsExp
                self.angles = angles
                self.shifts = shifts
                self.batch_size = batch_size
                self.dim = dim
                self.loadData()

            def loadData(self):
                def euler_to_rotation6d(angles, shifts):
                    mat =  xmippLib.Euler_angles2matrix(angles[0],angles[1],angles[2])
                    return np.concatenate((mat[1], mat[2], shifts))

                # Read all data in memory
                self.Xsim = np.zeros((len(self.angles), self.dim, self.dim, 1), dtype=np.float64)
                self.Xexp = np.zeros((len(self.angles), self.dim, self.dim, 1), dtype=np.float64)
                self.ysim = np.zeros((len(self.angles), 8), dtype=np.float64)
                readExp = len(fnImgsExp)>0
                for i in range(len(self.angles)):
                    Isim = np.reshape(xmippLib.Image(self.fnImgsSim[i]).getData(), (self.dim, self.dim, 1))
                    self.Xsim[i] = (Isim - np.mean(Isim)) / np.std(Isim)
                    self.ysim[i] = euler_to_rotation6d(self.angles[i], self.shifts[i])
                    if readExp:
                        Iexp = np.reshape(xmippLib.Image(self.fnImgsExp[i]).getData(), (self.dim, self.dim, 1))
                        self.Xexp[i] = (Iexp - np.mean(Iexp)) / np.std(Iexp)

                self.indexes = [i for i in range(len(self.fnImgsSim))]

                # K means of the projection directions
                kmeans = KMeans(n_clusters=10, random_state=0).fit(self.ysim[:,3:6])
                self.cluster_centers = normalize(kmeans.cluster_centers_)
                self.ylocation = np.dot(self.ysim[:,3:6], self.cluster_centers.T)

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

        def constructRESNET(inputLayer, outputSize, modelSize):
            x = conv_block(inputLayer, filters=64)
            x = Dropout(0.5)(x)
            x = conv_block(x, filters=128)
            x = Dropout(0.5)(x)
            if modelSize>=1:
                x = conv_block(x, filters=256)
                x = Dropout(0.5)(x)
            if modelSize>=2:
                x = conv_block(x, filters=512)
                x = Dropout(0.5)(x)
            if modelSize>=3:
                x = conv_block(x, filters=1024)
                x = Dropout(0.5)(x)
            x = GlobalAveragePooling2D()(x)
            x = Dense(64)(x)
            x = Dense(outputSize, name="output", activation="linear")(x)
            return x

        class ConcatenateZerosLayer(Layer):
            def __init__(self, num_zeros, **kwargs):
                super(ConcatenateZerosLayer, self).__init__(**kwargs)
                self.num_zeros = num_zeros

            def call(self, inputs, **kwargs):
                batch_size = tf.shape(inputs)[0]
                zeros = tf.zeros((batch_size, self.num_zeros), dtype=inputs.dtype)
                return tf.concat([zeros, inputs], axis=-1)

            def get_config(self):
                base_config = super(ConcatenateZerosLayer, self).get_config()
                base_config.update({'num_zeros': self.num_zeros})
                return {**base_config}

        def constructShiftModel(Xdim, modelSize):
            """RESNET architecture"""
            inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")
            x = constructRESNET(inputLayer, 2, modelSize)
            x = ConcatenateZerosLayer(6)(x)
            return Model(inputLayer, x)

        class ShiftImageLayer(Layer):
            def __init__(self, **kwargs):
                super(ShiftImageLayer, self).__init__(**kwargs)

            def call(self, inputs, **kwargs):
                images, shifts = inputs
                image_shape = tf.shape(images)
                batch_size, height, width, channels = image_shape[0], image_shape[1], image_shape[2], image_shape[3]

                # Create an individual transformation matrix for each image in the batch
                ones = tf.ones((batch_size,))
                zeros = tf.zeros((batch_size,))
                transforms = tf.stack([
                    ones, zeros, -shifts[:, -2],  # First row of the affine transformation matrix
                    zeros, ones, -shifts[:, -1],  # Second row of the affine transformation matrix
                    zeros, zeros  # Third row of the affine transformation matrix (unused in affine transform)
                ], axis=1)

                # Apply the transformation
                transformed_images = tf.raw_ops.ImageProjectiveTransformV3(
                    images=images,
                    transforms=transforms,
                    output_shape=[height, width],
                    fill_value=0,
                    interpolation='BILINEAR'
                )

                return transformed_images

            def get_config(self):
                base_config = super(ShiftImageLayer, self).get_config()
                return {**base_config}

        def constructAnglesModel(Xdim, modelSize, modelShift):
            """RESNET architecture"""
            inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")
            predicted_shift = modelShift(inputLayer)
            shifted_images = ShiftImageLayer()([inputLayer, predicted_shift])
            x = constructRESNET(shifted_images, 8, modelSize)
            x = Add()([x, predicted_shift])
            return Model(inputLayer, x)

        def constructLocationModel(Xdim, ydim, modelSize, modelShift):
            """RESNET architecture"""
            inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")
            predicted_shift = modelShift(inputLayer)
            shifted_images = ShiftImageLayer()([inputLayer, predicted_shift])
            x = constructRESNET(shifted_images, ydim, modelSize)
            return Model(inputLayer, x)

        def VAEsampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        def VAEencoder(inputLayer, latent_dim):
            x = Conv2D(32, 5, activation='relu', strides=2, padding='same')(inputLayer)
            x = Conv2D(64, 5, activation='relu', strides=2, padding='same')(x)
            x = Flatten()(x)
            x = Dense(16, activation='relu')(x)
            z_mean = Dense(latent_dim, name='z_mean')(x)
            z_log_var = Dense(latent_dim, name='z_log_var')(x)
            z = Lambda(VAEsampling)([z_mean, z_log_var])
            return Model(inputLayer, [z_mean, z_log_var, z])

        def VAEdecoder(latent_dim, Xdim):
            latent_inputs = Input(shape=(latent_dim,))
            x = Dense(16 * 16 * 64, activation='relu')(latent_inputs)
            x = Reshape((16, 16, 64))(x)
            x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
            x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
            x = Conv2DTranspose(1, 3, activation='linear', padding='same')(x)
            outputs = Resizing(Xdim, Xdim, interpolation='bilinear')(x)
            decoder = Model(latent_inputs, outputs)
            return decoder

        def vae_loss(inputs, outputs, z_mean, z_log_var):
            flattened_inputs = K.flatten(inputs)
            flattened_outputs = K.flatten(inputs)
            reconstruction_loss = mean_absolute_error(flattened_inputs, flattened_outputs)
            std_dev = K.std(flattened_inputs)
            return reconstruction_loss / (std_dev + K.epsilon())
            # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            # kl_loss = z_log_var - K.square(z_mean) - K.exp(z_log_var)
            # kl_loss = K.sum(kl_loss, axis=-1)
            # kl_loss *= -0.5
            # return K.mean(reconstruction_loss + kl_loss)

        def constructVAEModel(Xdim, modelShift, vae_loss):
            """VAE model"""
            inputLayer = Input(shape=(Xdim, Xdim, 1), name="input")
            predicted_shift = modelShift(inputLayer)
            shifted_images = ShiftImageLayer()([inputLayer, predicted_shift])
            latent_dim = 10
            encoder = VAEencoder(shifted_images, latent_dim)
            encoder.summary()
            decoder = VAEdecoder(latent_dim, Xdim)
            decoder.summary()
            output = decoder(encoder(inputLayer)[2])
            vae = Model(inputLayer, output)
            vae.add_loss(vae_loss(vae.inputs[0], vae.outputs[0],
                                  encoder.get_layer('z_mean').output,
                                  encoder.get_layer('z_log_var').output))
            vae.compile(optimizer='adam')
            return vae

        def get_labels(fnXmd):
            """Returns dimensions, images, angles and shifts values from images files"""
            md = xmippLib.MetaData(fnXmd)
            Xdim, _, _, _ = md.getImageSize()
            fnImg = md.getColumnValues(xmippLib.MDL_IMAGE)
            shiftX = md.getColumnValues(xmippLib.MDL_SHIFT_X)
            shiftY = md.getColumnValues(xmippLib.MDL_SHIFT_Y)
            rots = md.getColumnValues(xmippLib.MDL_ANGLE_ROT)
            tilts = md.getColumnValues(xmippLib.MDL_ANGLE_TILT)
            psis = md.getColumnValues(xmippLib.MDL_ANGLE_PSI)

            angles = [np.array((r,t,p)) for r, t, p in zip(rots, tilts, psis)]
            img_shift = [np.array((sX,sY)) for sX, sY in zip(shiftX, shiftY)]

            return Xdim, fnImg, angles, img_shift

        def custom_loss(y_true, y_pred):
            shift_true = y_true[:, 6:]  # Last 2 components
            shift_pred = y_pred[:, 6:]  # Last 2 components
            shift_error = tf.reduce_mean(tf.abs(shift_true - shift_pred))/Xdim
            error = shift_error

            if mode!=SHIFT_MODE:
                y_6d = y_pred[:, :6]
                y_6dtrue = y_true[:, :6]

                # Take care of symmetry
                num_rows = tf.shape(y_6d)[0]
                min_errors = tf.fill([num_rows], float('inf'))
                rotated_versions = tf.TensorArray(dtype=tf.float32, size=num_rows)
                for i, symmetry_matrix in enumerate(listSymmetryMatrices):
                    transformed = tf.matmul(y_6d, symmetry_matrix)
                    errors = tf.reduce_mean(tf.abs(y_6dtrue - transformed), axis=1)

                    # Update minimum errors and rotated versions
                    for j in tf.range(num_rows):
                        if errors[j] < min_errors[j]:
                            min_errors = tf.tensor_scatter_nd_update(min_errors, [[j]], [errors[j]])
                            rotated_versions = rotated_versions.write(j, transformed[j])
                y_6d = rotated_versions.stack()

                e3_true = y_6dtrue[:, 3:]  # Last 3 components
                e3_pred = y_6d[:, 3:]  # Last 3 components
                epsilon = 1e-7
                angle3 = tf.acos(tf.clip_by_value(tf.reduce_sum(e3_true * e3_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
                angular_error = tf.reduce_mean(tf.abs(angle3))
                Nangular=1

                if mode==FULL_MODE:
                    e2_true = y_6dtrue[:, :3]  # First 3 components
                    e2_pred = y_6d[:, :3]  # First 3 components

                    e1_true = tf.linalg.cross(e2_true, e3_true)

                    # Gram-Schmidt orthogonalization
                    e3_pred = tf.nn.l2_normalize(e3_pred, axis=-1)  # Normalize e3
                    projection = tf.reduce_sum(e2_pred * e3_pred, axis=-1, keepdims=True)
                    e2_pred = e2_pred - projection * e3_pred
                    e2_pred = tf.nn.l2_normalize(e2_pred, axis=-1)
                    e1_pred = tf.linalg.cross(e2_pred, e3_pred)
                    e1_pred = tf.nn.l2_normalize(e1_pred, axis=-1)

                    angle1 = tf.acos(tf.clip_by_value(tf.reduce_sum(e1_true * e1_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
                    angle2 = tf.acos(tf.clip_by_value(tf.reduce_sum(e2_true * e2_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))

                    angular_error += tf.reduce_mean(tf.abs(angle1))+tf.reduce_mean(tf.abs(angle2))
                    Nangular+=2

                error+=angular_error/Nangular

            return error

        class MySequence(keras.utils.all_utils.Sequence):
            def __init__(self, x_set, y_set, batch_size):
                self.x, self.y = x_set, y_set
                self.batch_size = batch_size

            def __len__(self):
                return int(np.ceil(len(self.x) / float(self.batch_size)))

            def __getitem__(self, idx):
                batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

                return batch_x, batch_y

        def trainModel(model, X, y, lossFunction, modeprec, saveModel=False):
            adam_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            sys.stdout.flush()
            model.compile(loss=lossFunction, optimizer=adam_opt)

            callbacks = []
            if saveModel:
                callbacks.append(save_best_model)

            generator = MySequence(X, y, batch_size)

            epoch = 0
            for i in range(maxEpochs):
                start_time = time()
                history = model.fit(generator, epochs=1, callbacks=callbacks, verbose=0)
                end_time = time()
                epoch += 1
                loss = history.history['loss'][-1]
                print("Epoch %d loss=%f trainingTime=%d" % (epoch, loss, int(end_time-start_time)), flush=True)
                if loss < modeprec:
                    break

        SL = xmippLib.SymList()
        listSymmetryMatrices = [tf.convert_to_tensor(np.kron(np.eye(2),np.transpose(np.array(R))), dtype=tf.float32)
                                for R in SL.getSymmetryMatrices(symmetry)]
        Xdim, fnImgsSim, angles, shifts = get_labels(fnXmdSim)
        if fnXmdExp!="":
            _, fnImgsExp, _, _ = get_labels(fnXmdExp)
        else:
            fnImgsExp=[]
        training_generator = DataGenerator(fnImgsSim, fnImgsExp, angles, shifts, batch_size, Xdim)

        for index in range(numModels):
            fnModelIndex = fnModel + "_angles"+str(index) + ".h5"
            if os.path.exists(fnModelIndex):
                continue
            save_best_model = ModelCheckpoint(fnModelIndex, monitor='loss', save_best_only=True)

            # Learn shift
            print("Learning shift")
            mode = SHIFT_MODE
            modeprec = 2 / 5 * precision  # 2/5 because the shift are 2 out of 5 numbers in the cost function
            modelShift = constructShiftModel(Xdim, modelSize)
            modelShift.summary()
            trainModel(modelShift, training_generator.Xsim, training_generator.ysim,
                       custom_loss, modeprec, saveModel=False)
            modelShift.trainable = False

            # Location model
            print("Learning approximate location")
            modelLocation = constructLocationModel(Xdim, training_generator.cluster_centers.shape[0],
                                                   modelSize, modelShift)
            modelLocation.summary()
            trainModel(modelLocation, training_generator.Xsim, training_generator.ylocation,
                       'mae', precision, saveModel=False)

            # VAE
            # vae = constructVAEModel(Xdim, modelShift, vae_loss)
            # trainModel(vae, training_generator.Xsim, vae_loss, 0.1, saveModel=False)

            # # Learn angles
            # mode = FULL_MODE
            # modeprec = precision
            # model = constructAnglesModel(Xdim, modelSize, modelShift)
            # trainModel(model, custom_loss, modeprec, saveModel=True)

if __name__ == '__main__':
    ScriptDeepGlobalAssignment().tryRun()