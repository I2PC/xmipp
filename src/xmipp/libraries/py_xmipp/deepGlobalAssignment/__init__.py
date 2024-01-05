# ***************************************************************************
# *
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# ****************************************************************************
""" This is an example of a module which can be imported by any Xmipp script.
    It will be installed at build/pylib/xmippPyModules.
"""

import numpy as np

class Redundancy():
    def __init__(cls):
        l1 = lambda x: [(-1) ** int(bit) for bit in (bin(x)[2:]).zfill(6)]
        l2 = lambda x: [2+(-1) ** int(bit) for bit in (bin(x)[2:]).zfill(6)]
        cls.A = np.array([l1(x) for x in range(32)]+[l2(x) for x in range(32)])
        cls.A = cls.A/np.linalg.norm(cls.A,axis=1)[:,np.newaxis]

        cls.Apinv = np.linalg.pinv(cls.A)

    def make_redundant(cls, x):
        return np.matmul(cls.A, x)

    def make_nonredundant(cls, x):
        return np.matmul(cls.Apinv, np.reshape(x,(cls.A.shape[0])))

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.losses import mean_absolute_error
    from keras.models import Model
    from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
        Activation, GlobalAveragePooling2D, Add, Layer, Dropout, Lambda, Resizing, Flatten, Reshape, \
        Conv2DTranspose, Concatenate, Layer

    SHIFT_MODE = 0
    FULL_MODE = 1

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

    class VAESampling(Layer):
        """Sampling layer for VAE."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        def get_config(self):
            return super().get_config()

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


    def constructRESNET(inputLayer, outputSize, modelSize, drop=0.0):
        x = conv_block(inputLayer, filters=64)
        if drop>0:
            x = Dropout(drop)(x)
        x = conv_block(x, filters=128)
        if drop>0:
            x = Dropout(drop)(x)
        if modelSize >= 1:
            x = conv_block(x, filters=256)
            if drop > 0:
                x = Dropout(drop)(x)
        if modelSize >= 2:
            x = conv_block(x, filters=512)
            if drop > 0:
                x = Dropout(drop)(x)
        if modelSize >= 3:
            x = conv_block(x, filters=1024)
            if drop > 0:
                x = Dropout(drop)(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(64)(x)
        x = Dense(outputSize, activation="linear")(x)
        return x


    def constructShiftModel(Xdim, modelSize):
        """RESNET architecture"""
        inputLayer = Input(shape=(Xdim, Xdim, 1))
        x = constructRESNET(inputLayer, 2, modelSize)
        x = ConcatenateZerosLayer(6)(x)
        return Model(inputLayer, x)


    def constructAnglesModel(Xdim, modelSize, modelShift, modelLocation, vaeEncoder):
        """RESNET architecture"""
        inputLayer = Input(shape=(Xdim, Xdim, 1))
        if modelShift is not None:
            predicted_shift = modelShift(inputLayer)
            shifted_images = ShiftImageLayer()([inputLayer, predicted_shift])
            x = constructRESNET(shifted_images, 8, modelSize)
            infoLayers = [x, predicted_shift]
        else:
            x = constructRESNET(inputLayer, 8, modelSize)
            infoLayers = [x]

        if modelLocation is not None:
            location = modelLocation(inputLayer)
            x2 = Dense(64)(location)
            x2 = Dense(8)(x2)
            infoLayers.append(x2)

        if vaeEncoder is not None:
            vae_outputs = vaeEncoder(inputLayer)
            z_mean, z_log_var, vae = vae_outputs
            x3 = Dense(64)(vae)
            x3 = Dense(8)(x3)
            infoLayers.append(x3)

        if len(infoLayers) > 1:
            concat_layer = Concatenate(axis=-1)  # Change axis if needed
            x = concat_layer(infoLayers)
            x = Dense(256, activation='relu')(x)
            x = Dense(8, activation='linear')(x)

        return Model(inputLayer, x)


    def constructLocationModel(Xdim, ydim, modelSize, modelShift):
        """RESNET architecture"""
        inputLayer = Input(shape=(Xdim, Xdim, 1))
        if modelShift is not None:
            predicted_shift = modelShift(inputLayer)
            shifted_images = ShiftImageLayer()([inputLayer, predicted_shift])
            x = constructRESNET(shifted_images, ydim, modelSize)
        else:
            x = constructRESNET(inputLayer, ydim, modelSize)
        return Model(inputLayer, x)


    def VAEencoder(inputLayer, latent_dim):
        x = Conv2D(32, 5, activation='relu', strides=2, padding='same')(inputLayer)
        x = Conv2D(64, 5, activation='relu', strides=2, padding='same')(x)
        x = Conv2D(64, 5, activation='relu', strides=2, padding='same')(x)
        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z = VAESampling()([z_mean, z_log_var])
        return Model(inputLayer, [z_mean, z_log_var, z])


    def VAEdecoder(latent_dim, Xdim):
        latent_inputs = Input(shape=(latent_dim,))
        x = Dense(16 * 16 * 64, activation='relu')(latent_inputs)
        x = Reshape((16, 16, 64))(x)
        x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
        x = Conv2DTranspose(1, 3, activation='linear', padding='same')(x)
        outputs = Resizing(Xdim, Xdim, interpolation='bilinear')(x)
        decoder = Model(latent_inputs, outputs)
        return decoder

    def vae_loss(inputs, outputs, z_mean, z_log_var):
        flattened_inputs = K.flatten(inputs)
        flattened_outputs = K.flatten(outputs)
        reconstruction_loss = mean_absolute_error(flattened_inputs, flattened_outputs)
        std_dev = K.std(flattened_inputs)
        return reconstruction_loss / (std_dev + K.epsilon())
        # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        # kl_loss = z_log_var - K.square(z_mean) - K.exp(z_log_var)
        # kl_loss = K.sum(kl_loss, axis=-1)
        # kl_loss *= -0.5
        # return K.mean(reconstruction_loss + kl_loss)

    def constructVAEModel(Xdim, modelShift):
        """VAE model"""
        inputLayer = Input(shape=(Xdim, Xdim, 1))
        latent_dim = 10
        if modelShift is not None:
            predicted_shift = modelShift(inputLayer)
            shifted_images = ShiftImageLayer()([inputLayer, predicted_shift])
            encoder = VAEencoder(shifted_images, latent_dim)
        else:
            encoder = VAEencoder(inputLayer, latent_dim)
        encoder.summary()
        decoder = VAEdecoder(latent_dim, Xdim)
        decoder.summary()
        output = decoder(encoder(inputLayer)[2])
        vae = Model(inputLayer, output)
        vae.add_loss(vae_loss(vae.inputs[0], vae.outputs[0],
                              encoder.get_layer('z_mean').output,
                              encoder.get_layer('z_log_var').output))
        vae.compile(optimizer='adam')
        return vae, encoder


    class AngularLoss(tf.keras.losses.Loss):
        def __init__(self, listSymmetryMatricesNP, Xdim, **kwargs):
            super().__init__(**kwargs)
            self.listSymmetryMatricesNP = listSymmetryMatricesNP # Standard rotation matrices as numpy arrays
            self.Xdim = Xdim
            self.mode = None

            self.listSymmetryMatrices = [
                tf.convert_to_tensor(np.kron(np.eye(2), np.transpose(np.array(R))), dtype=tf.float32)
                for R in self.listSymmetryMatricesNP]

        def setMode(self, mode):
            self.mode = mode

        def call(self, y_true, y_pred):
            shift_true = y_true[:, 6:]  # Last 2 components
            shift_pred = y_pred[:, 6:]  # Last 2 components
            shift_error = tf.reduce_mean(tf.abs(shift_true - shift_pred)) / self.Xdim
            error = shift_error

            if self.mode != SHIFT_MODE:
                y_6d = y_pred[:, :6]
                y_6dtrue = y_true[:, :6]

                # Take care of symmetry
                num_rows = tf.shape(y_6d)[0]
                min_errors = tf.fill([num_rows], float('inf'))
                rotated_versions = tf.TensorArray(dtype=tf.float32, size=num_rows)
                for i, symmetry_matrix in enumerate(self.listSymmetryMatrices):
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
                angle3 = tf.acos(
                    tf.clip_by_value(tf.reduce_sum(e3_true * e3_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
                angular_error = tf.reduce_mean(tf.abs(angle3))
                Nangular = 1

                if self.mode == FULL_MODE:
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

                    angle1 = tf.acos(
                        tf.clip_by_value(tf.reduce_sum(e1_true * e1_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
                    angle2 = tf.acos(
                        tf.clip_by_value(tf.reduce_sum(e2_true * e2_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))

                    angular_error += tf.reduce_mean(tf.abs(angle1)) + tf.reduce_mean(tf.abs(angle2))
                    Nangular += 2

                error += angular_error / Nangular

            return error

        def get_config(self):
            config = super().get_config()
            config.update({"listSymmetryMatricesNP": self.listSymmetryMatricesNP, "mode": self.mode})
            return config

        @classmethod
        def from_config(cls, config):
            # Reconstruct the loss object, especially reconstruct the list of symmetry tensors
            return cls(**config)

    KERAS_INSTALLED = True
except:
    KERAS_INSTALLED = False