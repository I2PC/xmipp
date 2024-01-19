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

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.losses import mean_absolute_error
    from keras.models import Model
    from keras.layers import Input, Conv2D, BatchNormalization, Dense, concatenate, \
        Activation, GlobalAveragePooling2D, Add, Layer, Dropout, Lambda, Resizing, Flatten, Reshape, \
        Conv2DTranspose, Concatenate, Layer, MaxPooling2D

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
                ones, zeros, shifts[:, -2],  # First row of the affine transformation matrix
                zeros, ones, shifts[:, -1],  # Second row of the affine transformation matrix
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

    class Angles2VectorLayer(Layer):
        def __init__(self, **kwargs):
            super(Angles2VectorLayer, self).__init__(**kwargs)

        def call(self, inputs):
            rot = inputs[:, 0]
            tilt = inputs[:, 1]
            transformed_output = tf.stack([
                -tf.cos(rot) * tf.sin(tilt),
                tf.sin(rot) * tf.sin(tilt),
                tf.cos(tilt)
            ], axis=1)
            return transformed_output

        def get_config(self):
            config = super(Angles2VectorLayer, self).get_config()
            return config

    def constructNN(inputLayer, outputSize, kernelSize, pooling=True):
        x = Conv2D(32, kernelSize, padding='same', activation='relu')(inputLayer)
        if pooling:
            x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, kernelSize, padding='same', activation='relu')(x)
        if pooling:
            x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, kernelSize, padding='same', activation='relu')(x)
        if pooling:
            x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(outputSize, activation="linear")(x)
        return x

    def constructNN2(inputLayer, outputSize, kernelSize, pooling=True):
        x = Conv2D(256, kernelSize, padding='same', strides=2, activation='relu')(inputLayer)
        x = Conv2D(16, 1, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, kernelSize, padding='same', strides=2, activation='relu')(x)
        x = Conv2D(16, 1, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        x = Dense(outputSize, activation="linear")(x)
        return x

    def constructShiftModel(Xdim):
        inputLayer = Input(shape=(Xdim, Xdim, 1))
        x = constructNN(inputLayer, 2, kernelSize=5, pooling=True)
        x = ConcatenateZerosLayer(6)(x)
        return Model(inputLayer, x)

    def connectShiftModel(input_tensor, modelShift):
        x = input_tensor
        for layer in modelShift.layers[1:]:
            layer.trainable = False
            x = layer(x)
        predicted_shift = x
        shifted_images = ShiftImageLayer()([input_tensor, predicted_shift])
        return shifted_images, predicted_shift

    import scipy.stats as st
    def gaussian_kernel(size: int, std: float):
        interval = (2 * std + 1.) / size
        x = np.linspace(-std - interval / 2., std + interval / 2., size)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def create_blur_filters(num_filters, max_std, filter_size):
        std_intervals = np.linspace(0.1, max_std, num_filters)
        filters = []
        for std in std_intervals:
            kernel = gaussian_kernel(filter_size, std)
            kernel = np.expand_dims(kernel, axis=-1)
            filters.append(kernel)
        filters = np.stack(filters, axis=-1)
        return tf.constant(filters, dtype=tf.float32)

    def constructAnglesModel(Xdim, modelShift):
        input_tensor = Input(shape=(Xdim, Xdim, 1))
        shifted_images, predicted_shift = connectShiftModel(input_tensor, modelShift)

        filters = create_blur_filters(16, 16, int(Xdim/2))
        blurred_images = tf.nn.depthwise_conv2d(shifted_images, filters, strides=[1, 1, 1, 1], padding='SAME')
        x = constructNN2(blurred_images, 8, kernelSize=11, pooling=True)
        infoLayers = [x, predicted_shift]

        concat_layer = Concatenate(axis=-1)  # Change axis if needed
        x = concat_layer(infoLayers)
        x = Dense(256, activation='relu')(x)
        x2 = Dense(2, activation='linear')(x)
        x2 = Angles2VectorLayer()(x2)
        x3 = Dense(2, activation='linear')(x)
        x3 = Angles2VectorLayer()(x3)
        deltaShift = Dense(2, activation='linear')(x)
        x = Concatenate(axis=-1)([x2, x3, deltaShift])
        return Model(input_tensor, x)

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
            shift_error = tf.reduce_mean(tf.abs(shift_true - shift_pred))
            error = shift_error
            # tf.print("y_true", y_true[0,])
            # tf.print("y_pred", y_pred[0,])

            if self.mode != SHIFT_MODE:
                y_6d = y_pred[:, :6]
                y_6dtrue = y_true[:, :6]

                # Take care of symmetry
                # num_rows = tf.shape(y_6d)[0]
                # min_errors = tf.fill([num_rows], float('inf'))
                # rotated_versions = tf.TensorArray(dtype=tf.float32, size=num_rows)
                # for i, symmetry_matrix in enumerate(self.listSymmetryMatrices):
                #     transformed = tf.matmul(y_6d, symmetry_matrix)
                #     errors = tf.reduce_mean(tf.abs(y_6dtrue - transformed), axis=1)
                #
                #     # Update minimum errors and rotated versions
                #     for j in tf.range(num_rows):
                #         if errors[j] < min_errors[j]:
                #             min_errors = tf.tensor_scatter_nd_update(min_errors, [[j]], [errors[j]])
                #             rotated_versions = rotated_versions.write(j, transformed[j])
                # y_6d = rotated_versions.stack()
                epsilon = 1e-7

                e3_true = y_6dtrue[:, 3:]  # Last 3 components
                # tf.print("e3_true", e3_true[0,])
                e3_pred = y_6d[:, 3:]  # Last 3 components
                # tf.print("e3_pred", e3_pred[0,])
                angle3 = tf.acos(
                    tf.clip_by_value(tf.reduce_sum(e3_true * e3_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
                angular_error = tf.reduce_mean(tf.abs(angle3))
                vector_error = tf.reduce_mean(tf.abs(e3_true-e3_pred))
                # tf.print("angle3", angle3[0,])
                Nangular = 1

                if self.mode == FULL_MODE:
                    e2_true = y_6dtrue[:, :3]  # First 3 components
                    e2_pred = y_6d[:, :3]  # First 3 components
                    vector_error += tf.reduce_mean(tf.abs(e2_true - e2_pred))

                    # tf.print("e2_true",e2_true[0,])
                    # tf.print("e2_pred",e2_pred[0,])

                    e1_true = tf.linalg.cross(e2_true, e3_true)
                    e1_pred = tf.linalg.cross(e2_pred, e3_pred)
                    vector_error += tf.reduce_mean(tf.abs(e1_true - e1_pred))

                    angle1 = tf.acos(
                        tf.clip_by_value(tf.reduce_sum(e1_true * e1_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
                    angle2 = tf.acos(
                        tf.clip_by_value(tf.reduce_sum(e2_true * e2_pred, axis=-1), -1.0 + epsilon, 1.0 - epsilon))
                    # tf.print("angle2",angle2[0,])
                    # tf.print("angle1",angle1[0,])

                    angular_error += tf.reduce_mean(tf.abs(angle1)) + tf.reduce_mean(tf.abs(angle2))
                    Nangular += 2

                error += 0.5*(angular_error / Nangular  + vector_error/Nangular)* self.Xdim/2
                # error += vector_error/Nangular * self.Xdim/2

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