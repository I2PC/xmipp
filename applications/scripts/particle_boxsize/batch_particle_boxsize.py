#!/usr/bin/env python
# **************************************************************************
# *
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************
#!/usr/bin/env python

import os
import xmipp3

import xmippLib as xmipp
import csv
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import VGG16
import numpy as np
import pickle as pickle
from sklearn.preprocessing import RobustScaler

class ImageScaler(object):
    def __init__(self, scaler):
        assert isinstance(x, RobustScaler), "Unexpected type for scaler" # Check that the scaler is of the expected type
        self.scaler = scaler

    def fit(self, X):
        self.scaler.fit(X.reshape((-1, 1)))

    def transform(self, X):
        input_shape = X.shape
        return self.scaler.transform(X.reshape((-1, 1))).reshape(*input_shape)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class ParticleSizeCnn(Sequential):
    def __init__(self, input_shape):
        super(ParticleSizeCnn, self).__init__()
        self._cnn_input_shape = input_shape

        self.add(Conv2D(3, kernel_size=(3, 3), activation='relu',
                        input_shape=input_shape, name="preproc_conv"))
        self.add(MaxPooling2D(pool_size=(2, 2), name="preproc_pooling"))
        output_shape = self.layers[-1].output_shape[1:]
        # Load pretrained VGG16
        self.conv_base = VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=output_shape)
        self.conv_base.trainable = False
        self.add(self.conv_base)

        # Custom top layer for regression
        self.add(Flatten())
        self.add(Dense(128, activation='relu'))
        self.add(Dense(1, activation=None))

    def set_trainable_last_conv_layer(self):
        self.conv_base.trainable = True
        set_trainable = False
        for layer in self.conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

    def save_weights(self, filepath, overwrite=True):
        # Set everything to trainable to save and load all weights
        self.set_trainable_last_conv_layer()
        assert self.conv_base == self.layers[2], "Model not consistent"  # TODO
        super(ParticleSizeCnn, self).save_weights(filepath, overwrite)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, reshape=False):
        self.set_trainable_last_conv_layer()
        super(ParticleSizeCnn, self).load_weights(filepath, by_name, skip_mismatch, reshape)
        self.conv_base = self.layers[2]
        self.conv_base.trainable = False


def get_image_config(img_size):
    config = {
        "img_rows": img_size,
        "img_cols": img_size,
        "data_format": K.image_data_format()
    }
    if config["data_format"] == 'channels_first':
        config["input_shape"] = (1, config["img_rows"], config["img_cols"])
    else:
        config["input_shape"] = (config["img_rows"], config["img_cols"], 1)
    return config


def crop(x, output_shape, shifts):
    assert len(shifts) == 2, "Invalid shift lengths"
    assert shifts[0] > 0 and shifts[1] > 0, "Negative shifts"

    nb_segments_i = np.ceil(x.shape[0] / float(output_shape[0]))
    nb_segments_j = np.ceil(x.shape[1] / float(output_shape[1]))
    if nb_segments_i > 1:
        delta_i_max = int(np.floor((x.shape[0] - output_shape[0]) / float(nb_segments_i - 1.0)))
        if shifts[0] > delta_i_max:
            # TODO: warning??
            shifts[0] = delta_i_max
    else:
        shifts[0] = 0
    if nb_segments_j > 1:
        delta_j_max = int(np.floor((x.shape[1] - output_shape[1]) / float(nb_segments_j - 1.0)))
        if shifts[1] > delta_j_max:
            # TODO: warning??
            shifts[1] = delta_j_max
    else:
        shifts[1] = 0
    start_i = 0
    end_i = start_i + output_shape[0]
    Xs = []
    for i in range(int(nb_segments_i)):
        start_j = 0
        end_j = start_j + output_shape[1]
        for j in range(int(nb_segments_j)):
            assert end_i <= x.shape[0], "end_i >= x_shape[0]"
            assert end_j <= x.shape[1], "end_j >= x_shape[1]"
            Xs.append(x[start_i:end_i, start_j:end_j])
            start_j += shifts[1]
            end_j = start_j + output_shape[1]
        start_i += shifts[0]
        end_i = start_i + output_shape[0]
    return Xs


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

class ScriptParticleBoxsize(xmipp3.XmippScript):
    def __init__(self):
        xmipp3.XmippScript.__init__(self)
        self.particle_boxsize = None
        
    def defineParams(self):
        self.addUsageLine('.')
        ## params
        self.addParamsLine('--img_size <IMG_SIZE> : Number of rows of the image. The image is assumed to have the same number ' +
                             'of rows and columns')
        self.addParamsLine('--weights <WEIGHS_FILE>   : Name of the weights file to be used for prediction')
        self.addParamsLine('--feature_scaler <FEATURE_SCALER>         : Path to feature scaler')
        self.addParamsLine('--micrographs <MICS_FILE>                 : Path to a TXT file that contains the complete paths of the micrographies that '
                                'should be processed')
        self.addParamsLine('--output <FILE>                          : Path to a TXT file that will contain the particle boxsize predicted for each '
                            'of the micrographs')
        ## examples
        ## TODO
        ## self.addExampleLine('   xmipp_particle_boxsize -p "Images/*xmp" -o all_images.sel -isstack')
            
    def run(self):
        # type: () -> object
        img_size = self.getParam('--img_size')
        weights = self.getParam('--weights')
        feature_scaler = self.getParam('--feature_scaler')
        micrographs = self.getParam('--micrographs')
        output_file = self.getParam('--output')
        
        config = get_image_config(int(img_size))

        # Default shifts for cropping the images. TODO: add as arguments?
        shift = np.max([config["img_rows"], 1])
        shifts = np.repeat(shift, 2)
        feature_scaler = pickle.load(open(feature_scaler, "rb"))

        model = ParticleSizeCnn(config["input_shape"])
        conv_base = model.conv_base
        model.load_weights(weights)

        predictions = []
        with open(micrographs, 'r') as micrographs_file:
            for micrograph_path in micrographs_file:
                micrograph_path = micrograph_path.rstrip('\n')
                img = xmipp.Image(micrograph_path).getData()
                img = feature_scaler.transform(img)
                Xs = crop(img, config["input_shape"], shifts)
                # reshape to match the number of channels and add an extra dimension
                # to account for the batch size
                Xs = [np.expand_dims(x.reshape(*config["input_shape"]), axis=0) 
                        for x in Xs
                ]
                micrograph_predictions= [float(model.predict(x)) for x in Xs]
                predictions.append(np.median(micrograph_predictions))
        self.particle_boxsize = int(np.round(np.median(predictions)))
        with open(output_file, 'w') as fp:
            fp.write(str(self.particle_boxsize) + '\n')
       
if __name__ == '__main__':
    ScriptParticleBoxsize().tryRun()
