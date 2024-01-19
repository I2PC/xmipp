#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:     Carlos Oscar Sorzano
 *
 * CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
"""

import numpy as np
import os
import tensorflow as tf
from xmipp_base import *
import xmippLib

class ScriptCropResizeGPU(XmippScript):

    def __init__(self):
        XmippScript.__init__(self)
        
    def defineParams(self):
        self.addUsageLine('Crop and resize a set of 2D images. The cropping to a given size is centered. '
                          'Then, after cropping the image is rescaled to the final size.')
        ## params
        self.addParamsLine(' -i <metadata>     : xmd file with the list of images')
        self.addParamsLine(' --oroot <root>    : Rootname for the output (.xmd and .mrcs)')
        self.addParamsLine(' --cropSize <dim>  : Crop size, 1st operation')
        self.addParamsLine(' --finalSize <dim> : Final size, 2nd operation')
        self.addParamsLine('[--block <N=2048>] : Size of the block of images to be processed simultaneously')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')

    def readBlock(self, b):
        i0=b*self.blockSize
        iF=(b+1)*self.blockSize if b<self.Nblocks else self.Nimgs
        image_list = []
        for i in range(i0, iF):
            image_data = xmippLib.Image(self.fnImgs[i]).getData()
            image_reshaped = np.reshape(image_data, (self.originalSize, self.originalSize, 1)).astype(np.float32)
            image_tensor = tf.convert_to_tensor(image_reshaped)
            image_list.append(image_tensor)
        self.image_tensor = tf.stack(image_list, axis=0)

    def writeBlock(self, b, transformed_images):
        i0=b*self.blockSize
        iF=(b+1)*self.blockSize if b<self.Nblocks else self.Nimgs
        j = 0
        I = xmippLib.Image()
        for i in range(i0, iF):
            I.setData(transformed_images[j,:,:,0].numpy())
            fnImgi = "%d@%s"%(i+1,self.fnStack)
            I.write(fnImgi)
            self.fnImgs[i]=fnImgi
            j+=1

    def processBlock(self, b):
        self.readBlock(b)
        transformed_images = tf.raw_ops.ImageProjectiveTransformV3(
            images=self.image_tensor,
            transforms=self.transformation_matrix,
            output_shape=[self.finalSize, self.finalSize],
            fill_mode='REFLECT',  # or 'constant', 'nearest', 'mirror', 'wrap'
            fill_value=0,
            interpolation='BILINEAR'  # or 'nearest'
        )
        self.writeBlock(b, transformed_images)

    def run(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        self.fnIn = self.getParam('-i')
        self.fnRoot = self.getParam('--oroot')
        self.cropSize = int(self.getParam('--cropSize'))
        self.finalSize = int(self.getParam('--finalSize'))
        self.blockSize = int(self.getParam('--block'))
        gpuId = self.getParam("--gpu")

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId


        self.mdIn = xmippLib.MetaData(self.fnIn)
        self.Nimgs=self.mdIn.size()
        self.originalSize, _, _, _ = self.mdIn.getImageSize()
        self.fnImgs = self.mdIn.getColumnValues(xmippLib.MDL_IMAGE)

        self.fnStack = self.fnRoot+".mrcs"
        xmippLib.createEmptyFile(self.fnStack, self.finalSize, self.finalSize, 1, self.Nimgs)

        # Compute the transformation matrix for center cropping and resizing
        # This is a 2x3 affine transformation matrix for image transformations
        scale = float(self.cropSize)/float(self.finalSize)
        crop_start = (self.originalSize - self.cropSize) // 2
        self.transformation_matrix = tf.constant([
            [scale, 0, crop_start],  # X-axis parameters
            [0, scale, crop_start]  # Y-axis parameters
        ], dtype=tf.float32)
        self.transformation_matrix = tf.reshape(self.transformation_matrix, [-1])
        self.transformation_matrix = tf.concat([self.transformation_matrix, [0, 0]], axis=0)
        self.transformation_matrix = self.transformation_matrix[None, :]

        self.Nblocks = self.Nimgs//self.blockSize
        for b in range(self.Nblocks):
            self.processBlock(b)
        if self.Nblocks*self.blockSize<self.Nimgs:
            self.processBlock(self.Nblocks)
        self.mdIn.setColumnValues(xmippLib.MDL_IMAGE, self.fnImgs)
        self.mdIn.write(self.fnRoot+".xmd")

if __name__ == '__main__':
    ScriptCropResizeGPU().tryRun()
