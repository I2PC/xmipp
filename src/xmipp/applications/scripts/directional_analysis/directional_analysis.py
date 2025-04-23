#!/usr/bin/env python3

# ***************************************************************************
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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
# *  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************/

import argparse
import os.path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

import xmippPyModules.swiftalign.image as image
import xmippPyModules.swiftalign.metadata as md
import xmippPyModules.swiftalign.operators as operators
import xmippPyModules.swiftalign.transform as transform
from xmippPyModules.swiftalign.pca import PCA

class DirectionalTransformer:
    def __init__(self, 
                 device: torch.device,
                 interpolation: str = 'bilinear',
                 padding: str = 'zeros' ):
        self.device = device
        self.transfer_stream = torch.cuda.Stream(device)
        self.compute_stream = torch.cuda.Stream(device)
        self.transfer_event = torch.cuda.Event()
        self.interpolation = interpolation
        self.padding = padding
        self.flattener = None
        self.direction = torch.tensor([0, 0, 1], dtype=torch.float32)
    
    def setup(self,
              direction_angles: torch.Tensor,
              mask: torch.Tensor ):
        
        with torch.cuda.stream(self.transfer_stream):
            mask = mask.to(self.device, non_blocking=True)
            self.transfer_event.record()
        
        direction_angles = torch.deg2rad(direction_angles)
        direction_quaternion = transform.euler_to_quaternion(
            direction_angles[...,0], 
            direction_angles[...,1],
            direction_angles[...,2]
        )
        self.inv_direction_quaternion = transform.quaternion_conj(direction_quaternion)
        
        with torch.cuda.stream(self.compute_stream):
            self.transfer_event.wait()
            self.flattener = operators.MaskFlattener(mask)
    
    def transform(self,
                  images: torch.Tensor,
                  angles: torch.Tensor,
                  shifts: torch.Tensor,
                  out: torch.Tensor ):
        
        with torch.cuda.stream(self.transfer_stream):
            images = images.to(self.device, non_blocking=True)
            
        centre = torch.tensor(images.shape[-2:]) / 2
        angles = torch.deg2rad(angles)
        quaternions = transform.euler_to_quaternion(
            angles[...,0], 
            angles[...,1],
            angles[...,2]
        )                
        
        quaternions = transform.quaternion_product(
            self.inv_direction_quaternion, 
            quaternions
        )
        matrices_3d = transform.quaternion_to_matrix(quaternions)
        matrices = transform.align_inplane(
            matrices_3d=matrices_3d,
            shifts=shifts,
            centre=centre,
        )

        with torch.cuda.stream(self.transfer_stream):
            matrices = matrices.to(images, non_blocking=True)
            self.transfer_event.record()
        
        with torch.cuda.stream(self.compute_stream):
            self.transfer_event.wait()
            images = transform.affine_2d(
                images=images, 
                matrices=matrices,
                interpolation=self.interpolation,
                padding=self.padding
            )
            
            out[...] = self.flattener(images)
        
        return out

def process_direction(direction_md: pd.DataFrame,
                      direction_angles: np.ndarray,
                      mask: np.ndarray,
                      transformer: DirectionalTransformer,
                      dimred,
                      batch_size: int = 256):
    images_paths = list(map(image.parse_path, direction_md[md.IMAGE]))
    images_dataset = image.torch_utils.Dataset(images_paths)
    angles = direction_md[[md.ANGLE_ROT, md.ANGLE_TILT, md.ANGLE_PSI]].to_numpy()
    shifts = direction_md[[md.SHIFT_X, md.SHIFT_Y]].to_numpy()

    start = 0
    n_elements = len(direction_md)
    
    data = torch.empty((n_elements, np.count_nonzero(mask)), device=transformer.device)
    transformer.setup(torch.as_tensor(direction_angles, dtype=torch.float32), torch.as_tensor(mask))

    while start < n_elements:
        end = min(start + batch_size, n_elements)
        
        images = torch.as_tensor(images_dataset[start:end]).pin_memory()
        transformer.transform(
            images=images,
            angles=torch.as_tensor(angles[start:end], dtype=torch.float32),
            shifts=torch.as_tensor(shifts[start:end], dtype=torch.float32),
            out=data[start:end]
        )
        
        start = end
    
    mean = data.mean(dim=0, keepdim=True)
    data -= mean
    data = dimred.fit_transform(data, mean=mean, mean_centered=True)
    data = data.to('cpu', non_blocking=True)
    
    return data, dimred

def run(directional_md_path: str, 
        output_root: str, 
        batch_size: int,
        device_names: list,
        n_components: int ):

    if device_names:
        devices = list(map(torch.device, device_names))
    else:
        devices = [torch.device('cpu')]
    
    transformer = DirectionalTransformer(device=devices[0])
        
    directional_md = md.read(directional_md_path)
    for i, row in directional_md.iterrows():
        direction_md_path = row[md.SELFILE]
        mask_path = image.parse_path(row[md.MASK])
        direction_angles = row[[md.ANGLE_ROT, md.ANGLE_TILT, md.ANGLE_PSI]]
        direction_md = md.read(direction_md_path)
        mask = image.read(mask_path.filename)
        if mask_path.position_in_stack is not None:
            mask = mask[mask_path.position_in_stack-1]
        mask = mask.astype(bool)
        
        pca = PCA(n_components=n_components)
        data, pca = process_direction(
            direction_md=direction_md,
            direction_angles=direction_angles,
            mask=mask,
            transformer=transformer,
            dimred=pca,
            batch_size=batch_size
        )
        
        mean = pca.mean_.squeeze().to('cpu').numpy()
        average_image = np.zeros_like(mask, dtype=mean.dtype)
        average_image[mask] = mean
        image.write(average_image, os.path.join(output_root, f'{i:06d}_average.mrc'))

        components = pca.components_.to('cpu').numpy()
        eigen_images = np.zeros((len(components), ) + mask.shape, dtype=components.dtype)
        eigen_images[:,mask] = components
        image.write(eigen_images, os.path.join(output_root, f'{i:06d}_eigen.mrcs'), image_stack=True)
        
        np.save(os.path.join(output_root, f'{i:06d}_data.npy'), data)
        
if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(prog = 'Directionally analyze images')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('-k', default=8, type=int)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--device', nargs='*')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        directional_md_path= args.i,
        output_root = args.o,
        batch_size = args.batch,
        device_names = args.device,
        n_components=args.k
    )
