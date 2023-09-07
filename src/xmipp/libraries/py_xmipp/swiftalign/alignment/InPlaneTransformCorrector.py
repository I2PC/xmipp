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

from typing import Iterable, Optional, Tuple
import pandas as pd
import torch

from .. import operators
from .. import transform
from .. import metadata as md

import matplotlib.pyplot as plt

class InPlaneTransformCorrector:
    def __init__(self,
                 flattener: operators.MaskFlattener,
                 interpolation: str = 'bilinear',
                 align_to_matrix: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None ) -> None:

        # Operations
        self.flattener = flattener
        self.interpolation = interpolation
        self.norm = None
        self.align_to_matrix = align_to_matrix
        
        # Device
        self.device = device
        
    def __call__(self, 
                 images: Iterable[Tuple[torch.Tensor, pd.DataFrame]]) -> torch.Tensor:
        
        if self.norm:
            raise NotImplementedError('Normalization is not implemented')

        angles = None
        transform_matrices_3d = None
        transform_matrices_2d = None
        transformed_images = None
        flattened_images = None
        
        TRANSFORM_LABELS = [md.ANGLE_PSI, md.SHIFT_X, md.SHIFT_Y]
        
        for batch_images, batch_md in images:
            if len(batch_images) != len(batch_md):
                raise RuntimeError('Metadata and image batch sizes do not match')
            
            if self.device is not None:
                batch_images = batch_images.to(self.device, non_blocking=True)
            
            # Read necessary metadata
            transformations = torch.as_tensor(batch_md[TRANSFORM_LABELS].to_numpy(), dtype=torch.float32)
            angles = torch.deg2rad(transformations[:,0], out=angles)
            shifts = transformations[:,1:]
            centre = torch.tensor(batch_images.shape[-2:]) / 2

            if self.align_to_matrix is not None:
                rot_tilt = torch.as_tensor(batch_md[[md.ANGLE_ROT, md.ANGLE_TILT]].to_numpy(), dtype=torch.float32)
                rot_tilt.deg2rad_()
                
                # Compute the transformation relative to 
                # the average
                transform_matrices_3d = transform.euler_to_matrix(
                    rot_tilt[:,0],
                    rot_tilt[:,1],
                    angles,
                    out=transform_matrices_3d
                )
                relative_transform = torch.matmul(
                    self.align_to_matrix.mT, 
                    transform_matrices_3d,
                )
                
                # Compute the in plane rotation difference relative to the 
                # average
                s = relative_transform[...,1,0]
                c = relative_transform[...,1,1]
                angles = torch.atan2(s, c, out=angles)

            transform_matrices_2d = transform.affine_matrix_2d(
                angles=angles,
                shifts=shifts,
                centre=centre,
                shift_first=True,
                out=transform_matrices_2d
            )

            transformed_images = transform.affine_2d(
                images=batch_images,
                matrices=transform_matrices_2d.to(batch_images, non_blocking=True),
                interpolation=self.interpolation,
                padding='zeros',
                out=transformed_images
            )
            
            flattened_images = self.flattener(
                transformed_images,
                out=flattened_images
            )              
            
            yield flattened_images