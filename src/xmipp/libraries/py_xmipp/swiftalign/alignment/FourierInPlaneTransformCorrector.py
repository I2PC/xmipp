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
import kornia

from .. import operators
from .. import math
from .. import metadata as md

def _create_rotation_matrix(angles: torch.Tensor,
                            out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    out = torch.empty((len(angles), 2, 2), out=out)
    
    # A day lost here :-)
    angles = torch.deg2rad(angles)
    
    out[:,0,0] = torch.cos(angles, out=out[:,0,0])
    out[:,1,0] = torch.sin(angles, out=out[:,0,1])
    out[:,0,1] = -out[:,1,0]
    out[:,1,1] = out[:,0,0]

    return out

def _create_affine_matrix(angles: torch.Tensor,
                          shifts: torch.Tensor,
                          centre: torch.Tensor,
                          out: Optional[torch.Tensor] = None ) -> torch.Tensor:

    batch_size = len(angles)

    if angles.shape != (batch_size, ):
        raise RuntimeError('angles has not the expected size')

    if shifts.shape != (batch_size, 2):
        raise RuntimeError('shifts has not the expected size')

    out = torch.empty((batch_size, 2, 3), out=out)

    # Compute the rotation matrix
    rotation_matrices = out[:,:2,:2]
    rotation_matrices = _create_rotation_matrix(
        angles=angles.to(out), 
        out=rotation_matrices
    )
    
    # Apply the shifts
    shifts = shifts - centre
    out[:,:,2,None] = torch.matmul(
        rotation_matrices, 
        shifts[...,None].to(out), 
        out=out[:,:,2,None]
    )
    out[:,:,2] += centre

    return out

def _affine_transform(images: torch.Tensor,
                      matrices: torch.Tensor,
                      interpolation: str = 'bilinear',
                      padding: str = 'zeros' ) -> torch.Tensor:
    images = images[:,None,:,:]
    result = kornia.geometry.transform.affine(
        images,
        matrix=matrices,
        mode=interpolation,
        padding_mode=padding
    )
    result = result[:,0,:,:]
    return result
    

class FourierInPlaneTransformCorrector:
    def __init__(self,
                 flattener: operators.SpectraFlattener,
                 weights: Optional[torch.Tensor] = None,
                 norm: Optional[str] = None,
                 interpolation: str = 'bilinear',
                 device: Optional[torch.device] = None ) -> None:

        # Operations
        self.fourier = operators.FourierTransformer2D()
        self.flattener = flattener
        self.weights = weights
        self.interpolation = interpolation
        self.norm = norm
        
        # Cached variables
        self.rotated_images = None
        self.rotated_fourier_transform = None
        self.rotated_band = None
        self.shifted_rotated_band = None
        
        # Device
        self.device = device
        
    def __call__(self, 
                 images: Iterable[Tuple[torch.Tensor, pd.DataFrame]]) -> torch.Tensor:
        
        if self.norm:
            raise NotImplementedError('Normalization is not implemented')

        transform_matrix = None
        fourier_transforms = None
        bands = None
        
        TRANSFORM_LABELS = [md.ANGLE_PSI, md.SHIFT_X, md.SHIFT_Y]
        
        for batch_images, batch_md in images:
            if len(batch_images) != len(batch_md):
                raise RuntimeError('Metadata and image batch sizes do not match')
            
            if self.device is not None:
                batch_images = batch_images.to(self.device, non_blocking=True)
            
            if all(map(batch_md.columns.__contains__, TRANSFORM_LABELS)):
                transformations = torch.from_numpy(batch_md[TRANSFORM_LABELS].to_numpy())
                angles = transformations[:,0]
                shifts = transformations[:,1:]
                centre = torch.tensor(batch_images.shape[-2:]) / 2

                transform_matrix = _create_affine_matrix(
                    angles=angles,
                    shifts=shifts,
                    centre=centre,
                    out=transform_matrix
                )
            
                batch_images = _affine_transform(
                    images=batch_images,
                    matrices=transform_matrix.to(batch_images, non_blocking=True),
                    interpolation=self.interpolation,
                    padding='zeros'
                )


            fourier_transforms = self.fourier(
                batch_images, 
                out=fourier_transforms
            )
            
            bands = self.flattener(
                fourier_transforms,
                out=bands
            )              
            
            if self.weights is not None:
                bands *= self.weights
                
            yield math.flat_view_as_real(bands)
            