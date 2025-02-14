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

class InPlaneTransformCorrector:
    def __init__(self,
                 flattener: operators.MaskFlattener,
                 interpolation: str = 'bilinear',
                 align_to_quaternion: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None ) -> None:

        # Operations
        self.flattener = flattener
        self.interpolation = interpolation
        self.norm = None
        self.align_to_quaternion = align_to_quaternion
        
        # Device
        self.device = device
        
    def __call__(self, 
                 images: Iterable[Tuple[torch.Tensor, pd.DataFrame]]) -> torch.Tensor:
        
        if self.norm:
            raise NotImplementedError('Normalization is not implemented')
    
        inverse_align_to_quaternion = transform.quaternion_conj(self.align_to_quaternion)
        
        angles = None
        quaternions = None
        relative_quaternions = None
        twist_quaternions = None
        mirror = None
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

            if inverse_align_to_quaternion is not None:
                rot_tilt = torch.as_tensor(batch_md[[md.ANGLE_ROT, md.ANGLE_TILT]].to_numpy(), dtype=torch.float32)
                rot_tilt.deg2rad_()
                
                # Compute the transformation relative to 
                # the average
                quaternions = transform.euler_to_quaternion(
                    rot_tilt[:,0],
                    rot_tilt[:,1],
                    angles,
                    out=quaternions
                )
                relative_quaternions = transform.quaternion_product(
                    inverse_align_to_quaternion[None],
                    quaternions, 
                    out=relative_quaternions
                )
                
                # Determine if it is a mirror by computing the ZZ component
                # of the transform matrix and checking its sign
                zz = 1 - 2*(torch.square(relative_quaternions[...,1]) + torch.square(relative_quaternions[...,2]))
                mirror = torch.lt(zz, 0.0, out=mirror)
                
                # Apply mirroring
                relative_quaternions[mirror] = -torch.roll(relative_quaternions[mirror], 2, dims=-1)
                
                # Compute the twist decomposition around Z
                twist_quaternions = transform.twist_decomposition(
                    relative_quaternions,
                    2, # Z
                    normalize_output=False, # Not required for atan2
                    out=twist_quaternions
                )
                
                
                # Obtain the angle of the twist
                torch.atan2(twist_quaternions[...,3], twist_quaternions[...,0], out=angles)
                angles *= 2
                
                
            transform_matrices_2d = transform.affine_matrix_2d(
                angles=angles,
                shifts=shifts,
                centre=centre,
                mirror=mirror if mirror is not None else False,
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