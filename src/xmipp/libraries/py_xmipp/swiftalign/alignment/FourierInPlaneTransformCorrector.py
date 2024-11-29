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
from .. import math
from .. import metadata as md

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

        angles = None
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
                transformations = torch.as_tensor(batch_md[TRANSFORM_LABELS].to_numpy(), dtype=torch.float32)
                angles = torch.deg2rad(transformations[:,0], out=angles)
                shifts = transformations[:,1:]
                centre = torch.tensor(batch_images.shape[-2:]) / 2

                transform_matrix = transform.affine_matrix_2d(
                    angles=angles,
                    shifts=shifts,
                    centre=centre,
                    shift_first=True,
                    out=transform_matrix
                )
            
                batch_images = transform.affine_2d(
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
            