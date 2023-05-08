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

from typing import Sequence, Iterable, Optional, Tuple
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from .. import operators
from .. import math
from .. import fourier
from .. import metadata as md

def _compute_frequencies(flattener: operators.SpectraFlattener,
                         dim: Sequence[int],
                         device: Optional[torch.device] = None ) -> torch.Tensor:
    d = 1.0/(2*torch.pi)
    frequency_grid = fourier.rfftnfreq(dim, d=d, device=device)
    return flattener(frequency_grid)

class FourierInPlaneTransformCorrector:
    def __init__(self,
                 dim: Sequence[int],
                 flattener: operators.SpectraFlattener,
                 weights: Optional[torch.Tensor] = None,
                 norm: Optional[str] = None,
                 interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
                 device: Optional[torch.device] = None ) -> None:

        self.frequencies = _compute_frequencies(flattener, dim, device=device)
        
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

        rotated_images = None
        rotated_fourier_transforms = None
        rotated_bands = None
        shift_filters = None
        shifted_rotated_bands = None
        
        for batch_images, batch_md in images:
            if self.device is not None:
                batch_images = batch_images.to(self.device, non_blocking=True)

            if md.ANGLE_PSI in batch_md.columns:
                angles = batch_md[md.ANGLE_PSI]
                
                # Individually rotate. #FIXME very slow
                rotated_images = torch.empty_like(batch_images)
                for angle, image, rotated_image in zip(-angles, batch_images, rotated_images):
                    rotated_image[None] = F.rotate(image[None], angle=float(angle), interpolation=self.interpolation)
                    
            else:
                # No need for rotation
                rotated_images = batch_images
                
            rotated_fourier_transforms = self.fourier(
                rotated_images, 
                out=rotated_fourier_transforms
            )
            
            rotated_bands = self.flattener(
                rotated_fourier_transforms,
                out=rotated_bands
            )              
            
            if self.weights is not None:
                rotated_bands *= self.weights
            
            if md.SHIFT_X in batch_md.columns and md.SHIFT_Y in batch_md.columns:
                shifts = torch.tensor(
                    batch_md[[md.SHIFT_X, md.SHIFT_Y]].to_numpy(), 
                    dtype=self.frequencies.dtype,
                    device=self.frequencies.device
                )
                shift_filters = fourier.time_shift_filter(shifts, self.frequencies, out=shift_filters)
                shifted_rotated_bands = torch.mul(rotated_bands, shift_filters, out=shifted_rotated_bands)

            else:
                # No need for shift correction
                shifted_rotated_bands = rotated_bands

            yield math.flat_view_as_real(shifted_rotated_bands)
            