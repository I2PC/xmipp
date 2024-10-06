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

from typing import Sequence, Iterable, Optional
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import itertools

from .. import operators
from .. import math
from .. import fourier
from .InPlaneTransformBatch import InPlaneTransformBatch

def _compute_shift_filters( shifts: torch.Tensor,
                            flattener: operators.SpectraFlattener,
                            dim: Sequence[int],
                            device: Optional[torch.device] = None ) -> torch.Tensor:
    d = 1.0/(2*torch.pi)
    frequency_grid = fourier.rfftnfreq(dim, d=d, device=device)
    frequency_coefficients = flattener(frequency_grid)
    return fourier.time_shift_filter(shifts.to(frequency_coefficients.device), frequency_coefficients)

class FourierInPlaneTransformGenerator:
    def __init__(self,
                 dim: Sequence[int],
                 angles: torch.Tensor,
                 shifts: torch.Tensor,
                 flattener: operators.SpectraFlattener,
                 ctfs: Optional[torch.Tensor] = None,
                 weights: Optional[torch.Tensor] = None,
                 norm: Optional[str] = None,
                 interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
                 device: Optional[torch.device] = None ) -> None:

        # Transforms
        self.angles = angles
        self.shifts = shifts
        self.shift_filters = _compute_shift_filters(-shifts, flattener, dim, device) # Invert shifts
        
        # Operations
        self.fourier = operators.FourierTransformer2D()
        self.flattener = flattener
        self.ctfs = ctfs
        self.weights = weights
        self.interpolation = interpolation
        self.norm = norm
        
        # Device
        self.device = device
        
    def __call__(self, 
                 images: Iterable[torch.Tensor]) -> InPlaneTransformBatch:
        
        if self.norm:
            raise NotImplementedError('Normalization is not implemented')
        
        indices = None
        rotated_images = None
        rotated_fourier_transforms = None
        rotated_bands = None
        ctf_bands = None
        shifted_bands = None
        
        start = 0
        for batch in images:
            if self.device is not None:
                batch = batch.to(self.device, non_blocking=True)
            
            end = start + len(batch)
            indices = torch.arange(start=start, end=end, out=indices)
            
            for angle in self.angles:
                rotated_images = F.rotate(
                    batch,
                    angle=float(angle),
                    interpolation=self.interpolation
                )
                
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
                
                ctfs = self.ctfs if self.ctfs is not None else itertools.repeat(None, times=1)
                for ctf in ctfs:
                    if ctf is not None:
                        # Apply the CTF when provided
                        ctf_bands = torch.mul(
                            rotated_bands, 
                            ctf, 
                            out=ctf_bands
                        )
                    else:
                        # No CTF
                        ctf_bands = rotated_bands
                    
                    for shift, shift_filter in zip(self.shifts, self.shift_filters):
                        shifted_bands = torch.mul(
                            ctf_bands,
                            shift_filter,
                            out=shifted_bands
                        )
                        
                        yield InPlaneTransformBatch(
                            indices=indices,
                            vectors=math.flat_view_as_real(shifted_bands),
                            angle=float(angle),
                            shift=shift
                        )
        
            # Advance the counter for the next iteration
            start = end
            