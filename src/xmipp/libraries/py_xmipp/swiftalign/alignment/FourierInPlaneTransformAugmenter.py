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

from typing import Iterable, Optional
import torch
import torchvision.transforms as T

import random

from .. import operators
from .. import math

class FourierInPlaneTransformAugmenter:
    def __init__(self,
                 max_psi: float,
                 max_shift: float,
                 flattener: operators.SpectraFlattener,
                 ctfs: Optional[torch.Tensor] = None,
                 weights: Optional[torch.Tensor] = None,
                 norm: Optional[str] = None,
                 interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR ) -> None:
        
        # Random affine transformer
        self.random_affine = T.RandomAffine(
            degrees=max_psi,
            translate=(max_shift, )*2,
            interpolation=interpolation
        )
        
        # Operations
        self.fourier = operators.FourierTransformer2D()
        self.flattener = flattener
        self.ctfs = ctfs
        self.weights = weights
        self.norm = norm
        
    def __call__(self, 
                 images: Iterable[torch.Tensor],
                 times: int = 1) -> torch.Tensor:
        
        if self.norm:
            raise NotImplementedError('Normalization is not implemented')
        
        images_affine = None
        images_fourier_transform = None
        images_band = None

        for batch in images:
            for _ in range(times):
                images_affine = self.random_affine(batch)
                images_fourier_transform = self.fourier(images_affine, out=images_fourier_transform)
                images_band = self.flattener(images_fourier_transform, out=images_band)

                if self.weights is not None:
                    images_band *= self.weights

                if self.ctfs is not None:
                    # Select a random CTF and apply it
                    ctf = self.ctfs[random.randrange(len(self.ctfs))]
                    images_band *= ctf

                yield math.flat_view_as_real(images_band)