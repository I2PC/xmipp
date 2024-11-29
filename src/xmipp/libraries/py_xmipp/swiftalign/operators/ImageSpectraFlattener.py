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

from typing import Optional

import torch

from .SpectraFlattener import SpectraFlattener
from .Transformer2D import Transformer2D
from .Weighter import Weighter

from .. import utils

class ImageSpectraFlattener:
    def __init__(self,
                 transform: Transformer2D,
                 flattener: SpectraFlattener,
                 weighter: Optional[Weighter] = None,
                 norm: Optional[str] = None ) -> None:
        
        self._transform = transform
        self._flattener = flattener
        self._weighter = weighter
        self._norm = norm
        
        self._transformed = None
        
    def __call__(self, 
                 batch: torch.Tensor, 
                 out: Optional[torch.Tensor] = None ) -> torch.Tensor:
        # Normalize image if requested
        if self._norm == 'image':
            batch = batch.clone()
            utils.normalize(batch, dim=(-2, -1))
        
        # Compute the fourier transform of the images and flatten and weighten it
        self._transformed = self._transform(batch, out=self._transformed)
        self._transformed_flat = self._flattener(self._transformed, out=self._transformed_flat)

        # Apply the weights
        if self._weighter is not None:
            self._transformed_flat = self._weighter(self._transformed_flat, out=self._transformed_flat)

        # Normalize complex numbers if requested
        if self._norm == 'complex':
            utils.complex_normalize(self._transformed_flat)

        # Elaborate the reference vectors
        out = utils.flat_view_as_real(self._transformed_flat)
                
        # Normalize reference vectors if requested
        if self._norm == 'vector':
            utils.l2_normalize(out, dim=-1)

        return out