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

from ..dct import dct_ii_basis, project_nd

from .Transformer2D import Transformer2D

class DctTransformer2D(Transformer2D):
    DIMS = (-1, -2) # Last two dimensions
    
    def __init__(self, dim: int, device: Optional[torch.device] = None) -> None:
        self._bases = (dct_ii_basis(dim).to(device), )*len(self.DIMS)
    
    def __call__(   self, 
                    input: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # To avoid warnings
        if out is not None:
            out.resize_(0)
            
        return project_nd(input, dims=self.DIMS, bases=self._bases, out=out)
    
    def has_complex_output(self) -> bool:
        return False