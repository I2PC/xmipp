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

from typing import Sequence, Optional
import torch

def _compute_padded_shape(shape: torch.Size, dim: Sequence[int], factor: int) -> torch.Size:
    shape_as_list = list(shape) # To mutate
    for d in dim:
        shape_as_list[d] *= factor
    return torch.Size(shape_as_list)

def zero_pad(x: torch.Tensor,
             dim: Sequence[int],
             factor: int,
             copy: bool = False,
             out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    if factor > 1 or copy:
        padded_size = _compute_padded_shape(x.shape, dim=dim, factor=factor)
        out = torch.zeros(
            size=padded_size,
            dtype=x.dtype,
            device=x.device,
            out=out
        )
        
        # Write
        write_slice = tuple(map(slice, x.shape))
        out[write_slice] = x
    
    else:
        out = x
        
    return out
    

