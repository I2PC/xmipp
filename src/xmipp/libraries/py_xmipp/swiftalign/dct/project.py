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

from typing import Optional, Iterable
import torch

def project(x: torch.Tensor, 
            dim: int, 
            basis: torch.Tensor, 
            out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    if out is x:
        raise Exception('Aliasing between x and out is not supported')
    
    def t(x: torch.Tensor) -> torch.Tensor:
        return torch.transpose(x, dim, -1)
    
    # Transpose the input to have dim
    # on the last axis
    x = t(x)
    if out is not None:
        out = t(out)
    
    # Perform the projection
    out = torch.matmul(basis, x, out=out)
    
    # Undo the transposition
    out = t(out) # Return dim to its place

    return out

def project_nd(x: torch.Tensor, 
               dims: Iterable[int], 
               bases: Iterable[torch.Tensor],
               out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    temp = None
    for i, (dim, basis) in enumerate(zip(dims, bases)):
        if i == 0:
            # First iteration
            out = project(x, dim, basis, out=out)
        else:
            assert(out is not None)
            if temp is None:
                temp = out.clone()
            else:
                temp[...] = out
            
            out = project(temp, dim, basis, out=out)
            
    return out