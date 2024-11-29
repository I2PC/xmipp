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

from typing import Optional, Sequence, Iterable, Callable
import torch

from .basis import dct_ii_basis, dct_iii_basis
from .project import project_nd

def bases_generator(shape: Sequence[int], 
                    dims: Iterable[int],
                    func: Callable[[int], torch.Tensor]) -> Iterable[torch.Tensor]:
    sizes = map(shape.__getitem__, dims)
    bases = map(func, sizes)
    return bases
    
def dct(x: torch.Tensor, 
        dims: Iterable[int], 
        out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    bases = bases_generator(x.shape, dims, dct_ii_basis)
    return project_nd(x, dims, bases, out=out)

def idct(x: torch.Tensor, 
         dims: Iterable[int], 
         out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    bases = bases_generator(x.shape, dims, dct_iii_basis)
    return project_nd(x, dims, bases, out=out)