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

import torch
import math

def _get_nk(N: int):
    n = torch.arange(N)
    k = n.view(N, 1)
    return n, k    

def dct_ii_basis(N: int, norm: bool = True) -> torch.Tensor:
    n, k = _get_nk(N)
    
    result = (n + 0.5) * k
    result *= torch.pi / N
    result = torch.cos(result, out=result)
    
    # Normalize
    if norm:
        result *= math.sqrt(1/N)
        result[1:,:] *= math.sqrt(2)
    
    return result

def dct_iii_basis(N: int, norm: bool = True) -> torch.Tensor:
    n, k = _get_nk(N)
    
    # TODO avoid computing result[:,0] twice
    result = (k + 0.5) * n
    result *= torch.pi / N
    result = torch.cos(result, out=result)
    
    if norm:
        result[:,0] = 1 / math.sqrt(2)
        result *= math.sqrt(2/N)
    else:
        result[:,0] = 0.5
        
    
    return result