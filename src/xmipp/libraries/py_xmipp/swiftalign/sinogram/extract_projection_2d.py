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
import math

def _extract_projection_2d_nearest(sinogram: torch.Tensor,
                                   index: float ) -> torch.Tensor:
    index = round(index)
    index %= len(sinogram)
    assert(abs(index) < len(sinogram))
    return sinogram[index]
    
def _extract_projection_2d_linear(sinogram: torch.Tensor,
                                  index: float,
                                  out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    prev_index = math.floor(index)
    frac_index = index - prev_index
    prev_index %= len(sinogram)
    assert(abs(prev_index) < len(sinogram))
    next_index = (prev_index + 1) % len(sinogram)
    assert(abs(next_index) < len(sinogram))
    
    projections = sinogram[[prev_index, next_index], :]
    weights = [1.0-frac_index, frac_index]
    return torch.dot(projections, weights, out=out)

def extract_projection_2d(sinogram: torch.Tensor,
                          index: float,
                          interpolation: str = 'linear',
                          out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    if interpolation == 'nearest':
        return _extract_projection_2d_nearest(sinogram=sinogram, index=index)   
    elif interpolation == 'linear':
        return _extract_projection_2d_linear(sinogram=sinogram, index=index, out=out)   
    else:
        raise RuntimeError('Interpolation must be nearest or linear')

    
