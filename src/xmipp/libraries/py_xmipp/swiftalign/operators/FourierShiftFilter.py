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

class FourierShiftFilter:
    def __init__(self,
                 dim: int,
                 shifts: torch.Tensor,
                 flattener: SpectraFlattener,
                 device: Optional[torch.device] = None):
        
        freq = self._get_freq(dim, flattener, device=device)
        angles = -torch.mm(shifts.to(device), freq)
        ONE = torch.tensor(1, dtype=angles.dtype, device=device)
        self._filters = torch.polar(ONE, angles)
        self._shifts = shifts
    
    def __call__(   self,
                    input: torch.Tensor,
                    index: int,
                    out: Optional[torch.Tensor] = None ):
        filter = self._filters[index,:]
        out = torch.mul(input, filter, out=out)
        return out
    
    def get_count(self) -> int:
        return self._filters.shape[0]
    
    def get_shift(self, index: int) -> torch.Tensor:
        return self._shifts[index]
    
    def _get_freq(  self,
                    dim: int,
                    flattener: SpectraFlattener,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        
        d = 0.5 / (dim*torch.pi)
        freq_x = torch.fft.rfftfreq(dim, d=d, device=device)
        freq_y = torch.fft.fftfreq(dim, d=d, device=device)
        grid = torch.stack(torch.meshgrid(freq_x, freq_y, indexing='xy'))
        return flattener(grid)
