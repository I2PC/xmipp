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

from typing import Optional, Sequence
import torch

from .SpectraFlattener import SpectraFlattener

class FourierLowPassFlattener(SpectraFlattener):
    def __init__(   self, 
                    dim: int, 
                    cutoff: float, 
                    device: Optional[torch.device] = None ):
        SpectraFlattener.__init__(
            self, 
            self._compute_mask(dim, cutoff), 
            device=device
        )
    
    def _compute_mask(  self, 
                        dim: int, 
                        cutoff: float ) -> torch.Tensor:
        
        # Compute the frequency grid
        freq_x = torch.fft.rfftfreq(dim)
        freq_y = torch.fft.fftfreq(dim)[...,None]
        freq2 = freq_x**2 + freq_y**2
        
        # Compute the mask
        cutoff2 = cutoff ** 2
        mask = freq2.less_equal(cutoff2)
        return mask
