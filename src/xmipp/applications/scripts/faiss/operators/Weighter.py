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

class Weighter:
    def __init__(self,
                 weights: torch.Tensor,
                 flattener: SpectraFlattener,
                 device: Optional[torch.device] = None):
        
        self._weights = flattener(weights[:,:flattener.get_mask().shape[-1]].to(device))
    
    def __call__(   self,
                    input: torch.Tensor,
                    out: Optional[torch.Tensor] = None ):
        torch.mul(input, self._weights, out=out)
        return out
    