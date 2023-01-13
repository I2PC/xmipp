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

class SpectraFlattener:
    def __init__(   self, 
                    mask: torch.Tensor,
                    device: Optional[torch.device]):
        self._mask = mask
        self._length = int(torch.count_nonzero(mask))
    
    def __call__(   self,
                    input: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # To avoid warnings
        if out is not None:
            out.resize_(0)
        
        out = torch.masked_select(input, self.get_mask(), out=out)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        return out
    
    def get_mask(self) -> torch.BoolTensor:
        return self._mask
    
    def get_length(self) -> int:
        return self._length
