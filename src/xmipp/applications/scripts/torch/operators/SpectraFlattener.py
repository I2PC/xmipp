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
                    padded_length: Optional[int] = None,
                    device: Optional[torch.device] = None):
        self._mask = mask
        self._indices = self._calculate_indices(mask, device=device)
        self._length = padded_length or len(self._indices)
    
    def __call__(   self,
                    input: torch.Tensor,
                    out: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Ensure output is defined
        out = self._alloc_output(input, out=out)
        
        # Flatten in the same dims as the mask
        flat_input = torch.flatten(input, start_dim=-len(self.get_mask().shape))

        # Write to the output
        indices = self.get_indices()
        k = len(indices)
        out[...,:k] = flat_input[...,indices]
        out[...,k:] = 0
        
        return out
    
    def get_mask(self) -> torch.BoolTensor:
        return self._mask
    
    def get_indices(self) -> torch.IntTensor:
        return self._indices
    
    def get_length(self) -> int:
        return self._length

    def _calculate_indices(self,
                           mask: torch.BoolTensor, 
                           device: Optional[torch.device] = None ) -> torch.IntTensor:
        flat_mask = torch.flatten(mask)
        indices = torch.argwhere(flat_mask)[:,0]
        return indices.to(device)
    
    def _alloc_output(self,
                      input: torch.Tensor,
                      out: Optional[torch.Tensor] = None ) -> torch.Tensor:
        
        batch_shape = input.shape[:-2]
        output_shape = batch_shape + (self.get_length(), )
        if out is None:
            out = torch.empty(output_shape, device=input.device, dtype=input.dtype)
        else:
            out.resize_(output_shape)
        