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
import torchvision

class ImageAffineTransformer:
    def __init__(   self,
                    angles: torch.Tensor,
                    shifts: torch.Tensor,
                    device: Optional[torch.device] = None ):
        self._angles = angles
        self._shifts = shifts
        
    def __call__(   self, 
                    input: torch.Tensor,
                    angle_index: int,
                    shift_index: int,
                    out: Optional[torch.Tensor] ) -> torch.Tensor:
        
        out = torchvision.transforms.functional.affine(
            input,
            self.get_angle(angle_index),
            self.get_shift(shift_index),
            1.0,
            0.0,
            torchvision.transforms.InterpolationMode.BILINEAR
        )
        return out

    def get_count(self) -> int:
        return len(self._angles)
    
    def get_angle(self, index: int) -> float:
        return float(self._angles[index])
    
    def get_shift(self, index: int) -> torch.Tensor:
        return self._shifts[index]
        