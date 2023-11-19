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
import torchvision.transforms as T

class ImageRotator:
    def __init__(   self,
                    angles: torch.Tensor,
                    device: Optional[torch.device] = None ):
        self._angles = angles
        
    def __call__(   self, 
                    input: torch.Tensor,
                    index: int,
                    out: Optional[torch.Tensor] ) -> torch.Tensor:
        
        # TODO use the matrix
        out = T.functional.rotate(
            input,
            self.get_angle(index),
            T.InterpolationMode.BILINEAR,
        )
        return out

    def get_count(self) -> int:
        return len(self._angles)
    
    def get_angle(self, index: int) -> float:
        return float(self._angles[index])
        