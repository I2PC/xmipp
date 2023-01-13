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

from transform import rotation_matrix_2d, combine_affine_2d, apply_affine

class ImageAffineTransformer:
    def __init__(   self,
                    angles: torch.Tensor,
                    shifts: torch.Tensor,
                    device: Optional[torch.device] = None ):
        self._angles = angles
        self._shifts = shifts
        
        self._rotation_matrices = rotation_matrix_2d(angles.to(device))
        self._shift_vectors = shifts.to(device)
        
    def __call__(   self, 
                    input: torch.Tensor,
                    angle_index: int,
                    shift_index: int,
                    affine_matrix: Optional[torch.Tensor],
                    out: Optional[torch.Tensor] ) -> torch.Tensor:
        
        # Ensemble the transform matrix
        rotation_matrix = self._rotation_matrices[angle_index]
        shift_vector = self._shift_vectors[shift_index]
        affine_matrix = combine_affine_2d(rotation_matrix, shift_vector, out=affine_matrix)
        
        # Perform the transform
        out=apply_affine(input, affine_matrix, out=out)
        
        return out
        

    def get_count(self) -> int:
        return len(self._angles)
    
    def get_angle(self, index: int) -> float:
        return float(self._angles[index])
    
    def get_shift(self, index: int) -> torch.Tensor:
        return self._shifts[index]
        