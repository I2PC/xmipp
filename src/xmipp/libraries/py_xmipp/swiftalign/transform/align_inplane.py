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

from .affine_matrix_2d import make_affine_matrix_2d

def align_inplane(matrices_3d: torch.Tensor,
                  shifts: torch.Tensor,
                  centre: torch.Tensor,
                  apply_streching: bool = False,
                  out: Optional[torch.Tensor] = None ) -> torch.Tensor:

    if apply_streching:
        inverse_matrices_3d = torch.linalg.inv(matrices_3d)
        matrices_2d = torch.linalg.inv(inverse_matrices_3d[..., :2,:2])
    else:
        u, _, vh = torch.linalg.svd(matrices_3d[..., :2,:2])
        matrices_2d = u @ vh
    
    return make_affine_matrix_2d(
        m22=matrices_2d,
        shifts=shifts,
        centre=centre,
        shift_first=True,
        out=out
    )
