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

from typing import Optional, Tuple

import math
import torch

def matrix_to_euler(m: torch.Tensor,
                    eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sy = torch.norm(m[...,2,0:2], dim=-1)
    
    gimbal_lock = sy < eps
    psi = -torch.where(
        gimbal_lock, 
        torch.atan2(-m[...,1,0], m[...,1,1]), 
        torch.atan2(m[...,2,1], m[...,2,0])
    )
    tilt = -torch.atan2(sy, m[...,2,2])
    rot = -torch.where(
        gimbal_lock, 
        torch.zeros(tuple(), dtype=m.dtype, device=m.device),
        torch.atan2(m[...,1,2], -m[...,0,2])
    )
    
    return rot, tilt, psi
