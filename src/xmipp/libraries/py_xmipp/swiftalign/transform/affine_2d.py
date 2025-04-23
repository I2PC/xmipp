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
import kornia

def affine_2d(images: torch.Tensor,
              matrices: torch.Tensor,
              interpolation: str = 'bilinear',
              padding: str = 'zeros',
              out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    images = images[:,None,:,:]
    out = kornia.geometry.transform.warp_affine(
        images,
        M=matrices,
        dsize=images.shape[-2:],
        mode=interpolation,
        padding_mode=padding
    )
    out = out[:,0,:,:]
    return out
    