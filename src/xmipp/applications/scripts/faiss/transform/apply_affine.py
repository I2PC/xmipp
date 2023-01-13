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
import math
import torch.nn.functional as F

def apply_affine(images: torch.Tensor,
                 matrix: torch.Tensor,
                 out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    #TODO implement this correctly
    raise NotImplementedError('This function has to be completed')
    batch_shape = images.shape[:-2]
    image_shape = images.shape[-2:]
    n_batch = math.prod(batch_shape)
 
    flattened_shape = (n_batch, 1) + image_shape
    flattened_images = images.view(flattened_shape)
    
    matrix = matrix.expand((n_batch, ) + matrix.shape)
    grid = F.affine_grid(matrix, flattened_images, align_corners=False)
    
    out = F.grid_sample(flattened_images, grid, align_corners=False)
    out = out.view(images.shape)
    
    return out