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

def complete_orthogonal_matrix(m32: torch.Tensor,
                               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    out = torch.empty(m32.shape[:-2] + (3, )*2, dtype=m32.dtype, device=m32.device, out=out)
    out[...,:2] = m32
    torch.linalg.cross(out[...,0], out[...,1], out=out[...,2])
    return out

def image_plane_vector_from_matrix(matrices: torch.Tensor) -> torch.Tensor:
    return matrices[...,2] # Extracts the Z column which is normal to the XY plane

def find_common_lines(planes0: torch.Tensor,
                      planes1: torch.Tensor,
                      normalize: bool = False,
                      out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # The cross product of the normal plane vectors represents
    # the direction of the common line of the two planes
    out = torch.cross(planes0, planes1, out=out)
    
    # L2 normalize the result
    if normalize:
        out = torch.nn.functional.normalize(out, dim=-1, out=out)
    
    return out
    
def unproject_to_image_plane(matrices: torch.Tensor,
                             vectors: torch.Tensor,
                             ignore_3d: bool = True,
                             out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    if matrices.shape[-2:] != (3, 3):
        raise RuntimeError('Matrices must be 3x3')

    if vectors.shape[-1] != 3:
        raise RuntimeError('Lines parameter must have 3 elements in the last dimension')
    
    # Skip the last column of the matrix, as we are not interested in the Z component
    if ignore_3d:
        matrices = matrices[...,:2]
    
    # Perform the unprojection
    if out is not None:
        out = out[...,None]
    return torch.matmul(matrices.mT, vectors[...,None], out=out)[...,0]