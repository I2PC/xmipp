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
import itertools

from geometry import *
from sinogram import *



def _random_rotation_matrix(d: int,
                            shape: torch.Size = torch.Size(), 
                            dtype: Optional[torch.dtype] = None,
                            device: Optional[torch.device] = None,
                            out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    # Allocate the output with the RVs
    out = torch.randn(shape + (d, d), dtype=dtype, device=device, out=out)
    
    # Select vectors at random and apply Gram-Schmidt
    for i in range(d):
        # Alias for the current column
        vi = out[...,:,i]
        
        # Apply Gram-Schmidt
        # TODO optimize this code
        for j in range(i):
            vj = out[...,:,j]
            proj = torch.matmul(vi[...,None,:], vj[...,:,None])[...,0]
            vi -= proj * vj
            
        # Make sure the vector is orthogonal
        torch.nn.functional.normalize(vi, dim=-1, out=vi)
    
    return out

def optimize_common_lines_monte_carlo(sinograms: torch.Tensor,
                                      n_iterations: int,
                                      batch: int ) -> torch.Tensor:
    
    best_error = None
    best_matrices = None
    
    matrices = None
    common_lines = None
    lines = torch.empty((batch, 2, 2), dtype=sinograms.dtype, device=sinograms.device)
    indices = torch.empty((batch, 2), dtype=sinograms.dtype, device=sinograms.device)
    projections = torch.empty((batch, 2, sinograms.shape[-1]), dtype=sinograms.dtype, device=sinograms.device)
    delta = None
    error = None
    for _ in range(n_iterations):
        # Compute random 3x3 matrices of shape [B, N, 3, 3]
        matrices = _random_rotation_matrix(
            3, (batch, len(sinograms)), 
            dtype=sinograms.dtype,
            device=sinograms.device,
            out=matrices
        )
        
        # Iterate over all pairs of images
        error = torch.zeros((batch, ), dtype=sinograms.dtype, device=sinograms.device, out=error)
        for idx0, idx1 in itertools.combinations(range(len(sinograms)), r=2):
            # Alias the current sinogram and angle
            sinogram0 = sinograms[idx0]
            sinogram1 = sinograms[idx1]
            matrices0 = matrices[...,idx0,:,:]
            matrices1 = matrices[...,idx1,:,:]
            
            # Compute the common lines for this pair
            common_lines = find_common_lines(
                image_plane_vector_from_matrix(matrices0),
                image_plane_vector_from_matrix(matrices1),
                normalize=False,
                out=common_lines
            )
            
            # Unproject the common lines to the image plane for each image of the pair
            lines0 = unproject_to_image_plane(
                matrices=matrices0,
                vectors=common_lines,
                out=lines[...,0,:]
            )
            lines1 = unproject_to_image_plane(
                matrices=matrices1,
                vectors=common_lines,
                out=lines[...,1,:]
            )
            
            # Compute the fractional index at the sinogram
            indices0 = index_from_line_2d(lines0, sinograms.shape[-2], out=indices[...,0])
            indices1 = index_from_line_2d(lines1, sinograms.shape[-2], out=indices[...,1])
            
            # Project the image in the direction
            projections0 = extract_projection_2d(sinogram0, indices=indices0, out=projections[...,0,:])
            projections1 = extract_projection_2d(sinogram1, indices=indices1, out=projections[...,1,:])

            # Accumulate the error for each try
            delta = torch.sub(projections0, projections1, out=delta)
            error += torch.bmm(delta[...,None,:], delta[...,:,None])[...,0,0]
        
        # Evaluate if there is any improvement
        batch_best = torch.argmin(error)
        batch_best_error = error[batch_best]
        if best_error is None or batch_best_error < best_error:
            best_error = error[batch_best]
            best_matrices = matrices[batch_best]
    
    return best_matrices, best_error
            