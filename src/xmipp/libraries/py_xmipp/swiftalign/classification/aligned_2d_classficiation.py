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

from typing import Iterable
import torch

def aligned_2d_classification(dataset: Iterable[torch.Tensor],
                              scratch: torch.Tensor,
                              q = 0.05):
    # Write
    start = 0
    for vectors in dataset:
        end = start + len(vectors)
        
        # Write 
        scratch[start:end,:] = vectors.to(scratch, non_blocking=True)

        # Setup next iteration
        start = end
        
    # Perform the PCA analysis
    avg = scratch.mean(dim=0)
    scratch -= avg
    _,_,v = torch.svd(scratch)
    direction = v[:,0]


    projections = torch.matmul(scratch, direction[...,None])[:,0]

    quantiles = torch.tensor([q, 1-q], device=projections.device)
    scales = torch.quantile(projections, quantiles)
    result = torch.matmul(scales[:,None], direction[None,:])
    result += avg

    return result, direction
