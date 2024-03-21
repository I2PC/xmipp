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

from typing import Iterable, Tuple
import torch

def _correct_amplitude(samples: torch.Tensor,
                       average: torch.Tensor ):

    ata = torch.empty((2, 2), dtype=average.dtype, device=average.device)
    ata[0,0] = average.square().sum()
    ata[1,0] = ata[0,1] = average.sum()
    ata[1,1] = len(average)
    
    pseudoinverse = torch.lianlg.inverse(ata)
    correlation = torch.matmul(samples, average)
    total = samples.sum(axis=-1)
    a = pseudoinverse[0,0]*correlation + pseudoinverse[0,1]*total
    b = pseudoinverse[1,0]*correlation + pseudoinverse[1,1]*total
    
    samples -= b[:,None]
    samples /= a[:,None]

def _mean_centered_pca(samples: torch.Tensor,
                       k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(samples.shape) != 2:
        raise RuntimeError('Sample matrix is expected to have 2 dimensions')
    
    if len(samples) >= 3*k:
        # Perform the computations with the smallest
        # covariance vector possible
        transpose = samples.shape[0] < samples.shape[1]
        
        # Compute the covariance according to
        # the transposition    
        if transpose:
            covariance = samples @ samples.T
        else:
            covariance = samples.T @ samples
        covariance /= len(samples) - 1
        assert(covariance.shape == (min(samples.shape), )*2)
            
        # Compute the largest eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = torch.lobpcg(
            covariance,
            k=k
        )
        
        # Undo the transposition
        if transpose:
            eigenvectors = samples.T @ eigenvectors
            eigenvectors /= torch.norm(eigenvectors, dim=0, keepdim=True)
            
    else:
        eigenvalues = torch.zeros((k, ), dtype=samples.dtype, device=samples.device)
        eigenvectors = torch.zeros((samples.shape[-1], k), dtype=samples.dtype, device=samples.device)

    return eigenvalues, eigenvectors
    

def aligned_2d_classification(dataset: Iterable[torch.Tensor],
                              scratch: torch.Tensor ):
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
    _correct_amplitude(scratch, avg)
    scratch -= avg
    _, v = _mean_centered_pca(scratch, k=1)
    direction = v[:,0]


    projections = torch.matmul(scratch, direction[...,None])[:,0]

    return avg, direction, projections
