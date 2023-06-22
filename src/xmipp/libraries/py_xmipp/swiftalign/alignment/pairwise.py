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

from typing import Iterable, Callable, Optional, Tuple
import torch

def condensed_array_size(m: int) -> int:
    return (m * (m - 1)) // 2

def pair_to_condensed_array_index(pairs: torch.IntTensor, m: int, pair_dim: int=-1) -> torch.IntTensor:
    """
    Based on:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    """
    if pairs.shape[pair_dim] != 2:
        raise RuntimeError('Tensor must have 2 elements on the pair axis')
    
    # Assure that the pair is ordered
    pairs = torch.sort(pairs, dim=pair_dim)
    
    # Select the indices separately
    i = pairs.select(dim=pair_dim, index=0)
    j = pairs.select(dim=pair_dim, index=1)
    assert(torch.all(torch.less(i, j)))
    
    return m * i + j - ((i + 2) * (i + 1)) // 2


def align_image(references: Iterable[torch.Tensor], 
                image: torch.Tensor, 
                out: Optional[Tuple[torch.IntTensor, torch.Tensor]] = None ) -> Tuple[torch.IntTensor, torch.Tensor]:
    
    out = torch.empty(len(references), dtype=int, device=image.device, out=out)
    
    q = torch.flatten(image)
    difference = None
    distances = None
    for i, reference in enumerate(references):
        r = torch.flatten(reference, start_dim=1)
        
        # Compute the closest point
        difference = torch.subtract(r, q, out=difference)
        distances = torch.norm(difference, axis=1, out=distances)
        out[0][i] = torch.argmin(distances, out=out[i])
    
    return out


def pairwise_align(images: torch.Tensor, 
                   transform_callback: Callable[[torch.Tensor], torch.Tensor],
                   out: Optional[torch.IntTensor] = None ) -> Tuple[torch.IntTensor, torch.Tensor]:
    out = torch.empty(
        condensed_array_size(len(images)), 
        device=images.device, dtype=images.dtype
    )
    
    references = []
    begin = 0
    for image in images:
        # Align the image with the existing gallery
        end = begin + len(references)
        if begin < end:
            align_image(references, image, out=out[begin:end])
        
        # Add its transformations to the reference gallery
        references.append(transform_callback(image))
        begin = end
    
    return out
