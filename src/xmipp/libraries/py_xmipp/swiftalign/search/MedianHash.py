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

from typing import Optional, List
import torch

from .Database import Database, SearchResult

class MedianHashDatabase(Database):
    def __init__(self, 
                 dim: int = 0) -> None:

        self._dim = dim
        self._median: Optional[torch.Tensor] = None
        self._hashes: List[torch.Tensor] = []
    
    def train(self, vectors: torch.Tensor) -> None:
        self._check_input(vectors)
        self._median = torch.median(vectors, dim=0).values
    
    def add(self, vectors: torch.Tensor) -> None:
        self._check_input(vectors)
        self._hashes.append(self.__compute_hash(vectors))
    
    def reset(self):
        self._hashes.clear()

    def search(self, vectors: torch.Tensor, k: int) -> SearchResult:
        if k != 1:
            raise NotImplementedError('KNN has been only implemented for k=1')
        
        self._check_input(vectors)

        # Compute the signature of the search vectors
        search_hashes = self.__compute_hash(vectors)

        result = None

        xors = None
        pop_counts = None
        best_candidate = None
        mask = None
        base_index = 0
        for reference_hashes in self._hashes:
            # Perform a XOR for all possible pairs
            xors = torch.logical_xor(
                search_hashes[:,None,:], 
                reference_hashes[None,:,:],
                out=xors
            )
            
            # Do pop-count
            pop_counts = torch.count_nonzero(xors, dim=-1) # TODO add out=count
            
            # Find the best candidates
            best_candidate = torch.min(pop_counts, dim=-1, out=best_candidate)
            
            # Evaluate new candidates
            if result is None:
                result = SearchResult(
                    indices=best_candidate.indices.clone(), 
                    distances=best_candidate.values.clone()
                )
            else:
                mask = torch.less(best_candidate.values, result.distances, out=mask)
                result.indices[mask] = best_candidate.indices[mask] + base_index
                result.distances[mask] = best_candidate.values[mask]
        
            # Update base index for next batch
            base_index += len(reference_hashes)
        
        # Add a dimension in the end
        return SearchResult(
            indices=result.indices[...,None],
            distances=result.distances[...,None]
        )
    
    def read(self, path: str):
        obj = torch.load(path)
        self._dim = obj['dim']
        self._median = obj['median']
        self._hashes = obj['hashes']
    
    def write(self, path: str):
        obj = {
            'dim': self._dim,
            'median': self._median,
            'hashes': self._hashes
        }
        torch.save(obj, path)

    def to_device(self, device: torch.device):
        def func(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if x is None:
                return None
            else:
                return x.to(device=device)
        
        self._median = func(self._median)
        self._hashes = list(map(func, self._hashes))
    
    def is_trained(self) -> bool:
        return self._median is not None

    def is_finalized(self) -> bool:
        return True
    
    def get_dim(self) -> int:
        return self._dim

    def get_item_count(self) -> int:
        return sum(map(len, self._hashes))

    def get_input_device(self) -> torch.device:
        return self._median.device
    
    def __compute_hash(self, 
                       vectors: torch.Tensor,
                       out: Optional[torch.BoolTensor] = None ) -> torch.BoolTensor:
        return torch.greater(self._median, vectors, out=out)