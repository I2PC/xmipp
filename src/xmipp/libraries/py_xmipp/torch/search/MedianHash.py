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
import math
import torch

from .Database import Database, SearchResult

class MedianHashDatabase(Database):
    def __init__(self) -> None:
        self._median: Optional[torch.Tensor] = None
        self._hashes: List[torch.Tensor] = []
    
    def train(self, vectors: torch.Tensor) -> None:
        self._median = torch.median(vectors, dim=-1, out=self._median)
    
    def add(self, vectors: torch.Tensor) -> None:
        self._hashes.append(self.__compute_hash(vectors))
    
    def reset(self):
        self._hashes.clear()

    def search(self, vectors: torch.Tensor, k: int) -> SearchResult:
        # Compute the signature of the search vectors
        search_hashes = self.__compute_hash(vectors)

        result = None

        xors = None
        counts = None
        best_indices = None
        best_distances = None
        mask = None
        for reference_hashes in self._hashes:
            # Perform a XOR for all possible pairs
            xors = torch.logical_xor(
                search_hashes[:,None,:], 
                reference_hashes[None,:,:],
                out=xors
            )
            
            # Do pop-count
            counts = torch.count_nonzero(xors, dim=-1, out=counts)
            
            # Find the best candidates
            best_distances, best_indices = torch.min(counts, dim=-1, out=(best_distances, best_indices))
            
            # Evaluate new candidates
            if result is None:
                result = SearchResult(indices=best_indices, distances=best_distances)
            else:
                mask = torch.less(best_distances, result.distances, out=mask)
                result.indices[mask] = best_indices[mask]
                result.distances[mask] = best_distances[mask]
                
        return result
    
    def read(self, path: str):
        obj = {
            'median': self._median,
            'hashes': self._hashes
        }
        torch.save(obj, path)
    
    def write(self, path: str):
        obj = torch.load(path)
        self._median = obj['median']
        self._hashes = obj['hashes']

    def is_trained(self) -> bool:
        return self._median is not None
    
    def get_item_count(self) -> int:
        return math.prod(map(len, self._hashes))
    
    def to_gpu(self, device: torch.device) -> MedianHashDatabase:
        pass
    
    def from_gpu(self) -> MedianHashDatabase:
        pass
    
    def __compute_hash(self, 
                       vectors: torch.Tensor,
                       out: Optional[torch.BoolTensor] ) -> torch.BoolTensor:
        return torch.greater(self._median, vectors, out=out)
        