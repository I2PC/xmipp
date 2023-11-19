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
import math
import torch
import faiss
import faiss.contrib.torch_utils

from .Database import Database, SearchResult

def opq_ifv_pq_recipe(dim: int, size: int = int(3e6), c: float = 16, norm=False):
    """ 
    Values selected using:
    https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    """
    # Determine the parameters
    PQ_BLOCK_SIZE = 4 # Coefficients per PQ partition
    PQ_MAX_BYTES_PER_VECTOR = 48 # For single precision
    PQ_MAX_VECTOR_SIZE = PQ_BLOCK_SIZE * PQ_MAX_BYTES_PER_VECTOR 
    d = min(PQ_MAX_VECTOR_SIZE, dim//PQ_BLOCK_SIZE*PQ_BLOCK_SIZE)
    m = d // PQ_BLOCK_SIZE # Bytes per vector. 48 is the max for GPU and single precision
    k = c * math.sqrt(size) # Number of clusters
    k = 2 ** math.ceil(math.log2(k)) # Use a power of 2
    
    # Elaborate the recipe for the factory
    opq = f'OPQ{m}_{d}' # Only supported on CPU
    ifv = f'IVF{k}' # In order to support GPU do not use HNSW32
    pq = f'PQ{m}'
    recipe = (opq, ifv, pq)
    
    if norm:
        l2norm = 'L2norm'
        recipe = (l2norm, ) + recipe
    
    return ','.join(recipe)

class FaissDatabase(Database):
    def __init__(self,
                 dim: int = 0, 
                 recipe: Optional[str] = None ) -> None:

        index: Optional[faiss.Index] = None
        if recipe and dim:
            index = faiss.index_factory(dim, recipe)
        
        self._index = index
    
    def train(self, vectors: torch.Tensor) -> None:
        self._check_input(vectors)        
        self._sync(vectors)
        self._index.train(vectors)
    
    def add(self, vectors: torch.Tensor) -> None:
        self._check_input(vectors)
        self._sync(vectors)
        self._index.add(vectors)
    
    def reset(self):
        self._index.reset()
    
    def search(self, vectors: torch.Tensor, k: int) -> SearchResult:
        self._check_input(vectors)
        self._sync(vectors)
        distances, indices = self._index.search(vectors, k)
        return SearchResult(indices=indices, distances=distances)
    
    def read(self, path: str):
        self._index = faiss.read_index(path)
    
    def write(self, path: str):
        faiss.write_index(self._index, path)
    
    def to_device(self, 
                  device: torch.device, 
                  use_f16: bool = False, 
                  reserve_vecs: int = 0,
                  use_precomputed = False ):
        if device.type == 'cuda':
            resources = faiss.StandardGpuResources()
            resources.setDefaultNullStreamAllDevices() # To interop with torch
            co = faiss.GpuClonerOptions()
            co.useFloat16 = use_f16
            co.useFloat16CoarseQuantizer = use_f16
            co.usePrecomputed = use_precomputed
            co.reserveVecs = reserve_vecs
            
            self._index = faiss.index_cpu_to_gpu(
                resources,
                device.index,
                self._index,
                co
            )
        
        elif device.type == 'cpu':
            self._index = faiss.index_gpu_to_cpu(self._index)
        
        else:
            raise ValueError('Input device must be CPU or CUDA')
    
    def is_trained(self) -> bool:
        return self._index.is_trained
    
    def is_finalized(self) -> bool:
        return True
    
    def get_dim(self) -> int:
        return self._index.d

    def get_item_count(self) -> int:
        return self._index.ntotal
    
    def get_input_device(self) -> torch.device:
        return torch.device('cpu') # TODO determine
    
    def set_metric_type(self, metric_type: int):
        self._index.metric_type = metric_type
    
    def get_metric_type(self) -> int:
        return self._index.metric_type
    
    def _sync(self, vectors: torch.Tensor):
        if vectors.device.type == 'cuda':
            stream = torch.cuda.current_stream(vectors.device)
            event = stream.record_event()
            raise NotImplementedError('We should sync here')