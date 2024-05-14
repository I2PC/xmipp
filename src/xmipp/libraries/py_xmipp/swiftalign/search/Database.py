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

from typing import NamedTuple
import torch

class SearchResult(NamedTuple):
    indices: torch.IntTensor
    distances: torch.Tensor

class Database:
    def __init__(self) -> None:
        pass
    
    def train(self, vectors: torch.Tensor) -> None:
        pass
    
    def add(self, vectors: torch.Tensor) -> None:
        pass
    
    def finalize(self):
        pass
    
    def reset(self):
        pass
    
    def search(self, vectors: torch.Tensor, k: int) -> SearchResult:
        pass
    
    def read(self, path: str):
        pass
    
    def write(self, path: str):
        pass

    def to_device(self, device: torch.device):
        pass
    
    def is_trained(self) -> bool:
        pass
    
    def is_populated(self) -> bool:
        return self.get_item_count() > 0

    def is_finalized(self) -> bool:
        pass

    def get_dim(self) -> int:
        pass

    def get_item_count(self) -> int:
        pass
    
    def get_input_device(self) -> torch.device:
        pass
    
    def _check_input(self, x: torch.Tensor):
        if len(x.shape) != 2:
            raise RuntimeError('Input should have 2 dimensions (batch and vector)')
        
        if x.shape[-1] != self.get_dim():
            raise RuntimeError('Input vectors have incorrect size')