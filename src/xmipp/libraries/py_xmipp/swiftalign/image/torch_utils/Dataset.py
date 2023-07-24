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

from typing import Sequence, Iterable, Optional, Union
import torch
import mrcfile
import numpy as np
import operator

from ..Path import Path
from ...utils import LruCache

def _read(filename: str):
    return mrcfile.mmap(filename, mode='r')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[Path], max_open=64):
        self._paths = paths
        self._cache = LruCache(_read, capacity=max_open)
        
    def __len__(self) -> int:
        return len(self._paths)
    
    def __getitem__(self, index: Union[int, Iterable[int]]) -> np.ndarray:
        if isinstance(index, Iterable):
            return self.get_batch(index)
        else:
            return self.get_single(index)
    
    def get_single(self, index: int) -> np.ndarray:
        path = self._paths[index]
        
        # Get referenced data
        mrc = self._cache(path.filename)
        
        # Extract the data
        data = mrc.data
        if mrc.is_image_stack() or mrc.is_volume_stack():
            data = data[path.position_in_stack-1]
            
        return data
        
    def get_batch(self, indices: Iterable[int]) -> np.ndarray:
        output_stacks = []

        current_filename = None
        current_end = None
        current_start = None
        
        def finish_run():
            if current_filename is not None:
                mrc = self._cache(current_filename)
                data = mrc.data
                
                if current_start is not None:
                    assert(current_end is not None)
                    output_stacks.append(data[current_start:current_end])
                else:
                    output_stacks.append(data[None])
                
        for index in indices:
            path = self._paths[index]
            filename = path.filename
            slice_index = None if path.position_in_stack is None else path.position_in_stack - 1
            
            if filename == current_filename and slice_index == current_end and current_end is not None:
                # We can continue with this run
                current_end += 1
                
            else:
                finish_run()

                current_filename = path.filename
                current_start = slice_index
                current_end = path.position_in_stack

        # Finish the last run
        finish_run()

        # Concatenate the arrays
        result = np.concatenate(output_stacks, axis=0)
        return result