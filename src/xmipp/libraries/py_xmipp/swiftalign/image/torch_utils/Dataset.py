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

from typing import Sequence, Iterable, Tuple, Union, Optional
import torch
import mrcfile
import numpy as np

from ..Path import Path
from ...utils import LruCache

def _read(filename: str):
    return mrcfile.mmap(filename, mode='r')

def _index_or_none(position_in_stack: Optional[int]) -> Optional[int]:
    return None if position_in_stack is None else position_in_stack - 1

def _batch_files(paths: Iterable[Path]) -> Tuple[str, Optional[slice]]:
    it = iter(paths)
    
    # Initialize with the first loop iteration
    path = next(it)
    current_filename = path.filename
    current_end = path.position_in_stack
    current_start = _index_or_none(current_end)
    
    for path in it:
        filename = path.filename
        index = _index_or_none(path.position_in_stack)
        
        if filename == current_filename and index == current_end and current_end is not None:
            # We can continue with this run
            current_end += 1
                
        else:
            # Finalize the last iteration and start over
            if current_start is not None:
                assert (current_end is not None)
                yield current_filename, slice(current_start, current_end)
            else:
                yield current_filename, None
            
            current_filename = path.filename
            current_end = path.position_in_stack
            current_start = _index_or_none(current_end)
     
    # Finalize the last iteration   
    if current_start is not None:
        assert (current_end is not None)
        yield current_filename, slice(current_start, current_end)
    else:
        yield current_filename, None
    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[Path], max_open=64):
        self._paths = paths
        self._cache = LruCache(_read, capacity=max_open)
        
    def __len__(self) -> int:
        return len(self._paths)
    
    def __getitem__(self, index: Union[int, Iterable[int], slice]) -> torch.Tensor:
        if isinstance(index, Iterable):
            return self.get_batch(index)
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop
            step = index.step or 1
            return self.get_batch(range(start, stop, step))
        else:
            return self.get_single(index)
    
    def get_single(self, index: int) -> torch.Tensor:
        path = self._paths[index]
        
        # Get referenced data
        mrc = self._cache(path.filename)
        
        # Extract the data
        data = mrc.data
        if mrc.is_image_stack() or mrc.is_volume_stack():
            data = data[path.position_in_stack-1]
            
        return data
        
    def get_batch(self, indices: Iterable[int]) -> torch.Tensor:
        output_stacks = []
        filenames = map(self._paths.__getitem__, indices)
        for filename, index_slice in _batch_files(filenames):
            mrc = self._cache(filename)
            data = mrc.data
            
            if mrc.is_image_stack() or mrc.is_volume_stack():
                if index_slice is None:
                    raise RuntimeError('Image index should be provided for image stacks')
                output_stacks.append(data[index_slice])
            else:
                output_stacks.append(data[None])

        # Concatenate the arrays
        if len(output_stacks) == 1:
            result = output_stacks[0]
        else:
            result = np.concatenate(output_stacks, axis=0)

        return result
