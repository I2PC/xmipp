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
import collections

from .Path import Path

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
    

class CachingReader:
    def __init__(self, max_open: int=64):
        self.max_open = max_open
        self._cache = collections.OrderedDict()
    
    def read(self, path: Path) -> torch.Tensor:
        # Get referenced data
        mrc = self.__read_file(path.filename)
        
        # Extract the data
        data = mrc.data
        if mrc.is_image_stack() or mrc.is_volume_stack():
            data = data[path.position_in_stack-1]
            
        return data
        
    def read_batch(self, 
                   paths: Iterable[Path],
                   pin_memory=False) -> torch.Tensor:
        output_stacks = []
        for filename, index_slice in _batch_files(paths):
            mrc = self.__read_file(filename)
            data = mrc.data
            
            if mrc.is_image_stack() or mrc.is_volume_stack():
                if index_slice is None:
                    raise RuntimeError('Image index should be provided for image stacks')
                output_stacks.append(data[index_slice])
            else:
                output_stacks.append(data[None])

        # Concatenate the arrays
        if len(output_stacks) == 1:
            result = torch.as_tensor(output_stacks[0])
            if pin_memory:
                result = result.pin_memory()
            
        else:
            n_images = sum(map(len, output_stacks))
            result_shape = (n_images, ) + output_stacks[0].shape[1:]
            result = torch.empty(result_shape, pin_memory=pin_memory)
            np.concatenate(output_stacks, axis=0, out=result.numpy())

        return torch.as_tensor(result)

    def __read_file(self, filename: str):
        item = self._cache.get(filename, None)
        
        # Update cache
        if item is None:
            # Not present, invoke the function
            item = mrcfile.mmap(filename, mode='r')
            self._cache[filename] = item
            if(len(self._cache) >= self.max_open):
                self._cache.popitem(last=False)
                
        else:
            # Put it on top
            self._cache.move_to_end(filename, last=True)

        assert(item is not None)
        return item

