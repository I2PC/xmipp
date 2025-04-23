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

from typing import Sequence, Iterable, Union

import torch

from ..CachingReader import CachingReader
from ..Path import Path

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[Path], max_open: int = 64):
        self._paths = paths
        self._cache = CachingReader(max_open=max_open)
        
    def __len__(self) -> int:
        return len(self._paths)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self._cache.read(self._paths[index])
    