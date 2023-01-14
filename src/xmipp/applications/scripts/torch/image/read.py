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

from typing import Iterable
import mrcfile
import numpy as np


def read_data(path: str, mmap: bool = False, permissive: bool = False) -> np.ndarray:
    open_fun = mrcfile.mmap if mmap else mrcfile.open
    with open_fun(path, mode='r', permissive=permissive) as mrc:
        return mrc.data
    
def read_data_batch(paths: Iterable[str]) -> np.ndarray:
    images = list(map(read_data, paths))
    return np.stack(images)

def read_header(path: str, permissive: bool = False):
    with mrcfile.open(path, mode='r', permissive=permissive, header_only=True) as mrc:
        return mrc.header
    
