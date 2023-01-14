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

from typing import Tuple

from . import read

def get_size(path: str) -> Tuple:
    image = read.read_data(path, mmap=True)
    return image.shape #TODO do with metadata

def decompose_path(path: str) -> Tuple[int, str]:
    STACK_INDEXER = '@'
    parts = path.split(STACK_INDEXER, maxsplit=1)
    
    if len(parts) == 2:
        return int(parts[0]), parts[1]
    else:
        assert(len(parts) == 1)
        return -1, parts[0]