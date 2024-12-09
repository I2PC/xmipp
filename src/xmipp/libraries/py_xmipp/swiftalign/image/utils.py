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

from .Path import Path

import mrcfile

def get_image2d_size(filename: str):
    with mrcfile.open(filename, mode='r', header_only=True) as mrc:
        header = mrc.header
        nx = int(header['nx'])
        ny = int(header['ny'])
        return ny, nx

def parse_path(f: str) -> Path:
    STACK_INDEXER = '@'
    parts = f.split(STACK_INDEXER, maxsplit=1)
    
    if len(parts) == 2:
        return Path(filename=parts[1], position_in_stack=int(parts[0]))
    else:
        assert(len(parts) == 1)
        return Path(filename=parts[0])