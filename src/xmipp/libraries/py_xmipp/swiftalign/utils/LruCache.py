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

from typing import Callable
from collections import OrderedDict

class LruCache:
    def __init__(self,
                 func: Callable,
                 capacity: int = 64 ) -> None:
        self._func = func
        self._capacity = capacity
        self._data = OrderedDict()
        
    def __call__(self, *args):        
        # Check if args are cached
        it = self._data.get(args, None)
        
        # Update cache
        if it is None:
            # Not present, invoke the function
            it = self._func(*args)
            self._data[args] = it # Appends it in the back
            if(len(self._data) >= self._capacity):
                self._data.popitem(last=False)
                
        else:
            # Put it on top
            self._data.move_to_end(args, last=True)
        assert(it is not None)

        return it