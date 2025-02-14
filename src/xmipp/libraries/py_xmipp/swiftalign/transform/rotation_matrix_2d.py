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
import torch

def rotation_matrix_2d(angles: torch.Tensor,
                       out: Optional[torch.Tensor] = None):

    out = torch.empty((len(angles), 2, 2), out=out)
    
    # The 2D rotation matrix is defined as:
    #|c -s|
    #|s  c|
    out[:,0,0] = torch.cos(angles, out=out[:,0,0]) #c
    out[:,1,0] = torch.sin(angles, out=out[:,0,1]) #s
    out[:,0,1] = -out[:,1,0] #-s
    out[:,1,1] = out[:,0,0] #c

    return out
