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

def time_shift_filter(shift: torch.Tensor,
                      freq: torch.Tensor,
                      out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    """Generates a multidimensional shift filter in Fourier space

    Args:
        shift (torch.Tensor): Shift in samples. (B, n) where shape, where B is the batch size and n is the dimensions 
        freq (torch.Tensor): Frequency grid in radians. (B, dn, ... dy, dx)
        out (Optional[torch.Tensor], optional): Preallocated tensor. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    
    # Fourier time shift theorem:
    # https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Some_discrete_Fourier_transform_pairs
    angles = -torch.matmul(shift, freq)
    gain = torch.tensor(1.0).to(angles) # TODO try to avoid using this
    out = torch.polar(gain, angles, out=out)
    return out
