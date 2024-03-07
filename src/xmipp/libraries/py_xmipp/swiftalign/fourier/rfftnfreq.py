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

from typing import Sequence, Optional
import torch

def rfftnfreq(dim: Sequence[int], 
              d: float = 1.0,
              dtype: Optional[type] = None,
              device: Optional[torch.device] = None) -> torch.Tensor:
    """Creates a multidimensional Fourier frequency grid

    Args:
        dim (Sequence[int]): Image size
        d (float, optional): Normalization. Defaults to 1.0.
        dtype (Optional[type], optional): Element type. Defaults to float32.
        device (Optional[torch.device], optional): Device. Defaults to CPU.

    Returns:
        torch.Tensor: _description_
    """
    
    def fftfreq(dim: int) -> torch.Tensor:
        return torch.fft.fftfreq(dim, d=d, dtype=dtype, device=device)
    
    def rfftfreq(dim: int) -> torch.Tensor:
        return torch.fft.rfftfreq(dim, d=d, dtype=dtype, device=device)
    
    # Compute the frequencies for each axis.
    # For the last axis use rfft
    axis_freq = list(map(fftfreq, dim[:-1]))
    axis_freq.append(rfftfreq(dim[-1]))
    
    mesh = torch.meshgrid(*reversed(axis_freq), indexing='xy')
    return torch.stack(mesh)
    
    
