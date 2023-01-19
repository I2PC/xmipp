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

from typing import Sequence
import torch

def nfft_freq2(shape: Sequence[int]) -> torch.Tensor:
    if len(shape) != 2:
        raise NotImplementedError('nfft_freq is only implemented for N=2')

    
    freq_x = torch.fft.rfftfreq(shape[-1])
    freq_y = torch.fft.fftfreq(shape[-2])[...,None]
    return freq_x**2 + freq_y**2
    
def nfft_freq(shape: Sequence[int]) -> torch.Tensor:
    out = nfft_freq2(shape)
    return torch.sqrt(out, out=out)