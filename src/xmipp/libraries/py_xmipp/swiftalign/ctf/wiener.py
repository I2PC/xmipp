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

from typing import Optional, Sequence
import torch

def wiener_2d( direct_filter: torch.Tensor,
               inverse_ssnr: Optional[torch.Tensor] = None,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    # Compute the filter power (|H|²) at the output
    if torch.is_complex(direct_filter):
        out = torch.abs(direct_filter, out=out)
        out.square_()
    else:
        out = torch.square(direct_filter, out=out)
    
    # Compute the default value for inverse SSNR if not
    # provided
    if inverse_ssnr is None:
        inverse_ssnr = torch.mean(out, dim=(-2, -1))
        inverse_ssnr *= 0.1
        inverse_ssnr = inverse_ssnr[...,None,None]
    
    # H* / (|H|² + N/S)
    out.add_(inverse_ssnr)
    torch.div(torch.conj(direct_filter), out, out=out)

    return out
