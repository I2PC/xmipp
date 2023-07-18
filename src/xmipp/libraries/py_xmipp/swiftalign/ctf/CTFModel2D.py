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
import math

def _compute_wavelength(voltage: float) -> float:
    return 1.23e-9 * math.sqrt(voltage*(1+1e-6*voltage)) # http://i2pc.es/coss/Articulos/Sorzano2007a.pdf

class CTFModel2D:
    def __init__(self, 
                 voltage: float,
                 spherical_aberration: float,
                 frequency_grid: torch.Tensor ) -> None:
        self._voltage = voltage
        self._spherical_aberration = spherical_aberration
        self._wavelength = _compute_wavelength(self._voltage)








def compute_ctf_image_2d(   frequency_grid: torch.Tensor,
                            defocus: torch.Tensor,
                            wavelength: float,
                            spherical_aberration: float,
                            phase_shift: float,
                            out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    # Compute the first term of the phase
    frequency2_grid = torch.square(frequency_grid)
    out = torch.tensordot(frequency2_grid, defocus, dims=((0, ), (-1, )), out=out)
    
    # Compute the squared frequency
    frequency2_grid = torch.sum(torch.square(frequency_grid), dim=0)
    
    # Compute defocus*frequency
    
    frequency2_grid = torch.tensordot(frequency_grid, frequency_grid, dim=(0, 0))
    
    out += (0.5*wavelength*wavelength*spherical_aberration)*frequency2_grid
    
    # TODO add the defocus term
    
    out *= frequency2
    out *= torch.pi * wavelength
    
    # Add the phase_shift. Also add pi to introduce the - sign
    out += phase_shift + torch.pi
    
    pass
