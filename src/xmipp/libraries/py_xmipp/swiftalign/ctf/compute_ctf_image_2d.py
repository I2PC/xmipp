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

def _compute_defocus_grid_2d(frequency_angle_grid: torch.Tensor,
                             defocus_average: torch.Tensor,
                             defocus_difference: torch.Tensor,
                             astigmatism_angle: torch.Tensor,
                             out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    out = torch.sub(frequency_angle_grid, astigmatism_angle[...,None], out=out)
    out *= 2
    out.cos_()    
    out *= defocus_difference[...,None]
    out += defocus_average[...,None]
    
    return out

def compute_ctf_image_2d(frequency_magnitude2_grid: torch.Tensor,
                         frequency_angle_grid: torch.Tensor,
                         defocus_average: torch.Tensor,
                         defocus_difference: torch.Tensor,
                         astigmatism_angle: torch.Tensor,
                         wavelength: float,
                         spherical_aberration: float,
                         phase_shift: float,
                         out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    k = 0.5 * spherical_aberration * wavelength * wavelength

    out = _compute_defocus_grid_2d(
        frequency_angle_grid=frequency_angle_grid,
        defocus_average=defocus_average,
        defocus_difference=defocus_difference,
        astigmatism_angle=astigmatism_angle,
        out=out
    )
    
    out -= k*frequency_magnitude2_grid
    out *= (torch.pi * wavelength) * frequency_magnitude2_grid
    out += torch.pi + phase_shift
    out.cos_()
    
    return out
    