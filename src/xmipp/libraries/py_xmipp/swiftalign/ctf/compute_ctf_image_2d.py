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

def _compute_defocus_grid_2d(frequency_angle_grid: torch.Tensor,
                             defocus_average: torch.Tensor,
                             defocus_difference: torch.Tensor,
                             astigmatism_angle: torch.Tensor,
                             out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    out = torch.sub(frequency_angle_grid, astigmatism_angle[...,None,None], out=out)
    out *= 2
    out.cos_()    
    out *= defocus_difference[...,None,None]
    out += defocus_average[...,None,None]
    
    return out

def _compute_beam_energy_spread(frequency_magnitude2_grid: torch.Tensor,
                                chromatic_aberration: torch.Tensor,
                                wavelength: float, 
                                energy_spread_coefficient: float,
                                lens_inestability_coefficient: float,
                                out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    # http://i2pc.es/coss/Articulos/Sorzano2007a.pdf
    # Equation 10
    k = torch.pi / 4 * wavelength * (energy_spread_coefficient + 2*lens_inestability_coefficient)
    x = chromatic_aberration * k
    x.square_()
    x *= -1.0 / math.log(2)
    
    out = torch.mul(x[...,None,None], frequency_magnitude2_grid.square())
    out.exp_()
    
    return out
    
def compute_ctf_image_2d(frequency_magnitude2_grid: torch.Tensor,
                         frequency_angle_grid: torch.Tensor,
                         defocus_average: torch.Tensor,
                         defocus_difference: torch.Tensor,
                         astigmatism_angle: torch.Tensor,
                         wavelength: float,
                         spherical_aberration: float,
                         q0: Optional[float] = None,
                         chromatic_aberration: Optional[torch.Tensor] = None,
                         energy_spread_coefficient: Optional[float] = None,
                         lens_inestability_coefficient: Optional[float] = None,
                         phase_shift: Optional[float] = None,
                         out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    k = 0.5 * spherical_aberration * wavelength * wavelength

    out = _compute_defocus_grid_2d(
        frequency_angle_grid=frequency_angle_grid,
        defocus_average=defocus_average,
        defocus_difference=defocus_difference,
        astigmatism_angle=astigmatism_angle,
        out=out
    )
    
    # Compute the phase
    out -= k*frequency_magnitude2_grid
    out *= (torch.pi * wavelength) * frequency_magnitude2_grid
    
    # Apply the phase shift if provided
    if phase_shift is not None: 
        out += phase_shift
    
    # Compute the sin, also considering the inelastic
    # difraction factor if provided
    if q0 is not None:
        out = out.sin() + q0*out.cos()
    else:
        out.sin_()
    
    # Apply energy spread envelope
    if (chromatic_aberration is not None) and \
       (energy_spread_coefficient is not None) and \
       (lens_inestability_coefficient is not None):
           
        beam_energy_spread = _compute_beam_energy_spread(
            frequency_magnitude2_grid=frequency_magnitude2_grid,
            chromatic_aberration=chromatic_aberration,
            wavelength=wavelength,
            energy_spread_coefficient=energy_spread_coefficient,
            lens_inestability_coefficient=lens_inestability_coefficient
        )
        out *= beam_energy_spread
        
    
    
    return out
    