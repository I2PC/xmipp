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

from typing import Optional, Union, Tuple

import torch

from .quaternion_arithmetic import quaternion_product, quaternion_conj

def twist_decomposition(quaternions: torch.Tensor,
                        directions: torch.Tensor,
                        assume_normalized: bool = True,
                        compute_quaternions: bool = False,
                        return_phasor: bool = False ):

    if not assume_normalized:
        raise NotImplementedError()
    
    proj = torch.sum(quaternions[...,1:4]*directions, axis=-1)
    half_angles = torch.complex(
        quaternions[...,0],
        proj
    )
    half_angles /= abs(half_angles)
    
    if return_phasor:
        angles = torch.square(half_angles)
    else:
        angles = 2*torch.acos(half_angles.real)
    
    if compute_quaternions:
        w = half_angles.real[...,None]
        v = half_angles.imag[...,None] * directions
        twist = torch.concat((w, v), axis=-1)
        return angles, twist
    else:
        return angles

def swing_decomposition(quaternions: torch.Tensor,
                        twists: torch.Tensor,
                        assume_normalized: bool = True,
                        out: Optional[torch.Tensor] = None ) -> torch.Tensor:

    if not assume_normalized:
        raise NotImplementedError()
                        
    return quaternion_product(
        quaternions,
        quaternion_conj(twists),
        out=out
    )

def twist_swing_decomposition(quaternions: torch.Tensor,
                              directions: Union[torch.Tensor],
                              assume_normalized: bool = True,
                              return_phasor: bool = False ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not assume_normalized:
        raise NotImplementedError()

    angles, twists = twist_decomposition(
        quaternions, 
        directions, 
        assume_normalized=True,
        compute_quaternions=True,
        return_phasor=return_phasor
    )
    swing = swing_decomposition(
        quaternions, 
        twists, 
        assume_normalized=True
    )
    
    return angles, twists, swing
