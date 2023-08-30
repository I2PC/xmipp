#!/usr/bin/env python

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
import argparse
import math

import xmippPyModules.torch.image as image

def run(map_volume_path: str,
        rec_volume_path: str,
        sigma2_volume_path: Optional[str],
        output_map_volume_path: str,
        output_sigma2_volume_path: str,
        gamma: float,
        nu: float,
        epsilon: float ):
    
    # Read input volumes
    map = image.read(map_volume_path)
    rec = image.read(rec_volume_path)
    if sigma2_volume_path is not None:
        sigma2 = image.read(sigma2_volume_path)
    else:
        sigma2 = torch.zeros_like(rec)
    
    # Compute the gradient
    grad = rec - map
    
    # Compute the magnitude
    sigma2 *= gamma
    sigma2 += (1.0 - gamma)*torch.square(grad)
    
    # Compute the gradient gain
    gain = nu / (torch.sqrt(sigma2) + epsilon)
    
    # Compute the next volume
    map += gain*grad
    
    # Write
    image.write(map, output_map_volume_path)
    image.write(sigma2, output_sigma2_volume_path)
    

if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'RMS Prop reconstruction' )
    parser.add_argument('--map', type=str, required=True)
    parser.add_argument('--rec', type=str, required=True)
    parser.add_argument('--sigma2', type=str)
    parser.add_argument('--omap', type=str, required=True)
    parser.add_argument('--osigma2', type=str, required=True)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--nu', type=float, default=0.001)
    parser.add_argument('--eps', type=float, default=1e-8)

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        map_volume_path = args.map,
        rec_volume_path = args.rec,
        sigma2_volume_path = args.sigma2,
        output_map_volume_path = args.omap,
        output_sigma2_volume_path = args.osigma2,
        gamma = args.gamma,
        nu = args.nu,
        epsilon = args.eps
    )