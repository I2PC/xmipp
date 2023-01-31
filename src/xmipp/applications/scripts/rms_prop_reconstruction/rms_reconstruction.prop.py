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



def run(prev_volume_path: str,
        rec_volume_path: str,
        output_path: str,
        gamma: float,
        nu: float,
        epsilon: float ):
    
    # Read input volumes
    prev = image.read(prev_volume_path)
    rec = image.read(rec_volume_path)
    
    # Compute the gradient
    grad = rec - prev
    
    # Compute the magnitude
    sigma2 = gamma*torch.var(prev) + (1.0 - gamma)*torch.sum(grad**2)
    
    # Compute the gradient gain
    gain = nu / (torch.sqrt(sigma2) + epsilon)
    
    # Compute the next volume
    next = prev + gain*grad
    
    # Write
    image.write(next, output_path)
    
    

if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'RMS Prop reconstruction' )
    parser.add_argument('--prev', type=str, required=True)
    parser.add_argument('--rec', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--nu', type=float, default=0.001)
    parser.add_argument('--eps', type=float, default=1e-8)

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        prev_volume_path = args.prev,
        rec_volume_path = args.rec,
        output_path = args.o,
        gamma = args.gamma,
        nu = args.nu,
        epsilon = args.eps  
    )