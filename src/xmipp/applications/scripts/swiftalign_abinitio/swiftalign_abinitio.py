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

import torch
import argparse

import xmippPyModules.swiftalign.metadata as md
import xmippPyModules.swiftalign.image as image
import xmippPyModules.swiftalign.commonlines as commonlines

def run(images_md_path: str,
        n_angles: int,
        batch: int,
        n_iter: int):
    images_md = md.sort_by_image_filename(md.read(images_md_path))
    images_paths = list(map(image.parse_path, images_md[md.IMAGE]))
    images_dataset = image.torch_utils.Dataset(images_paths)
    images = torch.utils.data.default_collate(list(images_dataset))
    
    sinogram = commonlines.compute_sinogram_2d(images, n_angles, 'bilinear')
    matrices, error = commonlines.optimize_common_lines_monte_carlo(sinogram, n_iter, batch)
    
if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Ab initio volume using common lines')
    parser.add_argument('-i', required=True)
    parser.add_argument('--sinogram_steps', type=int, default=72)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--iter', type=int, default=2**16)


    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        images_md_path=args.i,
        n_angles=args.sinogram_steps,
        batch=args.batch,
        n_iter=args.iter
    )