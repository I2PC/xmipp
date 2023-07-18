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

from typing import Optional, Tuple, Sequence
import torch
import argparse
import pathlib

import xmippPyModules.swiftalign.image as image
import xmippPyModules.swiftalign.fourier as fourier
import xmippPyModules.swiftalign.ctf as ctf
import xmippPyModules.swiftalign.metadata as md



def run(images_md_path: str, 
        output_md_path: str, 
        batch_size: int,
        device_names: list ):
    
    # Devices
    if device_names:
        devices = list(map(torch.device, device_names))
    else:
        devices = [torch.device('cpu')]
    transform_device = devices[0]
    
    # Read input files
    images_md = md.sort_by_image_filename(md.read(images_md_path))
    images_paths = list(map(image.parse_path, images_md[md.IMAGE]))
    images_dataset = image.torch_utils.Dataset(images_paths)
    images_loader = torch.utils.data.DataLoader(
        images_dataset,
        batch_size=batch_size,
        pin_memory=transform_device.type=='cuda'
    )
    
    # Process
    start = 0
    output_images = torch.empty(pin_memory=True) # TODO
    ctf_images = None
    wiener_filters = None
    for batch_images in images_loader:
        end = start + len(batch_images)
        batch_images = batch_images.to(transform_device, non_blocking=True)
        batch_slice = slice(start, end)
        batch_images_md = images_md.iloc[batch_slice]
        
        # Compute the CTF image
        # TODO
        
        # Compute the wiener filter
        wiener_filters = ctf.wiener_2d(ctf_images, out=wiener_filters)
        
        # Apply the filter to the images
        batch_images *= wiener_filters

        # Save the result
        output_images[batch_slice] = batch_images.to('cpu', non_blocking=True)
        
        start = end

    # Update metadata
    output_images_path = pathlib.Path(output_md_path).with_suffix('.mrc')
    
    
    # Write
    output_images = output_images.numpy()
    image.write(output_images, output_images_path)
    md.write(images_md, output_md_path)
    

if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Align Nearest Neighbor Training',
                        description = 'Align Cryo-EM images using a fast Nearest Neighbor approach')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--batch', type=int, default=8192)
    parser.add_argument('--devices', nargs='*')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        images_md_path = args.i,
        output_md_path = args.o,
        batch_size = args.batch,
        device_names = args.devices
    )