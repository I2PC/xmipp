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
import mrcfile

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
    pin_memory = transform_device.type=='cuda'
    
    # Read input files
    images_md = md.sort_by_image_filename(md.read(images_md_path))
    images_paths = list(map(image.parse_path, images_md[md.IMAGE]))
    images_dataset = image.torch_utils.Dataset(images_paths)
    images_loader = torch.utils.data.DataLoader(
        images_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory
    )
    image_size = md.get_image2d_size(images_md)
    
    # Create a MMAPed output file
    output_images_path = str(pathlib.Path(output_md_path).with_suffix('.mrc'))
    output_images = mrcfile.new_mmap(
        output_images_path, 
        shape=(len(images_md), 1) + image_size, 
        mrc_mode=2
    )

    # Process
    start = 0
    batch_images_fourier = None
    ctf_images = None
    wiener_filters = None
    for batch_images in images_loader:
        end = start + len(batch_images)
        batch_images: torch.Tensor = batch_images.to(transform_device, non_blocking=True)
        batch_slice = slice(start, end)
        batch_images_md = images_md.iloc[batch_slice]
        
        # Perform the FFT of the images
        batch_images_fourier = torch.fft.rfft2(batch_images, out=batch_images_fourier)
        
        # Compute the CTF image TODO
        ctf_images = ctf.compute_ctf_image_2d(
            frequency_magnitude2_grid=None,
            frequency_angle_grid=None,
            defocus_average=None,
            defocus_difference=None,
            astigmatism_angle=None,
            wavelength=None,
            spherical_aberration=None,
            phase_shift=None,
            out=ctf_images
        )
        
        # Compute the wiener filter
        wiener_filters = ctf.wiener_2d(ctf_images, out=wiener_filters)
        
        # Apply the filter to the images
        batch_images_fourier *= wiener_filters

        # Perform the inverse FFT computaion
        torch.fft.irfft2(batch_images_fourier, out=batch_images)

        # Store the result
        output_images.data[batch_slice,0] = batch_images.cpu().numpy()
        
        # Prepare for the next batch
        start = end
        progress = end / len(images_md)
        print('{:.2f}%'.format(100*progress))
        
    assert(end == len(images_md))

    # Update metadata
    images_md[md.IMAGE] = (images_md.index + 1).map(('{:06d}@' + output_images_path).format)
    md.write(images_md, output_md_path)
    

if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Align Nearest Neighbor Training',
                        description = 'Align Cryo-EM images using a fast Nearest Neighbor approach')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--device', nargs='*')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        images_md_path = args.i,
        output_md_path = args.o,
        batch_size = args.batch,
        device_names = args.device
    )