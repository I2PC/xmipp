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
import math

import xmippPyModules.swiftalign.image as image
import xmippPyModules.swiftalign.fourier as fourier
import xmippPyModules.swiftalign.ctf as ctf
import xmippPyModules.swiftalign.metadata as md

def _compute_polar_frequency_grid_2d(cartesian: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(cartesian)

    torch.sum(torch.square(cartesian), axis=0, out=result[0])
    torch.atan2(cartesian[1], cartesian[0], out=result[1])
    
    return result

def _compute_differential_defocus_inplace(defocus: torch.Tensor) -> torch.Tensor:
    # Compute the mean in the first column
    # (x1 + x2) / 2
    defocus[:,0] += defocus[:,1]
    defocus[:,0] *= 0.5
    
    # Compute the halved difference in the second column
    # (x1 + x2) / 2 - x2 = (x1 - x2) / 2
    torch.sub(defocus[:,0], defocus[:,1], out=defocus[:,1])
    
    return defocus
    
def _compute_wavelength(voltage: float) -> float:
    return 1.23e-9 / math.sqrt(voltage + 1e-6*voltage*voltage)
    
def run(images_md_path: str, 
        output_md_path: str, 
        pixel_size: float,
        spherical_aberration: float,
        voltage: float,
        phase_flipped: bool,
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
        mrc_mode=2,
        overwrite=True
    )
    
    # Convert units
    voltage *= 1e3 # kV to V
    wavelength = 1e10*_compute_wavelength(voltage)
    spherical_aberration *= 1e7 # mm to A
    
    # Compute frequency grid
    cartesian_frequency_grid = fourier.rfftnfreq(
        image_size,
        d=pixel_size,
        device=transform_device
    )
    polar_frequency_grid = _compute_polar_frequency_grid_2d(cartesian_frequency_grid)

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
        
        # Obtain defocus
        defocus = torch.as_tensor(batch_images_md[[md.CTF_DEFOCUS_U, md.CTF_DEFOCUS_V, md.CTF_DEFOCUS_ANGLE]].to_numpy())
        _compute_differential_defocus_inplace(defocus[:,:2])
        defocus[:,2].deg2rad_()
        defocus = defocus.to(transform_device, non_blocking=True)
        
        # Perform the FFT of the images
        batch_images_fourier = torch.fft.rfft2(batch_images, out=batch_images_fourier)
        
        # Compute the CTF image TODO
        ctf_images = ctf.compute_ctf_image_2d(
            frequency_magnitude2_grid=polar_frequency_grid[0],
            frequency_angle_grid=polar_frequency_grid[1],
            defocus_average=defocus[:,0],
            defocus_difference=defocus[:,1],
            astigmatism_angle=defocus[:,2],
            wavelength=wavelength,
            spherical_aberration=spherical_aberration,
            phase_shift=0.0,
            out=ctf_images
        )
        
        if phase_flipped:
            ctf_images.abs_()
        
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
    parser.add_argument('--pixel_size', type=float, required=True)
    parser.add_argument('--spherical_aberration', type=float, required=True)
    parser.add_argument('--voltage', type=float, required=True)
    parser.add_argument('--phase_flipped', action='store_true')
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--device', nargs='*')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        images_md_path = args.i,
        output_md_path = args.o,
        pixel_size=args.pixel_size,
        spherical_aberration=args.spherical_aberration,
        voltage=args.voltage,
        phase_flipped=args.phase_flipped,
        batch_size = args.batch,
        device_names = args.device
    )