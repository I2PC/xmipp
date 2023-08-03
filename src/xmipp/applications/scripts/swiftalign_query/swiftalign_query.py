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
import itertools
import time
import pandas as pd

import xmippPyModules.swiftalign.image as image
import xmippPyModules.swiftalign.search as search
import xmippPyModules.swiftalign.alignment as alignment
import xmippPyModules.swiftalign.operators as operators
import xmippPyModules.swiftalign.fourier as fourier
import xmippPyModules.swiftalign.metadata as md

def _dataframe_batch_generator(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    for i in range(0, len(df), batch_size):
        start = i
        end = start + batch_size
        yield df[start:end]

def _read_weights(path: Optional[str], 
                  flattener: operators.SpectraFlattener, 
                  device: Optional[torch.device] = None ) -> torch.Tensor:
    weights = None
    if path:
        weight_image = image.read(path)
        weight_image = torch.tensor(weight_image, device=device)
        weight_image = fourier.remove_symmetric_half(weight_image)
        weights = flattener(weight_image)
        weights = torch.sqrt(weights, out=weights)
    
    return weights

def _read_ctf(path: Optional[str],
              flattener: operators.SpectraFlattener,
              device: Optional[torch.device] = None ) -> torch.Tensor:
    ctfs = None
    ctf_md = None
    if path:
        ctf_md = md.read(path)
        ctf_paths = list(map(image.parse_path, ctf_md[md.IMAGE]))
        ctf_dataset = image.torch_utils.Dataset(ctf_paths)
        ctf_images = torch.utils.data.default_collate([ctf_dataset[i] for i in range(len(ctf_dataset))])
        ctf_images = fourier.remove_symmetric_half(ctf_images)
        ctfs = flattener(ctf_images.to(device))
        
    return ctfs

def _calculate_rotations(max_psi: float, 
                         n_rotations: int ) -> torch.Tensor:
    
    result = None
    if n_rotations > 1:
        if max_psi >= 180:
            # Consider [-180, +180)
            result = torch.linspace(-180.0, +180, n_rotations+1)[:-1]
        else:
            # Consider [-max_psi, +max_psi]
            result = torch.linspace(-max_psi, +max_psi, n_rotations)
    else:
        # No rotations
        result = torch.full((1, ), 0.0)
        
    return result
        

def _calculate_shifts(max_shift: float, 
                      n_shifts: int, 
                      image_size: Tuple[int, int]) -> torch.Tensor:
    max_shift_x = max_shift*image_size[0]
    max_shift_y = max_shift*image_size[1]
    shifts_x = torch.linspace(-max_shift_x, +max_shift_x, n_shifts)
    shifts_y = torch.linspace(-max_shift_y, +max_shift_y, n_shifts)
    shifts = torch.cartesian_prod(shifts_x, shifts_y)
    return shifts

def run(experimental_md_path: str, 
        reference_md_path: str, 
        index_path: str,
        ctf_md_path: Optional[str],
        weight_image_path: Optional[str],
        output_md_path: str,
        n_rotations : int,
        n_shifts : int,
        max_psi : float,
        max_shift : float,
        cutoff: float,
        batch_size: int,
        max_size: int,
        norm: Optional[str],
        local: bool,
        local_shift: bool,
        drop_na: bool,
        reference_labels: Sequence[str],
        k: int,
        device_names: list,
        use_f16: bool,
        use_precomputed: bool ):
    
    # Devices
    if device_names:
        devices = list(map(torch.device, device_names))
    else:
        devices = [torch.device('cpu')]
    
    transform_device = torch.device('cpu')
    db_device = devices[0]
    
    # Read input files
    experimental_md = md.sort_by_image_filename(md.read(experimental_md_path))
    reference_md = md.sort_by_image_filename(md.read(reference_md_path))
    image_size = md.get_image2d_size(experimental_md)
    
    # Read the database
    db = search.FaissDatabase()
    db.read(index_path)
    db.to_device(db_device, use_f16=use_f16, reserve_vecs=max_size, use_precomputed=use_precomputed)
    
    # Create the in-plane transforms
    angles = _calculate_rotations(max_psi=max_psi, n_rotations=n_rotations)   
    shifts = _calculate_shifts(max_shift=max_shift, n_shifts=n_shifts, image_size=image_size)
    n_transform = len(angles) * len(shifts)
    print(f'Performing {n_transform} transformations to each reference image')
    
    # Create the band flattener
    flattener = operators.FourierLowPassFlattener(
        dim=image_size,
        cutoff=cutoff,
        exclude_dc=True,
        device=transform_device
    )

    # Read weights
    weights = _read_weights(weight_image_path, flattener, transform_device)

    # Read CTFs
    ctfs = _read_ctf(ctf_md_path, flattener, transform_device)

    # Create the transformers
    reference_transformer = alignment.FourierInPlaneTransformGenerator(
        dim=image_size,
        angles=angles,
        shifts=shifts,
        flattener=flattener,
        ctfs=ctfs,
        weights=weights,
        norm=norm,
        device=transform_device
    )
    experimental_transformer = alignment.FourierInPlaneTransformCorrector(
        flattener=flattener,
        weights=weights,
        norm=norm,
        device=transform_device
    )

    # Create the reference dataset
    reference_paths = list(map(image.parse_path, reference_md[md.IMAGE]))
    reference_dataset = image.torch_utils.Dataset(reference_paths)
    experimental_paths = list(map(image.parse_path, experimental_md[md.IMAGE]))
    experimental_dataset = image.torch_utils.Dataset(experimental_paths)
    n_total = len(reference_dataset) * n_transform
    print(f'In total we will consider {n_total} transformed references')
    
    # Create the loaders
    pin_memory = transform_device.type == 'cuda'
    reference_loader = torch.utils.data.DataLoader(
        reference_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=1
    )
    reference_batch_iterator = iter(reference_transformer(reference_loader))
    
    alignment_md = None
    n_batches_per_iteration = max(1, max_size // min(batch_size, len(reference_dataset)))
    if local:
        local_columns = [md.ANGLE_PSI, md.SHIFT_X, md.SHIFT_Y]
    elif local_shift:
        local_columns = [md.SHIFT_X, md.SHIFT_Y]
    else:
        local_columns = []
    local_transform_md = experimental_md[local_columns]
    populate_time = 0.0
    alignment_time = 0.0
    while True:
        
        print('Uploading')
        start_time = time.perf_counter()
        projection_md = alignment.populate(
            db,
            dataset=itertools.islice(reference_batch_iterator, n_batches_per_iteration)
        )
        end_time = time.perf_counter()
        populate_time += end_time - start_time

        
        if len(projection_md) == 0:
            break
    
        experimental_loader = torch.utils.data.DataLoader(
            experimental_dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=1
        )

        print('Aligning')
        start_time = time.perf_counter()
        matches = alignment.align(
            db,
            experimental_transformer(zip(experimental_loader, _dataframe_batch_generator(local_transform_md, batch_size))),
            k=k
        )
    
        alignment_md = alignment.generate_alignment_metadata(
            experimental_md=experimental_md,
            reference_md=reference_md,
            projection_md=projection_md,
            matches=matches,
            local_transform_md=local_transform_md,
            reference_columns=reference_labels,
            output_md=alignment_md
        )
        end_time = time.perf_counter()
        alignment_time += end_time - start_time

    
    print('Populate time (s): ' + str(populate_time))
    print('Alignment time (s): ' + str(alignment_time))
    print('Alignment time per particle (ms/part.): ' + str(alignment_time*1e3/len(experimental_dataset)))
    
    if drop_na:
        alignment_md.dropna(inplace=True)
    
    alignment_md.sort_index(axis=0, inplace=True)
    md.write(alignment_md, output_md_path)



if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Align Nearest Neighbor Training',
                        description = 'Align Cryo-EM images using a fast Nearest Neighbor approach')
    parser.add_argument('-i', required=True)
    parser.add_argument('-r', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--weights')
    parser.add_argument('--index', required=True)
    parser.add_argument('--ctf', type=str)
    parser.add_argument('--rotations', type=int, required=True)
    parser.add_argument('--shifts', type=int, required=True)
    parser.add_argument('--max_shift', type=float, required=True)
    parser.add_argument('--max_psi', type=float, default=180.0)
    parser.add_argument('--max_frequency', type=float, required=True)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--norm', type=str)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--local_shift', action='store_true')
    parser.add_argument('--dropna', action='store_true')
    parser.add_argument('--reference_labels', type=str, nargs='*')
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('--devices', nargs='*')
    parser.add_argument('--max_size', type=int, default=int(2e6))
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_precomputed', action='store_true')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        experimental_md_path = args.i,
        reference_md_path = args.r,
        index_path = args.index,
        ctf_md_path = args.ctf,
        weight_image_path = args.weights,
        output_md_path = args.o,
        n_rotations = args.rotations,
        n_shifts = args.shifts,
        max_shift = args.max_shift,
        max_psi = args.max_psi,
        cutoff = args.max_frequency,
        batch_size = args.batch,
        max_size = args.max_size,
        local = args.local,
        local_shift = args.local_shift,
        norm = args.norm,
        drop_na = args.dropna,
        reference_labels = args.reference_labels,
        k = args.k,
        device_names = args.devices,
        use_f16 = args.fp16,
        use_precomputed=args.use_precomputed
    )