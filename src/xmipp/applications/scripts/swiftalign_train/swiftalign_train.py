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

from typing import Optional, Iterable
import argparse
import math
import torch

import xmippPyModules.swiftalign.image as image
import xmippPyModules.swiftalign.operators as operators
import xmippPyModules.swiftalign.fourier as fourier
import xmippPyModules.swiftalign.search as search
import xmippPyModules.swiftalign.alignment as alignment
import xmippPyModules.swiftalign.metadata as md

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

def run(reference_md_path: str, 
        index_path: str,
        recipe: str,
        ctf_md_path: Optional[str],
        weight_image_path: Optional[str],
        max_shift : float,
        max_psi: float,
        cutoff: float,
        norm: Optional[str],
        n_training: int,
        n_batch: int,
        device_names: list,
        scratch_path: Optional[str],
        use_f16: bool,
        use_precomputed: bool ):
   
    # Devices
    if device_names:
        devices = list(map(torch.device, device_names))
    else:
        devices = [torch.device('cpu')]

    transform_device = devices[0]
    db_device = devices[0]
    
    # Read input files
    reference_md =  md.sort_by_image_filename(md.read(reference_md_path))
    image_size = md.get_image2d_size(reference_md)

    # Create the flattener
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

    # Create the transformer
    transformer = alignment.FourierInPlaneTransformAugmenter(
        max_psi=max_psi,
        max_shift=max_shift,
        flattener=flattener,
        ctfs=ctfs,
        weights=weights,
        norm=norm
    )
    
    # Create the image loader
    image_paths = list(map(image.parse_path, reference_md[md.IMAGE]))
    dataset = image.torch_utils.Dataset(image_paths)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=n_batch,
        pin_memory=transform_device.type=='cuda',
        num_workers=1
    )
    
    # Create the image transformer
    n_repetitions = n_training // len(dataset)
    n_training = n_repetitions * len(dataset)
    
    # Create the DB to store the data
    dim = flattener.get_length()*2
    print(f'Data dimensions: {dim}')
    db = search.FaissDatabase(dim, recipe)
    db.to_device(db_device, use_f16=use_f16, use_precomputed=use_precomputed)
    
    # Create the storage for the training set.
    # This will be LARGE. Therefore provide a MMAP path
    training_set_shape = (n_training, dim)
    if scratch_path:
        size = math.prod(training_set_shape)
        storage = torch.FloatStorage.from_file(scratch_path, shared=True, size=size)
        training_set = torch.FloatTensor(storage=storage)
        training_set = training_set.view(training_set_shape)
    else:
        training_set = torch.empty(training_set_shape, device=torch.device('cpu'))

    # Run the training
    uploader = map(lambda x : x.to(transform_device, non_blocking=True), loader)
    alignment.train(
        db,
        dataset=transformer(uploader, times=n_repetitions),
        scratch=training_set
    )
    
    # Write to disk
    db.to_device(torch.device('cpu'))
    db.write(index_path)


if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Align Nearest Neighbor Training',
                        description = 'Align Cryo-EM images using a fast Nearest Neighbor approach')
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('--recipe', type=str, required=True)
    parser.add_argument('--ctf', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--max_shift', type=float, required=True)
    parser.add_argument('--max_psi', type=float, default=180.0)
    parser.add_argument('--max_frequency', type=float, required=True)
    parser.add_argument('--norm', type=str)
    parser.add_argument('--training', type=int, default=int(4e6))
    parser.add_argument('--batch', type=int, default=int(1024))
    parser.add_argument('--device', nargs='*')
    parser.add_argument('--scratch', type=str)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_precomputed', action='store_true')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        reference_md_path = args.i,
        index_path = args.o,
        recipe = args.recipe,
        ctf_md_path = args.ctf,
        weight_image_path = args.weights,
        max_shift = args.max_shift,
        max_psi = args.max_psi,
        cutoff = args.max_frequency,
        norm = args.norm,
        n_training = args.training,
        n_batch = args.batch,
        device_names = args.device,
        scratch_path=args.scratch,
        use_f16=args.fp16,
        use_precomputed=args.use_precomputed
    )