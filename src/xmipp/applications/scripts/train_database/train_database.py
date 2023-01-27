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
import faiss
import argparse

import xmippPyModules.torch.image as image
import xmippPyModules.torch.search as search
import xmippPyModules.torch.alignment as alignment
import xmippPyModules.torch.operators as operators
import xmippPyModules.torch.metadata as md



def run(reference_md_path: str, 
        weight_image_path: Optional[str],
        index_path: str,
        max_shift : float,
        n_training: int,
        n_samples: int,
        cutoff: float,
        method: str,
        norm: Optional[str],
        gpu: list ):
   
    # Devices
    if gpu:
        device = torch.device('cuda', int(gpu[0]))
    else:
        device = torch.device('cpu')

    transform_device = device
    db_device = device
    
    # Read input files
    reference_md = md.read(reference_md_path)
    image_size, _ = md.get_image_size(reference_md)
    
    # Create the transformer and flattener
    # according to the transform method
    if method == 'fourier':
        transformer = operators.FourierTransformer2D()
        flattener = operators.FourierLowPassFlattener(image_size, cutoff, device=transform_device)
    elif method == 'dct':
        transformer = operators.DctTransformer2D(image_size, device=transform_device)
        flattener = operators.DctLowPassFlattener(image_size, cutoff, device=transform_device)
        
    # Create the weighter
    weighter = None
    if weight_image_path:
        weights = torch.tensor(image.read(weight_image_path)) if weight_image_path else None
        weighter = operators.Weighter(weights, flattener, device=transform_device) 
    
    # Consider complex numbers
    dim = flattener.get_length()
    if transformer.has_complex_output():
        dim *= 2
    
    # Create the DB to store the data
    recipe = search.opq_ifv_pq_recipe(dim, n_samples)
    print(f'Data dimensions: {dim}')
    print(f'Database: {recipe}')
    db = search.FaissDatabase(dim, recipe)
    #db = search.MedianHashDatabase(dim)
    db.to_device(db_device)
    
    # Do some work
    print('Augmenting data')
    image_paths = list(map(image.parse_path, reference_md[md.IMAGE]))
    dataset = image.torch_utils.Dataset(image_paths)
    training_set = alignment.augment_data(
        db,
        dataset=dataset,
        transformer=transformer,
        flattener=flattener,
        weighter=weighter,
        norm=norm,
        count=n_training,
        max_rotation=180,
        max_shift=max_shift,
        batch_size=8192,
        transform_device=transform_device,
        store_device=torch.device('cpu') # Augmented dataset is very large
    )
    
    print('Training')
    db.train(training_set)
    
    # Write to disk
    db.to_device(torch.device('cpu'))
    db.write(index_path)


if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Align Nearest Neighbor Training',
                        description = 'Align Cryo-EM images using a fast Nearest Neighbor approach')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--weights')
    parser.add_argument('--max_shift', type=float, required=True)
    parser.add_argument('--training', type=int, default=int(4e6))
    parser.add_argument('--size', type=int, default=int(2e6))
    parser.add_argument('--max_frequency', type=float, required=True)
    parser.add_argument('--method', type=str, default='fourier')
    parser.add_argument('--norm', type=str)
    parser.add_argument('--gpu', nargs='*')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        reference_md_path = args.i,
        index_path = args.o,
        weight_image_path = args.weights,
        max_shift = args.max_shift,
        n_training = args.training,
        n_samples = args.size,
        cutoff = args.max_frequency,
        method = args.method,
        norm = args.norm,
        gpu = args.gpu
    )