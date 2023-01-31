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
import torchvision
import math

from .. import operators
from .. import utils
from .. import search

def augment_data(db: search.Database, 
                 dataset: torch.utils.data.Dataset,
                 transformer: operators.Transformer2D,
                 flattener: operators.SpectraFlattener,
                 weighter: Optional[operators.Weighter],
                 norm: Optional[str],
                 count: int,
                 max_rotation: float = 180,
                 max_shift: float = 0.1,
                 batch_size: int = 8192,
                 transform_device: Optional[torch.device] = None,
                 out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    # Decide the number of transformations
    n_references = len(dataset) # For the moment select all
    n_transforms = math.floor(count / n_references) # transforms for the rest
    count = n_references * n_transforms
    
    is_complex = transformer.has_complex_output()

    # Read all the images to be used as training data
    pin_memory = transform_device.type=='cuda'
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory
    )

    # Create the training set
    output_shape = (count, db.get_dim())
    if out is None:
        out = torch.empty(output_shape)
        
    # Create the transform randomizer
    random_affine = torchvision.transforms.RandomAffine(
        degrees=max_rotation,
        translate=(max_shift, )*2,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )

    # Fill the training set
    start = 0
    t_affine_images = None
    flat_t_affine_images = None
    for images in loader:
        for i in range(n_transforms):
            end = start + images.shape[0]
            
            # Upload to the device
            images: torch.Tensor = images.to(transform_device)
            
            # Normalize image if requested
            if norm == 'image':
                utils.normalize(images, dim=(-2, -1))
            
            # Transform the images
            transformed_images = random_affine(images)
            t_affine_images = transformer(transformed_images, out=t_affine_images)
            flat_t_affine_images = flattener(t_affine_images, out=flat_t_affine_images)
            if weighter:
                flat_t_affine_images = weighter(flat_t_affine_images, out=flat_t_affine_images)
            
            # Elaborate the train vectors
            train_vectors = flat_t_affine_images
            if is_complex:
                if norm == 'complex':
                    utils.complex_normalize(train_vectors)

                train_vectors = utils.flat_view_as_real(train_vectors)
                
            # Normalize train vectors if requested
            if norm == 'vector':
                utils.l2_normalize(train_vectors, dim=-1)

            # Write it to the destination array
            out[start:end,:] = train_vectors.to(out.device, non_blocking=True)
            
            # Update the index
            start = end
            
            if i % 256 == 0:
                utils.progress_bar(end, count)

    assert(end == count)
    utils.progress_bar(count, count)
    
    return out[:count,:]