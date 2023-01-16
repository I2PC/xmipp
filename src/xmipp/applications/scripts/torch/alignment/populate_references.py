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
import pandas as pd
import faiss
import faiss.contrib.torch_utils

import operators
import utils
import image

from .create_reference_metadata import create_reference_metadata


def populate_references(db: faiss.Index, 
                        dataset: image.torch_utils.Dataset,
                        rotations: operators.ImageRotator,
                        shifts: operators.ImageShifter,
                        transformer: operators.Transformer2D,
                        flattener: operators.SpectraFlattener,
                        weighter: operators.Weighter,
                        device: Optional[torch.device] = None,
                        batch_size: int = 1024 ) -> pd.DataFrame:
    
    n_transform = shifts.get_count() * rotations.get_count()

    is_complex = transformer.has_complex_output()

    # Create the data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x : torch.utils.data.default_collate(x).to(device, non_blocking=True)
    )

    # Create arrays for appending MD
    reference_indices = []
    psi_angles = []
    x_shifts = []
    y_shifts = []
    
    # Process in batches
    start = 0
    rotated_images = None
    transformed_images = None
    t_transformed_images = None
    flat_t_transformed_images = None
    utils.progress_bar(0, len(dataset)*n_transform)
    for images in loader:
        n_images = images.shape[0]
        end = start + n_images

        # Add the references as many times as their transformations
        for angle_index in range(rotations.get_count()):
            # Rotate the images
            rotated_images = rotations(images, angle_index, out=rotated_images)
            
            for shift_index in range(shifts.get_count()):
                # Shift the images
                transformed_images = shifts(rotated_images, shift_index, out=transformed_images)

                # Compute the transform of the images and flatten and weighten it
                t_transformed_images = transformer(transformed_images, out=t_transformed_images)
                flat_t_transformed_images = flattener(t_transformed_images, out=flat_t_transformed_images)
                flat_t_transformed_images = weighter(flat_t_transformed_images, out=flat_t_transformed_images)

                # Elaborate the reference vectors
                reference_vectors = flat_t_transformed_images
                if is_complex:
                    reference_vectors = utils.flat_view_as_real(reference_vectors)
                
                # Populate the database
                db.add(reference_vectors)
                
                # Add the current transform
                sx, sy = shifts.get_shift(shift_index)
                x_shifts += [-float(sx)] * n_images
                y_shifts += [-float(sy)] * n_images
                
            psi = rotations.get_angle(angle_index)
            psi_angles += [psi] * (n_images * shifts.get_count())
                
        reference_indices += list(range(start, end)) * n_transform
                
        # Advance indices
        start = end
        utils.progress_bar(end*n_transform, len(dataset)*n_transform)
        
    projection_md = create_reference_metadata(reference_indices, psi_angles, x_shifts, y_shifts)
    assert(len(projection_md) == db.ntotal)
    return projection_md
    
    