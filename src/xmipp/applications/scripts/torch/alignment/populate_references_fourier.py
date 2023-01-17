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



def populate_references_fourier(db: faiss.Index, 
                                dataset: image.torch_utils.Dataset,
                                rotations: operators.ImageRotator,
                                shifts: operators.FourierShiftFilter,
                                fourier: operators.FourierTransformer2D,
                                flattener: operators.FourierLowPassFlattener,
                                weighter: operators.Weighter,
                                norm: Optional[str],
                                device: Optional[torch.device] = None,
                                batch_size: int = 1024 ) -> pd.DataFrame:
    
    n_transform = rotations.get_count() * shifts.get_count()

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
    ft_rotated_images = None
    flat_ft_rotated_images = None
    shifted_ft = None
    utils.progress_bar(0, len(dataset)*n_transform)
    for images in loader:
        n_images = images.shape[0]
        end = start + n_images

        # Normalize image if requested
        if norm:
            utils.normalize(images, dim=(-2, -1))

        # Add the references as many times as their transformations
        reference_indices += list(range(start, end)) * n_transform
        for rot_index in range(rotations.get_count()):

            # Rotate the input image
            rotated_images = rotations(images, rot_index, out=rotated_images)

            # Compute the fourier transform of the images and flatten and weighten it
            ft_rotated_images = fourier(rotated_images, out=ft_rotated_images)
            flat_ft_rotated_images = flattener(ft_rotated_images, out=flat_ft_rotated_images)
            flat_ft_rotated_images = weighter(flat_ft_rotated_images, out=flat_ft_rotated_images)

            # Add the rotation angle as many times as shifts and images
            psi = rotations.get_angle(rot_index)
            psi_angles += [psi] * (n_images * shifts.get_count())
            for shift_index in range(shifts.get_count()):
                shifted_ft = shifts(flat_ft_rotated_images, shift_index, out=shifted_ft)
                
                # Elaborate the reference vectors
                reference_vectors = utils.flat_view_as_real(shifted_ft)
                
                # Normalize reference vectors if requested
                if norm == 'vector':
                    utils.l2_normalize(reference_vectors, dim=-1)

                # Populate the database
                db.add(reference_vectors)
                
                # Add the current shift for all images
                sx, sy = shifts.get_shift(shift_index)
                x_shifts += [-float(sx)] * n_images
                y_shifts += [-float(sy)] * n_images
                
        # Advance indices
        start = end
        utils.progress_bar(end*n_transform, len(dataset)*n_transform)
        
    projection_md = create_reference_metadata(reference_indices, psi_angles, x_shifts, y_shifts)
    assert(len(projection_md) == db.ntotal)
    return projection_md
    
    