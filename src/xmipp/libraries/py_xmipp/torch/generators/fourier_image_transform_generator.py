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
import torch

from .. import operators
from .. import utils

from .TransformedImages import TransformedImages

def fourier_image_transform_generator(dataset: Iterable[torch.Tensor],
                                      fourier: operators.FourierTransformer2D,
                                      flattener: operators.SpectraFlattener,
                                      rotations: Optional[operators.ImageRotator] = None,
                                      shifts: Optional[operators.ImageShifter] = None,
                                      weights: Optional[operators.Weighter] = None,
                                      norm: Optional[str] = None,
                                      device: Optional[torch.device] = None ) -> TransformedImages:

    # Process in batches
    start = 0
    indices = None
    rotated_images = None
    ft_rotated_images = None
    flat_ft_rotated_images = None
    shifted_ft = None
    for images in dataset:
        n_images = images.shape[0]
        end = start + n_images
        images: torch.Tensor = images.to(device)
        
        # Fill the indices
        indices = torch.arange(
            start=start, end=end, 
            out=indices
        )

        # Normalize image if requested
        if norm:
            utils.normalize(images, dim=(-2, -1))

        # Add the references as many times as their transformations
        for rot_index in range(rotations.get_count()):
            angle = rotations.get_angle(rot_index)
            
            # Rotate the input image
            rotated_images = rotations(images, rot_index, out=rotated_images)

            # Compute the fourier transform of the images and flatten and weighten it
            ft_rotated_images = fourier(rotated_images, out=ft_rotated_images)
            flat_ft_rotated_images = flattener(ft_rotated_images, out=flat_ft_rotated_images)
            if weights:
                flat_ft_rotated_images = weights(flat_ft_rotated_images, out=flat_ft_rotated_images)

            for shift_index in range(shifts.get_count()):
                shift = shifts.get_shift(shift_index)

                # Apply the shift
                shifted_ft = shifts(flat_ft_rotated_images, shift_index, out=shifted_ft)
                
                # Normalize complex numbers if requested
                if norm == 'complex':
                    utils.complex_normalize(shifted_ft)

                # Elaborate the reference vectors
                coefficients = utils.flat_view_as_real(shifted_ft)
                
                # Normalize reference vectors if requested
                if norm == 'vector':
                    utils.l2_normalize(coefficients, dim=-1)

                yield TransformedImages(
                    coefficients=coefficients,
                    indices=indices,
                    angle=angle,
                    shift=shift,
                )
                
        # Advance batch indices
        start = end
    