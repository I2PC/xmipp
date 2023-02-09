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

def fourier_generator(  dataset: Iterable[torch.Tensor],
                        fourier: operators.FourierTransformer2D,
                        flattener: operators.SpectraFlattener,
                        weights: Optional[operators.Weighter] = None,
                        norm: Optional[str] = None,
                        device: Optional[torch.device] = None ) -> torch.Tensor:

    # Process in batches
    ft_images = None
    flat_ft_images = None
    for images in dataset:
        images: torch.Tensor = images.to(device)
        
        # Normalize image if requested
        if norm == 'image':
            utils.normalize(images, dim=(-2, -1))

        # Compute the fourier transform of the images and flatten and weighten it
        ft_images = fourier(images, out=ft_images)
        flat_ft_images = flattener(ft_images, out=flat_ft_images)

        # Apply the weights
        if weights:
            flat_ft_images = weights(flat_ft_images, out=flat_ft_images)

        # Normalize complex numbers if requested
        if norm == 'complex':
            utils.complex_normalize(flat_ft_images)

        # Elaborate the reference vectors
        coefficients = utils.flat_view_as_real(flat_ft_images)
                
        # Normalize reference vectors if requested
        if norm == 'vector':
            utils.l2_normalize(coefficients, dim=-1)

        yield coefficients