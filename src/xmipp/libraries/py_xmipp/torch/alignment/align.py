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

from .. import operators
from .. import utils
from .. import image
from .. import search

        

def align(db: search.Database, 
          dataset: image.torch_utils.Dataset,
          transformer: operators.Transformer2D,
          flattener: operators.SpectraFlattener,
          weighter: operators.Weighter,
          norm: bool,
          k: int,
          transform_device: Optional[torch.device] = None,
          batch_size: int = 1024,
          queue_len: int = 16 ) -> search.SearchResult:

    # Read all the images to be used as training data
    pin_memory = transform_device.type=='cuda'
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory
    )

    is_complex = transformer.has_complex_output()
    database_device = db.get_input_device()

    index_vectors = []
    distance_vectors = []

    t_images = None
    flat_t_images = None
    search_vectors = None
    for images in loader:
        images: torch.Tensor = images.to(transform_device)
        
        # Normalize image if requested
        if norm == 'image':
            utils.normalize(images, dim=(-2, -1))

        # Compute the fourier transform of the images
        t_images = transformer(images, out=t_images)

        # Flatten
        flat_t_images = flattener(t_images, out=flat_t_images)
        if weighter:
            flat_t_images = weighter(flat_t_images, out=flat_t_images)
        
        # Elaborate the reference vectors
        search_vectors = flat_t_images
        if is_complex:
            if norm == 'complex':
                utils.complex_normalize(search_vectors)
            
            search_vectors = utils.flat_view_as_real(search_vectors)
        
        # Normalize search vectors if requested
        if norm == 'vector':
            utils.l2_normalize(search_vectors, dim=-1)

        # Search them
        s = db.search(search_vectors.to(database_device), k=k)
        
        # Add them to the result
        index_vectors.append(s.indices.cpu())
        distance_vectors.append(s.distances.cpu())
        
    # Concatenate all result vectors
    return search.SearchResult(
        indices=torch.cat(index_vectors, axis=0), 
        distances=torch.cat(distance_vectors, axis=0)
    )
    
    