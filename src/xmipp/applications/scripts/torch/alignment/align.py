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

from typing import Optional, Tuple
import torch
import torch.multiprocessing as mp
import pandas as pd
import faiss
import faiss.contrib.torch_utils

import operators
import utils
import image

def _image_transformer( loader: torch.utils.data.DataLoader,
                        q_out: mp.JoinableQueue,
                        transformer: operators.Transformer2D,
                        flattener: operators.SpectraFlattener,
                        weighter: operators.Weighter,
                        norm: bool,
                        device: torch.device ):

    is_complex = transformer.has_complex_output()

    t_images = None
    flat_t_images = None
    search_vectors = None
    for images in loader:
        # Compute the fourier transform of the images
        t_images = transformer(images, out=t_images)

        # Flatten
        flat_t_images = flattener(t_images, out=flat_t_images)
        flat_t_images = weighter(flat_t_images, out=flat_t_images)
        
        # Elaborate the reference vectors
        search_vectors = flat_t_images
        if is_complex:
            search_vectors = utils.flat_view_as_real(search_vectors)
        
        # Normalize if performing pearson's correlation
        if norm:
            utils.normalize(search_vectors, dim=-1)
        
        # Feed the queue
        q_out.put(search_vectors.to(device=device, non_blocking=True))
    
    # Finish processing
    q_out.put(None)
    q_out.join()
    
def _projection_searcher(q_in: mp.JoinableQueue,
                         db: faiss.Index, 
                         k: int,
                         device: torch.device ) -> Tuple[torch.Tensor, torch.Tensor]:

    index_vectors = []
    distance_vectors = []

    search_vectors: torch.Tensor = q_in.get()
    while search_vectors is not None:
        # Search them
        distances, indices = db.search(search_vectors, k=k)
        del search_vectors
        q_in.task_done()
        
        # Add them to the result
        index_vectors.append(indices)
        distance_vectors.append(distances)
        
        # Prepare fot the next batch
        search_vectors = q_in.get()

    else:
        q_in.task_done()
    
    # Concatenate all result vectors
    return torch.cat(index_vectors, axis=0), torch.cat(distance_vectors, axis=0)
        

def align(db: faiss.Index, 
          dataset: image.torch_utils.Dataset,
          transformer: operators.Transformer2D,
          flattener: operators.SpectraFlattener,
          weighter: operators.Weighter,
          norm: bool,
          k: int,
          device: Optional[torch.device] = None,
          batch_size: int = 1024,
          queue_len: int = 16 ) -> pd.DataFrame:

    # Create the exchange queue
    coefficient_queue = mp.JoinableQueue(maxsize=queue_len)
    
    # Read all the images to be used as training data
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
    )
    
    # Define the transformer process
    transformer_process = mp.Process(
        target=_image_transformer, 
        kwargs={
            'loader': loader, 
            'q_out': coefficient_queue, 
            'transformer': transformer, 
            'flattener': flattener, 
            'weighter': weighter,
            'norm': norm,
            'device': device
        },
        name='transformer'
    )
    
    # Run all the processes
    transformer_process.start()
    match_indices, match_distances = _projection_searcher(coefficient_queue, db, k, device)
    transformer_process.join()
    
    return match_indices, match_distances