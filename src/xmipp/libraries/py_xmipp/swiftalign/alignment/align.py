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

from typing import Iterable
import torch

from .. import search
        

def align(db: search.Database, 
          dataset: Iterable[torch.Tensor],
          k: int ) -> search.SearchResult:

    database_device = db.get_input_device()

    index_vectors = []
    distance_vectors = []

    for vectors in dataset:
        # Search them
        s = db.search(vectors.to(database_device), k=k)
        
        # Add them to the result
        index_vectors.append(s.indices.cpu())
        distance_vectors.append(s.distances.cpu())
        
    # Concatenate all result vectors
    return search.SearchResult(
        indices=torch.cat(index_vectors, axis=0), 
        distances=torch.cat(distance_vectors, axis=0)
    )
    
    