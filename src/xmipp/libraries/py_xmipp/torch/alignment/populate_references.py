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

from typing import Iterable, Sequence
import pandas as pd

from .. import generators
from .. import search
from .. import metadata as md

def _create_reference_metadata(reference_indices: Sequence[int],
                               psi_angles: Sequence[float],
                               shifts_x: Sequence[float],
                               shifts_y: Sequence[float] ) -> pd.DataFrame:
        
    # Create the output md
    COLUMNS = [
        md.REF,
        md.ANGLE_PSI,
        md.SHIFT_X,
        md.SHIFT_Y
    ]
    assert(len(reference_indices) == len(psi_angles))
    assert(len(reference_indices) == len(shifts_x))
    assert(len(reference_indices) == len(shifts_y))
    result = pd.DataFrame(
        data=zip(reference_indices, psi_angles, shifts_x, shifts_y),
        columns=COLUMNS
    )
    
    return result

def populate_references(db: search.Database, 
                        dataset: Iterable[generators.TransformedImages],
                        max_size: int ) -> pd.DataFrame:
    
    # Empty the database
    db.reset()
    
    database_device = db.get_input_device()

    # Create arrays for appending MD
    reference_indices = []
    angles = []
    shifts_x = []
    shifts_y = []
    
    # Add elements
    for batch in dataset:
        # yield if it is going to exceed max_size
        if (db.get_item_count() + len(batch)) > max_size:
            yield _create_reference_metadata(
                reference_indices, 
                angles, 
                shifts_x, 
                shifts_y
            )
            
            # Start over
            db.reset()
            reference_indices.clear()
            angles.clear()
            shifts_x.clear()
            shifts_y.clear()
        
        # Fill the metadata
        reference_indices += batch.indices.tolist()
        angles += [batch.angle] * len(batch.indices)
        shifts_x += [float(batch.shift[0])] * len(batch.indices)
        shifts_y += [float(batch.shift[1])] * len(batch.indices)
        
        # Populate the database
        db.add(batch.coefficients.to(database_device))
        

    # yield the rest
    if db.get_item_count() > 0:
        yield _create_reference_metadata(
            reference_indices, 
            angles, 
            shifts_x, 
            shifts_y
        )