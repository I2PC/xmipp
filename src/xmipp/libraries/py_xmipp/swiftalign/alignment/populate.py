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
import pandas as pd

from .. import search
from .. import metadata as md
from .InPlaneTransformBatch import InPlaneTransformBatch
 
def populate(db: search.Database, 
             dataset: Iterable[InPlaneTransformBatch] ) -> pd.DataFrame:

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
        # Populate the database
        db.add(batch.vectors.to(database_device))

        # Fill the metadata
        reference_indices += batch.indices.tolist()
        angles += [batch.angle] * len(batch.indices)
        shifts_x += [float(batch.shift[0])] * len(batch.indices)
        shifts_y += [float(batch.shift[1])] * len(batch.indices)
    
    # Create the output md
    COLUMNS = [
        md.REF,
        md.ANGLE_PSI,
        md.SHIFT_X,
        md.SHIFT_Y
    ]
    assert(len(reference_indices) == len(angles))
    assert(len(reference_indices) == len(shifts_x))
    assert(len(reference_indices) == len(shifts_y))
    
    result = pd.DataFrame(
        data=zip(reference_indices, angles, shifts_x, shifts_y),
        columns=COLUMNS
    )
    
    return result