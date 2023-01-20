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

from typing import Sequence
import pandas as pd

from .. import metadata as md

def create_reference_metadata(  reference_indices: Sequence[int],
                                psi_angles: Sequence[float],
                                x_shifts: Sequence[float],
                                y_shifts: Sequence[float] ) -> pd.DataFrame:
        
    # Create the output md
    COLUMNS = [
        md.REF,
        md.ANGLE_PSI,
        md.SHIFT_X,
        md.SHIFT_Y
    ]
    assert(len(reference_indices) == len(psi_angles))
    assert(len(reference_indices) == len(x_shifts))
    assert(len(reference_indices) == len(y_shifts))
    result = pd.DataFrame(
        data=zip(reference_indices, psi_angles, x_shifts, y_shifts),
        columns=COLUMNS
    )
    
    return result