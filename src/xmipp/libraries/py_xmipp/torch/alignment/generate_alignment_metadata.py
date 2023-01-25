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

import pandas as pd
import torch

from .. import metadata as md
from .. import search

def _ensemble_alignment_md(reference_md: pd.DataFrame,
                           projection_md: pd.DataFrame,
                           match_distances: torch.Tensor,
                           match_indices: torch.IntTensor ) -> pd.DataFrame:

    REFERENCE_COLUMNS = [md.ANGLE_ROT, md.ANGLE_TILT, md.REFERENCE_IMAGE]

    result = pd.DataFrame(match_distances.cpu(), columns=['distance'])
    
    # Left-join the projection metadata to the result
    result = result.join(projection_md, on=match_indices.cpu())
    
    # Left-join the reference metadata to the result
    result = result.join(reference_md[REFERENCE_COLUMNS], on=md.REF)
    
    # Drop the indexing columns
    result.drop(md.REF, axis=1, inplace=True)

    return result

def generate_alignment_metadata(experimental_md: pd.DataFrame,
                                reference_md: pd.DataFrame,
                                projection_md: pd.DataFrame,
                                matches: search.SearchResult ) -> pd.DataFrame:
    
    
    # Create the resulting array shifting old alignment values
    output_md = experimental_md.rename(columns={
        md.ANGLE_PSI: md.ANGLE_PSI2,
        md.ANGLE_ROT: md.ANGLE_ROT2,
        md.ANGLE_TILT: md.ANGLE_TILT2,
        md.SHIFT_X: md.SHIFT_X2,
        md.SHIFT_Y: md.SHIFT_Y2,
    })
    
    # Rename the reference image column to make it compatible 
    # with the resulting MD. (No duplicated IMAGE column)
    # Also add a invalid reference element
    reference_md = reference_md.rename(columns={
        md.IMAGE: md.REFERENCE_IMAGE,
    })
    
    # Use the first match
    alignment_md = _ensemble_alignment_md(
        reference_md=reference_md, 
        projection_md=projection_md, 
        match_distances=matches.distances[:,0], 
        match_indices=matches.indices[:,0]
    )
    
    # Add the alignment consensus to the output
    output_md = output_md.join(alignment_md)
    
    # Reorder columns for more convenient reading
    output_md = output_md.reindex(
        columns=experimental_md.columns.union(output_md.columns, sort=False)
    )
    
    return output_md
