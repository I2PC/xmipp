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

import pandas as pd
import torch
import numpy as np

from .. import metadata as md
from .. import search

def _ensemble_alignment_md(reference_md: pd.DataFrame,
                           projection_md: pd.DataFrame,
                           match_distances: np.ndarray,
                           match_indices: np.ndarray,
                           local_transform_md: Optional[pd.DataFrame] = None) -> pd.DataFrame:

    REFERENCE_COLUMNS = [
        md.ANGLE_ROT, 
        md.ANGLE_TILT, 
        md.REFERENCE_IMAGE, 
    ]
    if md.REF3D in reference_md.columns:
        REFERENCE_COLUMNS.append(md.REF3D) # Used for 3D classification

    result = pd.DataFrame(match_distances, columns=[md.COST])
    
    # Left-join the projection metadata to the result
    result = result.join(projection_md, on=match_indices)
    
    # Left-join the reference metadata to the result
    result = result.join(reference_md[REFERENCE_COLUMNS], on=md.REF)
    
    # Drop the indexing columns
    result.drop(md.REF, axis=1, inplace=True)

    # Accumulate transforms
    if local_transform_md is not None:
        result[local_transform_md.columns] += local_transform_md
        
        # Make psi in [-180, 180]
        if md.ANGLE_PSI in local_transform_md.columns:
            result[result[md.ANGLE_PSI] > +180.0] -= 360.0
            result[result[md.ANGLE_PSI] < -180.0] += 360.0

    return result

def _update_alignment_metadata(output_md: pd.DataFrame,
                               reference_md: pd.DataFrame,
                               projection_md: pd.DataFrame,
                               match_distances: np.ndarray,
                               match_indices: np.ndarray,
                               local_transform_md: Optional[pd.DataFrame] = None ) -> pd.DataFrame:
    # Select the rows to be updated
    selection = match_distances < output_md[md.COST]
    
    if local_transform_md is not None:
        local_transform_md = local_transform_md[selection]
    
    # Do an alignment for the selected rows
    alignment_md = _ensemble_alignment_md(
        reference_md=reference_md,
        projection_md=projection_md,
        match_distances=match_distances[selection],
        match_indices=match_indices[selection],
        local_transform_md=local_transform_md
    )
    
    # Update output
    output_md.loc[selection, alignment_md.columns] = alignment_md
    
    return output_md

def _create_alignment_metadata(experimental_md: pd.DataFrame,
                               reference_md: pd.DataFrame,
                               projection_md: pd.DataFrame,
                               match_distances: np.ndarray,
                               match_indices: np.ndarray,
                               local_transform_md: Optional[pd.DataFrame] = None ) -> pd.DataFrame:
    
    # Use the first match
    alignment_md = _ensemble_alignment_md(
        reference_md=reference_md, 
        projection_md=projection_md, 
        match_distances=match_distances, 
        match_indices=match_indices,
        local_transform_md=local_transform_md
    )
    
    # Add the alignment consensus to the output
    output_md = experimental_md.drop(columns=alignment_md.columns, errors='ignore')
    output_md = output_md.join(alignment_md)
    
    # Reorder columns for more convenient reading
    output_md = output_md.reindex(
        columns=experimental_md.columns.union(output_md.columns, sort=False)
    )
    
    return output_md

def generate_alignment_metadata(experimental_md: pd.DataFrame,
                                reference_md: pd.DataFrame,
                                projection_md: pd.DataFrame,
                                matches: search.SearchResult,
                                local_transform_md: Optional[pd.DataFrame] = None,
                                output_md: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    
    # Rename the reference image column to make it compatible 
    # with the resulting MD. (No duplicated IMAGE column)
    reference_md = reference_md.rename(columns={
        md.IMAGE: md.REFERENCE_IMAGE,
    })
    
    # Currently we only support kNN with k=1
    # Extract the best result if multiple provided
    match_distances = matches.distances[:,0].numpy()
    match_indices = matches.indices[:,0].numpy()
    
    # Update or generate depending on wether the output is provided
    if output_md is None:
        output_md = _create_alignment_metadata(
            experimental_md=experimental_md,
            reference_md=reference_md,
            projection_md=projection_md,
            match_distances=match_distances,
            match_indices=match_indices,
            local_transform_md=local_transform_md
        )
    else:
        output_md = _update_alignment_metadata(
            output_md=output_md,
            reference_md=reference_md,
            projection_md=projection_md,
            match_distances=match_distances,
            match_indices=match_indices,
            local_transform_md=local_transform_md
        )
    
    assert(output_md is not None)
    return output_md
