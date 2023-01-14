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
import torch

import metadata as md

def _ensemble_alignment_md(reference_md: pd.DataFrame,
                           projection_md: pd.DataFrame,
                           match_indices: torch.Tensor,
                           match_distances: torch.Tensor ) -> pd.DataFrame:

    REFERENCE_COLUMNS = [md.ANGLE_ROT, md.ANGLE_TILT, md.REFERENCE_IMAGE]

    result = pd.DataFrame(match_distances, columns=['distance'])
    
    # Left-join the projection metadata to the result
    result = result.join(projection_md, on=match_indices)
    
    # Left-join the reference metadata to the result
    result = result.join(reference_md[REFERENCE_COLUMNS], on=md.REF)
    
    # Drop the indexing columns
    result.drop(md.REF, axis=1, inplace=True)

    return result

def _do_pose_consensus(alignments: Sequence[pd.DataFrame]) -> pd.DataFrame:
    POSE_COLUMNS = [md.ANGLE_PSI, md.ANGLE_ROT, md.ANGLE_TILT]
    
    # Obtain the shifts from the alignments
    euler_angles = torch.stack(list(map(lambda x : torch.tensor(x[POSE_COLUMNS].values), alignments)))

    return alignments[0][POSE_COLUMNS]

def _do_shift_consensus(alignments: Sequence[pd.DataFrame]) -> pd.DataFrame:
    SHIFT_COLUMNS = [md.SHIFT_X, md.SHIFT_Y]

    # Obtain the shifts from the alignments
    shifts = torch.stack(list(map(lambda x : torch.tensor(x[SHIFT_COLUMNS].values), alignments)))
    distances = torch.stack(list(map(lambda x : torch.tensor(x['distance'].values), alignments)))
    
    # Compute the weights
    weights = torch.exp(-distances)
    #weights /= torch.sum(weights, dim=1)
    #print(weights)
    
    #weighted_shifts = torch.mul(shifts, weights)

    return alignments[0][SHIFT_COLUMNS]
    

def generate_alignment_metadata(experimental_md: pd.DataFrame,
                                reference_md: pd.DataFrame,
                                projection_md: pd.DataFrame,
                                match_indices: torch.Tensor,
                                match_distances: torch.Tensor ) -> pd.DataFrame:
    
    
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
    #reference_md.loc[-1, [md.ANGLE_ROT, md.ANGLE_TILT, md.REFERENCE_IMAGE]] = [0.0, 0.0, 'null']
    
    alignment_mds = []
    for i in range(match_indices.shape[-1]):
        alignment_mds.append(_ensemble_alignment_md(reference_md, projection_md, match_indices[:,i], match_distances[:,i]))
    
    # Add the alignment consensus to the output
    output_md = output_md.join(_do_pose_consensus(alignment_mds))
    output_md = output_md.join(_do_shift_consensus(alignment_mds))
    
    # Reorder columns for more convenient reading
    output_md = output_md.reindex(
        columns=experimental_md.columns.union(output_md.columns, sort=False)
    )
    
    return output_md
