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

from typing import Tuple

import torch

def svd_flip(u: torch.Tensor, vh: torch.Tensor, u_based_decision: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adjusts the signs of the singular vectors from the SVD decomposition for
    deterministic output.

    This method ensures that the output remains consistent across different
    runs.

    Args:
        u (torch.Tensor): Left singular vectors tensor.
        vh (torch.Tensor): Right singular vectors tensor.
        u_based_decision (bool, optional): If True, uses the left singular
            vectors to determine the sign flipping. Defaults to True.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular
            vectors tensors.
    """
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, :])
    else:
        max_abs_rows = torch.argmax(torch.abs(vh), dim=1)
        signs = torch.sign(vh[:, max_abs_rows])
    u *= signs
    vh *= signs[:, None]
    return u, vh