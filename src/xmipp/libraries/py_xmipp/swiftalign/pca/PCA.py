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
import torch.nn as nn

class PCA(nn.Module):
    def __init__(self, n_components: int):
        super().__init__()
        self.n_components = n_components

    def fit(self, 
            x: torch.Tensor, 
            mean: Optional[torch.Tensor] = None, 
            mean_centered: bool = False ):
        _, d = x.shape
        if self.n_components is not None:
            d = min(self.n_components, d)
        
        if mean is None:
            mean = x.mean(0, keepdim=True)
        self.register_buffer("mean_", mean)
        
        if not mean_centered:
            x = x - mean
        
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        self.register_buffer("explained_variance_", s[:d])
        self.register_buffer("components_", vh[:d])
        
        return self

    def forward(self, x: torch.Tensor):
        return self.transform(x)

    def transform(self, x: torch.Tensor, mean_centered: bool = False) -> torch.Tensor:
        if not hasattr(self, "mean_"):
            raise RuntimeError('fit must be called')
        
        if not mean_centered:
            x = x - self.mean_
        
        return torch.matmul(x, self.components_.t())

    def fit_transform(self, 
                      x: torch.Tensor, 
                      mean_centered: bool = False, 
                      mean: Optional[torch.Tensor] = None ) -> torch.Tensor:
        if mean is None:
            mean = x.mean(0, keepdim=True)
        
        if not mean_centered:
            x = x - mean
            
        self.fit(x, mean=mean, mean_centered=True)
        return self.transform(x, mean_centered=True)

    def inverse_transform(self, y: torch.Tensor):
        if not hasattr(self, "mean_"):
            raise RuntimeError('fit must be called')

        return torch.matmul(y, self.components_) + self.mean_
    