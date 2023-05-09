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
import numpy as np
import pandas as pd

from . import labels
from .. import image

def get_image2d_size(data: pd.DataFrame) -> Tuple:
    path = image.parse_path(data.loc[0, labels.IMAGE])
    return image.get_image2d_size(path.filename)

def sort_by_image_filename(df: pd.DataFrame, label: str = labels.IMAGE) -> pd.DataFrame:
    filenames = df[label].apply(image.parse_path)
    indices = np.argsort(filenames)
    return df.reindex(indices)
