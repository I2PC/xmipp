from typing import Tuple
import pandas as pd

from . import labels
import image

def get_image_size(data: pd.DataFrame) -> Tuple:
    path = data.loc[0, labels.IMAGE]
    return image.get_size(path)