from typing import Iterable
import mrcfile
import numpy as np


def read_data(path: str, mmap: bool = False, permissive: bool = False) -> np.ndarray:
    open_fun = mrcfile.mmap if mmap else mrcfile.open
    with open_fun(path, mode='r', permissive=permissive) as mrc:
        return mrc.data
    
def read_data_batch(paths: Iterable[str]) -> np.ndarray:
    images = list(map(read_data, paths))
    return np.stack(images)

def read_header(path: str, permissive: bool = False):
    with mrcfile.open(path, mode='r', permissive=permissive, header_only=True) as mrc:
        return mrc.header
    
