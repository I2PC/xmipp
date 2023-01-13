from typing import Tuple

from . import read

def get_size(path: str) -> Tuple:
    image = read.read_data(path, mmap=True)
    return image.shape #TODO do with metadata

def decompose_path(path: str) -> Tuple[int, str]:
    STACK_INDEXER = '@'
    parts = path.split(STACK_INDEXER, maxsplit=1)
    
    if len(parts) == 2:
        return int(parts[0]), parts[1]
    else:
        assert(len(parts) == 1)
        return -1, parts[0]