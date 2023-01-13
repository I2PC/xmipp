from typing import Optional, Sequence, Iterable, Callable
import torch

from basis import dct_ii_basis, dct_iii_basis
from project import project_nd

def bases_generator(shape: Sequence[int], 
                    dims: Iterable[int],
                    func: Callable[[int], torch.Tensor]) -> Iterable[torch.Tensor]:
    sizes = map(shape.__getitem__, dims)
    bases = map(func, sizes)
    return bases
    
def dct(x: torch.Tensor, 
        dims: Iterable[int], 
        out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    bases = bases_generator(x.shape, dims, dct_ii_basis)
    return project_nd(x, dims, bases, out=out)

def idct(x: torch.Tensor, 
         dims: Iterable[int], 
         out: Optional[torch.Tensor] = None ) -> torch.Tensor:
    
    bases = bases_generator(x.shape, dims, dct_iii_basis)
    return project_nd(x, dims, bases, out=out)