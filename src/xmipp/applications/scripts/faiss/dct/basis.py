import torch
import math

def _get_nk(N: int):
    n = torch.arange(N)
    k = n.view(N, 1)
    return n, k    

def dct_ii_basis(N: int, norm: bool = True) -> torch.Tensor:
    n, k = _get_nk(N)
    
    result = (n + 0.5) * k
    result *= torch.pi / N
    result = torch.cos(result, out=result)
    
    # Normalize
    if norm:
        result *= math.sqrt(1/N)
        result[1:,:] *= math.sqrt(2)
    
    return result

def dct_iii_basis(N: int, norm: bool = True) -> torch.Tensor:
    n, k = _get_nk(N)
    
    # TODO avoid computing result[:,0] twice
    result = (k + 0.5) * n
    result *= torch.pi / N
    result = torch.cos(result, out=result)
    
    if norm:
        result[:,0] = 1 / math.sqrt(2)
        result *= math.sqrt(2/N)
    else:
        result[:,0] = 0.5
        
    
    return result