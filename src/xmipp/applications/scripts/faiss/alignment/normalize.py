import torch

def normalize(data: torch.Tensor, dim: int = 1):
    if dim != 1:
        raise NotImplementedError('This function is not implemented form dim != 1')

    sigma2, mean = torch.var_mean(data, dim=dim)
    sigma = torch.sqrt(sigma2)
    data -= mean[...,None] # TODO for dim != 1
    data /= sigma[...,None]