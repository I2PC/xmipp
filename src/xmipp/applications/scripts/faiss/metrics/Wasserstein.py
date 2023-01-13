from typing import Sequence
import math

import torch
import numpy as np
import pywt

class Wasserstein:
    def __init__(self, shape: Sequence[int], wavelet: str = 'sym5'):
        self._shape = shape
        self._wavelet = wavelet
        self._weights = self._compute_level_weights(self._shape)
        
    def __call__(self, diff: torch.Tensor) -> torch.Tensor:
        # Shorthands
        weights = self._weights
        wavelet = self._wavelet
        n_levels = len(weights)
        n_batch_dim = len(diff.shape) - len(self._shape)
        
        # Compute the Discrete Wavelet Transform
        decomposition = pywt.wavedecn(
            diff.numpy(), 
            wavelet, 
            mode='zero', 
            level=n_levels,
            axes=range(n_batch_dim, len(diff.shape)) # Last axes
        )
        
        # Extract the detail coefficitents
        levels = np.flip(decomposition[1:])
        assert(len(levels) == n_levels)

        # Create an array with all the weighted coefficients
        weighted_coefficients = []
        for weight, level in zip(weights, levels):
            for coefficients in level.values():
                flat_coefficients = torch.flatten(torch.as_tensor(coefficients), start_dim=-2, end_dim=-1)
                weighted_coefficients.append(weight * flat_coefficients)

        # Compute the norm
        coefficients_vectors = torch.cat(weighted_coefficients, dim=-1)
        return torch.linalg.norm(coefficients_vectors, ord=1, dim=-1)

    def _compute_level_weights(self, shape: Sequence[int]) -> torch.Tensor:
        dimension_count = len(shape)
        level_count = math.ceil(math.log2(max(shape))) + 1
        
        l = 1 + (dimension_count / 2.0)
        exponents = l * torch.arange(level_count)
        return 2 ** exponents # TODO decide sign
