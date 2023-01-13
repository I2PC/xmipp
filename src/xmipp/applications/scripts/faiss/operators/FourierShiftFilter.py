from typing import Optional
import torch
import numpy as np

from SpectraFlattener import SpectraFlattener

class FourierShiftFilter:
    def __init__(self,
                 dim: int,
                 shifts: torch.Tensor,
                 flattener: SpectraFlattener,
                 device: Optional[torch.device] = None):
        
        freq = self._get_freq(dim, flattener, device=device)
        angles = -torch.mm(shifts.to(device), freq)
        ONE = torch.tensor(1, dtype=angles.dtype, device=device)
        self._filters = torch.polar(ONE, angles)
        self._shifts = shifts
    
    def __call__(   self,
                    input: torch.Tensor,
                    index: int,
                    out: Optional[torch.Tensor] = None ):
        filter = self._filters[index,:]
        out = torch.mul(input, filter, out=out)
        return out
    
    def get_count(self) -> int:
        return self._filters.shape[0]
    
    def get_shift(self, index: int) -> torch.Tensor:
        return self._shifts[index]
    
    def _get_freq(  self,
                    dim: int,
                    flattener: SpectraFlattener,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        
        freq_x = torch.fft.rfftfreq(dim, d=0.5/np.pi, device=device)
        freq_y = torch.fft.fftfreq(dim, d=0.5/np.pi, device=device)
        grid = torch.stack(torch.meshgrid(freq_x, freq_y, indexing='xy'))
        return flattener(grid)

"""
if __name__ == '__main__':
    flattener = LowPassFlattener(8, 0.25, device='cuda')
    shifts = torch.tensor([[0, 0], [2, 0], [1, 1], [0, 1]], dtype=torch.float32, device='cuda') 
    shifter = ShiftFilter(8, shifts, flattener, device='cuda')
"""