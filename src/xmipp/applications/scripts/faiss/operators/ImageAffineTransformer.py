from typing import Optional
import torch
import torchvision

class ImageAffineTransformer:
    def __init__(   self,
                    angles: torch.Tensor,
                    shifts: torch.Tensor,
                    device: Optional[torch.device] = None ):
        self._angles = angles
        self._shifts = shifts
        
    def __call__(   self, 
                    input: torch.Tensor,
                    angle_index: int,
                    shift_index: int,
                    out: Optional[torch.Tensor] ) -> torch.Tensor:
        
        out = torchvision.transforms.functional.affine(
            input,
            self.get_angle(angle_index),
            self.get_shift(shift_index),
            1.0,
            0.0,
            torchvision.transforms.InterpolationMode.BILINEAR
        )
        return out

    def get_count(self) -> int:
        return len(self._angles)
    
    def get_angle(self, index: int) -> float:
        return float(self._angles[index])
    
    def get_shift(self, index: int) -> torch.Tensor:
        return self._shifts[index]
        