from typing import Optional
import torch
import torchvision

class ImageRotator:
    def __init__(   self,
                    angles: torch.Tensor,
                    device: Optional[torch.device] = None ):
        self._angles = angles
        
    def __call__(   self, 
                    input: torch.Tensor,
                    index: int,
                    out: Optional[torch.Tensor] ) -> torch.Tensor:
        
        # TODO use the matrix
        out = torchvision.transforms.functional.rotate(
            input,
            self.get_angle(index),
            torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return out

    def get_count(self) -> int:
        return len(self._angles)
    
    def get_angle(self, index: int) -> float:
        return float(self._angles[index])
        