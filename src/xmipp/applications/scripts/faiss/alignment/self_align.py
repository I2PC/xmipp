import torch
import torchvision

import utils

from .align_in_plane import align_in_plane, DISTANCE_SLICE, N_SLICES

def self_align(images: torch.Tensor,
               angles: torch.Tensor,
               shifts: torch.Tensor,
               metric,
               interpolation: torchvision.transforms.InterpolationMode = torchvision.transforms.InterpolationMode.BILINEAR,
               batch_size: int = 1024 ) -> torch.Tensor:
    
    n_images = images.shape[0]
    n_comparisons = n_images * (n_images-1) // 2
    result = torch.zeros((N_SLICES, ) + (n_images, )*2)
    
    utils.progress_bar(0, n_comparisons)
    work_done = 0
    for idxRef in range(n_images):
        imageRef = images[idxRef,...]
        
        for idxExpStart in range(0, idxRef, batch_size):
            idxExpEnd = min(idxRef, idxExpStart+batch_size)
            imageExps = images[idxExpStart:idxExpEnd,...]
        
            alignment = align_in_plane(
                ref=imageRef, 
                exps=imageExps,
                angles=angles,
                shifts=shifts,
                metric=metric,
                interpolation=interpolation,
            )
            
            # Write everything anti-symmetrically
            result[:, idxRef, idxExpStart:idxExpEnd] = +alignment
            result[:, idxExpStart:idxExpEnd, idxRef] = -alignment
            
            # Symmetrize distance
            result[DISTANCE_SLICE, idxExpStart:idxExpEnd, idxRef] *= -1
            
        # Update progress bar
        work_done += idxRef
        utils.progress_bar(work_done, n_comparisons)
        
    
    return result