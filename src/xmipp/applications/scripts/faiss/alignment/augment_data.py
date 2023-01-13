from typing import Optional
import torch
import torchvision
import faiss
import faiss.contrib.torch_utils
import math

import operators
import utils
from .normalize import normalize

def augment_data(db: faiss.Index, 
                 dataset: torch.utils.data.Dataset,
                 transformer: operators.Transformer2D,
                 flattener: operators.SpectraFlattener,
                 weighter: operators.Weighter,
                 count: int,
                 max_rotation: float = 180,
                 max_shift: float = 0.1,
                 batch_size: int = 8192,
                 transform_device: Optional[torch.device] = None,
                 store_device: Optional[torch.device] = None ):
    
    # Decide the number of transformations
    n_references = len(dataset) # For the moment select all
    n_transforms = math.ceil(count / n_references) # transforms for the rest
    count = n_references * n_transforms
    
    # Read all the images to be used as training data
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x : torch.utils.data.default_collate(x).to(transform_device)
    )

    # Create the training set
    training_set = torch.empty(count, db.d, device=store_device)

    # Create the transform randomizer
    random_affine = torchvision.transforms.RandomAffine(
        degrees=max_rotation,
        translate=(max_shift, )*2,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )

    # Fill the training set
    start = 0
    t_affine_images = None
    flat_t_affine_images = None
    for images in loader:
        for i in range(n_transforms):
            end = start + images.shape[0]
            
            # Transform the images
            transformed_images = random_affine(images)
            t_affine_images = transformer(transformed_images, out=t_affine_images)
            flat_t_affine_images = flattener(t_affine_images, out=flat_t_affine_images)
            flat_t_affine_images = weighter(flat_t_affine_images, out=flat_t_affine_images)
            train_vectors = torch.view_as_real(flat_t_affine_images)
            train_vectors = torch.flatten(train_vectors, -2, -1)
            training_set[start:end,:] = train_vectors.to(training_set.device, non_blocking=True)
            normalize(training_set[start:end,:], dim=1)
            
            # Update the index
            start = end
            
            if i % 256 == 0:
                utils.progress_bar(end, count)

    assert(end == count)
    utils.progress_bar(count, count)
    
    # Wait for all downloads
    if transform_device != store_device:
        torch.cuda.synchronize(transform_device)
    
    return training_set