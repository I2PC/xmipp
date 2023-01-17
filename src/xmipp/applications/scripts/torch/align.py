import faiss
import torch
import argparse

import operators
import image
import search
import alignment
import metadata as md


def run(experimental_md_path: str, 
        reference_md_path: str, 
        index_path: str,
        weight_image_path: str,
        output_md_path: str,
        n_rotations : int,
        n_shifts : int,
        max_shift : float,
        cutoff: float,
        batch: int,
        method: str,
        norm: str,
        drop_na: bool,
        gpu: list ):
    
    # Devices
    if gpu:
        device = torch.device('cuda', int(gpu[0]))
    else:
        device = torch.device('cpu')
    transform_device = torch.device('cpu')
    db_device = device
    
    # Read input files
    experimental_md = md.read(experimental_md_path)
    reference_md = md.read(reference_md_path)
    weights = torch.tensor(image.read_data(weight_image_path))
    image_size, _ = md.get_image_size(experimental_md)
    
    
    print('Uploading')
    db = search.read_database(index_path)
    db = search.upload_database_to_device(db, db_device)

    # Create the transformer and flattener
    # according to the transform method
    dim = db.d
    if method == 'fourier':
        transformer = operators.FourierTransformer2D()
        flattener = operators.FourierLowPassFlattener(image_size, cutoff, padded_length=dim//2, device=transform_device)
    elif method == 'dct':
        transformer = operators.DctTransformer2D(image_size, device=transform_device)
        flattener = operators.DctLowPassFlattener(image_size, cutoff, padded_length=dim, device=transform_device)
        
    # Create the weighter
    weighter = operators.Weighter(weights, flattener, device=transform_device)
    
    # Create the in-plane transforms
    angles = torch.linspace(-180, 180, n_rotations+1)[:-1]
    rotation_transformer = operators.ImageRotator(angles, device=transform_device)

    axis_shifts = torch.linspace(-max_shift, max_shift, n_shifts)
    shifts = torch.cartesian_prod(axis_shifts, axis_shifts)
    if method == 'fourier':
        shift_transformer = operators.FourierShiftFilter(image_size, shifts, flattener, device=transform_device)
    else:
        shift_transformer = operators.ImageShifter(shifts, dim=image_size, device=transform_device)
    
    print('Projecting')
    reference_dataset = image.torch_utils.Dataset(reference_md[md.IMAGE])
    if method == 'fourier':
        projection_md = alignment.populate_references_fourier(
            db=db, 
            dataset=reference_dataset,
            rotations=rotation_transformer,
            shifts=shift_transformer,
            fourier=transformer,
            flattener=flattener,
            weighter=weighter,
            norm=norm,
            device=transform_device,
            batch_size=batch
        )
    else:
        projection_md = alignment.populate_references(
            db=db, 
            dataset=reference_dataset,
            rotations=rotation_transformer,
            shifts=shift_transformer,
            transformer=transformer,
            flattener=flattener,
            weighter=weighter,
            norm=norm,
            device=transform_device,
            batch_size=batch
        )
        
    print(f'Database contains {db.ntotal} entries')
    
    
    print('Aligning')
    experimental_dataset = image.torch_utils.Dataset(experimental_md[md.IMAGE])
    match_indices, match_distances = alignment.align(
        db=db, 
        dataset=experimental_dataset,
        transformer=transformer,
        flattener=flattener, 
        weighter=weighter,
        norm=norm,
        k=1,
        device=transform_device,
        batch_size=batch
    )
    assert(match_distances.shape[0] == len(experimental_md))
    assert(match_indices.shape[0] == len(experimental_md))
    
    
    print('Generating output')
    result_md = alignment.generate_alignment_metadata(
        experimental_md=experimental_md,
        reference_md=reference_md,
        projection_md=projection_md,
        match_indices=match_indices,
        match_distances=match_distances
    )
    
    if drop_na:
        result_md.dropna(inplace=True)
    
    # Denormalize shift
    result_md[[md.SHIFT_X, md.SHIFT_Y]] *= image_size
    
    md.write(result_md, output_md_path)



if __name__ == '__main__':
    # To avoid problems
    torch.multiprocessing.set_start_method('spawn')

    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Align Nearest Neighbor Training',
                        description = 'Align Cryo-EM images using a fast Nearest Neighbor approach')
    parser.add_argument('-i', required=True)
    parser.add_argument('-r', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--weights')
    parser.add_argument('--index', required=True)
    parser.add_argument('--rotations', type=int, required=True)
    parser.add_argument('--shifts', type=int, required=True)
    parser.add_argument('--max_shift', type=float, required=True)
    parser.add_argument('--max_frequency', type=float, required=True)
    parser.add_argument('--batch', type=int, default=16384)
    parser.add_argument('--method', type=str, default='fourier')
    parser.add_argument('--norm', type=str)
    parser.add_argument('--dropna', action='store_true')
    parser.add_argument('--gpu', nargs='*')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        experimental_md_path = args.i,
        reference_md_path = args.r,
        index_path = args.index,
        weight_image_path = args.weights,
        output_md_path = args.o,
        n_rotations = args.rotations,
        n_shifts = args.shifts,
        max_shift = args.max_shift,
        cutoff = args.max_frequency,
        batch = args.batch,
        method = args.method,
        norm = args.norm,
        drop_na = args.dropna,
        gpu = args.gpu
    )