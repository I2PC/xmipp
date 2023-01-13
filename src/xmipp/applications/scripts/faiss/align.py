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
        n_translations : int,
        max_shift : float,
        cutoff_f: float,
        batch: int ):
    
    # Variables
    # TODO make them parameters of deduce from data
    transform_device = torch.device('cpu')
    db_device = torch.device('cuda', 0)
    #db_device = torch.device('cpu')
    
    # Read input files
    experimental_md = md.read(experimental_md_path)
    reference_md = md.read(reference_md_path)
    weights = torch.tensor(image.read_data(weight_image_path))
    image_size, _ = md.get_image_size(experimental_md)
    
    # Create the Fourier Transformer and flattener
    fourier = operators.FourierTransformer2D()
    flattener = operators.FourierLowPassFlattener(image_size, cutoff_f, device=transform_device)
    weighter = operators.Weighter(weights, flattener, device=transform_device)
    
    # Create the in-plane transforms
    rotations = torch.linspace(-180, 180, n_rotations+1)[:-1]
    rotation_transformer = operators.ImageRotator(rotations, device=transform_device)
    max_shift_px = image_size*max_shift
    axis_shifts = torch.linspace(-max_shift_px, max_shift_px, n_translations)
    shifts = torch.cartesian_prod(axis_shifts, axis_shifts)
    shift_transformer = operators.FourierShiftFilter(image_size, shifts, flattener, device=transform_device)
    
    db = search.read_database(index_path)
    
    print('Uploading')
    db = search.upload_database_to_device(db, db_device)
    print('Projecting')
    reference_dataset = image.torch_utils.Dataset(reference_md[md.IMAGE])
    projection_md = alignment.populate_db(
        db=db, 
        dataset=reference_dataset,
        rotations=rotation_transformer,
        shifts=shift_transformer,
        fourier=fourier,
        flattener=flattener,
        weighter=weighter,
        device=transform_device,
        batch_size=batch
    )
    print(f'Database contains {db.ntotal} entries')
    print('Aligning')
    experimental_dataset = image.torch_utils.Dataset(experimental_md[md.IMAGE])
    match_indices, match_distances = alignment.align(
        db=db, 
        dataset=experimental_dataset,
        fourier=fourier, 
        flattener=flattener, 
        weighter=weighter,
        k=16,
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
    parser.add_argument('--translations', type=int, required=True)
    parser.add_argument('--max_shift', type=float, required=True)
    parser.add_argument('--max_frequency', type=float, required=True)
    parser.add_argument('--batch', type=int, default=16384)

    # Parse
    args = parser.parse_args()

    experimental_md_path = args.i
    reference_md_path = args.r
    index_path = args.index
    weight_image_path = args.weights
    output_md_path = args.o
    n_rotations = args.rotations
    n_translations = args.translations
    max_shift = args.max_shift
    cutoff_f = args.max_frequency
    batch = args.batch

    # Run the program
    run(
        experimental_md_path, reference_md_path,
        index_path, weight_image_path,
        output_md_path,
        n_rotations,n_translations, max_shift,
        cutoff_f, batch
    )