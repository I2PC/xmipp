import torch
import argparse

import operators
import image
import search
import alignment
import metadata as md

import faiss

def run(reference_md_path: str, 
        weight_image_path: str,
        output_path: str,
        max_shift : float,
        n_training: int,
        n_samples: int,
        cutoff: float ):
    
    # Devices
    transform_device = torch.device('cuda', 0)
    db_device = torch.device('cuda', 0)
    
    # Read input files
    reference_md = md.read(reference_md_path)
    weights = torch.tensor(image.read_data(weight_image_path))
    image_size = md.get_image_size(reference_md)[0]
    
    # Create the Fourier Transformer and flattener
    fourier = operators.FourierTransformer2D()
    flattener = operators.FourierLowPassFlattener(image_size, cutoff, device=transform_device)
    weighter = operators.Weighter(weights, flattener, device=transform_device)
    
    # Create the DB to store the data
    dim = 2*flattener.get_length()
    recipe = search.opq_ifv_pq_recipe(dim, n_samples)
    print(recipe)
    db = search.create_database(dim, recipe, metric_type=faiss.METRIC_L2)
    db = search.upload_database_to_device(db, db_device)
    
    # Do some work
    print('Augmenting data')
    dataset = image.torch_utils.Dataset(reference_md[md.IMAGE])
    training_set = alignment.augment_data(
        db,
        dataset=dataset,
        fourier=fourier,
        flattener=flattener,
        weighter=weighter,
        count=n_training,
        max_rotation=180,
        max_shift=max_shift,
        batch_size=8192,
        transform_device=transform_device,
        store_device=torch.device('cpu')
    )
    
    print('Training')
    db.train(training_set)
    
    # Write to disk
    db = search.download_database_from_device(db)  # TODO remove
    search.write_database(db, output_path + 'index')


if __name__ == '__main__':
    # To avoid problems
    torch.multiprocessing.set_start_method('spawn')

    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Align Nearest Neighbor Training',
                        description = 'Align Cryo-EM images using a fast Nearest Neighbor approach')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--weights')
    parser.add_argument('--max_shift', type=float, required=True)
    parser.add_argument('--training', type=int, default=int(4e6))
    parser.add_argument('--size', type=int, default=int(2e6))
    parser.add_argument('--max_frequency', type=float, required=True)

    # Parse
    args = parser.parse_args()

    reference_md_path = args.i
    output_path = args.o
    weight_image_path = args.weights
    max_shift = args.max_shift
    n_training = args.training
    n_samples = args.size
    cutoff = args.max_frequency

    # Run the program
    run(
        reference_md_path, weight_image_path,
        output_path,
        max_shift,
        n_training, n_samples,
        cutoff
    )