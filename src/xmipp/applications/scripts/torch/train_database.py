import torch
import argparse

import operators
import image
import search
import alignment
import metadata as md

import faiss

def _get_faiss_metric(metric: str):
    result = None
    
    if metric == 'euclidean':
        result = faiss.METRIC_L2
    elif metric == 'cosine':
        result = faiss.METRIC_INNER_PRODUCT
        
    return result

def run(reference_md_path: str, 
        weight_image_path: str,
        output_path: str,
        max_shift : float,
        n_training: int,
        n_samples: int,
        cutoff: float,
        method: str,
        metric: str,
        gpu: list ):
    
    # Devices
    if gpu:
        device = torch.device('cuda', int(gpu[0]))
    else:
        device = torch.device('cpu')

    transform_device = device
    db_device = device
    
    # Read input files
    reference_md = md.read(reference_md_path)
    weights = torch.tensor(image.read_data(weight_image_path))
    image_size = md.get_image_size(reference_md)[0]
    
    # Create the transformer and flattener
    # according to the transform method
    if method == 'fourier':
        transformer = operators.FourierTransformer2D()
        flattener = operators.FourierLowPassFlattener(image_size, cutoff, device=transform_device)
    elif method == 'dct':
        transformer = operators.DctTransformer2D(image_size, device=transform_device)
        flattener = operators.DctLowPassFlattener(image_size, cutoff, device=transform_device)
        
    # Create the weighter
    weighter = operators.Weighter(weights, flattener, device=transform_device)
    
    # Consider complex numbers
    dim = flattener.get_length()
    if transformer.has_complex_output():
        dim *= 2
    
    # Create the DB to store the data
    metric_type = _get_faiss_metric(metric)
    norm = 'vector' if (db.metric_type == faiss.METRIC_INNER_PRODUCT) else None
    recipe = search.opq_ifv_pq_recipe(dim, n_samples)
    print(f'Data dimensions: {dim}')
    print(f'Database: {recipe}')
    db = search.create_database(dim, recipe, metric_type=metric_type)
    db = search.upload_database_to_device(db, db_device)
    
    # Do some work
    print('Augmenting data')
    dataset = image.torch_utils.Dataset(reference_md[md.IMAGE])
    training_set = alignment.augment_data(
        db,
        dataset=dataset,
        transformer=transformer,
        flattener=flattener,
        weighter=weighter,
        norm=norm,
        count=n_training,
        max_rotation=180,
        max_shift=max_shift,
        batch_size=8192,
        transform_device=transform_device,
        store_device=torch.device('cpu') # Augmented dataset is very large
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
    parser.add_argument('--method', type=str, default='fourier')
    parser.add_argument('--metric', type=str, default='euclidean')
    parser.add_argument('--gpu', nargs='*')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        reference_md_path = args.i,
        output_path = args.o,
        weight_image_path = args.weights,
        max_shift = args.max_shift,
        n_training = args.training,
        n_samples = args.size,
        cutoff = args.max_frequency,
        method = args.method,
        metric = args.metric,
        gpu = args.gpu
    )