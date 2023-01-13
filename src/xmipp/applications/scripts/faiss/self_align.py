import faiss
import torch
import argparse

import operators
import image
import search
import metrics
import alignment
import metadata as md

def run(images_md_path: str, 
        output_md_path: str,
        n_rotations : int,
        n_translations : int,
        max_shift : float,
        batch: int ):
    
    # Variables
    # TODO make them parameters of deduce from data
    
    # Read input files
    images_md = md.read(images_md_path)
    image_size, _ = md.get_image_size(images_md)
    
    # Create the in-plane transforms
    rotations = torch.linspace(-180, 180, n_rotations+1)[:-1]
    max_shift_px = image_size*max_shift
    axis_shifts = torch.linspace(-max_shift_px, max_shift_px, n_translations)
    shifts = torch.cartesian_prod(axis_shifts, axis_shifts)
    
    images = torch.tensor(image.read_data_batch(images_md[md.IMAGE]))
    alignment.self_align(
        images, 
        angles=rotations,
        shifts=shifts,
        metric=metrics.Wasserstein((image_size, )*2),
        batch_size=batch
    )


if __name__ == '__main__':
    # To avoid problems
    torch.multiprocessing.set_start_method('spawn')

    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Self align images')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--rotations', type=int, required=True)
    parser.add_argument('--translations', type=int, required=True)
    parser.add_argument('--max_shift', type=float, required=True)
    parser.add_argument('--batch', type=int, default=8192)

    # Parse
    args = parser.parse_args()

    images_md_path = args.i
    output_md_path = args.o
    n_rotations = args.rotations
    n_translations = args.translations
    max_shift = args.max_shift
    batch = args.batch

    # Run the program
    run(
        images_md_path,
        output_md_path,
        n_rotations, n_translations, max_shift,
        batch
    )