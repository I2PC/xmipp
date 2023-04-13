import torch
import argparse

import operators
import image
import search
import utils
import scipy
import metadata as md



def run(ssnr_md_path: str, 
        output_image: str,
        sampling: float,
        size: int ):
    
    # Read the md
    ssnr_md = md.read(ssnr_md_path)

    # Create an interpolator from the data
    interpolator = scipy.interpolate.interp1d(
        x=ssnr_md[md.RESOLUTION_FREQ],
        y=ssnr_md[md.SIGMANOISE],
        kind='linear',
        bounds_error=False
    )
    
    # Compute the noise image
    freq = utils.nfft_freq((size, )*2) / sampling
    noise = interpolator(freq)
    weights = 1.0 / noise
    
    # Save it
    image.write(weights.astype('float32'), output_image)
    
    

if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Calculate weights' )
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--sampling', type=float, required=True)
    parser.add_argument('--size', type=int, required=True)
    
    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        ssnr_md_path = args.i, 
        output_image = args.o,
        sampling = args.sampling,
        size = args.size
    )