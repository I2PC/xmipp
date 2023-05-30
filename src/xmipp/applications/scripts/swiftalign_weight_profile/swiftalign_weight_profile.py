import argparse
import scipy
import torch

import xmippPyModules.swiftalign.image as image
import xmippPyModules.swiftalign.fourier as fourier
import xmippPyModules.swiftalign.metadata as md



def run(noise_md_path: str, 
        output_image: str,
        sampling: float,
        size: int ):
    
    # Read the md
    noise_md = md.read(noise_md_path)

    # Create an interpolator from the data
    interpolator = scipy.interpolate.interp1d(
        x=noise_md[md.RESOLUTION_FREQ],
        y=noise_md[md.SIGMANOISE],
        kind='linear',
        bounds_error=False
    )
    
    # Compute the noise image
    freq_grid = fourier.rfftnfreq((size, )*2)
    freq = torch.norm(freq_grid, axis=0) / sampling
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