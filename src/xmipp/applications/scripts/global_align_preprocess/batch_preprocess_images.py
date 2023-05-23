#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""

import mrcfile
import argparse
import sys, os
import numpy as np
import torch

torch.cuda.is_available()
torch.cuda.current_device()
cuda = torch.device('cuda:0')


def read_images(mrcfilename):

    with mrcfile.open(mrcfilename, permissive=True) as f:
         emImages = f.data.astype(np.float32).copy()
    return emImages 

def save_images(data, outfilename):
    data = data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        
        
def signal_to_noise_statistic(images, radius):
    
    #create circular mask
    dim = images.size(dim=1)
    
    if not radius:
        radius = dim // 2
    y, x = torch.meshgrid(torch.arange(dim), torch.arange(dim), indexing='ij')
    dist = torch.sqrt((x - radius)**2 + (y - radius)**2)
    mask = dist <= radius        
    mask = mask.float()       
    inv_mask = torch.logical_not(mask)
    
    mask = mask.bool()#.to(cuda)
    inv_mask = inv_mask.bool()#.to(cuda)
    
    pixels_in_mask = torch.masked_select(images, mask.unsqueeze(0))
    pixels_out_mask = torch.masked_select(images, inv_mask.unsqueeze(0))
    
    mean_value_in = torch.mean(pixels_in_mask)
    std_value_in = torch.std(pixels_in_mask)
    
    mean_value_out = torch.mean(pixels_out_mask)
    std_value_out = torch.std(pixels_out_mask)
    
    return(mean_value_in, std_value_in, mean_value_out, std_value_out)


def apply_scale(prjImages, expImages, radius):
    
    prjMeanSignal, prjStdSignal, prjMeanNoise, prjStdNoise = signal_to_noise_statistic(prjImages, radius)
    # print(prjMeanSignal, prjStdSignal, prjMeanNoise, prjStdNoise)
    expMeanSignal, expStdSignal, expMeanNoise, expStdNoise = signal_to_noise_statistic(expImages, radius)
    # print(expMeanSignal, expStdSignal, expMeanNoise, expStdNoise)
    
    a = prjStdSignal*prjStdSignal
    denom = torch.abs(expStdSignal*expStdSignal - expStdNoise*expStdNoise)
    if denom == 0:
        denom = 0.000000001
    a = torch.sqrt(a/denom)
    b = prjMeanSignal - a*(expMeanSignal-expMeanNoise)
    print(a,b)
      
    prjImages = (prjImages - b)/a
    
    return prjImages
  
       
if __name__=="__main__":
      
    parser = argparse.ArgumentParser(description="Program used to equalize the scales of the "
                                    "reference particles with respect to the experimental particles. "
                                     "If the volume used to generate the reference particles was created "
                                     "with XMIPP, this step is not necessary.")
    parser.add_argument("-i", "--exp", help="input mrc file for experimental images)", required=True)
    parser.add_argument("-r", "--ref", help="input mrc file for references images)", required=True)
    parser.add_argument("-o", "--output", help="Root directory for the output mrc preprocess images (example: references_scale.mrcs)", required=True)
    parser.add_argument("-rad", "--radius", type=float, help="Radius of the circular mask that will be used to define the background area (in pixels)")
    
    args = parser.parse_args()
    
    expFile = args.exp  
    prjFile = args.ref
    output = args.output
    radius =  args.radius

    #Read Images
    mmap = mrcfile.mmap(expFile, permissive=True)
    nExp = mmap.data.shape[0]
    prjImages = read_images(prjFile) 
    
    #convert ref images to tensor 
    tref= torch.from_numpy(prjImages).float().to("cpu")
    # tref= torch.from_numpy(prjImages).float().to(cuda)
    del(prjImages)

    num = 20000
    if nExp < num:
        num =  nExp
    
    print("Scaling particles")
    Texp = torch.from_numpy(mmap.data[:num].astype(np.float32)).to("cpu")
    # Texp = torch.from_numpy(mmap.data[:num].astype(np.float32)).to(cuda)
    tref = apply_scale(tref, Texp, radius)
    del(Texp)
    
        #save preprocess images
    save_images(tref.numpy(), output)


  














