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

from xmippPyModules.globalAlignFunction import (bnb_gpu, pca_gpu, assessment)

import time

def precalculateBands(nBand, dim, sampling, maxRes, minRes):

    vectorFreq = torch.fft.fftfreq(dim)
    freq_band = torch.full((dim,int(np.floor(dim/2))), 50)   

    maxFreq = sampling/maxRes
    minFreq = sampling/minRes
    factor = nBand * (1/maxFreq)  

    for x in range(dim):
        
        if vectorFreq[x] >= 0:
            wx = vectorFreq[x]

        for y in range(dim):
            wy = vectorFreq[y]

            w = torch.sqrt(wx**2 + wy**2)
            
            if (w > minFreq) and (w < maxFreq):
                freq_band[y][x] = torch.floor(w*factor)
    
    return freq_band


if __name__=="__main__":
       
    parser = argparse.ArgumentParser(description="Train PCA")
    parser.add_argument("-i", "--mrc", help="input mrc file for experimental images)", required=True)
    parser.add_argument("-n", "--bands", help="number of bands", required=True)
    parser.add_argument("-s", "--sampling", help="pixel size of the images", required=True)
    parser.add_argument("-hr", "--highres", help="highest resolution to consider", required=True)
    parser.add_argument("-lr", "--lowres", help="lowest resolution to consider", required=True)
    parser.add_argument("-p", "--perc", help="PCA percentage (between 0-1)", required=True)
    parser.add_argument("-t", "--training", help="percentage of image for training (between 0-1)", required=True)
    parser.add_argument("-o", "--output", help="Root directory for the output files", required=True)
    
    args = parser.parse_args()

    expFile = args.mrc  
    nBand = int(args.bands) 
    sampling = float(args.sampling)
    maxRes = float(args.highres)
    minRes = float(args.lowres)
    per_eig = float(args.perc)
    train = float(args.training)
    output = args.output
    
    torch.cuda.is_available()
    torch.cuda.current_device()
    cuda = torch.device('cuda:0')
    # cuda = torch.device('cpu')        
    
    print("Reading Images")
    mexp = mrcfile.mmap(expFile, permissive=True)
    dim = mexp.data.shape[1]
    nExp = int(train * mexp.data.shape[0])
    nExp = 30000 

    
    # Precalculate frequency bands  
    freq_band = precalculateBands(nBand, dim, sampling, maxRes, minRes) 
    torch.save(freq_band, "freq_bands.pt")

    coef = torch.zeros(nBand, dtype=int)
    for n in range(nBand):
        coef[n] = 2*torch.sum(freq_band==n)  
        
    bnb = BnBgpu(nBand)    
    expBatchSize = 5000  
    band = [torch.zeros(nExp, coef[n], device = cuda) for n in range(nBand)]      
     
    print("Select bands of experimental images")            
    for initBatch in range(0, nExp, expBatchSize):
        
        endBatch = initBatch+expBatchSize 
        if (endBatch > nExp):
            endBatch = nExp
        
        expImages = mexp.data[initBatch:endBatch].astype(np.float32)#.copy()
        Texp = torch.from_numpy(expImages).float().to(cuda)
        del(expImages)
        ft = torch.fft.rfft2(Texp, norm="forward")
        del(Texp)
        bandBatch = bnb.selectBandsRefs(ft, freq_band, coef)
        del(ft)
        for n in range(nBand):
            band[n][initBatch:endBatch] = bandBatch[n]
        del(bandBatch)
       
        
    #Training PCA 
    pca = PCAgpu(nBand)    
    mean, vals, vecs = pca.trainingPCAonline(band, coef, per_eig) 
    torch.save(mean, output + "_mean.pt")
    torch.save(vals, output + "_vals.pt")
    torch.save(vecs, output + "_vecs.pt")        
            
 
