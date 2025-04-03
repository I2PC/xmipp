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
# from functions.bnb_gpu import *
# from functions.assessment import *
from xmippPyModules.globalAlignFunction.bnb_gpu import *
from xmippPyModules.globalAlignFunction.pca_gpu import *
from xmippPyModules.globalAlignFunction.assessment import *


def read_images(mrcfilename):

    with mrcfile.open(mrcfilename, permissive=True) as f:
         emImages = f.data.astype(np.float32).copy()
    return emImages 


def flatGrid(freq_band, coef, nBand):
    
    grid_flat = [torch.zeros((2, int(coef[n]/2)), dtype=torch.float32, device = cuda) for n in range(nBand)]
    
    dim = freq_band.size(dim=0) 
    dimfreq = freq_band.size(dim=1)
    
    freq_x = torch.fft.rfftfreq(dim, d=0.5/np.pi, device=cuda)
    freq_y = torch.fft.fftfreq(dim, d=0.5/np.pi, device=cuda)

    grid = torch.meshgrid(freq_x, freq_y, indexing='xy')
    
    for n in range(nBand):
        grid_flat[n][0] = grid[0][:,:dimfreq][freq_band == n]      
        grid_flat[n][1] = grid[1][:,:dimfreq][freq_band == n]
        
    return(grid_flat) 
  
       
if __name__=="__main__":
      
    parser = argparse.ArgumentParser(description="align images")
    parser.add_argument("-i", "--exp", help="input mrc file for experimental images)", required=True)
    parser.add_argument("-r", "--ref", help="input mrc file for references images)", required=True)
    parser.add_argument("-s", "--sampling", type=float, help="pixel size of the images", required=True)
    parser.add_argument("-a", "--ang", type=float, help="rotation angle (in degree)", required=True)
    parser.add_argument("-amax", "--angmax", type=float, default=180.0, help="maximum rotation angle (in degree, default = 180)")
    parser.add_argument("-sh", "--shift", type=float, help="shift (px)", required=True)
    parser.add_argument("-msh", "--maxshift", type=float,help="maximum shift (px)", required=True)
    parser.add_argument("-b", "--bands", help="file with frequency bands, obtained from train step", required=True)
    parser.add_argument("-v", "--vecs", help="file with pretrain eigenvectors, obtained from train step", required=True)
    parser.add_argument("-o", "--output", help="Root directory for the output files", required=True)
    parser.add_argument("-stExp", "--sartExp", help="star file for experimental images", required=True)
    parser.add_argument("-stRef", "--starRef", help="star file for experimental images", required=True)
    parser.add_argument("-radius", type=int, help="radius for circular mask (in pixels)")       
    parser.add_argument("--apply_shifts",  action="store_true", help="Apply starfile shifts to experimental images")
    parser.add_argument("-nCl", "--numCl", type=int, default=1, help="number of classes for initial model")
    parser.add_argument("--save_class",  action="store_true", help="Save the corresponding class in output xmd")
    
    args = parser.parse_args()
    
    expFile = args.exp  
    prjFile = args.ref
    sampling = args.sampling
    ang = args.ang
    amax = args.angmax
    shiftMove = args.shift
    maxshift = args.maxshift
    bands = args.bands
    vecs = args.vecs
    output = args.output
    expStar = args.sartExp
    prjStar = args.starRef 
    radius = args.radius
    apply_shifts = args.apply_shifts
    numCl = args.numCl
    save_class = args.save_class
           
    torch.cuda.is_available()
    torch.cuda.current_device()
    cuda = torch.device('cuda')
    
    #load pca
    freqBn = torch.load(bands) 
    cvecs = torch.load(vecs)
    nBand = freqBn.unique().size(dim=0) - 1
    
    coef = torch.zeros(nBand, dtype=int)
    for n in range(nBand):
        coef[n] = 2*torch.sum(freqBn==n)
    
    grid_flat = flatGrid(freqBn, coef, nBand)

        
    bnb = BnBgpu(nBand)
    assess = evaluation()
    
    
    #Precomputed rotation and shift applied to references  
    angSet = (-amax, amax, ang)
    shiftSet = (-maxshift, maxshift+shiftMove, shiftMove)
    vectorRot, vectorshift = bnb.setRotAndShift(angSet, shiftSet)
    vectorRot.sort()         
    nShift = len(vectorshift)
    
    #Basename for multiple references
    if numCl > 1:
        prjFile_base = os.path.join(os.path.dirname(prjFile), os.path.splitext(os.path.basename(prjFile))[0].split('_class')[0])
        output_base = os.path.join(os.path.dirname(output), os.path.splitext(os.path.basename(output))[0].split("_class")[0])

    
    #Read Experimental Images
    mmap = mrcfile.mmap(expFile, permissive=True)
    nExp = mmap.data.shape[0]
    dim = mmap.data.shape[1]
    
    
    #Precomputing shift for particle centering
    if apply_shifts:
        print("Experimental particles will be centered")
        prev_shifts = assess.getShifts(expStar, nExp)
        

    print("---Precomputing the projections of the experimental images---")
    
    expBatchSize = 2048
    count = 0
    step = int(np.ceil(nExp/expBatchSize))
    batch_projExp_cpu = [0 for i in range(step)] 
                
    for initBatch in range(0, nExp, expBatchSize):
        
        expImages = mmap.data[initBatch:initBatch+expBatchSize].astype(np.float32)#.copy()
        Texp = torch.from_numpy(expImages).float().to(cuda)  
        if radius:
            Texp = Texp * bnb.create_mask(Texp, radius)    
        del(expImages)

        #center experimetal particles
        if apply_shifts:
            Texp = bnb.center_shifts(Texp, initBatch, expBatchSize, prev_shifts)

        batch_projExp = bnb.create_batchExp(Texp, freqBn, coef, cvecs) 
        del(Texp)
        batch_projExp = torch.stack(batch_projExp)
        batch_projExp_cpu[count] = batch_projExp.to("cpu")
        del(batch_projExp)
        count+=1 
    
    matches = [None] * numCl    
    for i in range(numCl):    
        #Reading references particles 
       
        if numCl > 1:   
            prjImages = read_images(f"{prjFile_base}_class{i}.mrcs")
            
            if save_class:
                output = f"{output_base}_classes.xmd"
            else:
                output = f"{output_base}_class{i}.xmd"
        else:
            prjImages = read_images(prjFile)
        
        #convert ref images to tensor 
        tref= torch.from_numpy(prjImages).float().to("cpu")
        if radius:
            tref = tref * bnb.create_mask(tref, radius)
        del(prjImages)
        
        
        matches[i] = torch.full((nExp, 5), float("Inf"), device = cuda)
        
        for rot in vectorRot:
    
            print("rotation angle  %s"%rot) 
            # print("---Computing the projections of the reference images---")      
            batch_projRef = bnb.precalculate_projection(tref, freqBn, grid_flat, coef, cvecs, rot, vectorshift)
            count = 0
            # print("matches")
            for initBatch in range(0, nExp, expBatchSize):
    
                batch_projExp = batch_projExp_cpu[count].to('cuda', non_blocking=True)
    
                matches[i] = bnb.match_batch(batch_projExp, batch_projRef, initBatch, matches[i], rot, nShift)

                del(batch_projExp)
                count+=1
            del(batch_projRef)
        
        # print(matches[i])
        # exit()
        matches[i] = bnb.match_batch_label_minScore(matches[i])
        print(matches[i])
        
        
        score = matches[i][:, 2].mean()
        print("mean score = %s" %score.item())
        #Write new starfile
        if not save_class:   
            assess.writeExpStar(prjStar, expStar, matches[i], vectorshift, nExp, apply_shifts, output)
            # assess.writeExpStar_minScore(prjStar, expStar, matches[i], vectorshift, nExp, apply_shifts, output)
    
    
    if save_class:
        matches_min = bnb.match_batch_with_class(matches)
        assess.writeExpStarClass(prjStar, expStar, matches_min, vectorshift, nExp, apply_shifts, output)










