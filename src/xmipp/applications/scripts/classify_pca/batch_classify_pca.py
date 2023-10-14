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
from xmippPyModules.classifyPcaFuntion.bnb_gpu import *
from xmippPyModules.classifyPcaFuntion.assessment import *


def read_images(mrcfilename):

    with mrcfile.open(mrcfilename, permissive=True) as f:
         emImages = f.data.astype(np.float32).copy()
    return emImages


def save_images(data, outfilename):
    data = data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        # if origin is not None:
        #     mrc.header['origin']['x'] = origin[0]
        #     mrc.header['origin']['y'] = origin[1]
        # mrc.update_header_from_data()
        # mrc.update_header_stats()


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
    parser.add_argument("-s", "--sampling", type=float, help="pixel size of the images", required=True)
    parser.add_argument("-c", "--classes", help="number of 2D classes", required=True)
    parser.add_argument("-r", "--ref", help="2D classes of external method")   
    parser.add_argument("-n", "--niter", help="number of iterations", required=True)
    parser.add_argument("-b", "--bands", help="file with frequency bands", required=True)
    parser.add_argument("-v", "--vecs", help="file with pretrain eigenvectors", required=True)
    parser.add_argument("--mask",  action="store_true", help="A Gaussian mask is used.")
    parser.add_argument("--sigma", type=float, help="value of sigma for the Gaussian mask. "
                                                    "It is only used if the --mask option is applied.")
    parser.add_argument("-o", "--output", help="Root directory for the output files", required=True)
    parser.add_argument("-stExp", "--sartExp", help="star file for images")

    
    args = parser.parse_args()
    
    expFile = args.exp
    sampling = args.sampling
    classes = int(args.classes)  
    refImages = args.ref
    niter = int(args.niter)
    bands = args.bands
    vecs = args.vecs
    mask = args.mask
    sigma = args.sigma
    output = args.output
    expStar = args.sartExp
           
    torch.cuda.is_available()
    torch.cuda.current_device()
    cuda = torch.device('cuda:0')

    #Read Images
    mmap = mrcfile.mmap(expFile, permissive=True)
    nExp = mmap.data.shape[0]
    dim = mmap.data.shape[1]
    
    if mask and sigma is None:
        sigma = dim/3
    
    expBatchSize = 5000
    numFirstBatch = 6
    initSubset = min(30000, nExp)
    refClas = torch.zeros(nExp)
    translation_vector = torch.zeros(nExp, 2)
    angles_deg = np.zeros(nExp)
    
    freqBn = torch.load(bands) 
    cvecs = torch.load(vecs)
    nBand = freqBn.unique().size(dim=0) - 1
    
    coef = torch.zeros(nBand, dtype=int)
    for n in range(nBand):
        coef[n] = 2*torch.sum(freqBn==n)

    grid_flat = flatGrid(freqBn, coef, nBand)

    bnb = BnBgpu(nBand)
       
    # print("---Precomputing the projections of the experimental images---")  

    if refImages: 
        initStep = -1
        clIm = read_images(refImages)
        cl = torch.from_numpy(clIm).float().to(cuda)
    else:
        initStep = int(min(numFirstBatch, np.ceil(nExp/expBatchSize)))
        cl = torch.zeros((classes, mmap.data.shape[1], mmap.data.shape[2]), device = cuda) 
 
        #create initial classes 
        div = int(initSubset/classes)
        resto = int(initSubset%classes)
    
        expBatchSizeClas = div+resto 
        
        count = 0
        for initBatch in range(0, initSubset, expBatchSizeClas):
            expImages = mmap.data[initBatch:initBatch+expBatchSizeClas].astype(np.float32)
            Texp = torch.from_numpy(expImages).float().to(cuda)
    
            #Averages classes
            if not refImages:
                cl[count] = torch.mean(Texp, 0)
                count+=1
            del(Texp)        
    
    # file = output+"_0.mrcs"    
    # save_images(cl.cpu().numpy(), file)
    
    num_batches = int(np.ceil(nExp / expBatchSize))
    mode = False
    
    batch_projExp_cpu = []
    for i in range(num_batches):
        initBatch = i * expBatchSize
        endBatch =  min( (i+1) * expBatchSize, nExp)
        
        expImages = mmap.data[initBatch:endBatch].astype(np.float32)
        Texp = torch.from_numpy(expImages).float().to(cuda)
        del(expImages)  
              
        if i < initStep:          
            batch_projExp_cpu.append( bnb.batchExpToCpu(Texp, freqBn, coef, cvecs) )           
            if i == initStep-1:
                mode = "create_classes"
        
        else:            
            # batch_projExp_cpu = bnb.batchExpToCpu(Texp, freqBn, coef, cvecs)
            batch_projExp_cpu = bnb.create_batchExp(Texp, freqBn, coef, cvecs)
            mode = "align_classes"
        
        if mode:
            # print(initBatch, endBatch)
            print(mode)

      
            #Initialization Transformation Matrix
            if mode == "create_classes":
                # subset = expBatchSize * initStep
                subset = endBatch
            else:
                subset = endBatch - initBatch

                 
            tMatrix = torch.eye(2, 3, device = cuda).repeat(subset, 1, 1)
            
            if mode == "align_classes":
                niter = 5
                
            for iter in range(niter):
                # print("-----Iteration %s for updating classes-------"%(iter+1))
                           
                matches = torch.full((subset, 5), float("Inf"), device = cuda)
                
                maxShift = round( (dim * 15)/100 )
                maxShift = (maxShift//4)*4
 
                if mode == "create_classes":
                    print("---Iter %s for creating classes---"%(iter+1))
                    if iter < 4:
                        # ang, shiftMove = (-180, 180, 6), (-12, 16, 4)
                        ang, shiftMove = (-180, 180, 6), (-maxShift, maxShift+4, 4)
                    elif iter < 7: 
                        ang, shiftMove = (-180, 180, 4), (-8, 10, 2)
                    elif iter < 10: 
                        ang, shiftMove = (-90, 90, 2), (-6, 8, 2)
                    elif iter < 13: 
                        ang, shiftMove = (-30, 31, 1), (-3, 4, 1)
                    elif iter < 15: 
                        ang, shiftMove = (-8, 8.5, 0.5), (-1.5, 2, 0.5)
                else:
                    print("---Iter %s for align to classes---"%(iter+1))
                    if iter < 1:
                        # ang, shiftMove = (-180, 180, 6), (-12, 16, 4)
                        ang, shiftMove = (-180, 180, 6), (-maxShift, maxShift+4, 4)
                    elif iter < 2: 
                        ang, shiftMove = (-180, 180, 4), (-8, 10, 2)
                    elif iter < 3: 
                        ang, shiftMove = (-90, 90, 2), (-6, 8, 2)
                    elif iter < 4: 
                        ang, shiftMove = (-30, 31, 1), (-3, 4, 1)
                    elif iter < 5: 
                        ang, shiftMove = (-8, 8.5, 0.5), (-1.5, 2, 0.5)
                    
                          
                vectorRot, vectorshift = bnb.setRotAndShift(ang, shiftMove)
                
                nShift = len(vectorshift)  
        
                for rot in vectorRot:            
        
                    # print("---Precomputing the projections of the reference images---")          
                    batch_projRef = bnb.precalculate_projection(cl, freqBn, grid_flat, coef, cvecs, float(rot), vectorshift)
            
                    count = 0  
                    steps = initStep if mode == "create_classes" else 1 
                                
                    for i in range(steps):
            
                        if mode == "create_classes":
                            init = i*expBatchSize
                            batch_projExp = batch_projExp_cpu[count].to('cuda', non_blocking=True)
                        else:
                            init = 0
                            batch_projExp = batch_projExp_cpu
                            
                        matches = bnb.match_batch(batch_projExp, batch_projRef, init, matches, rot, nShift)    
                        del(batch_projExp)
                        count+=1    
                del(batch_projRef)    
                
                #update classes        
                classes = len(cl)
        
                if mode == "create_classes":
                    cl, tMatrix, batch_projExp_cpu = bnb.create_classes(mmap, tMatrix, iter, subset, expBatchSize, matches, vectorshift, classes, freqBn, coef, cvecs, sampling, mask, sigma)
                else:
                    cl, tMatrix, batch_projExp_cpu = bnb.align_particles_to_classes(mmap.data[initBatch:endBatch], cl, tMatrix, iter, initBatch, subset, matches, vectorshift, classes, freqBn, coef, cvecs, sampling, mask, sigma)

                # #save classes
                # file = output+"_%s_%s.mrcs"%(initBatch,iter+1)
                # save_images(cl.cpu().detach().numpy(), file)
                
                
                if mode == "create_classes" and iter == 14:
                    
                    refClas[:endBatch] = matches[:, 1]
                                                          
                    #Applying TMT(inv). 
                    #This is done because the rotation is performed from the center of the image.
                    initial_shift = torch.tensor([[1.0, 0.0, -dim/2],
                                                  [0.0, 1.0, -dim/2],
                                                  [0.0, 0.0, 1.0]], device = tMatrix.device)
                    initial_shift = initial_shift.unsqueeze(0).expand(tMatrix.size(0), -1, -1)

                    tMatrix = torch.cat((tMatrix, torch.zeros((tMatrix.size(0), 1, 3), device=tMatrix.device)), dim=1)
                    tMatrix[:, 2, 2] = 1.0
                    tMatrix = torch.matmul(initial_shift, tMatrix)
                    tMatrix = torch.matmul(tMatrix, torch.inverse(initial_shift))
                    
                    #extract final angular and shift transformations
                    rotation_matrix = tMatrix[:, :2, :2]
                    translation_vector[:endBatch] = tMatrix[:, :2, 2]
                    angles_rad = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
                    angles_deg[:endBatch] = np.degrees(angles_rad.cpu().numpy())
                    
                elif mode == "align_classes" and iter == 4:
                    
                    refClas[initBatch:endBatch] = matches[:, 1]
                    
                    initial_shift = torch.tensor([[1.0, 0.0, -dim/2],
                                                  [0.0, 1.0, -dim/2],
                                                  [0.0, 0.0, 1.0]], device = tMatrix.device)
                    initial_shift = initial_shift.unsqueeze(0).expand(tMatrix.size(0), -1, -1)

                    tMatrix = torch.cat((tMatrix, torch.zeros((tMatrix.size(0), 1, 3), device=tMatrix.device)), dim=1)
                    tMatrix[:, 2, 2] = 1.0
                    tMatrix = torch.matmul(initial_shift, tMatrix)
                    tMatrix = torch.matmul(tMatrix, torch.inverse(initial_shift))
                    
                    rotation_matrix = tMatrix[:, :2, :2]
                    translation_vector[initBatch:endBatch] = tMatrix[:, :2, 2]
                    angles_rad = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
                    angles_deg[initBatch:endBatch] = np.degrees(angles_rad.cpu().numpy())
                    

    counts = torch.bincount(refClas.int(), minlength=classes) 
    
        #save classes
    file = output+".mrcs"
    save_images(cl.cpu().detach().numpy(), file)           
    
    # for number, count in enumerate(counts):
    #     if count > 0:
    #         print(number, count)

    print(counts.int())
    
    assess = evaluation()
    assess.updateExpStar(expStar, refClas, -angles_deg, translation_vector, output)
    assess.createClassesStar(classes, file, counts, output)



