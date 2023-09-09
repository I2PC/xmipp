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
    
    # nExp = 10000
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
       
    print("---Precomputing the projections of the experimental images---")  

    if refImages: 
        clIm = read_images(refImages)
        cl = torch.from_numpy(clIm).float().to(cuda)
    else:
        cl = torch.zeros((classes, mmap.data.shape[1], mmap.data.shape[2]), device = cuda) 
 
    #create initial random classes 
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
    
    # nExp = 30000
    expBatchSize = 6000 
    initStep = 5 
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
            print(initBatch, endBatch)
            print(mode)

      
            #Initialization Transformation Matrix
            if mode == "create_classes":
                subset = expBatchSize * initStep
            else:
                # initSubset = expBatchSize
                subset = endBatch - initBatch
                # print("escribo subset")
                # print(subset)
                 
            tMatrix = torch.eye(2, 3, device = cuda).repeat(subset, 1, 1)
            
            if mode == "align_classes":
                niter = initStep #5
                
            for iter in range(niter):
                print("-----Iteration %s for updating classes-------"%(iter+1))
                start = time.process_time()
                
                matches = torch.full((subset, 5), float("Inf"), device = cuda)
                
                maxShift = round( (dim * 15)/100 )
                maxShift = (maxShift//4)*4
 
                if mode == "create_classes":
                    if iter < 4:
                        # ang, shiftMove = (-180, 180, 6), (-12, 16, 4)
                        ang, shiftMove = (-180, 180, 6), (-maxShift, maxShift+4, 4)
                    elif iter < 8: 
                        ang, shiftMove = (-180, 180, 4), (-8, 10, 2)
                    elif iter < 12: 
                        ang, shiftMove = (-180, 180, 2), (-6, 8, 2)
                    elif iter < 16: 
                        ang, shiftMove = (-30, 31, 1), (-3, 4, 1)
                    elif iter < 18: 
                        ang, shiftMove = (-8, 8.5, 0.5), (-1.5, 2, 0.5)
                else:
                    if iter < 1:
                        # ang, shiftMove = (-180, 180, 6), (-12, 16, 4)
                        ang, shiftMove = (-180, 180, 6), (-maxShift, maxShift+4, 4)
                    elif iter < 2: 
                        ang, shiftMove = (-180, 180, 4), (-8, 10, 2)
                    elif iter < 3: 
                        ang, shiftMove = (-180, 180, 2), (-6, 8, 2)
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
                      
                print(time.process_time() - start)
                
                #update classes        
                start = time.process_time()
                classes = len(cl)
        
                if mode == "create_classes":
                    cl, tMatrix, batch_projExp_cpu = bnb.create_classes(mmap, tMatrix, iter, subset, expBatchSize, matches, vectorshift, classes, freqBn, coef, cvecs, sampling, mask, sigma)
                else:
                    cl, tMatrix, batch_projExp_cpu = bnb.align_particles_to_classes(mmap.data[initBatch:endBatch], cl, tMatrix, iter, initBatch, subset, matches, vectorshift, classes, freqBn, coef, cvecs, sampling, mask, sigma)
                print(time.process_time() - start)
                #save classes
                file = output+"_%s_%s.mrcs"%(initBatch,iter+1)
                save_images(cl.cpu().detach().numpy(), file)
                
                if mode == "create_classes" and iter == 17:
                    refClas[:endBatch] = matches[:, 1]
                    #extract final angular and shift transformations
                    rotation_matrix = tMatrix[:, :, :2]
                    translation_vector[:endBatch] = tMatrix[:, :, 2]
                    angles_rad = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
                    angles_deg[:endBatch] = np.degrees(angles_rad.cpu().numpy())
                    
                elif mode == "align_classes" and iter == 4:
                    refClas[initBatch:endBatch] = matches[:, 1]
                    
                    rotation_matrix = tMatrix[:, :, :2]
                    translation_vector[initBatch:endBatch] = tMatrix[:, :, 2]
                    angles_rad = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
                    angles_deg[initBatch:endBatch] = np.degrees(angles_rad.cpu().numpy())   
    
    
    # print(refClas)
    counts = torch.bincount(refClas.int(), minlength=classes)
    dim = torch.tensor([dim], dtype=torch.float32, device=translation_vector.device)
    translation_vector[:, 0] = translation_vector[:, 0] - ( dim * torch.trunc(translation_vector[:, 0]/dim) )
    translation_vector[:, 1] = translation_vector[:, 1] - ( dim * torch.trunc(translation_vector[:, 1]/dim) )
                    
    
    
    # for number, count in enumerate(counts):
    #     if count > 0:
    #         print(number, count)
    print(refClas.shape)
    print(counts.int())
    
    assess = evaluation()
    assess.updateExpStar(expStar, refClas, angles_deg, translation_vector, output)
    assess.createClassesStar(classes, file, counts, output)




