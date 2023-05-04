#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import torch
import time
import torchvision.transforms.functional as T
import torch.nn.functional as F
import kornia


class BnBgpu:
    
    def __init__(self, nBand):

        self.nBand = nBand 
        
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda:0')
        # self.cuda = torch.device('cpu')
    
    
    def setRotAndShift(self, angle, shift, shiftTotal):
        self.vectorRot = []
        self.vectorShift = []
        
        for rot in range (0, 360, angle):
            self.vectorRot.append(rot)
         
        self.vectorShift = [[0,0]]                   
        for tx in range (-shiftTotal, shiftTotal+1, shift):
            for ty in range (-shiftTotal, shiftTotal+1, shift):
                if (tx | ty != 0):
                    self.vectorShift.append( [tx,ty] )                  

        return self.vectorRot, self.vectorShift
    
    
    def setRotAndShift2(self, angle, shift):
        
        self.vectorRot = [0]
        for rot in np.arange(*angle):
            if rot < 0:
                rot = 360 + rot
            if rot != 0:
                self.vectorRot.append(rot)
         
        self.vectorShift = [[0,0]]                   
        for tx in np.arange(*shift):
            for ty in np.arange(*shift):
                if (tx or ty != 0):
                    self.vectorShift.append( [float(tx),float(ty)] )                  

        return self.vectorRot, self.vectorShift
    
    
    def signal_to_noise_statistic(self, images):
        
        #create mask
        dim = images.size(dim=1)
        radius = dim // 2
        y, x = torch.meshgrid(torch.arange(dim), torch.arange(dim))
        dist = torch.sqrt((x - radius)**2 + (y - radius)**2)
        mask = dist <= radius        
        mask = mask.float()       
        inv_mask = torch.logical_not(mask)
        
        mask = mask.bool()
        inv_mask = inv_mask.bool()
        
        pixels_in_mask = torch.masked_select(images, mask.unsqueeze(0))
        pixels_out_mask = torch.masked_select(images, inv_mask.unsqueeze(0))
        
        self.mean_value_in = torch.mean(pixels_in_mask)
        self.std_value_in = torch.std(pixels_in_mask)
        
        self.mean_value_out = torch.mean(pixels_out_mask)
        self.std_value_out = torch.std(pixels_out_mask)
        
        # mask = mask.expand(images.size(dim=0), dim, dim)
        # masked_images = images * mask
        # image_means = torch.mean(masked_images.view(images.size(dim=0), -1), dim=1)
        # images_std =  torch.std(masked_images.view(images.size(dim=0), -1), dim=1)
        # mean_of_means = torch.mean(image_means)
        # mean_of_std = torch.mean(images_std)
        # return masked_images
        # return(mean_of_means, mean_of_std)
        return(self.mean_value_in, self.std_value_in, self.mean_value_out, self.std_value_out)

    
    def calculate_dif_scale(self, prjImages, expImages):
        self.signal_to_noise_statistic(prjImages)
        self.signal_to_noise_statistic(expImages)
        
     
    
    def precShiftBand(self, ft, freq_band, grid_flat, coef, shift):
        
        fourier_band = self.selectFourierBands(ft, freq_band, coef)
        nRef = fourier_band[0].size(dim=0)
        nShift = shift.size(dim=0)
        
        band_shifted = [torch.zeros((nRef*nShift, coef[n]), device = self.cuda) for n in range(self.nBand)]
                     
        ONE = torch.tensor(1, dtype=torch.float32, device=self.cuda)
        
        for n in range (self.nBand):           
            angles = torch.mm(shift, grid_flat[n])     
            filter = torch.polar(ONE, angles)
            
            for i in range(nRef):
                temp = fourier_band[n][i].repeat(nShift,1)
                               
                band_shifted_complex = torch.mul(temp, filter)                
                band_shifted[n][i*nShift : (i*nShift)+nShift] = torch.cat((band_shifted_complex.real, band_shifted_complex.imag), dim=1)
                     
        return(band_shifted)

    
  
    def selectFourierBands(self, ft, freq_band, coef):

        dimFreq = freq_band.shape[1]

        fourier_band = [torch.zeros(int(coef[n]/2), dtype = ft.dtype, device = self.cuda) for n in range(self.nBand)]
        
        freq_band = freq_band.expand(ft.size(dim=0) ,freq_band.size(dim=0), freq_band.size(dim=1))
           
        for n in range(self.nBand):
            fourier_band[n] = ft[:,:,:dimFreq][freq_band == n]
            fourier_band[n] = fourier_band[n].reshape(ft.size(dim=0),int(coef[n]/2))
        # del(ft)
        # torch.cuda.empty_cache()            
        return fourier_band        
            


    def selectBandsRefs(self, ft, freq_band, coef): 
    
        dimfreq = freq_band.size(dim=1)
        batch_size = ft.size(dim=0)
   
        freq_band = freq_band.to(self.cuda)
        band = [torch.zeros(batch_size, coef[n], device = self.cuda) for n in range(self.nBand)]    
        
        freq_band = freq_band.expand(ft.size(dim=0) ,freq_band.size(dim=0), freq_band.size(dim=1))
        for n in range(self.nBand): 
        
            band_real = ft[:,:,:dimfreq][freq_band == n].real
            band_imag = ft[:,:,:dimfreq][freq_band == n].imag
            band_real = band_real.reshape(batch_size,int(coef[n]/2))
            band_imag = band_imag.reshape(batch_size,int(coef[n]/2))
            band[n] =  torch.cat((band_real, band_imag), dim=1)
    
        return band
    
    
    def phiProjRefs(self, band, vecs):
        
        nFT = band[0].size(dim=0)      
        proj = [torch.zeros(nFT, vecs[n].size(dim=1), device = self.cuda) for n in range(self.nBand)]

        for n in range(self.nBand):       
            proj[n] = torch.mm(band[n] , vecs[n])
        return proj
        
       
    #Applying rotation and shift
    def precalculate_projection(self, prjTensorCpu, freqBn, grid_flat, coef, cvecs, rot, shift):
                    
        shift_tensor = torch.Tensor(shift).to(self.cuda)       
        prjTensor = prjTensorCpu.to(self.cuda)
   
        rotRef = T.rotate(prjTensor, rot)
        del(prjTensor)
        torch.cuda.empty_cache()
        rotFFT = torch.fft.rfft2(rotRef, norm="forward")
        del(rotRef)
        torch.cuda.empty_cache()
        band_shifted = self.precShiftBand(rotFFT, freqBn, grid_flat, coef, shift_tensor)   
        del(rotFFT)  
        torch.cuda.empty_cache()
        projBatch = self.phiProjRefs(band_shifted, cvecs)
        del(band_shifted)
        torch.cuda.empty_cache()

        return(projBatch)
    
    
    def create_batchExp(self, Texp, freqBn, coef, vecs):
        
        batch_projExp = [torch.zeros((Texp.size(dim=0), vecs[n].size(dim=1)), device = self.cuda) for n in range(self.nBand)]
        
        expFFT = torch.fft.rfft2(Texp, norm="forward")
        del(Texp)
        bandExp = self.selectBandsRefs(expFFT, freqBn, coef)
        del(expFFT)
        batch_projExp = self.phiProjRefs(bandExp, vecs)
        
        torch.cuda.empty_cache()
        return(batch_projExp)
        
      
    def match_batch(self, batchExp, batchRef, initBatch, matches, rot, nShift):
        
        nExp = batchExp[0].size(dim=0) 
        part = initBatch+nExp
        nShift = torch.tensor(nShift, device=self.cuda)
                                  
        for n in range(self.nBand):
            # score = (torch.cdist(batchRef[n], batchExp[n])**2)
            score = torch.cdist(batchRef[n], batchExp[n])
            
        min_score, ref = torch.min(score,0)
            
        sel = (torch.floor(ref/nShift)).type(torch.int64)
        shift_location = (ref - (sel*nShift)).type(torch.int64)
        rotation = torch.full((nExp,1), rot, device = self.cuda)
        exp = torch.arange(initBatch, initBatch+nExp, 1, device = self.cuda).view(nExp,1)
        
        iter_matches = torch.cat((exp, sel.reshape(nExp,1), min_score.reshape(nExp,1), 
                                  rotation, shift_location.reshape(nExp,1)), dim=1)  
        
        cond = (iter_matches[:,2] < matches[initBatch:initBatch+nExp,2]).reshape(nExp,1)       
        matches_cond = torch.where(cond, iter_matches[:,:], matches[initBatch:initBatch+nExp,:])
        matches[initBatch:initBatch+nExp,:] = matches_cond
        
        # torch.cuda.empty_cache()        
        return(matches)
    
    
    def match_batch_correlation(self, batchExp, batchRef, initBatch, matches, rot, nShift):
        
        nExp = batchExp[0].size(dim=0) 
        part = initBatch+nExp
        nShift = torch.tensor(nShift, device=self.cuda)
                                  
        for n in range(self.nBand):
            # score = (torch.cdist(batchRef[n], batchExp[n])**2)

            # score = torch.matmul(batchRef[n], batchExp[n].T)
            # score = F.conv1d(a.view(2,1,3), b.view(2,1,3))
            score = F.conv1d(batchRef[n].view(batchRef[n].shape[0],1,batchRef[n].shape[1]), batchExp[n].view(batchExp[n].shape[0],1,batchExp[n].shape[1]))
            
        min_score, ref = torch.max(score,0)
            
        sel = (torch.floor(ref/nShift)).type(torch.int64)
        shift_location = (ref - (sel*nShift)).type(torch.int64)
        rotation = torch.full((nExp,1), rot, device = self.cuda)
        exp = torch.arange(initBatch, initBatch+nExp, 1, device = self.cuda).view(nExp,1)
        
        iter_matches = torch.cat((exp, sel.reshape(nExp,1), min_score.reshape(nExp,1), 
                                  rotation, shift_location.reshape(nExp,1)), dim=1)  
        
        cond = (iter_matches[:,2] < matches[initBatch:initBatch+nExp,2]).reshape(nExp,1)       
        matches_cond = torch.where(cond, iter_matches[:,:], matches[initBatch:initBatch+nExp,:])
        matches[initBatch:initBatch+nExp,:] = matches_cond
        
        # torch.cuda.empty_cache()        
        return(matches)
    
    
    def center_shifts(self, Texp, initBatch, expBatchSize, prev_shifts):
        batchsize = Texp.shape[0]
        translations = prev_shifts[initBatch:initBatch+expBatchSize]
        TexpView = Texp.view(batchsize, 1, Texp.shape[1], Texp.data.shape[2])
        transforTexp = kornia.geometry.translate(TexpView, translations)
        transforTexp = transforTexp.view(batchsize, Texp.shape[1], Texp.data.shape[2])
        del(Texp)
        del(TexpView)
        del(translations)     
        return(transforTexp)
    
    def center_shifts2(self, Texp, initBatch, expBatchSize, prevPosition):
        batchsize = Texp.shape[0]
        translations = prevPosition[initBatch:initBatch+expBatchSize,1:]
        TexpView = Texp.view(batchsize, 1, Texp.shape[1], Texp.data.shape[2])
        transforTexp = kornia.geometry.translate(TexpView, translations)
        transforTexp = transforTexp.view(batchsize, Texp.shape[1], Texp.data.shape[2])
        del(Texp)
        del(TexpView)
        del(translations)     
        return(transforTexp)
    
    def center_rot(self, Texp, initBatch, expBatchSize, prevPosition):
        batchsize = Texp.shape[0]
        rotation = -prevPosition[initBatch:initBatch+expBatchSize, 0]
        TexpView = Texp.view(batchsize, 1, Texp.shape[1], Texp.data.shape[2])
        transforTexp = kornia.geometry.rotate(TexpView, rotation)
        transforTexp = transforTexp.view(batchsize, Texp.shape[1], Texp.data.shape[2])
        del(Texp)
        del(TexpView)
        del(rotation)
        return(transforTexp)
    
    
    def center_particles(self, Texp, initBatch, expBatchSize, prevPosition):
        
        dim = Texp.size(dim=1)
        rotBatch = -prevPosition[initBatch:initBatch+expBatchSize, 0]   
        batchsize = rotBatch.size(dim=0)
        rotBatch = rotBatch.view(batchsize)  
        translations = prevPosition[initBatch:initBatch+expBatchSize,1:].view(batchsize,2,1)
        
        centerxy = torch.tensor([dim/2,dim/2], device = self.cuda)   
        center = torch.stack([centerxy]*batchsize, dim=0)      
        scale = torch.ones(batchsize, 2, device = self.cuda)
        
        #rotation matrix
        M = kornia.geometry.get_rotation_matrix2d(center, rotBatch, scale) 
        
        # Add translate
        shape = list(M.shape)
        shape[-1] -= 1
        M = M + torch.cat([torch.zeros(shape, device = self.cuda), translations], -1)  

        transforTexp = kornia.geometry.warp_affine(Texp.view(batchsize, 1, dim, dim), M, dsize=(dim, dim), mode='bilinear')
        transforTexp = transforTexp.view(batchsize, dim, dim)
        del(Texp)
        
        return(transforTexp)
        

        

            

    
    
