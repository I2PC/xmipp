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
        self.cuda = torch.device('cuda')
        # self.cuda = torch.device('cpu')
    
    
    def setRotAndShift(self, angle, shift):
        
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
     
    
    def precShiftBand(self, ft, freq_band, grid_flat, coef, shift):
        
        fourier_band = self.selectFourierBands(ft, freq_band, coef)
        nRef = fourier_band[0].size(dim=0)
        nShift = shift.size(dim=0)
        
        band_shifted = [torch.zeros((nRef*nShift, coef[n]), device = self.cuda) for n in range(self.nBand)]
                             
        for n in range (self.nBand):           
            angles = torch.matmul(shift, grid_flat[n])     
            filter = torch.polar(torch.ones(1, dtype=torch.float32, device=ft.device), angles)
            
            for i in range(nRef):
                temp = fourier_band[n][i].unsqueeze(0)
                               
                band_shifted_complex = torch.mul(temp, filter)    
                band_shifted_complex[:, int(coef[n] / 2):] = 0.0           
                band_shifted[n][i*nShift : (i*nShift)+nShift] = torch.cat((band_shifted_complex.real, band_shifted_complex.imag), dim=1)
                     
        return(band_shifted)

    
  
    def selectFourierBands(self, ft, freq_band, coef):

        dimFreq = freq_band.shape[1]

        fourier_band = [torch.zeros(int(coef[n]/2), dtype = ft.dtype, device = self.cuda) for n in range(self.nBand)]
        
        freq_band = freq_band.expand(ft.size(dim=0) ,freq_band.size(dim=0), freq_band.size(dim=1))
           
        for n in range(self.nBand):
            fourier_band[n] = ft[:,:,:dimFreq][freq_band == n].view(ft.size(dim=0),int(coef[n]/2))
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
        
            band_real = ft[:,:,:dimfreq][freq_band == n].real.view(batch_size, int(coef[n]/2))
            band_imag = ft[:,:,:dimfreq][freq_band == n].imag.view(batch_size, int(coef[n]/2))
            band[n] =  torch.cat((band_real, band_imag), dim=1)
        # del(band_real, band_imag, freq_band)
        return band
    
    
    
    def phiProjRefs(self, band, vecs):
        nFT = band[0].size(dim=0)
        proj = [torch.matmul(band[n], vecs[n]) for n in range(self.nBand)]
        return proj
        
       
    #Applying rotation and shift
    def precalculate_projection(self, prjTensorCpu, freqBn, grid_flat, coef, cvecs, rot, shift):
                    
        shift_tensor = torch.Tensor(shift).to(self.cuda)       
        prjTensor = prjTensorCpu.to(self.cuda)
   
        rotFFT = torch.fft.rfft2(T.rotate(prjTensor, rot), norm="forward")
        del prjTensor
        band_shifted = self.precShiftBand(rotFFT, freqBn, grid_flat, coef, shift_tensor) 
        del(rotFFT)  
        projBatch = self.phiProjRefs(band_shifted, cvecs)
        del(band_shifted)
        # torch.cuda.empty_cache()

        return(projBatch)
    
    
    def create_batchExp(self, Texp, freqBn, coef, vecs):
        
        batch_projExp = [torch.zeros((Texp.size(dim=0), vecs[n].size(dim=1)), device = self.cuda) for n in range(self.nBand)]
        
        expFFT = torch.fft.rfft2(Texp, norm="forward")      
        bandExp = self.selectBandsRefs(expFFT, freqBn, coef)
        # del(expFFT)
        batch_projExp = self.phiProjRefs(bandExp, vecs)
        # del(bandExp)
        # torch.cuda.empty_cache()
        return(batch_projExp)
        
      
    def match_batch(self, batchExp, batchRef, initBatch, matches, rot, nShift):
        
        nExp = batchExp[0].size(dim=0) 
        nShift = torch.tensor(nShift, device=self.cuda)       
                             
        for n in range(self.nBand):
            score = torch.cdist(batchRef[n], batchExp[n])
              
        min_score, ref = torch.min(score,0)
        del(score)
           
        sel = (torch.floor(ref/nShift)).type(torch.int64)
        shift_location = (ref - (sel*nShift)).type(torch.int64)
        rotation = torch.full((nExp,1), rot, device = self.cuda)
        exp = torch.arange(initBatch, initBatch+nExp, 1, device = self.cuda).view(nExp,1)
        
        iter_matches = torch.cat((exp, sel.view(nExp,1), min_score.view(nExp,1), 
                                  rotation, shift_location.view(nExp,1)), dim=1)  
        
        cond = iter_matches[:, 2] < matches[initBatch:initBatch + nExp, 2]
        matches[initBatch:initBatch + nExp] = torch.where(cond.view(nExp, 1), iter_matches, matches[initBatch:initBatch + nExp])       
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
    
    
    
    def center_particles_inverse(self, Texp, initBatch, expBatchSize, prevPosition):
        
        dim = Texp.size(dim=1)
        rotBatch = -prevPosition[initBatch:initBatch+expBatchSize, 0]   
        batchsize = rotBatch.size(dim=0)
        rotBatch = rotBatch.view(batchsize)  
        translations = prevPosition[initBatch:initBatch+expBatchSize,1:].view(batchsize,2,1)
        
        centerxy = torch.tensor([dim/2,dim/2], device = self.cuda) 
        center = torch.ones(batchsize, 2, device=self.cuda) * centerxy       
        scale = torch.ones(batchsize, 2, device = self.cuda)
        
        #translation matrix
        translation_matrix = torch.eye(3, device=self.cuda).unsqueeze(0).repeat(batchsize, 1, 1)
        translation_matrix[:, :2, 2] = translations.squeeze(-1)
        
        #rotation matrix
        rotation_matrix = kornia.geometry.get_rotation_matrix2d(center, rotBatch, scale)        
        M = torch.bmm(rotation_matrix, translation_matrix)
          
        transforTexp = kornia.geometry.warp_affine(Texp.view(batchsize, 1, dim, dim), M, dsize=(dim, dim), mode='bilinear')
        transforTexp = transforTexp.view(batchsize, dim, dim)
        del(Texp)
        
        return(transforTexp)
        
       
    def create_mask(self, images, rad):

        dim = images.size(dim=1)
        center = dim // 2
        y, x = torch.meshgrid(torch.arange(dim) - center, torch.arange(dim) - center, indexing='ij')
        dist = torch.sqrt(x**2 + y**2).to(images.device)

        self.mask = dist <= rad
        self.mask = self.mask.float() 
        
        return self.mask
    
    
    # def create_gaussian_mask(self, images, sigma):
    #     dim = images.size(dim=1)
    #     center = dim // 2
    #     y, x = torch.meshgrid(torch.arange(dim) - center, torch.arange(dim) - center, indexing='ij')
    #     dist = torch.sqrt(x**2 + y**2).float().to(images.device)  
    #
    #     sigma2 = sigma**2
    #     K = 1. / (torch.sqrt(2 * torch.tensor(np.pi)) * sigma)**2
    #
    #     self.mask = K * torch.exp(-0.5 * (dist**2 / sigma2))
    #     self.mask = self.mask / self.mask[center, center].clone()
    #
    #     return self.mask 
            

    
    
