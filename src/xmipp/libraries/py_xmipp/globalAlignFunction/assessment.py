#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import starfile
import torch

class evaluation:
    
    def __init__(self):
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda:0')
        
    #for experimental images with starfile module
    def getValue(self, prjStar):
        
        star=starfile.read(prjStar)
        self.angle_triplet = []
        
        for i in range(len(star)):
            
            phi = star["anglePsi"][i]
            rot = star["angleRot"][i]
            tilt = star["angleTilt"][i]
            angles = (phi, rot, tilt)
            self.angle_triplet.append(angles)

        return self.angle_triplet
    
    
    def getShifts(self, expStar, nExp):
        
        self.expShifts = torch.zeros(nExp, 2, device = self.cuda)       
        star=starfile.read(expStar)
        
        for i in range(nExp):
            
            shX = star["shiftX"][i]
            shY = star["shiftY"][i]           
            self.expShifts[i] = torch.tensor((shX, shY)).view(1,2) 

        return self.expShifts
    
    
    def getPosition(self, expStar, nExp):
    
        self.expPosition = torch.zeros(nExp, 3, device = self.cuda)       
        star=starfile.read(expStar)
        
        for i in range(nExp):
            
            rot = star["anglePsi"][i]
            shX = star["shiftX"][i]
            shY = star["shiftY"][i]           
            self.expPosition[i] = torch.tensor((rot, shX, shY)).view(1,3) 

        return self.expPosition
    
    
    def getShiftsRelion(self, expStar, sampling, nExp):
        
        self.expShifts = torch.zeros(nExp, 2, device = self.cuda)
        
        star=starfile.read(expStar)
        
        for i in range(nExp):
            
            shX = round(star["particles"]["rlnOriginXAngst"][i]/sampling)
            shY = round(star["particles"]["rlnOriginYAngst"][i]/sampling)
            
            self.expShifts[i] = torch.tensor((shX, shY)).view(1,2) 

        return self.expShifts
    
    
        #for experimental images with starfile module
    def writeExpStar(self, prjStar, expStar, matchPair, shiftVec, nExp, apply_shifts, output):
        
        self.getValue(prjStar)
        self.getShifts(expStar, nExp)
        expShifts = self.expShifts.cpu().numpy()
        # self.getPosition(expStar, nExp)
        # expRot = self.expPosition[:,0].cpu().numpy()
        # expShifts = self.expPosition[:,1:].cpu().numpy()
        star=starfile.read(expStar)
        new = output #+ "newStar_exp.xmd"
        
        #Initializing columns
        star["anglePsi"] = 0.0
        star["angleRot"] = 0.0
        star["angleTilt"] = 0.0 
        star["shiftX"] = 0.0
        star["shiftY"] = 0.0
        star["shiftZ"] = 0.0

        for i in range(len(star)):
                                
            id = int(matchPair[i][1])
            new_psi = matchPair[i][3]
            posS = int(matchPair[i][4])           
            new_shiftX = float(shiftVec[posS][0])
            new_shiftY = float(shiftVec[posS][1])
            
            if(new_psi < 180):
                new_psi = new_psi
            else:
                new_psi = new_psi - 360
            # if(new_psi > 180):
            #     new_psi = 360 - new_psi
            # else:
            #     new_psi = -new_psi 
                
            # if(expRot[i] < 180):
            #     expRot[i] = expRot[i]
            # else:
            #     expRot[i] = expRot[i] - 360
        
             
            # if apply_shifts:
            #     star.at[i, "anglePsi"] = new_psi + self.angle_triplet[id][0] + expRot[i]
            # else:
            star.at[i, "anglePsi"] = new_psi + self.angle_triplet[id][0]
            star.at[i, "angleRot"] = self.angle_triplet[id][1]
            star.at[i, "angleTilt"] = self.angle_triplet[id][2]

            if apply_shifts:
                star.at[i, "shiftX"] = new_shiftX + expShifts[i][0]
                star.at[i, "shiftY"] = new_shiftY + expShifts[i][1]
            else:
                star.at[i, "shiftX"] = new_shiftX
                star.at[i, "shiftY"] = new_shiftY 
        
        starfile.write(star, new)
        
        
    def writeExpStarRelion(self, prjStar, expStar, matchPair, shiftVec, sampling, nExp, apply_shifts, output):
        
        self.getValue(prjStar)
        self.getShiftsRelion2(expStar, sampling, nExp)
        self.expShifts = self.expShifts.cpu().numpy()
        star=starfile.read(expStar)
        new = output #+ "newStar_exp.star"
        
        #Initializing columns
        star["particles"]["rlnAnglePsi"] = 0.0
        star["particles"]["rlnAngleRot"] = 0.0
        star["particles"]["rlnAngleTilt"] = 0.0
        star["particles"]["rlnOriginXAngst"] = 0.0
        star["particles"]["rlnOriginYAngst"] = 0.0
        star["particles"]["rlnOriginZAngst"] = 0.0

        for i in range(len(star)):
                                
            id = int(matchPair[i][1])
            new_psi = matchPair[i][3]
            posS = int(matchPair[i][4])           
            new_shiftX = float(shiftVec[posS][0])
            new_shiftY = float(shiftVec[posS][1])
            
            if(new_psi < 180):
                new_psi = new_psi
            else:
                new_psi = new_psi - 360

            star["particles"].at[i, "rlnAnglePsi"] = new_psi + self.angle_triplet[id][0]
            star["particles"].at[i, "rlnAngleRot"] = self.angle_triplet[id][1]
            star["particles"].at[i, "rlnAngleTilt"] = self.angle_triplet[id][2]
            
            if apply_shifts:
                star["particles"].at[i, "rlnOriginXAngst"] = (new_shiftX + self.expShifts[i][0])*sampling
                star["particles"].at[i, "rlnOriginYAngst"] = (new_shiftY + self.expShifts[i][1])*sampling
            else:
                star["particles"][i, "rlnOriginXAngst"] = new_shiftX*sampling
                star["particles"][i, "rlnOriginYAngst"] = new_shiftY*sampling 
            
        #priors  
        star["particles"].loc[:,"rlnOriginXPriorAngst"] = star["particles"]["rlnOriginXAngst"] 
        star["particles"].loc[:,"rlnOriginYPriorAngst"] = star["particles"]["rlnOriginYAngst"]
          
        star["particles"].loc[:,"rlnAnglePsiPrior"] = star["particles"]["rlnAnglePsi"] 
        star["particles"].loc[:,"rlnAngleRotPrior"] = star["particles"]["rlnAngleRot"]
        star["particles"].loc[:,"rlnAngleTiltPrior"] = star["particles"]["rlnAngleTilt"]
           
        starfile.write(star, new)
    
   
   
   
    
    #for experimental images with starfile module
    def writeExpStar2(self, prjStar, expStar, matchPair, bnb, move):
        
        rotVec, shiftVec = bnb.setRotAndShift(move[0], move[1], move[2])
        self.getValue(prjStar)
        star=starfile.read(expStar)
        new = "newStar_exp.star"
        count = 0
        for i in range(len(star)):
                                
            id = matchPair[i][1]-1
            posR = matchPair[i][2]-1
            posS = matchPair[i][3]-1
            
            new_psi = float(rotVec[posR])
            new_shiftX = float(shiftVec[posS][0])
            new_shiftY = float(shiftVec[posS][1])
            
            if(new_psi < 180):
                new_psi = new_psi
            else:
                new_psi = new_psi - 360
            
            star["anglePsi"][i] = new_psi + self.angle_triplet[id][0]
            star["angleRot"][i] = self.angle_triplet[id][1]
            star["angleTilt"][i] = self.angle_triplet[id][2] 

            star["shiftX"][i] = -new_shiftX  
            star["shiftY"][i] = -new_shiftY
        
        starfile.write(star, new)


