#!/usr/bin/env python2
"""/***************************************************************************
 *
 * Authors:     Erney Ramirez Aportela
 *
 * CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
"""

from keras.models import load_model
from keras.utils import Sequence
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import xmipp3

# The method accepts as input a 3D crioEM map and the mask
# both with sampling rate of 1 A/pixel for network 1 or 0.5 A/pixel for network 2

boxDim = 13
boxDim2 = boxDim//2
maxSize = 1000


def getBox(V,z,y,x):
    boxDim2 = boxDim//2
    box = V[z-boxDim2:z+boxDim2+1,y-boxDim2:y+boxDim2+1,x-boxDim2:x+boxDim2+1]
    box = box/np.linalg.norm(box)
    return box 

class VolumeManager(Sequence):
    def __init__(self, fnVol, fnMask):
        self.V = xmipp3.Image(fnVol).getData()
        self.M = xmipp3.Image(fnMask).getData()
        self.Zdim, self.Ydim, self.Xdim = self.V.shape

        #calculate total voxels (inside mask)
        vx=0
        for self.z in range(boxDim2,self.Zdim-boxDim2):
           for self.y in range(boxDim2,self.Ydim-boxDim2):
              for self.x in range(boxDim2,self.Xdim-boxDim2):
                  if self.M[self.z,self.y,self.x]>0.15 and self.V[self.z,self.y,self.x]>0.00015:
                        if ((self.x+self.y+self.z)%2)==0:
                             vx=vx+1
        #print vx
        self.st=vx/maxSize
        if vx % maxSize >0:
           self.st=self.st+1
        #print self.st


        self.x = boxDim2
        self.y = boxDim2
        self.z = boxDim2


        #print (self.x   ,   self.y  ,  self.z, self.M[self.z,self.y,self.x])
        if self.M[self.z,self.y,self.x]<=0.15:
            self.advance()


    def __len__(self):
        return maxSize

    def getNumberOfBlocks(self):
        return self.st

    def advancePos(self):
         self.x+=1
         if self.x==self.Xdim-boxDim2:
            self.x=boxDim2
            self.y+=1
            if self.y==self.Ydim-boxDim2:
                self.x = boxDim2
                self.y = boxDim2
                self.z+=1
                if self.z==self.Zdim-boxDim2:
                    return False
         return True

    def advance(self):
        ok=self.advancePos()
        while ok:
            if self.M[self.z,self.y,self.x]>0.15 and self.V[self.z,self.y,self.x]>0.00015:
                    if (self.x+self.y+self.z)%2==0:
                        break
            ok=self.advancePos()
        return ok 


    def __getitem__(self,idx):
        count=0;
        batchX = []
        ok = True 
        while (count<maxSize and ok):
            batchX.append(getBox(self.V, self.z, self.y, self.x))
            #print (count   ,    self.x   ,   self.y  ,  self.z)
            ok=self.advance()
            count+=1
	batchX=np.asarray(batchX).astype("float32")
        #print("count = ", count) 
        batchX = batchX.reshape(count, batchX.shape[1], batchX.shape[2], batchX.shape[3], 1)      

        #print("batchX.shape = ", batchX.shape)
        return (batchX)
   


def produceOutput(fnVolIn, fnMask, model, sampling, Y, fnVolOut):
    Vxmipp = xmipp3.Image(fnVolIn)
    Mask= xmipp3.Image(fnMask)
    V = Vxmipp.getData()
    Orig = V
    M = Mask.getData()
    V = V*0
    Zdim, Ydim, Xdim = V.shape
    idx = 0

    if model==1:
       if 2*sampling > 2.5:
          minValue=2*sampling
       else:
          minValue=2.5
    if model==2:
       if 2*sampling > 1.5:
          minValue=2*sampling
       else:
          minValue=1.5        

    boxDim2 = boxDim//2
    for z in range(boxDim2,Zdim-boxDim2):
        for y in range(boxDim2,Ydim-boxDim2):
            for x in range(boxDim2,Xdim-boxDim2):
                if M[z,y,x]>0.15 and Orig[z,y,x]>0.00015:
                    if ((x+y+z)%2)==0:

                        if model==1:
                           if Y[idx]>12.9:
                              Y[idx]=12.9
                           if Y[idx]<minValue:
                              Y[idx]=minValue
                        if model==2:
                           if Y[idx]>5.9:
                              Y[idx]=5.9
                           if Y[idx]<minValue:
                              Y[idx]=minValue

                        V[z,y,x]=Y[idx]
                        idx=idx+1

    ###mean
    for z in range(boxDim2+1,Zdim-boxDim2):
        for y in range(boxDim2+1,Ydim-boxDim2):
            for x in range(boxDim2+1,Xdim-boxDim2):
                if M[z,y,x]>0.15 and Orig[z,y,x]>0.00015:
                    if ((x+y+z)%2)!=0:
                       col=0
                       ct=0
                       if V[z+1,y,x]>0:
                          col+=V[z+1,y,x]
                          ct+=1
                       if V[z-1,y,x]>0:
                          col+=V[z-1,y,x]
                          ct+=1
                       if V[z,y+1,x]>0:
                          col+=V[z,y+1,x]
                          ct+=1
                       if V[z,y-1,x]>0:
                          col+=V[z,y-1,x]
                          ct+=1
                       if V[z,y,x+1]>0:
                          col+=V[z,y,x+1]
                          ct+=1
                       if V[z,y,x-1]>0:
                          col+=V[z,y,x-1]
                          ct+=1
                       if ct==0:
                          V[z,y,x]=0
                          ct=1
                       else:
                          meansum=col/ct
                          V[z,y,x]=meansum         


    Vxmipp.setData(V)
    Vxmipp.write(fnVolOut)

if __name__=="__main__":
    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    checkIf_tf_keras_installed()
    parser = argparse.ArgumentParser(description="determine the local resoluction")
    parser.add_argument("-dl", "--dl_model", help="input deep learning model", required=True)
    parser.add_argument("-i", "--map", help="input map", required=True)
    parser.add_argument("-m", "--mask", help="input mask", required=True)
    parser.add_argument("-s", "--sampling", help="sampling rate", required=True)
    parser.add_argument("-o", "--output", help="output resolution map", required=True)   
    args = parser.parse_args()

    fnModel = args.dl_model
    fnVolIn = args.map 
    fnMask  = args.mask
    sampling = float(args.sampling)
    fnVolOut = args.output

    model = load_model(fnModel)
    manager = VolumeManager(fnVolIn,fnMask)
    Y = model.predict_generator(manager, manager.getNumberOfBlocks())

    if fnModel==xmipp3.Plugin.getModel("deepRes", "model_w13.h5"):
       model=1
    if fnModel==xmipp3.Plugin.getModel("deepRes", "model_w7.h5"):
       model=2
    produceOutput(fnVolIn, fnMask, model, sampling, Y, fnVolOut)


