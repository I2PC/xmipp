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
import xmipp
import xmippLib

boxDim = 13
boxDim2 = boxDim//2
maxSize = 1000


def getBox(V,z,y,x):
    boxDim2 = boxDim//2
    box = V[z-boxDim2:z+boxDim2+1,y-boxDim2:y+boxDim2+1,x-boxDim2:x+boxDim2+1]
    box = box/np.linalg.norm(box)
    return box
    #return box.reshape(boxDim, boxDim, boxDim, 1) 

class VolumeManager(Sequence):
    def __init__(self, fnVol, fnMask):
        self.V = xmipp.Image(fnVol).getData()
        self.M = xmipp.Image(fnMask).getData()
        self.Zdim, self.Ydim, self.Xdim = self.V.shape

        #calculate total voxels (inside mask)
        vx=0
        for self.z in range(boxDim2,self.Zdim-boxDim2):
           for self.y in range(boxDim2,self.Ydim-boxDim2):
              for self.x in range(boxDim2,self.Xdim-boxDim2):
                  if self.M[self.z,self.y,self.x]>0.2:
                        if ((self.x+self.y+self.z)%2)==0:
                             vx=vx+1
        print vx
        self.st=vx/maxSize
        if vx % maxSize >0:
           self.st=self.st+1
        print self.st


        self.x = boxDim2
        self.y = boxDim2
        self.z = boxDim2


        #print (self.x   ,   self.y  ,  self.z, self.M[self.z,self.y,self.x])
        #print (self.M[20,20,20])
        if self.M[self.z,self.y,self.x]<=0.2:
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
            if self.M[self.z,self.y,self.x]>0.2:
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
   


def produceOutput(fnVolIn, fnMask, Y, fnVolOut):
    Vxmipp = xmipp.Image(fnVolIn)
    Mask= xmipp.Image(fnMask)
    V = Vxmipp.getData()
    M = Mask.getData()
    V = V*0
    Zdim, Ydim, Xdim = V.shape
    idx = 0
    boxDim2 = boxDim//2
    for z in range(boxDim2,Zdim-boxDim2):
        for y in range(boxDim2,Ydim-boxDim2):
            for x in range(boxDim2,Xdim-boxDim2):
                if M[z,y,x]>0.2:
                    if ((x+y+z)%2)==0:
                        if Y[idx]>12.9:
                           Y[idx]=12.9
                        if Y[idx]<2.5:
                           Y[idx]=2.5
                        V[z,y,x]=Y[idx]
                        idx=idx+1

    ###mean
    for z in range(boxDim2+1,Zdim-boxDim2):
        for y in range(boxDim2+1,Ydim-boxDim2):
            for x in range(boxDim2+1,Xdim-boxDim2):
                if M[z,y,x]>0.2:
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

    parser = argparse.ArgumentParser(description="determine the local resoluction")
    parser.add_argument("-dl", "--dl_model", help="input deep learning model", required=True)
    parser.add_argument("-i", "--map", help="input map", required=True)
    parser.add_argument("-m", "--mask", help="input mask", required=True)
    parser.add_argument("-s", "--sampling", help="sampling rate", required=True)
    parser.add_argument("-o", "--output", help="output resolution map", required=True)   
    args = parser.parse_args()

    fnModel = args.dl_model
#    fnVolOrig = args.map 
#    fnMaskOrig  = args.mask
    fnVolIn = args.map 
    fnMask  = args.mask
    init_sampling = float(args.sampling)
    fnVolOut = args.output

#    Vx = xmipp.Image(fnVolOrig)
#    Vi = Vx.getData()
#    Zdim, Ydim, Xdim = Vi.shape
    #print Zdim

    #change sampling rate to 1
#    fnVolIn = "volume_temp.vol"
#    fnMask = "mask_temp.vol"
#    samplingFactor = float(init_sampling)/1.0
#    fourierValue = float(init_sampling)/(2*1.0)

#    if init_sampling > 1.0:
#	os.system("xmipp_image_resize -i %s -o %s --factor %s  -v 0"%(fnVolOrig,fnVolIn,samplingFactor))
#	os.system("xmipp_image_resize -i %s -o %s --factor %s  -v 0"%(fnMaskOrig,fnMask,samplingFactor))
#    else:
#	os.system("xmipp_transform_filter -i %s -o %s --fourier low_pass %s  -v 0"%(fnVolOrig,fnVolIn,fourierValue))
#	os.system("xmipp_transform_filter -i %s -o %s --fourier low_pass %s  -v 0"%(fnMaskOrig,fnMask,fourierValue))
#	os.system("xmipp_image_resize -i %s  --factor %s  -v 0"%(fnVolIn,samplingFactor))
#	os.system("xmipp_image_resize -i %s  --factor %s  -v 0"%(fnMask,samplingFactor))

    model = load_model(fnModel)
    manager = VolumeManager(fnVolIn,fnMask)
    Y = model.predict_generator(manager, manager.getNumberOfBlocks())

    #print Y.shape
    produceOutput(fnVolIn, fnMask, Y, fnVolOut)

#    Vout = xmipp.Image(fnVolOut)
#    Vo = Vout.getData()
#    ZdimOut, YdimOut, XdimOut = Vo.shape

#    dFactor = float(ZdimOut)/float(Zdim)

#   if init_sampling < 1.0:
#	os.system("xmipp_image_resize -i %s -o %s --dim %f -v 0" %(fnVolOut,fnVolOut,Zdim))
#	os.system("xmipp_image_header -i %s -s %f -v 0" %(fnVolOut,init_sampling))
#    else:
#       fourierValueOut=1.0/(2*dFactor)
#	os.system("xmipp_transform_filter -i %s -o %s --fourier low_pass %s  -v 0"%(fnVolOut,fnVolOut,fourierValueOut))
#	os.system("xmipp_image_resize -i %s -o %s --dim %d -v 0" %(fnVolOut,fnVolOut,Zdim))
#	os.system("xmipp_image_header -i %s -s %f -v 0" %(fnVolOut,init_sampling))

#    os.system("rm -f %s %s "%(fnVolIn,fnMask))


    ###############Histogram#############
    #plt.hist(Y, range=(4,12))#, bins=25)
    plt.hist(Y, bins=25) 
    plt.xlabel('Resolution (A)')
    plt.ylabel('Count')
    #plt.axis([2, 8, 0, 140000]) 
    #plt.xticks(range(2, 7))
    #plt.savefig('hist.png')
    #plt.show()
    #plt.close()

    media=np.mean(Y)
    td=np.std(Y)
    minimo=Y.min()
    maximo=Y.max()

    print ("min resolution = %f" %(minimo))
    print ("max resolution = %f" %(maximo))
    print ("medium resolution = %f   y   std = %f" %(media, td))

