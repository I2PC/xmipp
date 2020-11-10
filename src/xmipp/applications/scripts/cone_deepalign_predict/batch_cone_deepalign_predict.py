#!/usr/bin/env python2


#import cv2
import math
import numpy as np
import os
import random
import string
import sys
import xmippLib
import time
from shutil import copy
import pyworkflow.em.metadata as md


def loadData(mdIn, mdExp):
    XProj=None
    XExp=None
    Nproj = mdIn.size()
    Nexp = mdExp.size()
    idx = 0
    for objId in mdIn:
        fnImg = mdIn.getValue(xmippLib.MDL_IMAGE,objId)
        I = xmippLib.Image(fnImg)
        if XProj is None:
            Xdim, Ydim, _, _ = I.getDimensions()
            XProj = np.zeros((Nproj,Xdim,Ydim,1),dtype=np.float64)
	XProj[idx,:,:,0] = I.getData()
        idx+=1

    idx = 0
    for objId in mdExp:
        fnImg = mdExp.getValue(xmippLib.MDL_IMAGE,objId)
        I = xmippLib.Image(fnImg)
        if XExp is None:
            Xdim, Ydim, _, _ = I.getDimensions()
            XExp = np.zeros((Nexp,Xdim,Ydim,1),dtype=np.float64)
	XExp[idx,:,:,0] = I.getData()
        idx+=1

    return XProj, XExp, Xdim, Ydim, Nproj, Nexp


if __name__=="__main__":

    from xmippPyModules.deepLearningToolkitUtils.utils import checkIf_tf_keras_installed
    checkIf_tf_keras_installed()

    fnExp = sys.argv[1]
    #fnOName = sys.argv[2]
    fnODir = sys.argv[2]
    Xdim = int(sys.argv[3])
    numClassif = int(sys.argv[4])
    numMax=int(sys.argv[5])
    gpuId = sys.argv[6]

    if not gpuId.startswith('-1'):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

    from keras.callbacks import TensorBoard, ModelCheckpoint
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
    from keras.optimizers import Adam
    from keras.models import load_model
    import tensorflow as tf
    
    print('Predict mode')
    newImage = xmippLib.Image()
    #copy(fnExp, fnOName)
    mdExp = xmippLib.MetaData(fnExp)
    #XProj, XExp, Xdim, Ydim, Nproj, Nexp = loadData(mdProj, mdExp)
    #AJ no quiero cargar todas las imagenes en memoria
    #si puedo cargar todas las projs e iterar por las exp

    sizeBatch=1000
    Nexp = mdExp.size()
    maxBatchs=np.ceil(float(Nexp)/float(sizeBatch))
    Ypred = np.zeros((Nexp),dtype=np.float64)   
    refPred = np.zeros((Nexp,(numMax*2)+1),dtype=np.float64)    
    models=[]
    for i in range(numClassif):
	if os.path.exists(os.path.join(fnODir,'modelCone%d.h5'%(i+1))):
	    models.append(load_model(os.path.join(fnODir,'modelCone%d.h5'%(i+1))))
	else:
	    models.append(None)
    if Nexp>sizeBatch:
        oneXExp = np.zeros((sizeBatch,Xdim,Xdim,1),dtype=np.float64)
        YpredAux = np.zeros((sizeBatch,numClassif),dtype=np.float64)

    idxExp = 0
    countBatch=0
    numBatch = 0
    done = 0
    for row in md.iterRows(mdExp):
	objIdExp = row.getObjId()
	#print("AAAA", objIdExp)
	#print("BBBB", mdExp.getValue(xmippLib.MDL_ITEM_ID,objIdExp))
	if numBatch==(maxBatchs-1) and done==0:
	    oneXExp = np.zeros((Nexp-idxExp,Xdim,Xdim,1),dtype=np.float64)
            YpredAux = np.zeros((Nexp-idxExp,numClassif),dtype=np.float64)
	    done=1
	fnExp = mdExp.getValue(xmippLib.MDL_IMAGE,objIdExp)
	Iexp = xmippLib.Image(fnExp)
        oneXExp[countBatch,:,:,0] = Iexp.getData()
	oneXExp[countBatch,:,:,0] = (oneXExp[countBatch,:,:,0]-np.mean(oneXExp[countBatch,:,:,0]))/np.std(oneXExp[countBatch,:,:,0])
	countBatch+=1
	idxExp+=1
	refPred[idxExp-1,0] = mdExp.getValue(xmippLib.MDL_ITEM_ID,objIdExp)
	if ((idxExp%sizeBatch)==0 or idxExp==Nexp):
	    countBatch = 0
            for i in range(numClassif):
	        model = models[i]
		if model is not None:
                    out = model.predict([oneXExp])
		else:
		    myDim= YpredAux.shape
		    myDim = myDim[0]
		    out = -1.0*np.zeros((myDim,1),dtype=np.float64)
		YpredAux[:,i] = out[:,0]
	    if numBatch==(maxBatchs-1):
		for n in range(numMax):
                    Ypred[numBatch*sizeBatch:Nexp] = np.max(YpredAux, axis=1)
	            auxPos = np.argmax(YpredAux, axis=1)
	            refPred[numBatch*sizeBatch:Nexp, (n*2)+1] = np.argmax(YpredAux, axis=1)+1
	            refPred[numBatch*sizeBatch:Nexp, (n*2)+2] = Ypred[numBatch*sizeBatch:Nexp]
		    for i,pos in enumerate(auxPos):
		        YpredAux[i,pos]=0.0
	    else:
		for n in range(numMax):
                    Ypred[idxExp-sizeBatch:idxExp] = np.max(YpredAux, axis=1)
	            auxPos = np.argmax(YpredAux, axis=1)
   		    #print(n, YpredAux, len(auxPos))
	            refPred[idxExp-sizeBatch:idxExp, (n*2)+1] = np.argmax(YpredAux, axis=1)+1
	            refPred[idxExp-sizeBatch:idxExp, (n*2)+2] = Ypred[idxExp-sizeBatch:idxExp]
		    for i,pos in enumerate(auxPos):
		        YpredAux[i,pos]=0.0
	    numBatch+=1

    np.savetxt(os.path.join(fnODir,'conePrediction.txt'), refPred)








