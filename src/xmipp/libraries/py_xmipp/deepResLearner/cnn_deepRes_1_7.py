from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import Sequence, to_categorical
import glob
import numpy as np
import os
import random
import string
import sys
import xmipp
import xmippLib

boxDim=13
boxDim2 = boxDim//2
fnDir = sys.argv[1] # /dataset/PDB/3Dcomplex_set/

allPDBs = glob.glob(fnDir+"/*pdb")

# generator
def Gen(idx):

	while True:

		n = np.random.randint(0,len(allPDBs))

		maxRes = round(np.random.uniform(0.9,7.0),1)

		#VOLUMENES
		ok = False
		fnRandom = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(32)])
		fnHash = "tmp"+fnRandom
		while not ok:
		    ok = True
		    ok = os.system("xmipp_volume_from_pdb  -i %s -o %s --sampling 0.5 --centerPDB -v 0"%(allPDBs[n],fnHash))==0
		    if ok:
			ok = os.system("xmipp_transform_filter -i %s.vol -o %sFiltered.vol --fourier low_pass %f 0.02 --sampling 0.5 -v 0"%(fnHash,fnHash,maxRes))==0
		    if ok:
			ok = os.system("xmipp_transform_threshold -i %s.vol -o %sMask.vol --select below 0.2 --substitute binarize -v 0"%(fnHash,fnHash))==0
		    n = np.random.randint(0,len(allPDBs))

		Vf = xmipp.Image("%sFiltered.vol"%fnHash).getData()		 
		Vmask = xmipp.Image("%sMask.vol"%fnHash).getData()
		os.system("rm -f %s.vol %sFiltered.vol %sMask.vol"%(fnHash,fnHash,fnHash))

		batchX = []
		batchY = []


		#BOX
	        Zdim, Ydim, Xdim = Vf.shape
	        boxDim2 = boxDim//2

                cont1=0
                for z in range(boxDim2,Zdim-boxDim2, 2):
                  for y in range(boxDim2,Ydim-boxDim2, 2):
                    for x in range(boxDim2,Xdim-boxDim2, 2):
                       if Vmask[z,y,x]>0:
                          cont1=cont1+1

                if cont1>1000:
                   batch_size=1000
                else:
                   batch_size=cont1

                cont2=0
                for z in range(boxDim2,Zdim-boxDim2, 2):
                  for y in range(boxDim2,Ydim-boxDim2, 2):
                    for x in range(boxDim2,Xdim-boxDim2, 2):
                       if Vmask[z,y,x]>0:
                        
                          if (cont2<batch_size):

                              box = Vf[z-boxDim2:z+boxDim2+1,y-boxDim2:y+boxDim2+1,x-boxDim2:x+boxDim2+1]
 		              boxnorm=(box/np.linalg.norm(box))
		              boxnorm=boxnorm.reshape(boxDim, boxDim, boxDim, 1) 
                          
		              batchX.append(boxnorm)
		              batchY.append(maxRes)

                              cont2=cont2+1

                          else:
                              break



		batchX=np.asarray(batchX).astype("float32")
		batchY=np.asarray(batchY).astype("float32")


		yield (batchX, batchY)

def constructModel(boxDim):
    model = Sequential()     
    model.add(Conv3D(32, (13,13,13), activation='relu', input_shape=(boxDim,boxDim,boxDim,1), padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.summary()
    return model



if __name__ == '__main__':

    model = constructModel(boxDim)
    model.compile(loss='mean_squared_error', optimizer='adam')

    pdbM = Gen(allPDBs)

# checkpoint
    filepath="/home/erney/data/test_allboxes_dense_512_1-7A_window13_paral/model_w7.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)

    callbacks_list = [checkpoint] 

#fit
    model.fit_generator(generator=pdbM,  epochs=5000, steps_per_epoch=50, callbacks=callbacks_list, verbose=1, use_multiprocessing=True, workers=4)
    model.save('/home/erney/data/test_allboxes_dense_512_1-7A_window13_paral/model_w7_all.h5')
  
