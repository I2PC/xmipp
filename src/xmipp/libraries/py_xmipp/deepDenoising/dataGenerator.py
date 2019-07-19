import pyworkflow.em.metadata as md
import xmippLib
import numpy as np
import random

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from .augmentators import (_random_flip_leftright, _random_flip_updown, _mismatch_projection, 
                          _random_90degrees_rotation, _random_rotation,generateEmptyParticlesFunction)

BATCH_SIZE= 16
def getDataGenerator( imgsMdXmd, masksMdXmd, xmdEmptyParts=None, augmentData=True, nEpochs=-1, isTrain=True, valFraction=0.1, 
            batchSize= BATCH_SIZE, doTanhNormalize=True, simulateEmptyParts=True, addMismatch=False, path_size_fraction=None): 

  if nEpochs<1: 
    nEpochs= 9999999
  mdImgs  = md.MetaData(imgsMdXmd)
  mdMasks = md.MetaData(masksMdXmd)
    
  nImages= int(mdImgs.size())

  I= xmippLib.Image()    

  imgFnames = mdImgs.getColumnValues(md.MDL_IMAGE)
  maskFnames= mdMasks.getColumnValues(md.MDL_IMAGE)
  
  I.read( imgFnames[0] )
  shape= I.getData().shape+ (1,)
  if not path_size_fraction is None and 0<path_size_fraction<1:
    shape= tuple([int(path_size_fraction*elem) for elem in shape[:-1] ])+(1,)
    
  if not xmdEmptyParts is None:
    mdEmpty= md.MetaData(xmdEmptyParts)
    nImages+= int(mdEmpty.size())
    emptyFnames= mdEmpty.getColumnValues(md.MDL_IMAGE)
    imgFnames+= emptyFnames
    maskFnames+= [None]*len(emptyFnames)
  
  stepsPerEpoch= nImages//batchSize
    

  if augmentData:
    augmentFuns= [_random_flip_leftright, _random_flip_updown, _random_90degrees_rotation, _random_rotation ]
    if simulateEmptyParts==True:
      augmentFuns+= [generateEmptyParticlesFunction(shape, prob=0.2)]
    if addMismatch==True:
      augmentFuns+= [_mismatch_projection]
    def augmentBatch( batchX, batchY):
      for fun in augmentFuns:
        if bool(random.getrandbits(1)):
          batchX, batchY= fun(batchX, batchY)
      return batchX, batchY
  else:
    def augmentBatch( batchX, batchY): return batchX, batchY
    
  if valFraction>0:
    (imgFnames_train, imgFnames_val, maskFnames_train,
     maskFnames_val) = train_test_split(imgFnames, maskFnames, test_size=valFraction, random_state=121)  
    if isTrain:
      imgFnames, maskFnames= imgFnames_train, maskFnames_train
    else:
      imgFnames, maskFnames= imgFnames_val, maskFnames_val

  def readOneImage(fname):
    I.read(fname)
    return I.getData()


  def readImgAndMask(fnImageImg, fnImageMask):
    img= normalization( np.expand_dims(readOneImage(fnImageImg), -1), sigmoidInsteadTanh= not doTanhNormalize)
    if fnImageMask is None:
      mask= -1*np.ones_like(img)
    else:
      mask= normalization(np.expand_dims(readOneImage(fnImageMask), -1), sigmoidInsteadTanh=not doTanhNormalize)
    return img, mask
      
  def extractPatch(img, mask):
    halfShape0= shape[0]//2
    halfShape0Right= halfShape0 + shape[0]%2
    halfShape1= shape[1]//2
    halfShape1Right= halfShape1 + shape[1]%2  
    hpos= random.randint(halfShape0, img.shape[0]-halfShape0Right)
    wpos= random.randint(halfShape1, img.shape[1]-halfShape1Right)
    
    img= img[hpos-halfShape0: hpos+halfShape0Right, wpos-halfShape1: wpos+halfShape1Right]
    mask= mask[hpos-halfShape0: hpos+halfShape0Right, wpos-halfShape1: wpos+halfShape1Right]
    return img, mask
      
  def dataIterator(imgFnames, maskFnames, nBatches=None):
    
    batchStack = np.zeros((batchSize,)+shape )
    batchLabels = np.zeros((batchSize,)+shape )
    currNBatches=0 
    for epoch in range(nEpochs):
      if isTrain:
        imgFnames, maskFnames= shuffle(imgFnames, maskFnames)
      n=0
      for fnImageImg, fnImageMask in zip(imgFnames, maskFnames):
        img, mask= readImgAndMask(fnImageImg, fnImageMask)
        if not path_size_fraction is None:
          img, mask= extractPatch(img, mask)
#          print(img.shape, mask.shape)
        batchStack[n,...], batchLabels[n,...]= img, mask
        n+=1
        if n>=batchSize:
          yield augmentBatch(batchStack, batchLabels)
          n=0
          currNBatches+=1
          if nBatches and currNBatches>=nBatches:
            break
      if n>0:
        yield augmentBatch(batchStack[:n,...], batchLabels[:n,...])
        
  return dataIterator(imgFnames, maskFnames), stepsPerEpoch
  
def extractNBatches(valIterator, maxBatches=-1):
  x_val=[]
  y_val=[]
  for i, (x, y) in enumerate(valIterator):
    x_val.append(x)
    y_val.append(y)
    if i== maxBatches:
      break
  return ( np.concatenate(x_val, axis=0), np.concatenate(y_val, axis=0 ))

def normalizationV1( img, sigmoidInsteadTanh=True):
  normData= (img -np.min(img))/ (np.max(img)-np.min(img))
  if not sigmoidInsteadTanh:
    normData= 2*normData -1
  if np.any( np.isnan(normData)):
    normData= np.zeros_like(normData)
  return normData

#from scipy.stats import iqr
#def normalizationV2(img, sigmoidInsteadTanh=True):
#  '''
#  Proposed alternative normalization. Seems to be worse
#  '''
#  iqr_val= iqr(img, rng=(10,90) )
#  if iqr_val==0:
#      iqr_val= (np.max(img)-np.min(img)) + 1e-12
#  newImg=(img- np.median(img))/iqr_val
#  if sigmoidInsteadTanh:
#    newImg=1 / (1 + np.exp(-newImg))
#  else:
#    newImg= np.tanh(newImg)
#  return newImg

normalization= normalizationV1

def normalizeImgs(batch_img, sigmoidInsteadTanh=True):
  for i in range(batch_img.shape[0]):
    batch_img[i]= normalization(batch_img[i], sigmoidInsteadTanh)
  return batch_img
  
if __name__=="__main__":
  import sys, os
  import matplotlib.pyplot as plt
  runsPath="/home/rsanchez/ScipionUserData/projects/tryDenoiser"
  xmdParticles=os.path.join(runsPath, "Runs/004808_XmippProtDeepDenoising/extra/resizedParticles.xmd")
  xmdProjections=os.path.join(runsPath,"Runs/004808_XmippProtDeepDenoising/extra/resizedProjections.xmd")
  xmdEmptyParts=None
  os.chdir(runsPath)
  trainIterator, stepsPerEpoch= getDataGenerator(xmdParticles, xmdProjections, xmdEmptyParts=xmdEmptyParts,
                                                 isTrain=True, augmentData=True,
                                                 valFraction=0.1, batchSize=32, doTanhNormalize=True)

  for patch_x, patch_y in trainIterator:
#  for patch_x, patch_y, fnames in one_gen:
    print(patch_x.shape, patch_y.shape)
    print(patch_x.mean(), patch_y.mean())
    for x,y in zip( patch_x, patch_y ):
#    for x,y,fname in zip( patch_x, patch_y, fnames ):
      fig, axarr = plt.subplots(1, 3)
#      fig.suptitle(fname)
      if len(axarr.shape)==1:
        axarr= np.expand_dims(axarr, axis=0)
      k=0
      axarr[k,0].imshow(  np.squeeze(x), cmap="gray" )
      axarr[k,0].set_title("particle")
#      axarr[k,0].axis('off')
      axarr[k,1].imshow(  np.squeeze(y), cmap="gray" )
      axarr[k,1].set_title("projection")
#      axarr[k,1].axis('off')
      axarr[k,2].imshow(  np.squeeze(x*y), cmap="gray" )
      axarr[k,2].set_title("particle*projection")
#      axarr[k,2].axis('off')
      plt.show()
  print("DONE")
