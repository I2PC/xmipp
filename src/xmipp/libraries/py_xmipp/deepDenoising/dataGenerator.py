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
    if simulateEmptyParts:
      augmentFuns+= [generateEmptyParticlesFunction(shape, prob=0.2)]
    if addMismatch:
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

def normalization( img, sigmoidInsteadTanh=True):
  normData= (img -np.min(img))/ (np.max(img)-np.min(img))
  if not sigmoidInsteadTanh:
    normData= 2*normData -1
  if np.any( np.isnan(normData)):
    normData= np.zeros_like(normData)
  return normData

def normalizeImgs(batch_img, sigmoidInsteadTanh=True):
  for i in range(batch_img.shape[0]):
    batch_img[i]= normalization(batch_img[i], sigmoidInsteadTanh)
  return batch_img