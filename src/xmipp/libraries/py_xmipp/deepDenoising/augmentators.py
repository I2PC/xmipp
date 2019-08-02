import numpy as np
import random, scipy
import scipy.ndimage

def _random_flip_leftright( batchX, batchY):
  for i in range(batchX.shape[0]):
    if bool(random.getrandbits(1)):
      batchX[i] = np.fliplr(batchX[i])
      batchY[i] = np.fliplr(batchY[i])
  return batchX, batchY

def _random_flip_updown( batchX, batchY):
  for i in range(batchX.shape[0]):
    if bool(random.getrandbits(1)):
      batchX[i] = np.flipud(batchX[i])
      batchY[i] = np.flipud(batchY[i])
  return batchX, batchY

def _random_90degrees_rotation( batchX, batchY, rotations=[0, 1, 2, 3]):
  for i in range(batchX.shape[0]):
    num_rotations = random.choice(rotations)
    batchX[i] = np.rot90(batchX[i], num_rotations)
    batchY[i] = np.rot90(batchY[i], num_rotations)
  return batchX, batchY

def _random_rotation( batchX, batchY, max_angle=25.):
  for i in range(batchX.shape[0]):
    if bool(random.getrandbits(1)):
      # Random angle
      angle = random.uniform(-max_angle, max_angle)
      batchX[i] = scipy.ndimage.interpolation.rotate(batchX[i], angle,reshape=False, mode="reflect")
      batchY[i] = scipy.ndimage.interpolation.rotate(batchY[i], angle,reshape=False, mode="reflect")
  return batchX, batchY

def _random_blur( batchX, batchY, sigma_max):
  for i in range(batchX.shape[0]):
    if bool(random.getrandbits(1)):
      # Random sigma
      sigma = random.uniform(0., sigma_max)
      batchX[i] = scipy.ndimage.filters.gaussian_filter(batchX[i], sigma)
      batchY[i] = scipy.ndimage.filters.gaussian_filter(batchY[i], sigma)
  return batchX, batchY


def _mismatch_projection( batchX, batchY, p=0.05):
  maxMismatch= 1+int(batchX.shape[1]*.01)
  for i in range(batchX.shape[0]):
    if np.random.rand()<p:
      angle = random.uniform(-maxMismatch,-maxMismatch)
      batchX[i] = scipy.ndimage.interpolation.rotate(batchX[i], angle,reshape=False, mode="reflect")
  return batchX, batchY
  

def initPointsRadius(img_shape, aboveInsteadBelow=True, radiusFraction=0.9):

  xv, yv,= np.meshgrid(np.arange( img_shape[0] ), np.arange( img_shape[1] ), indexing="xy")
  allCoords= np.stack([xv, yv], axis=-1).reshape((-1,2))
  central_coords= np.array([img_shape[0]//2, img_shape[1]//2] )
  distMat= np.squeeze( np.sqrt( np.sum((central_coords- allCoords)**2, axis=1))  )
  distThr= int(radiusFraction*(img_shape[0]/2))
  if aboveInsteadBelow:
    return allCoords[ distMat>distThr ].astype( np.int64)
  else:
    return allCoords[ distMat<distThr ].astype( np.int64)

  
def generateReverseNormalizationFunction(img_shape, radiusFraction=0.8):
  '''
  This function is used to generate a function that will compute the inverse operation of normalization given an
  stack of treated images and an stack of the original images.
  :param img_shape: tuple
  :param radiusFraction: the fraction of the image size to consider as the radius of the particle
  :return: function reverseNormalization
  '''
  pointsBelow= initPointsRadius(img_shape, aboveInsteadBelow=False, radiusFraction=radiusFraction)
  def reverseNormalization(imgs, refs):
    '''
    Given an stack of treated images and an stack of the original images, computes the inverse operation of the
    normalization
    :param imgs: stack of treated images (denoised)
    :param refs: stack of original images
    :return: stack of denormalized images
    '''
    imgsDeNorm= np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
      img, ref= imgs[i], refs[i]
      #compute the statistics of the original particles and the background (inside/outside radius)
      innerParRef= ref[ pointsBelow[...,0], pointsBelow[...,1] ] #pointsBelow are the a matrix of [i,j] that are inside the radius
      innerParImg= img[ pointsBelow[...,0], pointsBelow[...,1] ]
      innerMeanRef, innerStdRef= np.mean(innerParRef), np.std(innerParRef)
      innerMeanImg, innerStdImg= np.mean(innerParImg), np.std(innerParImg)
      imgsDeNorm[i]= (img - (innerMeanImg - innerMeanRef)) / (innerStdImg / innerStdRef)
    return imgsDeNorm
  return reverseNormalization
    
def generateEmptyParticlesFunction(img_shape, prob=0.2):

  cornerSize= 8
  imgSide= min(img_shape[:2])
  nTilesPerSide= imgSide//cornerSize
  
  pointsAboveRadius= initPointsRadius(img_shape, radiusFraction=0.9)

  def noisifyImg( img):
    dataSlice= img[ pointsAboveRadius[...,0], pointsAboveRadius[...,1] ] 
    meanVal= np.mean( dataSlice )
    stdVal= np.std( dataSlice )
    noise_img= np.random.normal(meanVal, stdVal, img.shape )
    return noise_img
    

  def swapParticlesForEmptyParticles1( batchX, batchY):
    for i in range(batchX.shape[0]):
      if np.random.rand()<prob:
        batchX[i]= noisifyImg( batchX[i] )
        batchY[i]= -1*np.ones_like( batchY[i] )
    return batchX, batchY
  
  
  if imgSide%cornerSize!=0:
    print("noisify with mu sigma")
    return swapParticlesForEmptyParticles1
  else:
    print("noisify with mu sigma and corner patches")
    def swapParticlesForEmptyParticles2( batchX, batchY):

      corners= np.concatenate( _random_90degrees_rotation(batchX[:, 0:cornerSize, 0:cornerSize, ...], 
                                                          batchX[:, -cornerSize:, -cornerSize:,...] ) +
                               _random_90degrees_rotation(batchX[:, 0:cornerSize, -cornerSize:,...], 
                                                          batchX[:, -cornerSize:, 0:cornerSize,...]) )
      np.random.shuffle( corners)
      nToTake= 4*(nTilesPerSide**2)
      corners= np.tile( corners, (nToTake//corners.shape[0]+1,)+tuple([1]*(len(corners.shape)-1) ) )                                              
      imgsNChans= batchX.shape[-1]
      corners= corners[: nToTake, ...]
      corners_reshaped1=  np.reshape(corners, (-1, nTilesPerSide**2, cornerSize,cornerSize,imgsNChans))
      corners_reshaped2=  np.reshape(corners_reshaped1, (corners_reshaped1.shape[0], cornerSize*nTilesPerSide,
                                                         cornerSize*nTilesPerSide,imgsNChans))
      nNoisyParticles= corners_reshaped2.shape[0]
      k=0
      for i in range(batchX.shape[0]):
        if np.random.rand()<prob:
          if k<nNoisyParticles and np.random.rand()<0.5:
            batchX[i]= corners_reshaped2[k]
            k+=1
          else:
            batchX[i]= noisifyImg( batchX[i] )
          batchY[i]= -1*np.ones_like( batchY[i] )

      return batchX, batchY

    return swapParticlesForEmptyParticles2
      
