#################################################################################################################
#
#    MODIFIED FROM https://github.com/JGuillaumin/SuperResGAN-keras
#                  https://arxiv.org/pdf/1609.04802.pdf
#    by rsanchez@cnb.csic.es
#
#################################################################################################################

import sys, os

import random, math
import numpy as np
import keras
from keras.models import Model, Sequential, load_model
from keras.utils import multi_gpu_model
from keras.layers import (Input, Conv2D, BatchNormalization, LeakyReLU, Activation,
                         Lambda, Dropout, Flatten, Reshape, Dense, MaxPooling2D)
from keras.optimizers import Adam
from keras.layers.merge import _Merge
from functools import partial
import keras.backend as K
import time


from skimage.transform import rotate

import xmippLib
import matplotlib.pyplot as plt
import pwem.metadata as md
from .DeepLearningGeneric import DeepLearningModel
from .dataGenerator import normalizeImgs, getDataGenerator, extractNBatches

from .unet import build_UNet

BATCH_SIZE= 32
NUM_BATCHES_PER_EPOCH= 128

TRAINING_RATIO= 5
LOSS_WEIGHT= 3
PATCH_SIZE_FRACTION= None #0.45

class GAN(DeepLearningModel):
  def __init__(self, boxSize, saveModelFname, gpuList="0", batchSize=BATCH_SIZE, modelDepth=4, generatorLoss= "MSE",
                  training_DG_ratio= TRAINING_RATIO, loss_logWeight=LOSS_WEIGHT, trainingDataMode="ParticlesAndSyntheticNoise",
                  regularizationStrength=1e-5):

    DeepLearningModel.__init__(self,boxSize, saveModelFname, gpuList, batchSize, generatorLoss= generatorLoss,
                                trainingDataMode=trainingDataMode,  regularizationStrength= regularizationStrength)
                                
    self.epochSize= NUM_BATCHES_PER_EPOCH
    self.trainingDataMode= trainingDataMode
    self.trainingRatio=training_DG_ratio
    self.loss_logWeight= loss_logWeight
    
  def _setShape(self, boxSize):
    self.img_rows = boxSize
    self.img_cols = boxSize
    self.channels = 1
    self.shape = self.img_rows * self.img_cols
    if not PATCH_SIZE_FRACTION is None:
      self.img_shape = (int(self.img_rows*PATCH_SIZE_FRACTION), int(self.img_cols*PATCH_SIZE_FRACTION), self.channels)
    else:
      self.img_shape = (self.img_rows, self.img_cols, self.channels)
    return self.shape, self.img_shape

  def buildGenerator(self):

    if os.path.exists(self.saveModelFname):
      print("loading previously saved model")
      generator = keras.models.load_model(self.saveModelFname, custom_objects={self.generatorLoss.__name__: self.generatorLoss})
      print("model loaded")
    else:
      generator = build_UNet( self.img_shape, out_ch=1, start_ch=32, depth=self.modelDepth, inc_rate=2., activation='relu', 
                        dropout=0.5, batchnorm=True, residual=True, lastActivation="tanh", l1l2_reg=self.regularizationStrength)
    return generator
                        
  def buildDiscriminator(self):
    return build_discriminator( self.img_shape, self.modelDepth )
  

  def train(self, learningRate, nEpochs, xmdParticles, xmdProjections, xmdEmptyParts=None):
    saveImagesPath= self.createSaveImgsPath(xmdParticles)
    generator = self.buildGenerator()
    discriminator = self.buildDiscriminator()

    input_discriminator = Input(shape=self.img_shape, name='input_discriminator')
    output_discriminator = discriminator(input_discriminator)

    discriminator_model = Model(input_discriminator, output_discriminator)
    discriminator_model.name = 'discriminator'
    optimizer_discriminator_model= Adam(learningRate*0.5, beta_1=0.8)
    discriminator_model.compile(loss="binary_crossentropy", metrics=["acc"], optimizer= optimizer_discriminator_model )

    discriminator_model.trainable = False
    input_generator_gan = Input(shape=self.img_shape, name='input_generator_gan')
    output_generator_gan = generator(input_generator_gan)
    output_discriminator_gan = discriminator_model(output_generator_gan)

    generatorGAN_model = Model(inputs=input_generator_gan, outputs=[output_generator_gan, output_discriminator_gan])


    N_GPUs= len(self.gpuList.split(','))
    if N_GPUs > 1:
      generatorGAN_model = multi_gpu_model(generatorGAN_model, gpus= N_GPUs)

    optimizer_generatorGAN_model= Adam(learningRate*0.5, beta_1=0.8)
    generatorGAN_model.compile(loss=[self.generatorLoss, "binary_crossentropy"], loss_weights=[ 10**(self.loss_logWeight), 1.],
                          optimizer=optimizer_generatorGAN_model)

    def generateLabels(nLabels, exact=False, fake_first=True, corruptLabelsProb=0):
      if exact:
        fake_labels = np.zeros( (nLabels,1) )
        real_labels = np.ones( (nLabels,1) ) 
      else:
        fake_labels = np.random.uniform(0.0, 0.3, size=nLabels).astype(np.float32)
        real_labels = np.random.uniform(0.7, 1.2, size=nLabels).astype(np.float32)

      if corruptLabelsProb>0:
        labelsToExchange= np.random.randint(0, nLabels, int(nLabels*corruptLabelsProb))
        for idx in labelsToExchange:
          fake_l= fake_labels[idx]
          fake_labels[idx]= real_labels[idx]
          real_labels[idx]= fake_l
        
      if fake_first:
        return fake_labels, real_labels
      else:
        return real_labels, fake_labels
    
    real_labels_no_random = np.ones((self.batchSize, 1), dtype = np.float32)

    superBatchSize= int((1+self.trainingRatio)*self.batchSize)
    trainIterator, stepsPerEpoch= getDataGenerator(xmdParticles, xmdProjections, xmdEmptyParts=xmdEmptyParts,
                                                   isTrain=True, augmentData=True, path_size_fraction=PATCH_SIZE_FRACTION,
                                                   valFraction=0.1, batchSize=superBatchSize,
                                                   simulateEmptyParts=self.addSyntheticEmpty) 
    nEpochs_init= nEpochs
    nEpochs= int(max(1, nEpochs_init*float(stepsPerEpoch)/self.epochSize ))
    if nEpochs<200: 
      print("WARNING: The number of epochs is probably too small to train a gan. If bad results, try to use more epochs")
                                                              
    print("nEpochs : %.1f --> Epochs: %d.\nTraining begins: Epoch 0/%d"%(nEpochs_init, nEpochs, nEpochs))
    sys.stdout.flush()
    

    valIterator, valStepsPerEpoch= getDataGenerator(xmdParticles, xmdProjections, isTrain=False, valFraction=0.1,
                                           path_size_fraction=PATCH_SIZE_FRACTION,
                                           augmentData=False, nEpochs= 1, batchSize=100 )

    particles_val, projections_val = extractNBatches(valIterator, 2) ; del valIterator

    bestValidationLoss = sys.maxsize

    roundsToEarlyStopping=40
    roundsToLR_decay=15
    
    remainingRoundsToTrainDiscr= 0 #To use in case some handicap is desired
    remainingBatchesToTrainGen= 0  #To use in case some handicap is desired
    roundsNoImprovementSinceLRDecay=0
    roundsNoImprovement=0
    currTime= time.time()
    for epoch in range(nEpochs):
      discriminatorLoss_list=[]
      generatorLoss_list=[]
      discriminatorInGeneratorLoss_list=[]
      for batchNum, (X_particles, Y_projections) in enumerate( trainIterator ):
        if X_particles.shape[0]!= superBatchSize:
          continue
        
        if remainingRoundsToTrainDiscr<=0:
          discriminator_model.trainable = True
          for idx in range(0, superBatchSize-self.batchSize, self.batchSize):
            X_particles1= X_particles[idx: idx+self.batchSize]
            Y_projections1= Y_projections[idx: idx+self.batchSize]
            generated_projections, __= generatorGAN_model.predict(X_particles1)

            fake_labels, real_labels= generateLabels(X_particles1.shape[0], exact=False)
            if bool( random.getrandbits(1)):
              d_loss_fake, acc_fake = discriminator_model.train_on_batch(generated_projections, fake_labels)
              d_loss_real, acc_real = discriminator_model.train_on_batch(Y_projections1, real_labels)
            else:
              d_loss_real, acc_real = discriminator_model.train_on_batch(Y_projections1, real_labels)
              d_loss_fake, acc_fake = discriminator_model.train_on_batch(generated_projections, fake_labels)

            discriminatorLoss_list.append( 0.5*(d_loss_fake+ d_loss_real) )
            discriminator_model.trainable = False
        else:
          discriminatorLoss_list.append( np.nan )
          remainingRoundsToTrainDiscr-=1
          idx=superBatchSize-2*self.batchSize
          
        if remainingBatchesToTrainGen<=0:
          X_particles2= X_particles[idx+self.batchSize:]
          Y_projections2= Y_projections[idx+self.batchSize:]
          generatorLoss= generatorGAN_model.train_on_batch( X_particles2, [Y_projections2, real_labels_no_random] )
          generatorLoss_list.append(generatorLoss[1])
          discriminatorInGeneratorLoss_list.append(generatorLoss[-1])
        else:
          generatorLoss_list.append(np.nan)
          discriminatorInGeneratorLoss_list.append(np.nan)
          remainingBatchesToTrainGen-=1
          
        if batchNum>= self.epochSize:
          break
      
      generatorValLoss= generatorGAN_model.evaluate(particles_val, [projections_val, 
                                                                     np.ones(projections_val.shape[0])], verbose=0)

      discriminatorInGeneratorValLoss= generatorValLoss[-1]
      generatorValLoss= generatorValLoss[1]
      generated_imgs, __= generatorGAN_model.predict(particles_val, batch_size= self.batchSize, verbose=0)
      X_forDiscriminator= np.concatenate([generated_imgs, projections_val])
      Y_forDiscriminator= np.concatenate(generateLabels( generated_imgs.shape[0], exact=True, fake_first=True))
      disc_val_loss, disc_val_acc = discriminator_model.evaluate(X_forDiscriminator, Y_forDiscriminator, verbose=0)
      
      # Plot the progress at the end of each epoch
      meanDiscrLoss= self.takeListMean(discriminatorLoss_list)
      meanGenLoss= self.takeListMean(generatorLoss_list)
      newTime= time.time()
      print(("\nEpoch %d/%d ended (%2.1f s). discr_loss= %2.5f  generator_loss= %2.5f val_discr_acc= %2.5f  "+
             "val_generator_mse= %2.5f")%(epoch+1,nEpochs, newTime-currTime, meanDiscrLoss, meanGenLoss,
                                                    disc_val_acc, generatorValLoss) )
      currTime= newTime

      if generatorValLoss < bestValidationLoss:
        print("Saving model. validation meanLoss improved from %2.6f to %2.6f"%(bestValidationLoss, generatorValLoss ) )
        generator.save(self.saveModelFname)
        bestValidationLoss= generatorValLoss
        roundsNoImprovement= 0
        roundsNoImprovementSinceLRDecay=0
      else:
        print("Validation meanLoss did not improve from %s"%(bestValidationLoss ) )
        roundsNoImprovement+=1
        roundsNoImprovementSinceLRDecay+=1
      self.save_imgs([generated_imgs, particles_val, projections_val], saveImagesPath, epoch)
      sys.stdout.flush()
      
      if roundsNoImprovement>= roundsToEarlyStopping:
        print("Early stopping")
        break
      elif roundsNoImprovementSinceLRDecay== roundsToLR_decay:
        new_lr= max(1e-9, 0.05* learningRate)
        print("Decreasing learning rate to %.2E"%(learningRate) )
        K.set_value(optimizer_discriminator_model.lr, new_lr)
        K.set_value(optimizer_generatorGAN_model.lr, new_lr)
        learningRate= new_lr
        roundsNoImprovementSinceLRDecay=0

      if epoch>= nEpochs:
        break
      print("------------------------------------------------------------------------")

def build_discriminator( img_shape, nConvLayers= 4):
  assert math.log(img_shape[1], 2)> nConvLayers, "Error, too small images: input %s. Min size %s"%(img_shape, 2**nConvLayers)
    
  model = Sequential()
  model.add( Conv2D(2**3, 5, activation="linear", padding="same", input_shape= img_shape) )
  model.add( BatchNormalization() )
  model.add( LeakyReLU(0.2) )
  for i in range(nConvLayers-1):
    model.add( Conv2D(2**(4+i),3, activation="linear", padding="same") )
    model.add( BatchNormalization() )
    model.add( LeakyReLU(0.2) )
    model.add( MaxPooling2D(pool_size=2) )
  model.add(Flatten())
  model.add(Dense( min(128, np.prod(img_shape)) ) )
  model.add(LeakyReLU(alpha=0.2))
  model.add( Dropout(0.5) )
  model.add(Dense(1, activation='sigmoid'))
  return model