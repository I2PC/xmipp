import sys, os
import keras
import math
from keras.models import load_model
import tensorflow as tf

from .DeepLearningGeneric import DeepLearningModel, save_imgs
from .dataGenerator import getDataGenerator, extractNBatches, BATCH_SIZE

NUM_BATCHES_PER_EPOCH= 256

class UNET(DeepLearningModel):
  
  def __init__(self, boxSize, saveModelFname, gpuList="0", batchSize=BATCH_SIZE, modelDepth=4, generatorLoss="MSE",
                     trainingDataMode="ParticlesAndSyntheticNoise", regularizationStrength=1e-5):
  
    DeepLearningModel.__init__(self,boxSize, saveModelFname, gpuList, batchSize, generatorLoss= generatorLoss,
                                trainingDataMode=trainingDataMode,  regularizationStrength= regularizationStrength)
    self.epochSize= NUM_BATCHES_PER_EPOCH
    
  def _setShape(self, boxSize):
    self.img_shape= (boxSize,boxSize,1)
    return self.img_shape
    
  def train(self, learningRate, nEpochs, xmdParticles, xmdProjections, xmdEmptyParts=None):

    saveImagesPath= self.createSaveImgsPath(xmdParticles)
    N_GPUs= len(self.gpuList.split(',')) 
    if os.path.isfile(self.saveModelFname):
      print("loading previous model")
      if N_GPUs>1:
        with tf.device('/cpu:0'):
          model_1gpu = load_model(self.saveModelFname, custom_objects= CUSTOM_OBJECTS)
        model= keras.utils.multi_gpu_model(model_1gpu, gpus= N_GPUs)
      else:
        model_1gpu = load_model(self.saveModelFname, custom_objects= {self.generatorLoss.__name__: self.generatorLoss})
        model= model_1gpu
    else:
      paramsNewUnet={"img_shape":self.img_shape, "out_ch":1, "start_ch":32, "depth": self.modelDepth,
                     "inc_rate":2., "activation":"relu", "dropout":0.5, "batchnorm":True, "l1l2_reg":self.regularizationStrength, 
                     "maxpool":True, "upconv":True, "residual":True, "lastActivation":"tanh"}      
      if N_GPUs>1:
        with tf.device('/cpu:0'):
          model_1gpu= build_UNet( **paramsNewUnet)
        model= keras.utils.multi_gpu_model(model_1gpu, gpus= N_GPUs)
      else:
        model_1gpu= build_UNet( **paramsNewUnet)
        model= model_1gpu
    
    optimizer= keras.optimizers.Adam(lr= learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0)
    model.compile(loss=self.generatorLoss, metrics=["mse"], optimizer=optimizer)

    print("train/val split"); sys.stdout.flush()
    valIterator, stepsPerEpoch_val= getDataGenerator(xmdParticles, xmdProjections, augmentData=False, 
                                                     isTrain=False, valFraction=0.1, batchSize=self.batchSize)
    valData= extractNBatches(valIterator, min(10, stepsPerEpoch_val)); del valIterator
    
    trainIterator, stepsPerEpoch= getDataGenerator(xmdParticles, xmdProjections, xmdEmptyParts=xmdEmptyParts,
                                                   isTrain=True, augmentData=True, valFraction=0.1, batchSize=self.batchSize,
                                                   simulateEmptyParts=self.addSyntheticEmpty)
    cBacks=[]
    cBacks+= [ keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_lr=1e-8) ]
    cBacks+= [ keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto') ]
    cBacks+= [ AltModelCheckpoint(self.saveModelFname, model_1gpu, monitor='val_loss', verbose=1, save_best_only=True) ]
    cBacks+= [ WriteImageCBack(valData[0][:self.batchSize], valData[1][:self.batchSize], 
                                saveImagesPath, batch_size=self.batchSize) ]
    
    nEpochs_init= nEpochs
    nEpochs= max(1, nEpochs_init*float(stepsPerEpoch)/self.epochSize)
    print("nEpochs : %.1f --> Epochs: %d.\nTraining begins: Epoch 0/%d"%(nEpochs_init, nEpochs, nEpochs))
    sys.stdout.flush()
    
    model.fit_generator(trainIterator, epochs= nEpochs, steps_per_epoch=self.epochSize, #steps_per_epoch=stepsPerEpoch,
                      verbose=2, callbacks=cBacks, validation_data=valData, max_queue_size=12, 
                      workers=1, use_multiprocessing=False)
                        


def build_UNet( img_shape, out_ch=1, start_ch=32, depth=3, inc_rate=2., activation='relu', dropout=0.5, batchnorm=False,
                maxpool=True, upconv=True, residual=False, lastActivation="linear", l1l2_reg=1e-5):

    if l1l2_reg is not None:
      regularizerFun= lambda: keras.regularizers.l1_l2(l1=l1l2_reg, l2=l1l2_reg)
    else:
      regularizerFun= lambda: None
    
    def conv_block( m, dim, acti, bn, res, do=0):
        n = keras.layers.Conv2D(dim, 3, activation=acti, padding='same',kernel_regularizer=regularizerFun())(m)
        n = keras.layers.BatchNormalization()(n) if bn else n
        n = keras.layers.Dropout(do)(n) if do else n
#        n = keras.layers.SpatialDropout2D(do)(n) if do else n
        n = keras.layers.Conv2D(dim, 3, activation=acti, padding='same',kernel_regularizer=regularizerFun())(n)
        n = keras.layers.BatchNormalization()(n) if bn else n
        return keras.layers.Concatenate()([m, n]) if res else n

    def level_block( m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = conv_block(m, dim, acti, bn, res)
            m = keras.layers.MaxPooling2D()(n) if mp else keras.layers.Conv2D(dim, 3, strides=2,
                                                              padding='same',kernel_regularizer=regularizerFun())(n)
            m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
            if up:
                m = keras.layers.UpSampling2D()(m)
                m = keras.layers.Conv2D(dim, 2, activation=acti, padding='same', kernel_regularizer=regularizerFun())(m)
            else:
                m = keras.layers.Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            n = keras.layers.Concatenate()([n, m])
            m = conv_block(n, dim, acti, bn, res)
        else:
            m = conv_block(m, dim, acti, bn, res, do)
        return m
    
    i = keras.layers.Input(shape=img_shape)

    all_pad_size= ( 2**(int(math.ceil(math.log(img_shape[1], 2) )))-img_shape[1])
    pad_size_left=  all_pad_size//2
    pad_size_right=  all_pad_size//2 + all_pad_size%2

    x = keras.layers.ZeroPadding2D( [(pad_size_left, pad_size_right), (pad_size_left, pad_size_right) ])( i ) #Padded to 2**N

    o = level_block(x, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = keras.layers.Conv2D(out_ch, 1, activation=lastActivation)(o)
    if pad_size_right>0:
      o = keras.layers.Lambda(lambda m: m[:,pad_size_left:-pad_size_right,pad_size_left:-pad_size_right,:] )( o )
    return  keras.models.Model(inputs=i, outputs=o)

#####################################################################################################################
# AltModelCheckpoint taken from https://github.com/TextpertAi/alt-model-checkpoint/blob/master/alt_model_checkpoint/__init__.py#L9     #
#####################################################################################################################
from keras.callbacks import ModelCheckpoint


class AltModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, alternate_model, **kwargs):
        """
        Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.
        :param filepath:
        :param alternate_model: Keras model to save instead of the default. This is used especially when training multi-
                                gpu models built with Keras multi_gpu_model(). In that case, you would pass the original
                                "template model" to be saved each checkpoint.
        :param kwargs:          Passed to ModelCheckpoint.
        """

        self.alternate_model = alternate_model
        super(type(self),self).__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.alternate_model
        super(type(self),self).on_epoch_end(epoch, logs)
        self.model = model_before
        

class WriteImageCBack(keras.callbacks.Callback):
  """"
  callback to observe the output of the network
  """
  def __init__(self, x, y, saveImagesPath, batch_size= 16):
    self.x = x
    self.y = y
    self.batch_size= batch_size
    self.saveImagesPath= saveImagesPath
    self.currentEpoch=0
  def on_epoch_end(self, epoch, logs={}):
      self.currentEpoch+=1
      pred= self.model.predict(self.x, batch_size= self.batch_size, verbose=0)
      save_imgs([pred, self.x, self.y], self.saveImagesPath, self.currentEpoch, basenameTemplate="denoise_%d.png")

        
