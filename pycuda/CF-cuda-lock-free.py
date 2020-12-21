from tensorflow import keras
from keras.datasets import cifar10
import os
import time
import numpy as np
# Library for plot the output and save to file                                  
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import math
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.autoinit import context



baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)


# Normalize the data by subtract the mean image                                 
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage



xVal -= meanImage
xTest -= meanImage

# Reshape data from channel to rows                                             
xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))
# Add bias dimension columns                                                    
xTrain = np.hstack([xTrain, np.ones((xTrain.shape[0], 1))])
xVal = np.hstack([xVal, np.ones((xVal.shape[0], 1))])
xTest = np.hstack([xTest, np.ones((xTest.shape[0], 1))])


class Svm (object):
    """" Svm classifier """
    # (xTrain.shape[1], numClasses)
    def __init__ (self, inputDim, outputDim):
        self.W = None
        sigma =0.01
        self.W = sigma * np.random.randn(inputDim,outputDim)


    def sgd (self, xTrain, yTrain, learning_rate,
             reg_strength, max_epochs=200, batchSize=1, verbose=False):

        # train model with cuda    

        length_of_features = xTrain.shape[1]
        num_examples = xTrain.shape[0]
        dim0 = 1024
        block_dim = (dim0, 1, 1)
        grid_dim = ( math.ceil(num_examples/dim0),1, 1)
        print(grid_dim)
        print(self.W[0])
        prg_sgd = SourceModule(open("../kernels/sgd_cifar_lock_free_2.cu", "r").read())
        func = prg_sgd.get_function("sgd")

        time_gpu = 0
        
        xTrain, yTrain = shuffle(xTrain, yTrain)
        loss = np.zeros(num_examples).astype(np.float32)
        _start = cuda.Event()
        _end = cuda.Event()
        # Transfer variables onto device

        d_x = gpuarray.to_gpu(xTrain.astype(np.float32))
        d_y = gpuarray.to_gpu(yTrain.astype(np.float32))
        d_weights = gpuarray.to_gpu(self.W.astype(np.float32))
        d_loss = gpuarray.to_gpu(loss.astype(np.float32))
#        d_weights = cuda.managed_empty(shape=num_examples*10,
#                                  dtype=np.float32,
#                                  mem_flags=cuda.mem_attach_flags.GLOBAL)
#        d_weights = self.W.flatten()
        _start.record()
        func(d_x, d_y, d_weights,
             d_loss,
             np.float32(reg_strength),
             np.float32(learning_rate),
             np.int32(num_examples),
             np.int32(max_epochs),
             block=block_dim,
             grid=grid_dim)
            
        _end.record()
        _end.synchronize()
        context.synchronize()
        
        self.W = d_weights.get()#.reshape(num_examples, 10)
        loss = d_loss.get()
        print(loss[0])
        print("Epoch = ", max_epochs, ", Loss = ", np.sum(loss), "\n")
        time_gpu =  _start.time_till(_end) # in ms
        print(" THE GPU TIME IS :  " , time_gpu *(1e-03))    
        print(self.W[0])
        
    def predict (self, x,):
        yPred = np.zeros(x.shape[0])
        s = x.dot(self.W)
        yPred = np.argmax(s, axis=1)
        return yPred
    
    def calAccuracy (self, x, y):
        acc = 0
        yPred = self.predict(x)
        acc = np.mean(y == yPred)*100
        return acc


# Training classifier

numClasses = np.max(yTrain) + 1
classifier = Svm(xTrain.shape[1], numClasses)    
startTime = time.time()

# for lock-free, batchSize is not used.
classifier.sgd(xTrain, yTrain,
               learning_rate=1e-7,
               reg_strength=5e4,
               max_epochs=500,
               batchSize=200,
               verbose=True)

print ('Training time: {0}'.format(time.time() - startTime))
print ('Training acc:   {0}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print ('Validating acc: {0}%'.format(classifier.calAccuracy(xVal, yVal)))
print ('Testing acc:    {0}%'.format(classifier.calAccuracy(xTest, yTest)))
