#!/usr/bin/env python
# coding: utf-8

# In[8]:

from tensorflow import keras
from keras.datasets import cifar10
import os
import time
import numpy as np
# Library for plot the output and save to file                                  
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time

import math
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit


# In[26]:


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:



baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)


# In[24]:



# In[32]:



# In[33]:


# Pre processing data                                                           
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

    def __init__ (self, inputDim, outputDim):
        self.W = None
        sigma =0.01
        self.W = sigma * np.random.randn(inputDim,outputDim)
        self.s_dummy = np.zeros((200,10)).astype(np.float32)
        self.time1=0
        self.time2=0
        self.time3=0
        self.time4=0
        self.time5=0
        self.time6=0
        self.time7=0
        self.time8=0
        self.final_time=0

    def calLoss (self, x, y, reg): # x is a 2D array. num of rows is the num of examples


        ### 1st kerel #############################
        dW = np.zeros_like(self.W).astype(np.float32)

        # tiled-version of dot product of x and w
        # we have too kernels ready, x_dot_w.cu and x_dot_w_tile.cu
        
        prg_sgd1 = SourceModule(open("../kernels/x_dot_w.cu", "r").read())
        func1 = prg_sgd1.get_function("x_dot_w")
        dim0 = 10
        dim1 = 100
        batchSize = x.shape[0]
        featureSize = x.shape[1]
        

        block_dim = (dim0, dim1, 1)
        grid_dim = (1, math.ceil(x.shape[0]/dim1), 1)

        d_x = gpuarray.to_gpu(x.astype(np.float32))
        d_w = gpuarray.to_gpu(self.W.astype(np.float32))
        d_s = gpuarray.empty(self.s_dummy.shape ,dtype=np.float32)
        a_start = cuda.Event()
        a_end = cuda.Event()
        a_start.record()
        func1( d_x , d_w , d_s,
                np.int32(batchSize),
                np.int32(featureSize),
                np.int32(10),
                block=block_dim,
                grid=grid_dim)
        
        a_end.record()
        a_end.synchronize()
        # time1 is the execution of the 1st kernel.
        self.time1= a_start.time_till(a_end) *(1e-3)
        s = d_s.get()
        a_s = time.time()
        

        s_yi = s[np.arange(x.shape[0]),y]
        ae = time.time()
        # time2 is to calculate s_yi
        self.time2= ae- a_s



        #####  2nd kernel ############################
        ## not using max_epoch ###
        prg_sgd2 = SourceModule(open("../kernels/delta.cu", "r").read())
        func2 = prg_sgd2.get_function("delta")
        
        d_s =  gpuarray.to_gpu(s.astype(np.float32))
        d_y =  gpuarray.to_gpu(s_yi.astype(np.float32))
        d_delta = gpuarray.empty(s.shape ,dtype=np.float32)
        
        b_start = cuda.Event()
        b_end =cuda.Event()
        b_start.record()
        func2( d_s , d_y , d_delta,
                np.int32(batchSize),
                np.int32(10),
                block=block_dim,
               grid=grid_dim)
        b_end.record()
        b_end.synchronize()
        # time3 is the execution time of 2nd kernel
        self.time3= b_start.time_till(b_end)*(1e-3)

        delta = d_delta.get()
        ds = np.zeros_like(delta)
        
        ####   3rd kernel  #############################
        # not using max_epochs #
        prg_sgd3 = SourceModule(open("../kernels/ds.cu", "r").read())
        func3 = prg_sgd3.get_function("ds")

        d_ds =  gpuarray.to_gpu(ds.astype(np.float32))
        d_y =  gpuarray.to_gpu(y.astype(np.int32))
        d_delta = gpuarray.to_gpu(delta.astype(np.float32))

        c_start =cuda.Event()
        c_end = cuda.Event()
        c_start.record()
        func3( d_ds , d_y , d_delta,
                   np.int32(batchSize),
                   np.int32(10),
                   block=block_dim,
                   grid=grid_dim)
        c_end.record()
        c_end.synchronize()
        # time5 is the execution time of the 3rd kernel
        self.time5= c_start.time_till(c_end)*(1e-3)        
        ds = d_ds.get()

        c_s =time.time()
        ds[np.arange(x.shape[0]),y] = -np.sum(ds, axis=1)
        c_e = time.time()
        # time6 is to calculate ds[] 
        self.time6= c_e - c_s


        ### 4th kernel  ####################
        # not using max_epoch
        prg_sgd6 = SourceModule(open("../kernels/xT.cu", "r").read())

        dim00 = 32
        dim11= 32

        block_dim2 = (dim00, dim11, 1)
        grid_dim2 = (math.ceil(x.shape[0]/dim00), math.ceil(x.shape[1]/dim11) , 1)

        func6 = prg_sgd6.get_function("xT")
        d_xp  = gpuarray.to_gpu(x.astype(np.float32))
        d_XT  = gpuarray.empty((3073,200), dtype=np.float32 )
        d_start =cuda.Event()
        d_end= cuda.Event()
        d_start.record()
        func6( d_xp, d_XT, 
               np.int32(batchSize),
               np.int32(featureSize),
               block=block_dim2,  grid=grid_dim2)
        d_end.record()
        d_end.synchronize()
        # time4 is for the execution time for the 4th kernel
        self.time4= d_start.time_till(d_end)*(1e-3)


        #####   5th kernel #################################
        ## have 2 versions of kernel, get_w_combo.cu and get_w_combo_tiled.cu 
        prg_sgd5 = SourceModule(open("../kernels/get_w_combo.cu", "r").read())
        dim00 =10
        dim11= 100
        block_dim2 = (dim00, dim11, 1)
        grid_dim2 = (math.ceil(dW.shape[1]/dim00), math.ceil(dW.shape[0]/dim11) , 1)
        
        func5 = prg_sgd5.get_function("get_w_combo")
        
        d_xp  = gpuarray.to_gpu(d_XT.get().astype(np.float32))
        d_DS  = gpuarray.to_gpu(ds.astype(np.float32))
        d_W = gpuarray.to_gpu(self.W.astype(np.float32))
        
        d_start =cuda.Event()
        d_end= cuda.Event()
        d_start.record()
        
        func5( d_xp, d_DS, d_W,
               np.int32(featureSize),
               np.int32(batchSize),
               np.int32(10),
               block=block_dim2,  grid=grid_dim2)
        
        d_end.record()
        d_end.synchronize()
        ## time7 is for the 5th kernel.
        self.time7= d_start.time_till(d_end)*(1e-3)
        

        # total time for execution.
        self.final_time+= self.time1 + self.time2 + self.time3 + self.time4 +self.time5 +self.time6 +self.time7 
        self.W = d_W.get()
        return self.W 
  
    def train (self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        # Run stochastic gradient descent to optimize W. 
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None
            # draw num_train samples from x, num_train is a 1-D array contains integeres
            num_train = np.random.choice(x.shape[0], batchSize)
            xBatch = x[num_train] # xBatch is a 2D array, each row is a data point,
            yBatch = y[num_train]
            w_g  = self.calLoss(xBatch,yBatch,reg)

        print(" EXE time" , self.final_time)
        return lossHistory
    
    
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




numClasses = np.max(yTrain) + 1
classifier = Svm(xTrain.shape[1], numClasses)




# Training classifier
startTime = time.time()

classifier.train(xTrain, yTrain, lr=1e-7, reg=5e4, iter=500 ,verbose=True)


print ('Training time: {0}'.format(time.time() - startTime))
print ('Training acc:   {0}%'.format(classifier.calAccuracy(xTrain, yTrain)))
print ('Validating acc: {0}%'.format(classifier.calAccuracy(xVal, yVal)))
print ('Testing acc:    {0}%'.format(classifier.calAccuracy(xTest, yTest)))




