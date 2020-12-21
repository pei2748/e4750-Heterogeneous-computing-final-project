# Acceleration of SVM algorithm on GPU using PyCuda 

### Directory Explanation
Our repository has the following structure. It has 6 subdirectories at the top level. 


<img src="tree.png" width="400">


#### Directory 1: kernels/ 

This diectory contains the cuda kernels to be executed on GPU for Breast-Cancer and CiFAR-10 dataset.\
1.) sgd_kerne.cu \
    This kernel is used by BC-cuda-naive.py program. It performs the sgd on BC dataset where each thread works on one feature of the datapoint.\
    
2.) sgd_bc_sh_mem.cu \
This kernel is used by BC-cuda-shared-mem.py program. It performs the sgd on BC dataset where each thread works on one feature of the datapoint. This kernel also used the shared memory and tiled approach where a batch of data points are first loaded into shared memory and then computed upon.\

3.) sgd_kernel_no_locks.cu\
  This kernel is used by BC-cuda-no-lock.py program. It performs the sgd on BC dataset where each thread works on one datapoint. So in a lock free manner all the datapoints will be updating the weights in every epoch.\

4.) sgd_cifar_single_blk_normal_mult.cu\
This kernel is used by CF-cuda-naive.py program. It is to implement SVM on cifar dataset, which includes tiled multiplication and use of shared memory to perform computation, each thread works on each class.\

5.) sgd_cifar_lock_free_2.cu\
This is the kernel code used to implement SVM on cifar dataset, each thread works on one datapoint and they update the weights in alock free manner for every epoch. \

6.) x_dot_w.cu, xT.cu, ds.cu, delta.cu, get_w_combo.cu \
  These 4 kernels work for the CF-cuda-multi-kernel.py pycuda Program.\
  
7) x_dot_w_tile.cu, xT.cu, ds.cu, delta.cu, get_w_combo_tiled.cu \
  These 5 kernels work for the CF-cuda-multi-kernel-tiled.py pycuda Program.\

# Directory 2: python_code

This directory contains 3 files:

> svm_BC-SGD.py : 

  native python implementation of sgd on BC dataset
  
> svm_BC-opencv.py : 

  openCV library implementation of sgd on BC dataset

>train-cifar10-svm.py : 

 native python implementation of sgd on cifar 10 dataset

# Directory 3: PyCuda Source Codes 

These are PyCuda source code which implement different kernels


>BC-cuda-lock-free.py

>BC-cuda-naive.py

>BC-cuda-shared-mem.py

>CF-cuda-lock-free.py

>CF-cuda-single-blk.py




# Commands

Run commands in svm/ folder.

Commands to run nvprovf:

1, nvprof -o nvprof/BC-cuda-naive.nvprof python BC-cuda-naive.py

2, nvprof -o nvprof/BC-cuda-lock-free.nvprof python BC-cuda-lock-free.py 

3, nvprof -o nvprof/BC-cuda-shared-mem.nvprof python BC-cuda-shared-mem.py 

4, nvprof -o nvprof/CF-cuda-single-blk.nvprof python CF-cuda-single-blk.py 

5, nvprof -o nvprof/CF-cuda-lock-free.nvprof python CF-cuda-lock-free.py

Commands to view nvprof:
viewprofile nvprof/BC-cuda-naive.nvprof





