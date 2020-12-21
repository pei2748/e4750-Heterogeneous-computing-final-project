This Project Repo is for GPU Project.
The Project is about Acceleration of SVM algorithm on GPU using PyCuda 


Repo Structure and File Organization:

This Repository has 2 directories, 5 python files 

# Directory 1: kernels 

This diectory contains the cuda kernels to be executed on GPU for BC and CiFAR10 dataset.

### BC Dataset Kernels

1.) sgd_kerne.cu

This kernel file performs the sgd on BC dataset where each thread works on one feature of the datapoint.

2.) sgd_bc_sh_mem.cu

This kernel file performs the sgd on BC dataset where each thread works on one feature of the datapoint. This kernel also used the shared memory and tiled approach where a batch of data points are first loaded into shared memory and then computed upon.

3.) sgd_kernel_no_locks.cu

This kernel file performs the sgd on BC dataset where each thread works on one datapoint. So in a lock free manner all the datapoints will be updating the weights in every epoch.

### CiFAR10 Dataset Kernels

1.) sgd_cifar_single_blk_normal_mult.cu

This is the kernel code used to implement SVM on cifar dataset, it included tiled multiplication and use of shared memory to perform computation, each thread works on each class.

2.) sgd_cifar_lock_free_2.cu

This is the kernel code used to implement SVM on cifar dataset, each thread works on one datapoint and they update the weights in alock free manner for every epoch.

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





