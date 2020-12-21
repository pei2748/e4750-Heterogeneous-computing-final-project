# Acceleration of SVM algorithm on GPU using PyCuda 
.\
├── INSTRUCTIONS.md\
├── README.md\
├── figures\
│   ├── cloud-using-1.png\
│   ├── cloud-using-2.png\
│   └── cloud-using-3.png\
├── kernels\
│   ├── delta.cu\
│   ├── ds.cu\
│   ├── get_w_combo.cu\
│   ├── get_w_combo_tiled.cu\
│   ├── sgd_bc_sh_mem.cu\
│   ├── sgd_cifar_lock_free_2.cu\
│   ├── sgd_cifar_single_blk_normal_mult.cu\
│   ├── sgd_kernel.cu\
│   ├── sgd_kernel_no_locks.cu\
│   ├── xT.cu\
│   ├── x_dot_w.cu\
│   └── x_dot_w_tile.cu\
├── nvprof\
│   ├── BC-cuda-lock-free.nvprof\
│   ├── BC-cuda-naive.nvprof\
│   ├── BC-cuda-shared-mem.nvprof\
│   ├── CF-cuda-lock-free.nvprof\
│   ├── CF-cuda-multi-kernel-tiled.nvprof\
│   ├── CF-cuda-multi-kernel.nvprof\
│   ├── CF-cuda-single-blk.nvprof\
│   ├── cifar_time.nvprof\
│   └── nvprof-result-figures/\
├── pycuda\
│   ├── BC-cuda-lock-free.py\
│   ├── BC-cuda-naive.py\
│   ├── BC-cuda-shared-mem.py\
│   ├── CF-cuda-lock-free.py\
│   ├── CF-cuda-multi-kernel-tiled.ipynb\
│   ├── CF-cuda-multi-kernel-tiled.py\
│   ├── CF-cuda-multi-kernel.ipynb\
│   ├── CF-cuda-multi-kernel.py\
│   ├── CF-cuda-naive.ipynb\
│   ├── CF-cuda-naive.py\
│   ├── CF-data-comparison.ipynb\
│   └── data.csv\
├── python_code\
│   ├── data.csv\
│   ├── svm_BC-SGD.ipynb\
│   ├── svm_BC-SGD.py\
│   ├── svm_BC-opencv.ipynb\
│   ├── svm_BC-opencv.py\
│   ├── train-cifar10-svm.ipynb\
│   └── train-cifar10-svm.py\
└── result-plots/\






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





