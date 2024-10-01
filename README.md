# CUDA-Accelerated Image Convolution with Shared Memory and cuDNN
```
Author: Rishabh Srivastava (rs4489)
```
This repository contains a CUDA-based implementation of image convolution using a set of filters. The project demonstrates the use of parallel processing to accelerate the convolution operation, commonly used in image processing tasks. By leveraging the power of GPUs, the code efficiently applies multiple filters to an image, showcasing significant speed improvements over traditional CPU-based implementations.

## Experiments
The task involves implementing convolution in CUDA, with three specific versions: a basic convolution without shared memory (`c1.cu`), a tiled convolution using shared memory (`c2.cu`), and a cuDNN-based convolution (`c3.cu`). 

### Basic Convolution without Shared Memory
`c1.cu` implements convolution using global memory access in CUDA. The input tensor `I`, padded version `I0`, and filters `F` are allocated on the device memory. The convolution is computed by iterating over the filter elements and performing the dot product with the corresponding elements in `I0`.

### Tiled Convolution with Shared Memory
`c2.cu` optimizes the convolution by loading chunks of the input tensor into shared memory to reduce global memory access latency. Shared memory is faster but limited in size, so we use **tiling** to divide the input into smaller regions.

### Convolution with cuDNN
`c3.cu` uses the **cuDNN** library, a GPU-accelerated library for deep learning. The convolution is performed using cuDNN’s CUDNN_CONVOLUTION_FWD_PREFER_FASTEST algorithm, which provides optimized performance.

## Steps To Run
- Install CUDA and cuDNN
- Compile the CUDA code using the command below (add the `-lcudnn` flag when running `c3.cu` to use cuDNN).
```
nvcc -o c1 c1.cu
```
- Run the executable
```
./c1
```

Each program will print out two values:
- The checksum of the output tensor `O`.
- The execution time of the CUDA kernel (in milliseconds).

## Results
Each implementation uses different CUDA techniques, with the tiled version offering more optimization than the basic version by leveraging shared memory, while the cuDNN version takes advantage of NVIDIA's optimized library.
All three implementations should output the same checksum, verifying that the convolution results are correct.
- During the experiments, the tiled convolution completed execution in the fastest time - **33.246 ms**.
- Using cuDNN library resulted in an execution time of **38.338 ms**.
- As expected, the basic convolution implementation was the slowest, with an execution time of **47.087 ms**.

