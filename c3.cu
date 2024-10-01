// Command to build this file: nvcc -o c3 c3.cu -lcudnn

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cudnn.h>

#define C 3
#define H 1024
#define W 1024
#define FH 3
#define FW 3
#define K 64
#define P 1
#define MILLION 1000000L

int main() {
    // Generate input tensor I and convolution filters F
    int nI = C * H * W;
    int nF = K * C * FH * FW;
    int nI0 = C * (W + 2 * P) * (H + 2 * P);
    int nO = K * W * H;

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    double *I = (double*)malloc(nI * sizeof(double));
    double *F = (double*)malloc(nF * sizeof(double));
    // Initialize I and F
    for(int c=0;c<C;c++){
        for(int x=0;x<H;x++){
            for(int y=0;y<W;y++){
                I[c * H * W + x * W + y] = c * (x + y);
            }
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    // Pad input tensor I to obtain I0
    double *I0 = (double*)malloc(nI0 * sizeof(double));
    // Pad I to obtain I0
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H + 2 * P; ++x) {
            for (int y = 0; y < W + 2 * P; ++y) {
                if (x < P || x >= H + P || y < P || y >= W + P) {
                    I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = 0.0; // Padding with zeros
                } else {
                    I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = I[c * H * W + (x - P) * W + (y - P)];
                }
            }
        }
    }

    double *O = (double*)malloc(nO * sizeof(double));

    double *d_I0;
    double *d_F;
    double *d_O;

    cudaMalloc(&d_I0, nI0 * sizeof(double));
    cudaMalloc(&d_F, nF * sizeof(double));
    cudaMalloc(&d_O, nO * sizeof(double));

    cudaMemcpy(d_I0, I0, nI0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, nF * sizeof(double), cudaMemcpyHostToDevice);

    // Create tensor descriptors
    cudnnTensorDescriptor_t xDesc;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H + 2 * P, W + 2 * P);

    cudnnFilterDescriptor_t wDesc;
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW);

    cudnnTensorDescriptor_t oDesc;
    cudnnCreateTensorDescriptor(&oDesc);
    cudnnSetTensor4dDescriptor(oDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W);

    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);

    // Get the number of algorithms
    int numAlgos;
    cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &numAlgos);

    // Allocate memory for the array of algorithm performance structures
    cudnnConvolutionFwdAlgoPerf_t *algoPerf = (cudnnConvolutionFwdAlgoPerf_t *)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t) * numAlgos);

    // Get the algorithms and their performance
    cudnnFindConvolutionForwardAlgorithm(cudnn, xDesc, wDesc, convDesc, oDesc, numAlgos, &numAlgos, algoPerf);

    // Choose the fastest algorithm
    cudnnConvolutionFwdAlgo_t algo = algoPerf[0].algo;

    size_t workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, xDesc, wDesc, convDesc, oDesc, algo, &workspaceSize);

    void *workspace;
    cudaMalloc(&workspace, workspaceSize);

    double alpha = 1.0, beta = 0.0;
    // warm-up
    cudnnConvolutionForward(cudnn, &alpha, xDesc, d_I0, wDesc, d_F, convDesc, algo, workspace, workspaceSize, &beta, oDesc, d_O);
    cudaDeviceSynchronize();
    
    struct timespec start, end;
    double total_time = 0.0;

    // Execute convolution
    clock_gettime(CLOCK_MONOTONIC, &start);
    cudnnConvolutionForward(cudnn, &alpha, xDesc, d_I0, wDesc, d_F, convDesc, algo, workspace, workspaceSize, &beta, oDesc, d_O);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = (end.tv_sec - start.tv_sec)*1000 + (end.tv_nsec - start.tv_nsec) / (double)MILLION;
    

    cudaMemcpy(O, d_O, nO * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute checksum
    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                checksum += O[k * W * H + x * H + y];
            }
        }
    }
    printf("C3_checksum: %.3f, C3_execution_time: %.3f ms\n", checksum, total_time);

    // Free memory
    free(I);
    free(F);
    free(I0);
    free(O);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroyTensorDescriptor(oDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaFree(workspace);
    cudnnDestroy(cudnn);

    return 0;
}
