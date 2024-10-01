#include <iostream>
#include <stdlib.h>
#include <time.h>

#define C 3
#define H 1024
#define W 1024
#define FH 3
#define FW 3
#define K 64
#define P 1
#define MILLION 1000000L

__global__ void convolution(double *I0, double *F, double *O) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (x < W && y < H) {
        double sum = 0.0;
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    sum += F[k * C * FH * FW + c * FH * FW + (FW - 1 - i) * FW + (FH - 1 - j)] * I0[c * (W + 2 * P) * (H + 2 * P) + (x + i) * (H + 2 * P) + y + j];
                }
            }
        }
        O[k * W * H + x * H + y] = sum;
    }
}

int main() {
    // Generate input tensor I and convolution filters F
    int nI = C * H * W;
    int nF = K * C * FH * FW;
    int nI0 = C * (W + 2 * P) * (H + 2 * P);
    int nO = K * W * H;

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

    // Allocate memory on GPU
    double *d_I0, *d_F, *d_O;
    cudaMalloc(&d_I0, sizeof(double) * nI0);
    cudaMalloc(&d_F, sizeof(double) * nF);
    cudaMalloc(&d_O, sizeof(double) * nO);

    // Copy data from host to GPU
    cudaMemcpy(d_I0, I0, sizeof(double) * nI0, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, sizeof(double) * nF, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, K);

    // Launch the kernel, warm-up
    convolution<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    // Synchronize threads
    cudaDeviceSynchronize();

    struct timespec start, end;
    double total_time = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
    convolution<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = (end.tv_sec - start.tv_sec)*1000 + (end.tv_nsec - start.tv_nsec) / (double)MILLION;

    // Copy the result back to host
    double *O = (double*)malloc(nO * sizeof(double));
    cudaMemcpy(O, d_O, sizeof(double) * nO, cudaMemcpyDeviceToHost);

    // Compute checksum
    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                checksum += O[k * W * H + x * H + y];
            }
        }
    }
    printf("C1_checksum: %.3f, C1_execution_time: %.3f ms\n", checksum, total_time);

    // Free memory
    free(I);
    free(F);
    free(I0);
    free(O);
    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);

    return 0;
}
