#include <stdio.h>
#include <stdlib.h>

// #include "utils.h"
#include "../../timing/dphpc_timing.h"

__global__ void scalar_update_kernel(double *x, int i, double *b, double *dp, double *L, int N) {
    x[i] = (b[i] - dp[0]) / L[i * N + i];
}

__global__ void dot_product_kernel(double *L, double *x, double *dp, int i, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < i) {
        dp[j] = L[i * N + j] * x[j]; 
        __syncthreads();

        for (int stride = 1; j + stride < i; stride <<= 1) {
            if (j % (2 * stride) == 0){
                dp[j] += dp[j + stride]; 
            }
            __syncthreads();
        }
    }
}

void kernel(double *L, double *x, double *b, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    double *dp;
    cudaMalloc((void **)&dp, N * sizeof(double));

    for(int i = 0; i < N; i++) {
        dot_product_kernel<<<numBlocks, blockSize>>>(L, x, dp, i, N);
        scalar_update_kernel<<<1, 1>>>(x, i, b, dp, L, N);
    }

    cudaDeviceSynchronize();
}

void reset(double *L, double *x, double *b, double *d_L, double *d_x, double *d_b, int N) {
    cudaMemcpy(d_L, L, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void initialize(int N, double *L, double *x, double *b) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            L[i * N + j] = (i + N - j + 1) * 2.0 / N;
        }
        x[i] = 0;
        b[i] = (double)i;
    }
}

void run_bm(int N, const char *preset) {
    double *L = (double*)malloc(sizeof(double) * N * N);
    double *x = (double*)malloc(sizeof(double) * N);
    double *b = (double*)malloc(sizeof(double) * N);

    initialize(N, L, x, b);

    double *d_L, *d_x, *d_b;
    cudaMalloc((void **)&d_L, N * N * sizeof(double));
    cudaMalloc((void **)&d_x, N * sizeof(double));
    cudaMalloc((void **)&d_b, N * sizeof(double));

    reset(L, x, b, d_L, d_x, d_b, N);
    kernel(d_L, d_x, d_b, N);
    cudaDeviceSynchronize();


    dphpc_time3(
        reset(L, x, b, d_L, d_x, d_b, N),
        kernel(d_L, d_x, d_b, N),
        preset
    );


    cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    printf("Done \n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", x[i]);
    }
    
    printf("%f ", x[N-1]);

    cudaFree(d_L);
    cudaFree(d_x);
    cudaFree(d_b);
    free(L);
    free(x);
    free(b);
}

int main() {
    run_bm(2000, "S");
    run_bm(5000, "M");
    run_bm(14000, "L");
    run_bm(16000, "paper");

    return 0;
}


