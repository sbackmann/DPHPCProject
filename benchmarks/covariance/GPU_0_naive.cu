#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"

#define DEV_MODE 0
#define TIME 1

// CUDA kernel to calculate covariance matrix
__global__ void dot_prod_store_kernel(int M, int N, double* data, double* cov) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < M) {
        double partial_sum = 0.0;

        for (int k = 0; k < N; k++) {
            partial_sum += data[k * M + i] * data[k * M + j];
        }

        cov[i * M + j] = partial_sum / ((double)N - 1.0);
    }
}

__global__ void mean_kernel(double *data, double *mean, int N, int M) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int j = col_idx; j < M; j += stride) {
        double local_sum = 0.0;

        for (int row_idx = 0; row_idx < N; ++row_idx) {
            local_sum += data[row_idx * M + j];
        }

        mean[j] = local_sum / N;
    }
}

__global__ void subtract_kernel(double *data, double *mean, int N, int M) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int j = col_idx; j < M; j += stride) {
        for (int i = 0; i < N; i++) {
            data[i * M + j] -= mean[j];
        }
    }
}

void reset(int M, int N, double* data, double* d_data) {
    cudaMemcpy(d_data, data, N * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void kernel(int M, int N, double* d_data, double* d_cov) {
    double *d_mean;
    cudaMalloc((void**)&d_mean, N * M * sizeof(double));

    int threads_per_block = 256;
    int blocks = (M + threads_per_block - 1) / threads_per_block;

    mean_kernel<<<blocks, threads_per_block>>>(d_data, d_mean, N, M);
    cudaDeviceSynchronize();

    subtract_kernel<<<blocks, threads_per_block>>>(d_data, d_mean, N, M);
    cudaDeviceSynchronize();

    int t = 16;
    dim3 blockDim(t, t);
    dim3 gridDim((M + t - 1) / t, (M + t - 1) / t);

    dot_prod_store_kernel<<<gridDim, blockDim>>>(M, N, d_data, d_cov);
    cudaDeviceSynchronize();

    // cudaMemcpy(cov, d_cov, M * M * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaFree(d_data);
    // cudaFree(d_cov);
    // free(mean);
}

void run_bm(int M, int N, const char* preset) {
    double *data = (double*)malloc(N * M * sizeof(double));
    double *cov = (double*)malloc(M * M * sizeof(double));

    double *d_data, *d_cov;
    cudaMalloc((void**)&d_cov, M * M * sizeof(double));
    cudaMalloc((void**)&d_data, N * M * sizeof(double));


    initialize(M, N, data);
    reset(M, N, data, d_data);
    kernel(M, N, d_data, d_cov);

    cudaMemcpy(cov, d_cov, M * M * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    if (!is_correct(M, cov, preset)) {
        printf("Validation failed for preset: %s \n", preset);
        exit(1);
    } else {
        printf("Validation passed for preset: %s \n", preset);
    }

    dphpc_time3(
        reset(M, N, data, d_data),
        kernel(M, N, d_data, d_cov),
        preset
    );

    // Do a post validation 
    cudaMemcpy(cov, d_cov, M * M * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    if (!is_correct(M, cov, preset)) {
        printf("Validation failed for preset: %s \n", preset);
        exit(1);
    } else {
        printf("Validation passed for preset: %s \n", preset);
    }

    cudaFree(d_data);
    cudaFree(d_cov);
    free(data);
    free(cov);
}

int main() {
    run_bm(500, 600, "S");
    run_bm(1400, 1800, "M");
    run_bm(3200, 4000, "L");
    run_bm(1200, 1400, "paper");

    return 0;
}
