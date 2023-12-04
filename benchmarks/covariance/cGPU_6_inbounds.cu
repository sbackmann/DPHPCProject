#include <stdio.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"

#define DEV_MODE 0
#define TIME 1

// CUDA kernel to calculate covariance matrix
__global__ void covariance_kernel(int M, int N, double* data, double* cov) {
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

__global__ void mean_adjust_kernel(double *data, int N, int M) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int j = col_idx; j < M; j += stride) {
        double local_sum = 0.0;

        for (int row_idx = 0; row_idx < N; ++row_idx) {
            local_sum += data[row_idx * M + j];
        }

        double mean_j = local_sum / N;

        for (int row_idx = 0; row_idx < N; ++row_idx) {
            data[row_idx * M + j] -= mean_j;
        }
    }
}


void reset(int M, int N, double* data, double* d_data) {
    cudaMemcpy(d_data, data, N * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void initialize(int M, int N, double* data) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            data[i * M + j] = (double)(i * j) / M;
        }
    }
}

void printMatrix(int n, int m, double *matrix) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.6lf ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

void kernel(int M, int N, double* d_data, double* d_cov) {
    int threads_per_block = 256;
    int blocks = (M + threads_per_block - 1) / threads_per_block;

    mean_adjust_kernel<<<blocks, threads_per_block>>>(d_data, N, M);
    cudaDeviceSynchronize();


    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    covariance_kernel<<<gridDim, blockDim>>>(M, N, d_data, d_cov);
    cudaDeviceSynchronize();

    // cudaMemcpy(cov, d_cov, M * M * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaFree(d_data);
    // cudaFree(d_cov);
    // free(mean);
}

void run_bm(int M, int N, const char* preset) {
    double *data = (double*)malloc(N * M * sizeof(double));
    double *cov = (double*)malloc(M * M * sizeof(double));
    initialize(M, N, data);

    double *d_data, *d_cov;
    cudaMalloc((void**)&d_cov, M * M * sizeof(double));
    cudaMalloc((void**)&d_data, N * M * sizeof(double));


    #if DEV_MODE
    printf("Data matrix: \n");
    printMatrix(N, M, data);
    #endif

    #if TIME
        dphpc_time3(
            reset(M, N, data, d_data),
            kernel(M, N, d_data, d_cov),
            preset
        );
    #else
        reset(M, N, data, d_data);
        kernel(M, N, d_data, d_cov);
    #endif

    cudaMemcpy(cov, d_cov, M * M * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    #if DEV_MODE
    printf("Covariance matrix: \n");
    printMatrix(M, M, cov);
    printf("END\n\n");
    #endif

    cudaFree(d_data);
    cudaFree(d_cov);
    free(data);
    free(cov);
}

int main() {
    #if DEV_MODE
        run_bm(3, 4, "M");
        run_bm(5, 7, "M");
        // run_bm(3, 4, "S");
        // run_bm(5, 5, "S");
        // run_bm(5, 7, "M");
    #else
        run_bm(500, 600, "S");
        run_bm(1400, 1800, "M");
        run_bm(3200, 4000, "L");
        run_bm(1200, 1400, "paper");
    #endif

    return 0;
}
