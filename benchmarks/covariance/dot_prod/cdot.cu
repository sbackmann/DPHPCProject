#include <stdio.h>
#include <stdlib.h>

#include "../../../timing/dphpc_timing.h"

#define DEV_MODE 0
#define TIME 1

// CUDA kernel to calculate covariance matrix
__global__ void covariance_kernel(int M, double* data, double* cov) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < M) {
        double partial_sum = 0.0;

        for (int k = 0; k < M; k++) {
            partial_sum += data[k * M + i] * data[k * M + j];
        }

        cov[i * M + j] = partial_sum;
    }
}


void reset(int M, double* data, double* d_data) {
    cudaMemcpy(d_data, data, M * M * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void initialize(int M, double* data) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            data[i * M + j] = (double)(i * j) / M;
        }
    }
}

void printMatrix(int m, double *matrix) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.6lf ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

void kernel(int M, double* d_data, double* d_cov) {
    int t = 16;
    dim3 blockDim(t, t);
    dim3 gridDim((M + t - 1) / t, (M + t - 1) / t);

    covariance_kernel<<<gridDim, blockDim>>>(M, d_data, d_cov);
    cudaDeviceSynchronize();
    // cudaMemcpy(cov, d_cov, M * M * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaFree(d_data);
    // cudaFree(d_cov);
    // free(mean);
}

void run_bm(int M,  const char* preset) {
    double *data = (double*)malloc(M * M * sizeof(double));
    double *cov = (double*)malloc(M * M * sizeof(double));
    initialize(M, data);

    double *d_data, *d_cov;
    cudaMalloc((void**)&d_cov, M * M * sizeof(double));
    cudaMalloc((void**)&d_data, M * M * sizeof(double));


    #if DEV_MODE
    printf("Data matrix: \n");
    printMatrix(N, M, data);
    #endif

    #if TIME
        dphpc_time3(
            reset(M, data, d_data),
            kernel(M, d_data, d_cov),
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
        // run_bm(500, 600, "M");
        // run_bm(3, 4, "S");
        // run_bm(5, 5, "S");
        // run_bm(5, 7, "M");
    #else
        run_bm(1400, "missing");
        // run_bm(500, "S");
        // run_bm(1400, "M");
        // run_bm(3200, "L");
        // run_bm(1200, "paper");
    #endif

    return 0;
}
