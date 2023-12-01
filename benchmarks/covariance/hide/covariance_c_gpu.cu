#include <stdio.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"

#define DEV_MODE 1
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
        // cov[j * M + i] = cov[i * M + j];
    }
}

void reset() {}

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

void kernel(int M, int N, double* data, double* cov) {
    double* mean = (double*)malloc(M * sizeof(double));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mean[j] += data[i * M + j];
        }
    }

    for (int i = 0; i < M; i++) {
        mean[i] /= (double)N;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            data[i * M + j] -= mean[j];
        }
    }

    double *d_data, *d_cov;
    cudaMalloc((void**)&d_data, N * M * sizeof(double));
    cudaMalloc((void**)&d_cov, M * M * sizeof(double));

    cudaMemcpy(d_data, data, N * M * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    covariance_kernel<<<gridDim, blockDim>>>(M, N, d_data, d_cov);

    cudaMemcpy(cov, d_cov, M * M * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_cov);

    free(mean);
}

void run_bm(int M, int N, const char* preset) {
    double *data = (double*)malloc(N * M * sizeof(double));
    double *cov = (double*)malloc(M * M * sizeof(double));
    initialize(M, N, data);

    #if DEV_MODE
    printf("Data matrix: \n");
    printMatrix(N, M, data);
    #endif

    #if TIME
        dphpc_time3(
            reset(),
            kernel(M, N, data, cov),
            preset
        );
    #else
        kernel(M, N, data, cov);
    #endif

    #if DEV_MODE
    printf("Covariance matrix: \n");
    printMatrix(M, M, cov);
    printf("END\n\n");
    #endif

    free(data);
    free(cov);
}

int main() {
    #if DEV_MODE
        run_bm(5, 5, "S");
        run_bm(5, 7, "M");
    #else
        run_bm(500, 600, "S");
        run_bm(1400, 1800, "M");
        run_bm(3200, 4000, "L");
        run_bm(1200, 1400, "paper");
    #endif
    return 0;
}
