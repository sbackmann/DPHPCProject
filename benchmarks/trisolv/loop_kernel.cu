#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "../../timing/dphpc_timing.h"

bool VALIDATE = false;

__global__ void kernel(double *L, double *x, double *b, int i, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j  = i + id - 1;
    int d = i-1;

    if (d >= 1 && j <= N) {
        x[j-1] += x[d-1]*L[(j-1)*N + d-1];
    }

    if (id == 1) {
        x[i-1] = (b[i-1] - x[i-1]) / L[(i-1)*N + i-1];
    }
}

void kernel(double *L, double *x, double *b, int N) {
    int threads = 64;
    

    for(int i = 1; i <= N; i++) {
        int numBlocks = (N - i) / threads + 1;
        kernel<<<numBlocks, threads>>>(L, x, b, N, i);
    }

    cudaDeviceSynchronize();
}

void reset(double *L, double *x, double *b, double *d_L, double *d_x, double *d_b, int N) {
    initialize(N, L, x, b);

    cudaMemcpy(d_L, L, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void run_bm(int N, const char *preset) {
    double *L = (double*)malloc(sizeof(double) * N * N);
    double *x = (double*)malloc(sizeof(double) * N);
    double *b = (double*)malloc(sizeof(double) * N);

    double *d_L, *d_x, *d_b;
    cudaMalloc((void **)&d_L, N * N * sizeof(double));
    cudaMalloc((void **)&d_x, N * sizeof(double));
    cudaMalloc((void **)&d_b, N * sizeof(double));

    if (VALIDATE) {
        reset(L, x, b, d_L, d_x, d_b, N);

        cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        kernel(d_L, d_x, d_b, N);

        cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        }

        if (!is_correct(N, x, preset)) {
            printf("Validation failed for preset: %s \n", preset);
            exit(1);
        } else {
            printf("Validation passed for preset: %s \n", preset);
        }
    }

    dphpc_time3(
        reset(L, x, b, d_L, d_x, d_b, N),
        kernel(d_L, d_x, d_b, N),
        preset
    );

    cudaFree(d_L);
    cudaFree(d_x);
    cudaFree(d_b);
    free(L);
    free(x);
    free(b);
}

int main() {
    const char *presets[] = {"S", "M", "L", "paper"};

    for (int i = 0; i < 4; i++) {
        const char* preset = presets[i];
        int n = get_params(preset)[0];
        run_bm(n, preset);
    }

    return 0;
}


