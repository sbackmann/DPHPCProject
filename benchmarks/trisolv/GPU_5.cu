#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "utils.h"
#include "../../timing/dphpc_timing.h"

#define DEV true
#define TIME false
#define DEBUG false

// Function to perform the inversion of the diagonal elements
__global__ void inv_diag_kernel(double *matrix, double *diag, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        diag[i] = 1.0 / matrix[i * N + i];
    }
}

// Function to perform scalar multiplication
__global__ void scalar_mult_kernel(double *L, double *Lx, double scalar, int j, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Lx[i * N + j] = L[i * N + j] * scalar;
    }
}

// Function to perform the main computation kernel
__global__ void pre_comp_kernel(double *L, double *Lx, double *x, int j, int N) {
    int start_row = j;
    int i = blockIdx.x * blockDim.x + threadIdx.x + start_row;
    if (i < N) {
        Lx[i * N + j] = L[i * N + j] * x[j] + Lx[i * N + (j - 1)];
    }
}

// Function to update the solution vector
__global__ void scalar_update_kernel(int N, double *x, int i, double *Lx, double *inv_diag) {
    x[i] -= (Lx[i * N + (i - 1)] * inv_diag[i]);
}

__global__ void element_wise_mult_kernel(double *a, double *b, double *dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < N) {
        dst[i] = a[i] * b[i];
    }
}

void kernel(double *d_L, double *d_x, double *d_b, int N) {
    int t = 256;
    int blocks = (N + t - 1) / t;

    double *d_inv_diag, *d_b_prod_inv_diag, *d_Lx;
    cudaMalloc((void **)&d_inv_diag, N * sizeof(double));
    cudaMalloc((void **)&d_b_prod_inv_diag, N * sizeof(double));
    cudaMalloc((void **)&d_Lx, N * N * sizeof(double));

    inv_diag_kernel<<<blocks, t>>>(d_L, d_inv_diag, N);
    cudaDeviceSynchronize();

    element_wise_mult_kernel<<<blocks, t>>>(d_b, d_inv_diag, d_b_prod_inv_diag, N);
    cudaDeviceSynchronize();

    // now wrong
    // d_x = d_b_prod_inv_diag;

    scalar_mult_kernel<<<blocks, t>>>(d_L, d_Lx, d_b_prod_inv_diag[0], 0, N);
    cudaDeviceSynchronize();

    // Main computation loop
    for (int i = 1; i < N; i++) {
        // Update x
        scalar_update_kernel<<<1, 1>>>(N, d_x, i, d_Lx, d_inv_diag);
        cudaDeviceSynchronize();

        // Pre-compute kernel
        blocks = (N - i + t - 1) / t;
        pre_comp_kernel<<<blocks, t>>>(d_L, d_Lx, d_x, i, N);
        cudaDeviceSynchronize();
    }

    cudaFree(d_inv_diag);
    cudaFree(d_b_prod_inv_diag);
    cudaFree(d_Lx);
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
        x[i] = -999.0;
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
    // kernel(N, L, x, b);
    // if (!is_correct(N, x, preset)) {
    //     printf("Validation failed for preset: %s \n", preset);
    //     exit(1);
    // } else {
    //     printf("Validation passed for preset: %s \n", preset);
    // }

    dphpc_time3(
        reset(L, x, b, d_L, d_x, d_b, N),
        kernel(d_L, d_x, d_b, N),
        preset
    );

    cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

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
