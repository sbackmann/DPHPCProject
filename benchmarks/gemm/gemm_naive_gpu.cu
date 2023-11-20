#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../timing/dphpc_timing.h"


#define alpha 1.5 
#define beta 1.2


__global__ void gemm_kernal(int N, int M, int K, double *A, double *B, double *C){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k; 

    if (i < N && j < M) {
        C[i * M + j] *= beta;
        for (k = 0; k < K; k++) {
            C[i * M + j] += alpha * A[i * K + k] * B[k * M + j];
        }
    }
}


// initialize matrices A, B, and C with random values
void init_matrices(int N, int M, int K, double* A, double* B, double* C){
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            A[i*K + j] = (double) ((i*j+1) % K) / K;
    
    for (int i = 0; i < K; i++)
        for (int j = 0; j < M; j++)
            B[i*M + j] = (double) ((i*j+1) % M) / M;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            C[i*M + j] = (double) ((i*j+1) % M) / M;
}


void run_gemm_kernel(int N, int M, int K, double *A, double *B, double *C) {
    dim3 block(16, 16);
    dim3 grid((M+block.x -1)/block.x, (N+block.y-1)/block.y);
    gemm_kernal<<<grid,block>>>(N, M, K, A, B, C); 
    cudaDeviceSynchronize();
}


void run_bm(int N, int M, int K, const char* preset) {

    double* A = (double *)malloc(N*K*sizeof(double));
    double* B = (double *)malloc(K*M*sizeof(double));
    double* C = (double *)malloc(N*M*sizeof(double));

    double *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, N*K*sizeof(double));
    cudaMalloc((void**) &B_d, K*M*sizeof(double));
    cudaMalloc((void**) &C_d, N*M*sizeof(double));

    init_matrices(N, M, K, A, B, C);

    
    cudaMemcpy((void*) A_d, (void*) A, N*K*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) B_d, (void*) B, K*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) C_d, (void*) C, N*M*sizeof(double), cudaMemcpyHostToDevice);


    dphpc_time3(
       cudaMemcpy(C_d, C, N*M*sizeof(double), cudaMemcpyHostToDevice),
       run_gemm_kernel(N, M, K, A_d, B_d, C_d),
        preset
    );

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C); 
}


int main(){
    
    run_bm(1000, 1100, 1200, "S"); 
    run_bm(2500, 2750, 3000, "M"); 
    run_bm(7000, 7500, 8000, "L"); 
    run_bm(2000, 2300, 2600, "paper"); 
    
    return 0; 
}