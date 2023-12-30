#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../timing/dphpc_timing.h"


#define alpha 1.5 
#define beta 1.2

cudaError_t cudaStatus;
//#define VALIDATION  // comment or uncomment to toggle 


// Arrays are stored in row-major but in the kernel they are being read as column major 
// Been validated to be correct 

__global__ void gemm_kernel(int N, int M, int K, double *A, double *B, double *C){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k; 

   if (i < N && j < M) {
        C[N * j + i] *= beta;
        for (int k = 0; k < K; k++) {
            C[N * j + i] += alpha * A[N * k + i] * B[K * j + k];
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
    gemm_kernel<<<grid,block>>>(N, M, K, A, B, C); 
    cudaDeviceSynchronize();

     #ifdef VALIDATION
    cudaStatus = cudaGetLastError();
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }
    #endif

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

// ************************ Validation **********************************

void validation_init_matrices(int N, int M, int K, double* A, double* B, double* C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            A[i * K + j] = 0.5; 

    for (int i = 0; i < K; i++)
        for (int j = 0; j < M; j++)
            B[i * M + j] = 0.7;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            C[i * M + j] = 0.3; 
}


void validation(int N, int M, int K) {

    double* A = (double *)malloc(N*K*sizeof(double));
    double* B = (double *)malloc(K*M*sizeof(double));
    double* C = (double *)malloc(N*M*sizeof(double));

    double *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, N*K*sizeof(double));
    cudaMalloc((void**) &B_d, K*M*sizeof(double));
    cudaMalloc((void**) &C_d, N*M*sizeof(double));

    // init with predefined matrices
    validation_init_matrices(N, M, K, A, B, C);

    
    cudaMemcpy((void*) A_d, (void*) A, N*K*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) B_d, (void*) B, K*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) C_d, (void*) C, N*M*sizeof(double), cudaMemcpyHostToDevice);

    // run kernal 
    run_gemm_kernel(N, M, K, A_d, B_d, C_d);

    // copy C from device back to host 
    cudaMemcpy(C, C_d, N*M*sizeof(double), cudaMemcpyDeviceToHost); 
    
    // write C to file called gemm_unrolledx4_acc_gpu
    FILE *outputFile;
    char fileName[] = "gemm_naive_gpu.txt";

    // Open the file for writing
    outputFile = fopen(fileName, "w");

    if (outputFile == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        exit(EXIT_FAILURE);
    }

    // Print the matrix C to the file
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            fprintf(outputFile, "%.2f ", C[i * M + j]);
        }
        fprintf(outputFile, "\n");
    }

    // Close the file
    fclose(outputFile);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    free(A);
    free(B);
    free(C); 
}

int main(){
    
   #ifdef VALIDATION 

    validation(30, 40, 50);

    #else

    run_bm(1000, 1100, 1200, "S"); 
    run_bm(2500, 2750, 3000, "M"); 
    run_bm(7000, 7500, 8000, "L"); 
    run_bm(2000, 2300, 2600, "paper"); 

    #endif
    
    return 0; 
}