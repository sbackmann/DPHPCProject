#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../timing/dphpc_timing.h"

// similar to polybench implemenation 
// Dynamic Arrays 


// gemm: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/  


#define alpha 1.5 
#define beta 1.2

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

void gemm(int N, int M, int K, double* A, double* B, double* C){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            C[i*M + j] *= beta;
            for (int k = 0; k < K; k++) {
                C[i*M + j] += alpha * A[i*K + k] * B[k*M + j];
            }
        }
    }
}


void run_bm(int N, int M, int K, const char* preset) {

    double* A = (double *)malloc(N*K*sizeof(double));
    double* B = (double *)malloc(K*M*sizeof(double));
    double* C = (double *)malloc(N*M*sizeof(double));
    

    dphpc_time3(
       init_matrices(N, M, K, A, B, C),
       gemm(N, M, K, A, B, C),
        preset
    );

    free(A);
    free(B);
    free(C); 
}


int main(){
    
    run_bm(50, 60, 70, "S"); 
    run_bm(100, 110, 120, "M"); 
    run_bm(600, 610, 620, "L"); 
    run_bm(1000, 1100, 1200, "paper"); 

    return 0; 
}