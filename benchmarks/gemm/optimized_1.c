#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../timing/dphpc_timing.h"

// similar to polybench implemenation 
// Dynamic Arrays 


// gemm: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/  


#define alpha 1.5 
#define beta 1.2

//#define VALIDATION 

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
    
    validation_init_matrices(N, M, K, A, B, C); 
    gemm(N, M, K, A, B, C);

    // print C's content here 

    // write C to file called gemm_unrolledx4_acc_gpu
    FILE *outputFile;
    char fileName[] = "optimized_1_output.txt";

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

    free(A);
    free(B);
    free(C); 
}



int main(){
    
    #ifdef VALIDATION  

    validation(30,40,50);

    #else 

    run_bm(1000, 1100, 1200, "S"); 
    run_bm(2500, 2750, 3000, "M"); 
    run_bm(7000, 7500, 8000, "L"); 
    run_bm(2000, 2300, 2600, "paper"); 

    #endif

    
    return 0; 
}