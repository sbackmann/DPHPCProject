#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../../timing/dphpc_timing.h"

// similar to polybench implemenation 
// Note: only works with small size arrays 


// gemm: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/  


#define alpha 1.5 
#define beta 1.2

// initialize matrices A, B, and C with random values
void init_matrices(int N, int M, int K, double A[N][K], double B[K][M], double C[N][M]){
    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            A[i][j] = (double) ((i*j+1) % K) / K;
    
    for (int i = 0; i < K; i++)
        for (int j = 0; j < M; j++)
            B[i][j] = (double) ((i*j+1) % M) / M;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            C[i][j] = (double) ((i*j+1) % M) / M;

}

void gemm(int N, int M, int K, double A[N][K], double B[K][M], double C[N][M]){

    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            C[i][j] *= beta;
            for (k = 0; k < K; k++) {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }

}


void run_bm(int N, int M, int K, const char* preset) {

    
    double (*A)[N][K];
    double (*B)[K][M];
    double (*C)[N][M]; 

    A = (double(*)[N][K]) malloc(N*K*sizeof(double));
    B = (double(*)[K][M]) malloc(K*M*sizeof(double));
    C = (double(*)[N][M]) malloc(N*M*sizeof(double));


    dphpc_time3(
       init_matrices(N, M, K, *A, *B, *C),
       gemm(N, M, K, *A, *B, *C),
        preset
    );

    free(A);
    free(B);
    free(C); 
}


int main(int argc, char** argv){

    run_bm(1000, 1100, 1200, "S"); 
    run_bm(2500, 2750, 3000, "M"); 
    run_bm(7000, 7500, 8000, "L"); 
    run_bm(2000, 2300, 2600, "paper"); 
    
return 0;


}