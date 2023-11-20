#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"



__global__ void lu_kernal(int N, double* A){

 int i = blockIdx.x * blockDim.x + threadIdx.x;
 int j, k;

    if (i < N) {
        for (j = 0; j < i; j++) {
            for (k = 0; k < j; k++) {
                A[i * N + j] = A[i * N + j] - (A[i * N + k] * A[k * N + j]);
            }
            A[i * N + j] = A[i * N + j] / A[j * N + j];
        }
        for (j = i; j < N; j++) {
            for (k = 0; k < i; k++) {
                A[i * N + j] = A[i * N + j] - (A[i * N + k] * A[k * N + j]);
            }
        }
    }
}


void init_array(int N, double A[N][N]) {

  // create lower triangle of matrix 
  for (int i = 0; i < N; i++) {
    // initialize the lower triangle 
    for (int j = 0; j <= i; j++) {
      A[i][j] = (double)(-j % N) / N + 1;
    }
    // set upper triangle to zero 
    for (int j = i + 1; j < N; j++) {
      A[i][j] = 0;
    }
    // set elements on the diagonal to 1 
    A[i][i] = 1;
  }

  double (*B)[N][N];
  B = (double(*)[N][N])malloc(N*N* sizeof(double));

  for (int r = 0; r < N; ++r)
    for (int s = 0; s < N; ++s)
      (*B)[r][s] = 0;

  // multiply A by A^T and save the result in B
  // result is a symmetric matrix 
  for (int t = 0; t < N; ++t)
    for (int r = 0; r < N; ++r)
      for (int s = 0; s < N; ++s)
        (*B)[r][s] = (*B)[r][s] + (A[r][t] * A[s][t]);

  for (int r = 0; r < N; ++r)
    for (int s = 0; s < N; ++s)
      A[r][s] = (*B)[r][s];

  free((void*)B);

}

void run_lu_kernel(int N, double* A) {
    dim3 block(16, 16);
    dim3 grid((N+block.x -1)/block.x, (N+block.y-1)/block.y);
    lu_kernal<<<grid,block>>>(N, A); 
    cudaDeviceSynchronize();
}

void run_bm(int N, int M, int K, const char* preset) {

    double* A = (double *)malloc(N*K*sizeof(double));

    double *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, N*K*sizeof(double));
   
    init_matrices(N, A);

    
    cudaMemcpy((void*) A_d, (void*) A, N*K*sizeof(double), cudaMemcpyHostToDevice);
    
    dphpc_time3(
       cudaMemcpy((void*) A_d, (void*) A, N*K*sizeof(double), cudaMemcpyHostToDevice),
       run_lu_kernel(N, A_d),
        preset
    );

    cudaFree(A_d);
  
    free(A);

}

int main(int argc, char** argv) {

    run_bm(60, "S"); 
    run_bm(220, "M"); 
    run_bm(700, "L"); 
    run_bm(2000, "paper"); 

  return 0;
}



