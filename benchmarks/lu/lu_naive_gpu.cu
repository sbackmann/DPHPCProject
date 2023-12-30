#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"

//#define VALIDATION // toggle to turn on/off 

// Diffcult to parallelize as its prone to race condition (A is modified while being read)
// This approach works but the GPU block is 1D, I don't know how that will effect perf on larger inpute 
// Lower triangle is not correct 


__global__ void lu_kernel(int N, double* A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++) {
                A[i * N + j] = A[i * N + j] - (A[i * N + k] * A[k * N + j]);
            }
            A[i * N + j] = A[i * N + j] / A[j * N + j];
        }

        __syncthreads();  // Ensure previous calculations are complete before proceeding

        for (int j = i; j < N; j++) {
            for (int k = 0; k < i; k++) {
               A[i * N + j] = A[i * N + j] - A[i * N + k] * A[k * N + j];
            }
        }
    }
}

// All values are not NaN and postive; upper triangle is correct but not the lower 
void run_lu_kernel(int N, double* A) {
   
    // Define thread block and grid dimensions

    // Note, the blocksize that I was using was an issue 
    // 1D block works 
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    lu<<<blocksPerGrid, threadsPerBlock>>>(N, A);
    cudaDeviceSynchronize();

}



void init_array(int N, double* A) {

  double* B = (double*)malloc(N * N * sizeof(double));

  // create lower triangle of matrix 
  for (int i = 0; i < N; i++) {
    // initialize the lower triangle 
    for (int j = 0; j <= i; j++) {
      A[i * N + j] = (double)(-j % N) / N + 1;
    }
    // set upper triangle to zero 
    for (int j = i + 1; j < N; j++) {
      A[i * N + j] = 0;
    }
    // set elements on the diagonal to 1 
    A[i * N + i] = 1;
  }

  for (int r = 0; r < N; ++r)
    for (int s = 0; s < N; ++s)
        B[r*N + s] = 0;

  // multiply A by A^T and save the result in B
  // result is a symmetric matrix 
  for (int t = 0; t < N; ++t)
    for (int r = 0; r < N; ++r)
      for (int s = 0; s < N; ++s)
        B[r * N + s] = B[r * N + s] + (A[r * N + t] * A[s * N + t]);

  // Copy the result back to A
  for (int r = 0; r < N; ++r)
    for (int s = 0; s < N; ++s)
      A[r * N + s] = B[r * N + s];

  // Free the dynamically allocated memory for A and B
  free(B);
}



void run_bm(int N, const char* preset) {

    double* A = (double *)malloc(N*N*sizeof(double));
    double *A_d;

    cudaMalloc((void**) &A_d, N*N*sizeof(double));
   
    init_array(N, A);

    
    cudaMemcpy((void*) A_d, (void*) A, N*N*sizeof(double), cudaMemcpyHostToDevice);
    
    dphpc_time3(
       cudaMemcpy((void*) A_d, (void*) A, N*N*sizeof(double), cudaMemcpyHostToDevice),
       run_lu_kernel(N, A_d),
        preset
    );

    cudaFree(A_d);
  
    free(A);

}

//***************************************VALIDATION***************************************

void validation(int N) {

    double* A = (double *)malloc(N*N*sizeof(double));

    double *A_d;
    cudaMalloc((void**) &A_d, N*N*sizeof(double));

    // init with predefined matrices
    init_array(N,A);

    
    cudaMemcpy((void*) A_d, (void*) A, N*N*sizeof(double), cudaMemcpyHostToDevice);

    // run kernal 
    run_lu_kernel(N, A_d);

    // copy C from device back to host 
    cudaMemcpy(A, A_d, N*N*sizeof(double), cudaMemcpyDeviceToHost); 
    
    // write C to file called gemm_unrolledx4_acc_gpu
    FILE *outputFile;
    char fileName[] = "test_lu.txt";

    // Open the file for writing
    outputFile = fopen(fileName, "w");

    if (outputFile == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        exit(EXIT_FAILURE);
    }

    // Print the matrix C to the file
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(outputFile, "%.2f ", A[i * N + j]);
        }
        fprintf(outputFile, "\n");
    }

    // Close the file
    fclose(outputFile);

    cudaFree(A_d);

    free(A);
    
}


int main(int argc, char** argv) {

  #ifdef VALIDATION 

    validation(30);

    #else

    run_bm(60, "S"); 
    run_bm(220, "M"); 
    run_bm(700, "L"); 
    run_bm(2000, "paper"); 

   #endif 

  return 0;
}





