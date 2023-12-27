#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"

#define VALIDATION // toggle to turn on/off 

// Diffcult to parallelize as its prone to race condition (A is modified while being read)
// There are many nested loops so the naive approach is to call the kernel in a loop
// Doesn't work yet 
__global__ void lu_kernel_1(int N, double* A, int i){

 //int i = blockIdx.x * blockDim.x + threadIdx.x;
 int j = blockIdx.y * blockDim.y + threadIdx.y;
 int k;

    if (j<i) {
        for (k = 0; k < j; k++) {
            A[i * N + j] = A[i * N + j] - (A[i * N + k] * A[k * N + j]);
            } 
           // __syncthreads();

    }

}

__global__ void lu_kernel_2(int N, double* A, int i){

 //int i = blockIdx.x * blockDim.x + threadIdx.x;
 int j = blockIdx.y * blockDim.y + threadIdx.y; // column

    if (j<i) {
        A[i * N + j] = A[i * N + j] / A[j * N + j];
         //__syncthreads();
    }

}


__global__ void lu_kernel_3(int N, double* A, int i){

//int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y; // column
int k;

    if (j<=i) {
    for (k = 0; k < i; k++) {
        A[i * N + j] = A[i * N + j] - (A[i * N + k] * A[k * N + j]);
       
        }    
       // __syncthreads();
    
}

}


__global__ void lu_kernel_test(int N, double* A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;

    if (i < N && j < N) {
        if (j < i) {
            for (k = 0; k < j; k++) {
                A[i * N + j] = A[i * N + j] - (A[i * N + k] * A[k * N + j]);
              
            }
            A[i * N + j] = A[i * N + j] / A[j * N + j];

        } else {
            for (k = 0; k < i; k++) {
                A[i * N + j] = A[i * N + j] - A[i * N + k] * A[k * N + j];
                
            }
        }
    }
}

// Using different blocks and cudaDeviceSync after each kernel call doesn't effet the result 
void run_lu_kernel(int N, double* A) {
    dim3 block1(16, 16);
    dim3 grid1((N+block1.x -1)/block1.x, (N+block1.y-1)/block1.y);

    dim3 block2(16, 16);
    dim3 grid2((N+block2.x -1)/block2.x, (N+block2.y-1)/block2.y);

    //dim3 block3(16, 16);
    //dim3 grid3((N+block3.x -1)/block3.x, (N+block3.y-1)/block3.y);
    // lu_kernel_test<<<grid,block>>>(N, A); 

    for (int i=0; i<N; i++){

    lu_kernel_1<<<grid1,block1>>>(N, A, i);
    cudaDeviceSynchronize(); 
    lu_kernel_2<<<grid1,block1>>>(N, A, i); 
    cudaDeviceSynchronize();
    lu_kernel_3<<<grid2,block2>>>(N, A, i);
    cudaDeviceSynchronize(); 


    }


    //cudaDeviceSynchronize();
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





