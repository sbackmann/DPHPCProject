#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"
#include "_parameters.h"

// #define VALIDATION // toggle to turn on/off 

__global__ void set_zero(double* s, int iB) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < iB)
      s[i] = 0;
}

__global__ void kernel1(double* A, int N, int i, int B) {
    int m = B < i ? B-1 : i-1;
    double s;
    for (int j=1; j <= m; j++) {
        s = 0.0;
        for (int k = 1; k <= j-1; k++) {
            s += A[(i-1)*N + k-1] * A[(k-1)*N + j-1];
        }
        A[(i-1)*N + j-1] = (A[(i-1)*N + j-1] - s) / A[(j-1)*N + j-1];
    }
}

__global__ void kernel2(double* A, double* s, int N, int i, int B, int k) {
    int id = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = id-1+B;

    if (B <= j && j <= i-1) {
        s[j-B+1-1] += A[(i-1)*N + k-1] * A[(k-1)*N + j-1];
    }
}

__global__ void kernel3(double* A, int N, double* s, int i, int B) {
    for (int j = B; j <= i-1; j++) {
        for (int k = B; k <= j-1; k++) {
            s[j-B+1-1] += A[(i-1)*N + k-1] * A[(k-1)*N + j-1];
        }
        A[(i-1)*N + j-1] = (A[(i-1)*N + j-1] - s[j-B+1-1]) / A[(j-1)*N + j-1];
    }
}


__global__ void kernel4(double* A, int N, int i, int k) {
    int id = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = id-1+i;

    if (i <= j && j <= N) {
        A[(i-1)*N + j-1] -= A[(i-1)*N + k-1] * A[(k-1)*N + j-1];
    }
}


// Using different blocks and cudaDeviceSync after each kernel call doesn't effet the result 
void run_lu_kernel(int N, double* A) {
    int B = (int) (0.3*N);
    double *s; cudaMalloc((void**) &s, (N-B)*sizeof(double));
    int threadsPerBlock = 64;
    int n;
    int blocks;
    
    for (int i=1; i <= N; i++) {

        kernel1<<<1,1>>>(A, N, i, B);

        if (i-1 >= B) {

            n = i-B;
            blocks = (n-1) / threadsPerBlock + 1;
            set_zero<<<blocks, threadsPerBlock>>>(s, i-B);

            for (int k = 1; k <= B-1; k++) {
                n = i-B;
                blocks = (n-1) / threadsPerBlock + 1;
                kernel2<<<blocks, threadsPerBlock>>>(A, s, N, i, B, k);
            }

            kernel3<<<1, 1>>>(A, N, s, i, B);
        }

        for (int k=1; k <= i-1; k++) {
            n = N-i+1;
            blocks = (n-1) / threadsPerBlock + 1;
            kernel4<<<blocks, threadsPerBlock>>>(A, N, i, k);
        }
    }
    cudaDeviceSynchronize();
    cudaFree(s);

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

    for (int i = N-5; i<N; i++) {
        for (int j=N-5; j<N; j++) {
            printf("%f, ", A[i*N+j]);
        }
        printf("\n");
    }
    
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

    

    const char *presets[] = {"S", "M", "L", "paper"};

    for (int i = 0; i < 4; i++) {
        const char* preset = presets[i];
        int n = get_params(preset)[0];
        run_bm(n, preset);
    }

   #endif 

  return 0;
}





