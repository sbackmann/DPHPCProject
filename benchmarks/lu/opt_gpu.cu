#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"
#include "_parameters.h"

// #define VALIDATION // toggle to turn on/off 


__global__ void kernel_col(double* A, int N, int i) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
    
    if (j < N)
    {
        A[j * N + i] /= A[i * N + i];
    }
}

__global__ void kernel_submat(double* A, int N, int i) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + i + 1;
    
    if (j < N && k < N)
    {
        A[j * N + k] -= (A[i * N + k] * A[j * N + i]);
    }
}

void run_lu_kernel(int N, double* A) {
    int threadsPerBlock1D = 256;
    int blocks1D;
    dim3 threadsPerBlock2D(16, 16);   // threads per block: 256
    dim3 blocks2D;
    
    // based on restructured core loop:
    // for (int i = 0; i < N; i++) {
    //     for (int j = i + 1; j < N; j++) {
    //         A[j][i] /= A[i][i]; // lower triangle
    //     }
        
    //     for (int j = i + 1; j < N; j++) {
    //         for (int k = i + 1; k < N; k++) {
    //             A[j][k] -= (A[i][k] * A[j][i]); // upper triangle + diagonal
    //         }   
    //     }
    // }
    
    for (int i = 0; i < N; i++) {
        blocks1D = (N - i - 2) / threadsPerBlock1D + 1;
        kernel_col<<<blocks1D, threadsPerBlock1D>>>(A, N, i);
        
        blocks2D = dim3((N - i - 2) / threadsPerBlock2D.x + 1, (N - i - 2) / threadsPerBlock2D.y + 1);
        kernel_submat<<<blocks2D, threadsPerBlock2D>>>(A, N, i);   
    }
    cudaDeviceSynchronize();
}



void init_array(int N, double* A) {

    double *B = (double *) calloc(N * N, sizeof(double));

    // create lower triangle of matrix 
    for (int i = 0; i < N; i++) {
        // initialize the lower triangle 
        for (int j = 0; j < i; j++) {
            A[i * N + j] = (double)(-j % N) / N + 1;
        }
        
        // set elements on the diagonal to 1 
        A[i * N + i] = 1;
        
        // set upper triangle to zero 
        for (int j = i + 1; j < N; j++) {
            A[i * N + j] = 0;
        }
    }

    // multiply A by A^T and save the result in B
    // result is a symmetric matrix 
    for (int t = 0; t < N; t++) {
        for (int r = 0; r < N; r++) {
            for (int s = 0; s < N; s++) {
                B[r * N + s] += A[r * N + t] * A[s * N + t];
            }
        }
    }

    // Copy the result back to A
    for (int r = 0; r < N; r++) {
        for (int s = 0; s < N; s++) {
            A[r * N + s] = B[r * N + s];
        }
    }

    // Free the dynamically allocated memory for B
    free(B);
}



void run_bm(int N, const char* preset) {

    double *A = (double *) malloc(N * N * sizeof(double));
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
void print_array(int n, double *A)
{   
    puts("matrix A:");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f ", A[i * n + j]);
        }
        printf("\n");
    }
}

void validation(int N) {

    double *A = (double *) malloc(N * N * sizeof(double));

    double *A_d;
    cudaMalloc((void**) &A_d, N*N*sizeof(double));

    // init with predefined matrices
    init_array(N, A);
    
    // print to stdout
    //print_array(N, A);
    
    cudaMemcpy((void*) A_d, (void*) A, N*N*sizeof(double), cudaMemcpyHostToDevice);

    // run kernal 
    run_lu_kernel(N, A_d);

    // copy C from device back to host 
    cudaMemcpy(A, A_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    
    // print to stdout
    //print_array(N, A);
    
    // write C to file
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





