#include "../../timing/dphpc_timing.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define ASSERT 1

__global__ void kernel_j2d(int n, double *A, double *B)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < (n - 1) && j > 0 && j < (n - 1))
    {
        B[i * n + j] = 0.2 * (A[i * n + j] + A[i * n + (j - 1)] + A[i * n + (j + 1)] + A[(i + 1) * n + j] + A[(i - 1) * n + j]);
    }
        
}

void init_arrays(int n, double *A, double *B)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = ((double) i * (j + 2) + 2) / n;
            B[i * n + j] = ((double) i * (j + 3) + 3) / n;
        }
    }
}

void print_arrays(int n, double *A, double *B)
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
    
    puts("\nmatrix B:");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.2f ", B[i * n + j]);
        }
        printf("\n");
    }
}


void run_kernel_j2d(int tsteps, int n, double *A, double *B)
{   
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(n / 16 + 1, n / 16 + 1);
    for (int t = 0; t < tsteps; t++)
    {
        // after 3 rows have been done, the reverse accumulation can begin simultaneously, in theory
        // could also put in kernel, with if guards. kinda difficult with the blocking though. maybe prev block row can start?
        kernel_j2d<<<numBlocks,threadsPerBlock>>>(n, A, B); 
        cudaDeviceSynchronize();
        kernel_j2d<<<numBlocks,threadsPerBlock>>>(n, B, A);
        cudaDeviceSynchronize();
    }
}

void reset(int n, double *A, double *A_d, double *B, double *B_d)
{
    cudaMemcpy((void*) A_d, (void*) A, sizeof(*A) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy((void*) B_d, (void*) B, sizeof(*B) * n * n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void run_bm(int tsteps, int n, const char* preset)
{
    double *A = (double*) malloc(sizeof(*A) * n * n);
    double *B = (double*) malloc(sizeof(*B) * n * n);

    double *A_d, *B_d;
    cudaMalloc((void**) &A_d, sizeof(*A) * n * n);
    cudaMalloc((void**) &B_d, sizeof(*B) * n * n);

    init_arrays(n, A, B);

    cudaMemcpy((void*) A_d, (void*) A, sizeof(*A) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy((void*) B_d, (void*) B, sizeof(*B) * n * n, cudaMemcpyHostToDevice);

    dphpc_time3(
        reset(n, A, B, A_d, B_d),
        run_kernel_j2d(tsteps, n, A_d, B_d),
        preset
    );
    
    if (ASSERT && strcmp(preset, "S") == 0)
    {
        print_arrays(n, *A_d, *B_d);
        print_arrays(n, *A, *B);
    }

    cudaFree((void*) A_d);
    cudaFree((void*) B_d);

    free((void*) A);
    free((void*) B);
}

#ifndef MAIN_HANDLED
int main(int argc, char** argv)
{   
    run_bm(50, 150, "S");   // steps 50, n 150
    run_bm(80, 350, "M");   // steps 80, n 350
    run_bm(200, 700, "L");   // steps 200, n 700
    run_bm(1000, 2800, "paper");  // steps 1000, n 2800

    return 0;
}
#endif
