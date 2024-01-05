#define PARAMETERSH // dont include definition of get_params function again
#include "utils.h"
#include "../../timing/dphpc_timing.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <stdlib.h>


void init_array(int nr, int nq, int np,
  double *A,
  double *C4)
{

    for (int i = 0; i < nr; i++)
        for (int j = 0; j < nq; j++)
            for (int k = 0; k < np; k++)
                A[i * nq * np + j * np + k] = (double) ((i*j + k)%np) / np;
    for (int i = 0; i < np; i++)
        for (int j = 0; j < np; j++)
            C4[i * np + j] = (double) (i*j % np) / np;
}


void reset(int nr, int nq, int np, double *A, double *A_gpu, double *sum_gpu) {
    cudaMemcpy(A_gpu, A, nr * nq * np * sizeof(double), cudaMemcpyHostToDevice),
    cudaMemset(sum_gpu, 0, nr * nq * np * sizeof(double));
    cudaDeviceSynchronize();
}


__global__ void kernel_doitgen_test(int nr, int nq, int np,
      double *A,
      double *C4,
      double *sum)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < nr && q < nq) {
        for(int p = 0; p < np; p++) {
            double tempSum = 0.0;
            for (int s = 0; s < np; s++) {
                tempSum += A[r * nq * np + q * np + s] * C4[s * np + p];
            }
            sum[r * nq * np + q * np + p] = tempSum;
        }
        for (int p = 0; p < np; p++) {
            A[r * nq * np + q * np + p] = sum[r * nq * np + q * np + p];
        }
    }

}
void run_doitgen_gpu_test(int nr, int nq, int np,
      double *A,
      double *C4,
      double *sum)
{
    dim3 threadsPerBlock(16, 16);
    int n = max(nr, nq);
    dim3 numBlocks(n / 16 + 1, n / 16 + 1);
    kernel_doitgen_test<<<numBlocks,threadsPerBlock>>>(nr, nq, np, A, C4, sum);
    cudaDeviceSynchronize();
}

void assertCorrectness(int nr, int nq, int np,
    double *A, const char *prefix) {

    fprintf(stderr, "A[0][0][1] = %f\n", prefix, A[1]);

    double *A_test;
    A_test = (double*)malloc (nr * nq * np* sizeof(double));
    double *C4_test;
    C4_test = (double*)malloc (np * np * sizeof(double));

    double *A_test_gpu, *C4_test_gpu, *sum_test_gpu;
    cudaMalloc((void**) &A_test_gpu, nr * nq * np * sizeof(double));
    cudaMalloc((void**) &C4_test_gpu, np * np * sizeof(double));
    cudaMalloc((void**) &sum_test_gpu, nr * nq * np * sizeof(double));

    init_array(nr, nq, np, A_test, C4_test);
    cudaMemcpy(A_test_gpu, A_test, nr * nq * np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C4_test_gpu, C4_test, np * np * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    run_doitgen_gpu_test(nr, nq, np, A_test_gpu, C4_test_gpu, sum_test_gpu);
    cudaMemcpy(A_test, A_test_gpu, nr * nq * np * sizeof(double), cudaMemcpyDeviceToHost);
    fprintf(stderr, "A_test[0][0][1] = %f\n", prefix, A_test[1]);
    //fprintf(stderr, "A[%d][%d][%d] = %f\n", nr /2, nq / 2, np / 2, A[(nr / 2) * nq * np + (nq / 2) * np + (np / 2)]);
    //fprintf(stderr, "A_test[%d][%d][%d] = %f\n", nr /2, nq / 2, np / 2, A_test[(nr / 2) * nq * np + (nq / 2) * np + (np / 2)]);
    //fprintf(stderr, "A_test[0][96][45] = %f\n", A_test[0 * nq * np + 96 * np + 45]);
    // Perform assertion
    for (int r = 0; r < nr; r++) {
        for (int q = 0; q < nq; q++) {
            for (int p = 0; p < np; p++){
                if (A[r * nq * np + q * np + p] != A_test[r * nq * np + q * np + p]) {
                    printf("A[%d][%d][%d] = %f\n", r, q, p, A[r * nq * np + q * np + p]);
                    printf("A_test[%d][%d][%d] = %f\n", r, q, p, A_test[r * nq * np + q * np + p]);
                    printf("Arrays are not equal.\n");
                    exit(1);
                }
            }
        }
    }
    printf("Arrays are equal.\n");
    cudaFree(A_test_gpu);
    cudaFree(C4_test_gpu);
    cudaFree(sum_test_gpu);
    free(A_test);
    free(C4_test);
}


void run_bm(int nr, int nq, int np, const char* preset, void (*kernel)(int, int, int, double*, double*, double*), int ASSERT) {
    
    double *A, *C4;
    A = (double*)malloc (nr * nq * np* sizeof(double));
    C4 = (double*)malloc (np * np * sizeof(double));

    double *A_gpu, *C4_gpu, *sum_gpu;

    cudaMalloc((void**) &A_gpu, nr * nq * np * sizeof(double));
    cudaMalloc((void**) &C4_gpu, np * np * sizeof(double));
    cudaMalloc((void**) &sum_gpu, nr * nq * np * sizeof(double));

    init_array (nr, nq, np,
        A,
        C4);
    cudaMemcpy(C4_gpu, C4, np * np * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    dphpc_time3(
        reset(nr, nq, np, A, A_gpu, sum_gpu),
        (*kernel)(nr, nq, np, A_gpu, C4_gpu, sum_gpu),
        preset
    );

    if (ASSERT && should_run_preset(preset)) {
        cudaMemcpy(A, A_gpu, nr * nq * np * sizeof(double), cudaMemcpyDeviceToHost);
        assertCorrectness(nr, nq, np, A, preset);
    }

    cudaFree(A_gpu);
    cudaFree(C4_gpu);
    cudaFree(sum_gpu);

    free(A);
    free(C4);
}