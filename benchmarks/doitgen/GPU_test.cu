
#include "../../timing/dphpc_timing.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <stdlib.h>

#define ASSERT 0

static
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

/*

static
void print_array(int nr, int nq, int np,
   double *A)
{
    
    int i, j, k;

    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "A");
    for (i = 0; i < nr; i++)
        for (j = 0; j < nq; j++)
            for (k = 0; k < np; k++) {
                if ((i*nq*np+j*np+k) % 20 == 0) fprintf (stderr, "\n");
                fprintf (stderr, "%0.5lf ", A[i * nq * np + j * np + k]);
        }
    fprintf(stderr, "\nend   dump: %s\n", "A");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}
*/

void assertCorrectness(int nr, int nq, int np,
    double *A, const char *prefix) {

    // Somehow only works when a.exe run manually
    // When path is amended to "test_cases/%s.dat", no output is produced

    fprintf(stderr, "A[0][0][1] = %f\n", A[1]);
    char path[100];
    snprintf(path, sizeof(path), "benchmarks/doitgen/test_cases/%s.dat", prefix);
    printf("Reading from %s\n", path);
    FILE *file = fopen(path, "rb");
    printf("Array deserialized.\n");
    if (!file)
        perror("fopen");

    double A_test[nr][nq][np];

    if (fread(A_test, sizeof(double), nr * nq * np, file) != nr * nq * np) {
        printf("Error reading file.\n");
        exit(1);
    }

    fclose(file);
    // Perform assertion
    for (int r = 0; r < nr; r++) {
        for (int q = 0; q < nq; q++) {
            for (int p = 0; p < np; p++){
                if (A[r * nq * np + q * np + p] != A_test[r][q][p]) {
                    printf("A[%d][%d][%d] = %f\n", r, q, p, A[r * nq * np + q * np + p]);
                    printf("A_test[%d][%d][%d] = %f\n", r, q, p, A_test[r][q][p]);
                    printf("Arrays are not equal.\n");
                    exit(1);
                }
            }
        }
    }
    printf("Arrays are equal.\n");
}

__global__ void kernel_doitgen(int nr, int nq, int np,
      double *A,
      double *C4,
      double *sum)
{
    // Very bad performance - optimization replace writing to sum with temporary variable
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < nr && q < nq) {
        for(int p = 0; p < np; p++) {
            extern __shared__ double C[];
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
void run_doitgen_gpu(int nr, int nq, int np,
      double *A,
      double *C4,
      double *sum)
{
    dim3 threadsPerBlock(16, 16);
    int n = max(nr, nq);
    dim3 numBlocks(n / 16 + 1, n / 16 + 1);
    sharedMemorySize
    kernel_doitgen<<<numBlocks,threadsPerBlock>>>(nr, nq, np, A, C4, sum);
    cudaDeviceSynchronize();
}

void reset(int nr, int nq, int np, double *A, double *A_gpu) {
    cudaMemcpy(A_gpu, A, nr * nq * np * sizeof(double), cudaMemcpyHostToDevice),
    cudaDeviceSynchronize();
}

void run_bm(int nr, int nq, int np, const char* preset) {
    
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
    cudaMemcpy(A_gpu, A, nr * nq * np * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(C4_gpu, C4, np * np * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    dphpc_time3(
        reset(nr, nq, np, A, A_gpu),
        run_doitgen_gpu(nr, nq, np, A_gpu, C4_gpu, sum_gpu),
        preset
    );

    if (ASSERT && strcmp(preset, "S") == 0) {
        cudaMemcpy(A, A_gpu, nr * nq * np * sizeof(double), cudaMemcpyDeviceToHost);
        assertCorrectness(nr, nq, np, A, preset);
    }

    cudaFree(A_gpu);
    cudaFree(C4_gpu);
    cudaFree(sum_gpu);

    free(A);
    free(C4);
}

int main(int argc, char** argv)
{

    int nr = 60;
    int nq = 60;
    int np = 128;
    run_bm(nr, nq, np, "S");
    
    nr = 110;
    nq = 125;
    np = 256;
    run_bm(nr, nq, np, "M");

    nr = 220;
    nq = 250;
    np = 512;
    run_bm(nr, nq, np, "L");

    nr = 220;
    nq = 250;
    np = 270;
    run_bm(nr, nq, np, "paper");

    return 0;
}
