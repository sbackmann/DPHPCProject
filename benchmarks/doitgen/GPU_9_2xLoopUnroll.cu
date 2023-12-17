#include "utils.h"

#define ASSERT 1
#define THREADS 32


__global__ void kernel_doitgen(int nr, int nq, int np,
      double *A,
      double *C4,
      double *sum) {

    int q = blockIdx.x * THREADS + (threadIdx.x / THREADS);
    int p = blockIdx.y * THREADS + (threadIdx.x % THREADS);
    int odd = np % 2 == 1;

    if (p < np && q < nq) {
        for(int r = 0; r < nr; r++) {
            double tempSum = 0.0;
            for (int s = 0; s < np-1; s+=2) {
                tempSum += A[r * nq * np + q * np + s] * C4[s * np + p];
                tempSum += A[r * nq * np + q * np + s+1] * C4[(s+1) * np + p];
            }
            if (odd) {
                tempSum += A[r * nq * np + q * np + np-1] * C4[(np-1) * np + p];
            }
            sum[r * nq * np + q * np + p] = tempSum;
        }        
    }
}


void run_doitgen_gpu(int nr, int nq, int np,
      double *A,
      double *C4,
      double *sum) {

    dim3 threadsPerBlock(THREADS * THREADS);
    dim3 numBlocks(CEIL_DIV(nq, THREADS), CEIL_DIV(np, THREADS));
    kernel_doitgen<<<numBlocks,threadsPerBlock>>>(nr, nq, np, A, C4, sum);
    cudaMemcpy(A, sum, nr * nq * np * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}


int main(int argc, char** argv) {

    int nr = NR_S; int nq = NQ_S; int np = NP_S;
    run_bm(nr, nq, np, "S", run_doitgen_gpu, ASSERT);
    
    nr = NR_M; nq = NQ_M; np = NP_M;
    run_bm(nr, nq, np, "M", run_doitgen_gpu, ASSERT);

    nr = NR_L; nq = NQ_L; np = NP_L;
    run_bm(nr, nq, np, "L", run_doitgen_gpu, ASSERT);

    nr = NR_PAPER; nq = NQ_PAPER; np = NP_PAPER;
    run_bm(nr, nq, np, "paper", run_doitgen_gpu, ASSERT);

    return 0;
}
