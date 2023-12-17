#include "utils.h"

#define ASSERT 1
#define THREADS 32


__global__ void kernel_doitgen(int nr, int nq, int np,
      double *A,
      double *C4,
      double *sum) {
                    
        __shared__ double As[THREADS * THREADS];
        __shared__ double C4s[THREADS * THREADS];

        const int threadCol = threadIdx.x % THREADS;
        const int threadRow = threadIdx.x / THREADS;

        //if (blockIdx.x * BLOCKSIZE + threadRow < nq && blockIdx.y * BLOCKSIZE + threadCol < np) {
            for(int r = 0; r < nr; r++) {
                int q = blockIdx.x * THREADS * np;
                int p = blockIdx.y * THREADS;
                int res = blockIdx.x * THREADS * np + blockIdx.y * THREADS;
                double tempSum = 0.0;
                for (int blkIdx = 0; blkIdx < np; blkIdx += THREADS) {
                    if (blkIdx + threadCol < np && blockIdx.x * THREADS + threadRow < nq)
                        As[threadRow * THREADS + threadCol] = A[r * nq * np + q + threadRow * np + threadCol];
                    else
                        As[threadRow * THREADS + threadCol] = 0.0;
                    if (blkIdx + threadRow < np && blockIdx.y * THREADS + threadCol < np)
                        C4s[threadRow * THREADS + threadCol] = C4[p + threadRow * np + threadCol];
                    else
                        C4s[threadRow * THREADS + threadCol] = 0.0;
                        
                    __syncthreads();

                    q += THREADS;
                    p += THREADS * np;

                    for (int s = 0; s < THREADS; ++s) {
                        tempSum += As[threadRow * THREADS + s] * C4s[s * THREADS + threadCol];
                    }
                    __syncthreads();
                    
                }
                
                sum[r * nq * np + res + threadRow * np + threadCol] = tempSum;
            }
        
        
}


void run_doitgen_gpu(int nr, int nq, int np,
      double *A,
      double *C4,
      double *sum) {

    dim3 threadsPerBlock(32 * 32);
    int n = max(nq, np);
    dim3 numBlocks(CEIL_DIV(nq, 32), CEIL_DIV(np, 32));
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
