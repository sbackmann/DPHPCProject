
#include "utils.h"

#define ASSERT 1


__global__ void kernel_floyd_warshall(int n, int *graph) {

  int tmp, tmp1, tmp2, tmp3;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    for (int k = 0; k < n-3; k+=4){
      tmp = graph[i * n + k] + graph[k * n + j];

      tmp1 = graph[i * n + k + 1] + graph[(k + 1) * n + j];
      
      tmp2 = graph[i * n + k + 2] + graph[(k + 2) * n + j];
      
      tmp3 = graph[i * n + k + 3] + graph[(k + 3) * n + j];
      if (tmp > tmp1) tmp = tmp1;
      if (tmp > tmp2) tmp = tmp2;
      if (tmp > tmp3) tmp = tmp3;
      
      if (tmp < graph[i * n + j]) {
        graph[i * n + j] = tmp;
      }
    }

    for (int k = n - 3; k < n; k++) {
      tmp = graph[i * n + k] + graph[k * n + j];
      if (tmp < graph[i * n + j]) {
        graph[i * n + j] = tmp;
      }
    }

  }
}


void run_floyd_warshall_gpu(int n, int *graph) {
  
  int threads = 16;
  dim3 threadsPerBlock(threads, threads);
  dim3 numBlocks(n / threads + 1, n / threads + 1);
  kernel_floyd_warshall<<<numBlocks,threadsPerBlock>>>(n, graph);
  cudaDeviceSynchronize();
}


int main(int argc, char** argv) {
  
  run_bm(N_S, "S", run_floyd_warshall_gpu, ASSERT);
  run_bm(N_M, "M", run_floyd_warshall_gpu, ASSERT);
  run_bm(N_L, "L", run_floyd_warshall_gpu, ASSERT);
  run_bm(N_PAPER, "paper", run_floyd_warshall_gpu, ASSERT);
  
  return 0;
}