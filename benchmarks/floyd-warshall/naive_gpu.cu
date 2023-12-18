
#include "utils.h"

#define ASSERT 1


__global__ void kernel_floyd_warshall(int n, int *graph) {

  int tmp;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    for (int k = 0; k < n; k++){
      tmp = graph[i * n + j] < graph[i * n + k] + graph[k * n + j];
        if (tmp==1){
          graph[i * n + j] = graph[i * n + j];
        }
        else {
          graph[i * n + j] = graph[i * n + k] + graph[k * n + j];
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