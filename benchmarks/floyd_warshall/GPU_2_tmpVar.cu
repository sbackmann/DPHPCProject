#include "utils.h"

#define ASSERT 1

__global__ void kernel_floyd_warshall(int n, int *graph, int k) {

  int tmp;
  int tmp1;
  int tmp2;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    tmp1 = graph[i * n + j];
    tmp2 = graph[i * n + k] + graph[k * n + j];
    tmp = tmp1 < tmp2;
    if (tmp==1){
      graph[i * n + j] = tmp1;}
    else {
      graph[i * n + j] = tmp2;
    }
  }
}


void run_floyd_warshall_gpu(int n, int *graph) {
  
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(n / 16 + 1, n / 16 + 1);
  for (int k = 0; k < n; k++) {
    kernel_floyd_warshall<<<numBlocks,threadsPerBlock>>>(n, graph, k);
    cudaDeviceSynchronize();
  }
}
/*
__global__ void kernel_floyd_warshall(int n, int *graph) {

  int tmp;
  int tmp1;
  int tmp2;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    for (int k = 0; k < n; k++){
      tmp1 = graph[i * n + j];
      tmp2 = graph[i * n + k] + graph[k * n + j];
      tmp = tmp1 < tmp2;
        if (tmp==1){
          graph[i * n + j] = tmp1;}
        else {
          graph[i * n + j] = tmp2;
        }
    }
  }
}


void run_floyd_warshall_gpu(int n, int *graph) {
  
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(n / 16 + 1, n / 16 + 1);
  kernel_floyd_warshall<<<numBlocks,threadsPerBlock>>>(n, graph);
  cudaDeviceSynchronize();
}
*/
int main(int argc, char** argv) {
  
  run_bm(N_S, "S", run_floyd_warshall_gpu, ASSERT);
  run_bm(N_M, "M", run_floyd_warshall_gpu, ASSERT);
  run_bm(N_L, "L", run_floyd_warshall_gpu, ASSERT);
  run_bm(N_PAPER, "paper", run_floyd_warshall_gpu, ASSERT);
  
  return 0;
}
