
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
          graph[i * n + j] = graph[i * n + j];}
        else {
          graph[i * n + j] = graph[i * n + k] + graph[k * n + j];
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

int main(int argc, char** argv)
{
  run_bm(200, "S", run_floyd_warshall_gpu, ASSERT);
  run_bm(400, "M", run_floyd_warshall_gpu, ASSERT);
  run_bm(850, "L", run_floyd_warshall_gpu, ASSERT);
  run_bm(2800, "paper", run_floyd_warshall_gpu, ASSERT);
  
  return 0;
}