
#include "utils.h"

#define ASSERT 1


__global__ void kernel_floyd_warshall(int n, int *graph) {

  int tmp, tmp1, tmp2, tmp3;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

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
    tmp = graph[i * n + (n - 1)] + graph[(n - 1) * n + j];
      if (tmp < graph[i * n + j]) {
        graph[i * n + j] = tmp;
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