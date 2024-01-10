
#include "utils.h"
#include <stdio.h>

#define ASSERT 1
#define BLOCK_SIZE 32
#define MAX_DIST 1000000


__global__ void kernel_floyd_warshall_1(int n, int *graph, const int blockId) {

  __shared__ int graphShared[BLOCK_SIZE][BLOCK_SIZE];

  int idx = threadIdx.x;
  int idy = threadIdx.y;

  int i = BLOCK_SIZE * blockId + idy;
  int j = BLOCK_SIZE * blockId + idx;

  if (i < n && j < n) {
      graphShared[idy][idx] = graph[i * n + j];
  } else {
      graphShared[idy][idx] = MAX_DIST;
  }
  __syncthreads();

  int tmp;
  for (int k = 0; k < BLOCK_SIZE; k++) {
      tmp = graphShared[idy][k] + graphShared[k][idx];
      __syncthreads();
      if (tmp < graphShared[idy][idx]) {
          graphShared[idy][idx] = tmp;
      }
      __syncthreads();
  }

  if (i < n && j < n) {
      graph[i * n + j] = graphShared[idy][idx];
  }

}

static __global__ void kernel_floyd_warshall_2(int n, int* graph, const int blockId) {
    if (blockIdx.x == blockId) {
        return;
    }

    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int i = BLOCK_SIZE * blockId + idy;
    int j = BLOCK_SIZE * blockId + idx;

    __shared__ int graphSharedTmp[BLOCK_SIZE][BLOCK_SIZE];

    if (i < n && j < n) {
        graphSharedTmp[idy][idx] = graph[i * n + j];
    } else {
        graphSharedTmp[idy][idx] = MAX_DIST;
    }

    if (blockIdx.y == 0) {
        j = BLOCK_SIZE * blockIdx.x + idx;
    } else {
        i = BLOCK_SIZE * blockIdx.x + idy;
    }

    int bestRoute;

    if (i < n && j < n) {
        bestRoute = graph[i * n + j];
    } else {
        bestRoute = MAX_DIST;
    }
    __shared__ int graphShared[BLOCK_SIZE][BLOCK_SIZE];
    graphShared[idy][idx] = bestRoute;
    __syncthreads();

    int tmp;

    if (blockIdx.y == 0) {
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp = graphSharedTmp[idy][k] + graphShared[k][idx];

            if (tmp < bestRoute) {
                bestRoute = tmp;
            }

            __syncthreads();
            graphShared[idy][idx] = bestRoute;
            __syncthreads();
        }
    } else {
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp = graphShared[idy][k] + graphSharedTmp[k][idx];

            if (tmp < bestRoute) {
                bestRoute = tmp;
            }

            __syncthreads();
            graphShared[idy][idx] = bestRoute;
            __syncthreads();
        }
    }

    if (i < n && j < n) {
        graph[i * n + j] = bestRoute;
    }
}

static __global__
void kernel_floyd_warshall_3(int n, int* graph, const int blockId) {
    if (blockIdx.x == blockId || blockIdx.y == blockId) {
        return;
    }
    __shared__ int graphSharedTmpRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int graphSharedTmpCol[BLOCK_SIZE][BLOCK_SIZE];

    int idx = threadIdx.x;
    int idy = threadIdx.y;

    int i = blockDim.y * blockIdx.y + idy;
    int j = blockDim.x * blockIdx.x + idx;

    int iRow = BLOCK_SIZE * blockId + idy;
    int jCol = BLOCK_SIZE * blockId + idx;

    if (i  < n && jCol < n) {
        graphSharedTmpCol[idy][idx] = graph[i * n + jCol];
    }
    else {
        graphSharedTmpCol[idy][idx] = MAX_DIST;
    }

    if (iRow < n && j < n) {

        graphSharedTmpRow[idy][idx] = graph[iRow * n + j];
    }
    else {
        graphSharedTmpRow[idy][idx] = MAX_DIST;
    }

    __syncthreads();

    if (i  < n && j < n) {

        int bestRoute = graph[i * n + j];
        int tmp;
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp = graphSharedTmpCol[idy][k] + graphSharedTmpRow[k][idx];
            if (bestRoute > tmp) {
                bestRoute = tmp;
            }
        }
        graph[i * n + j] = bestRoute;
    }
}

void run_floyd_warshall_gpu(int n, int* graph) {

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks1(1, 1);
    dim3 numBlocks2(CEIL_DIV(n, BLOCK_SIZE), 2);
    dim3 numBlocks3(CEIL_DIV(n, BLOCK_SIZE), CEIL_DIV(n, BLOCK_SIZE));

    int totalBlocks = CEIL_DIV(n, BLOCK_SIZE);

    for(int blockID = 0; blockID < totalBlocks; blockID++) {
        kernel_floyd_warshall_1<<<numBlocks1, threadsPerBlock>>>(n, graph, blockID);

        kernel_floyd_warshall_2<<<numBlocks2, threadsPerBlock>>>(n, graph, blockID);

        kernel_floyd_warshall_3<<<numBlocks3, threadsPerBlock>>>(n, graph, blockID);
    }
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
  
  run_bm(N_S, "S", run_floyd_warshall_gpu, ASSERT);
  run_bm(N_M, "M", run_floyd_warshall_gpu, ASSERT);
  run_bm(N_L, "L", run_floyd_warshall_gpu, ASSERT);
  run_bm(N_PAPER, "paper", run_floyd_warshall_gpu, ASSERT);
  
  return 0;
}