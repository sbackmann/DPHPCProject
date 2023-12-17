
#include "utils.h"
#include "../../timing/dphpc_timing.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <stdlib.h>


void init_array (int n,
   int *graph)
{
  int tmp;
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++) {
      graph[i * n + j] = i*j%7+1;
      tmp=(i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0;
      if (tmp==0){
         graph[i * n + j] = 999;}
    }}
}

__global__ void kernel_floyd_warshall_test(int n, int *graph) {

  int tmp;
  int tmp1;
  int tmp2;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

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

void run_floyd_warshall_gpu_test(int n, int *graph) {
  
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(n / 16 + 1, n / 16 + 1);
  kernel_floyd_warshall_test<<<numBlocks,threadsPerBlock>>>(n, graph);
  cudaDeviceSynchronize();
}


void assertCorrectness(int n,
    int *graph, const char *prefix) {

    fprintf(stderr, "Preset %s, graph[0][1] = %d\n", prefix, graph[1]);
    int *graph_test;
    graph_test = (int*) malloc(n * n * sizeof(int));

    int *graph_test_gpu;
    cudaMalloc((void**) &graph_test_gpu, n * n * sizeof(int));

    init_array(n, graph_test);
    cudaMemcpy(graph_test_gpu, graph_test, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    run_floyd_warshall_gpu_test(n, graph_test_gpu);
    cudaMemcpy(graph_test, graph_test_gpu, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    
    // Perform assertion
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graph[i * n + j] != graph_test[i * n + j]) {
                fprintf(stderr, "graph[%d][%d] = %d\n", i, j, graph[i * n + j]);
                fprintf(stderr, "graph_test[%d][%d] = %d\n", i, j, graph_test[i* n + j]);
                fprintf(stderr, "Arrays are not equal.\n");
                exit(1);
            }
        }
    }
    fprintf(stderr, "Arrays are equal.\n");
    cudaFree(graph_test_gpu);
    free(graph_test);
}

void reset(int n, int *graph, int *graph_gpu) {
    cudaMemcpy(graph_gpu, graph, n * n * sizeof(int), cudaMemcpyHostToDevice),
    cudaDeviceSynchronize();
}

void run_bm(int n, const char* preset, void (*kernel)(int, int*), int ASSERT) {                                   
  int *graph;                                                                       
  graph = (int*) malloc(n * n * sizeof(int));                                       
  init_array(n, graph);                                                             
  int *graph_gpu;                                                                   
  cudaMalloc((void**) &graph_gpu, n * n * sizeof(int));                      
  cudaDeviceSynchronize();                                                          
                                                                                    
  dphpc_time3(                                                                      
      reset(n, graph, graph_gpu),                                                   
      (*kernel)(n, graph_gpu),                                         
      preset                                                                        
  );                                                                                
                                                                                    
  if (ASSERT && should_run_preset(preset)) {                                        
      cudaMemcpy(graph, graph_gpu, n * n * sizeof(int), cudaMemcpyDeviceToHost);    
      assertCorrectness(n, graph, preset);                                          
  }                                                                                 
                                                                                    
  cudaFree(graph_gpu);                                                              
  free(graph);                                                                      
}
