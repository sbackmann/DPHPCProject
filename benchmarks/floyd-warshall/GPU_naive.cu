

#include "../../timing/dphpc_timing.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <stdlib.h>

#define ASSERT 0

static
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



/*
static
void print_array(int n,
   int *graph)

{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "graph");
  for (i = 0; i < 1; i++){
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
      fprintf (stderr, "%d ", graph[i * n + j]);
    }}
  fprintf(stderr, "\nend   dump: %s\n", "graph");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}
*/
void assertCorrectness(int n,
    int *graph, const char *prefix) {

    // Somehow only works when a.exe run manually
    // When graph is amended to "test_cases/%s.dat", no output is produced

    fprintf(stderr, "graph[0][1] = %d\n", graph[1]);
    char path[100];
    snprintf(path, sizeof(path), "benchmarks/floyd-warshall/test_cases/%s.dat", prefix);
    printf("Reading from %s\n", path);
    FILE *file = fopen(path, "rb");
    printf("Array deserialized.\n");
    if (!file)
        perror("fopen");

    int graph_test[n][n];

    if (fread(graph_test, sizeof(int), n * n, file) != n * n) {
        printf("Error reading file.\n");
        exit(1);
    }

    fclose(file);
    // Perform assertion
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graph[i * n + j] != graph_test[i][j]) {
                printf("graph[%d][%d] = %d\n", i, j, graph[i * n + j]);
                printf("graph_test[%d][%d] = %d\n", i, j, graph_test[i][j]);
                printf("Arrays are not equal.\n");
                exit(1);
            }
        }
    }
    printf("Arrays are equal.\n");
}

__global__ void kernel_floyd_warshall(int n, int *graph) {

  int tmp;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n && j < n) {
    for (int k = 0; k < n; k++){
      tmp = graph[i * n + j] < graph[i * n + k] + graph[k * n + j];
        if (tmp==1){
          graph[i * n + j] =  graph[i * n + j];}
        else{
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


void run_bm(int n, const char* preset) {

    
  int *graph;
  graph = (int*) malloc(n * n * sizeof(int));

  int *graph_gpu;
  cudaMalloc((void**) &graph_gpu, n * n * sizeof(int));

  init_array(n, graph);
  cudaMemcpy(graph_gpu, graph, n * n * sizeof(int), cudaMemcpyHostToDevice);

  dphpc_time3(
      cudaMemcpy(graph_gpu, graph, n * n * sizeof(int), cudaMemcpyHostToDevice),
      run_floyd_warshall_gpu(n, graph_gpu),
      preset
  );
  if (ASSERT && strcmp(preset, "S") == 0) {
      cudaMemcpy(graph, graph_gpu, n * n * sizeof(int), cudaMemcpyDeviceToHost);
      assertCorrectness(n, graph, preset);
  }
  cudaFree(graph_gpu);
  free(graph);
}

int main(int argc, char** argv)
{
  run_bm(200, "S");
  run_bm(400, "M");
  run_bm(850, "L");
  run_bm(2800, "paper");
  
  return 0;
}
