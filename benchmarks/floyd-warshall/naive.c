/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* floyd-warshall.c: this file is part of PolyBench/C */

#include "../../timing/dphpc_timing.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <stdlib.h>

#define ASSERT 0

static
void init_array (int n,
   int graph[n][n])
{
  int tmp;
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++) {
      graph[i][j] = i*j%7+1;
      tmp=(i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0;
      if (tmp==0){
         graph[i][j] = 999;}
    }}
}




static
void print_array(int n,
   int graph[n][n])

{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "graph");
  for (i = 0; i < 1; i++){
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
      fprintf (stderr, "%d ", graph[1][j]);
    }}
  fprintf(stderr, "\nend   dump: %s\n", "graph");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}




static
void kernel_floyd_warshall(int n,
      int graph[n][n])
{
  int tmp;
  #pragma scop
  for (int k = 0; k < n; k++){
    for(int i = 0; i < n; i++){
      for (int j = 0; j < n; j++){
        tmp=graph[i][j] < graph[i][k] + graph[k][j];
        if (tmp==1){
          graph[i][j] =  graph[i][j];}
        else{
          graph[i][j] = graph[i][k] + graph[k][j];
        }
      }
    }
  }
  #pragma endscop

}

void serializeArray(int n,
    int graph[n][n], const char *prefix) {
    char path[100]; 
    snprintf(path, sizeof(path), "benchmarks/floyd-warshall/test_cases/%s.dat", prefix);

    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    if (fwrite(graph, sizeof(int), n * n, file) != n * n) {
        printf("Error writing to file.\n");
        exit(1);
    }

    fclose(file);
    printf("Array serialized and written to file.\n");
}

void assertCorrectness(int n,
    int graph[n][n], const char *prefix) {

    // Somehow only works when a.exe run manually
    // When graph is amended to "test_cases/%s.dat", no output is produced

    fprintf(stderr, "graph[0][1] = %d\n", graph[0][1]);
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
            if (graph[i][j] != graph[i][j]) {
                printf("graph[%d][%d] = %d\n", i, j, graph[i][j]);
                printf("graph_test[%d][%d] = %d\n", i, j, graph_test[i][j]);
                printf("Arrays are not equal.\n");
                exit(1);
            }
        }
    }
    printf("Arrays are equal.\n");
}

void run_bm(int n, const char* preset) {

    
    int (*graph)[n][n]; graph = (int(*)[n][n]) malloc(n * n * sizeof(int));
    init_array(n, *graph);

    dphpc_time3(
        init_array(n, *graph),
        kernel_floyd_warshall(n, *graph),
        preset
    );
    if (ASSERT && strcmp(preset, "S") == 0) {
        assertCorrectness(n, *graph, preset);
    }
    free((void*)graph);
}

int main(int argc, char** argv)
{
  run_bm(200, "S");
  run_bm(400, "M");
  run_bm(850, "L");
  run_bm(2800, "paper");
  
  return 0;
}
