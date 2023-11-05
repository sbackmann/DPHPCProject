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

static
void init_array (int n,
   int path[n][n])
{
  int tmp;
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++) {
      path[i][j] = i*j%7+1;
      tmp=(i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0;
      if (tmp==0){
         path[i][j] = 999;}
    }}
}




static
void print_array(int n,
   int path[n][n])

{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "path");
  for (i = 0; i < n; i++){
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
      fprintf (stderr, "%d ", path[i][j]);
    }}
  fprintf(stderr, "\nend   dump: %s\n", "path");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}




static
void kernel_floyd_warshall(int n,
      int path[n][n])
{
  int tmp;
  #pragma scop
  for (int k = 0; k < n; k++){
    for(int i = 0; i < n; i++){
      for (int j = 0; j < n; j++){
        tmp=path[i][j] < path[i][k] + path[k][j];
        if (tmp==1){
          path[i][j] =  path[i][j];}
        else{
          path[i][j] = path[i][k] + path[k][j];
        }
      }
    }
  }
  #pragma endscop

}

void run_bm(int n, const char* preset) {

    
    int (*path)[n][n]; path = (int(*)[n][n]) malloc(n * n * sizeof(int));
    init_array(n, *path);

    dphpc_time3(
        init_array(n, *path),
        kernel_floyd_warshall(n, *path),
        preset
    );

    free((void*)path);
}

int main(int argc, char** argv)
{
  run_bm(200, "S");
  run_bm(400, "M");
  run_bm(850, "L");
  run_bm(2800, "paper");
  /*
  int N[4] = {200, 400, 850, 2800};
  char sizes[4][10] = {"S", "M", "L", "paper"};
  for (int i = 0; i < 2; i++) {
    int n = N[i];
    char size[10] = "";
    strcpy(size, sizes[i]);

    int (*path)[2800 + 0][2800 + 0]; path = (int(*)[2800 + 0][2800 + 0])malloc ((2800 + 0) * (2800 + 0)* sizeof(int));;
    dphpc_time3(
      ,
      floyd_warshall(n, *path),
      size
    );

    if (argc > 42 && ! strcmp(argv[0], ""))
      print_array(n, *path);

    free((void*)path);;
  }*/
  return 0;
}
