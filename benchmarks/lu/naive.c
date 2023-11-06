#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"

void init_array(int N, double A[N][N]) {

  for (int i = 0; i < N; i++) {
    for (int j = 0; j <= i; j++) {
      A[i][j] = (double)(-j % N) / N + 1;
    }
    for (int j = i + 1; j < N; j++) {
      A[i][j] = 0;
    }
    A[i][i] = 1;
  }

  double (*B)[N][N];
  B = (double(*)[N][N])malloc(N*N* sizeof(double));

  for (int r = 0; r < N; ++r)
    for (int s = 0; s < N; ++s)
      (*B)[r][s] = 0;
  for (int t = 0; t < N; ++t)
    for (int r = 0; r < N; ++r)
      for (int s = 0; s < N; ++s)
        (*B)[r][s] = (*B)[r][s] + (A[r][t] * A[s][t]);

  for (int r = 0; r < N; ++r)
    for (int s = 0; s < N; ++s)
      A[r][s] = (*B)[r][s];

  free((void*)B);

}


void lu(int N, double A[N][N]) {

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++) {
        A[i][j] = A[i][j] - (A[i][k] * A[k][j]);
      }
      A[i][j] = A[i][j] / A[j][j];
    }
    for (int j = i; j < N; j++) {
      for (int k = 0; k < i; k++) {
        A[i][j] = A[i][j] - (A[i][k] * A[k][j]);
      }
    }
  }

}



void run_bm(int N, const char* preset) {

    
   double (*A)[N][N]; 
    A = (double(*)[N][N]) malloc(N*N*sizeof(double));


    dphpc_time3(
       init_array(N, *A),
       lu(N, *A),
        preset
    );

    free((void*)A); 
  
}


int main(int argc, char** argv) {

    run_bm(50, "S"); 
    run_bm(600, "M"); 
    run_bm(1500, "L"); 
    run_bm(2000, "paper"); 

  return 0;
}
