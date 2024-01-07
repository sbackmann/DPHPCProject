#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"
#include "_parameters.h"

void init_array(int N, double A[N][N]) {

  // create lower triangle of matrix 
  for (int i = 0; i < N; i++) {
    // initialize the lower triangle 
    for (int j = 0; j <= i; j++) {
      A[i][j] = (double)(-j % N) / N + 1;
    }
    // set upper triangle to zero 
    for (int j = i + 1; j < N; j++) {
      A[i][j] = 0;
    }
    // set elements on the diagonal to 1 
    A[i][i] = 1;
  }

  double (*B)[N][N];
  B = (double(*)[N][N])malloc(N*N* sizeof(double));

  for (int r = 0; r < N; ++r)
    for (int s = 0; s < N; ++s)
      (*B)[r][s] = 0;

  // multiply A by A^T and save the result in B
  // result is a symmetric matrix 
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

// first nested loop calculates the upper triangle
// second nested loop calculates the lower trinagle 
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


void print2DArray(int N, double arr[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf\t", arr[i][j]);
        }
        printf("\n");
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

    // for testing
    // int N = 30;
    // double (*A)[N][N]; 
    // A = (double(*)[N][N]) malloc(N*N*sizeof(double));

    // init_array(N, *A);
    // lu(N,*A);
    // print2DArray(N, *A); 

    const char *presets[] = {"S", "M", "L", "paper"};

    for (int i = 0; i < 4; i++) {
        const char* preset = presets[i];
        int n = get_params(preset)[0];
        run_bm(n, preset);
    }

    return 0;
}
