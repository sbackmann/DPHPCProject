#include "../../timing/dphpc_timing.h"



void init_array(int n, double A[n][n], double B[n][n], double out[n][n])
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = (double) ((i+1)*(j+1)*3 % n);
            B[i][j] = (double) ((i+1)*(j+1)*7 % n);
            out[i][j] = 0.0;
        }
    }
}

void example(
    int n, 
    double A[n][n], double B[n][n], double out[n][n])
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                out[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void run_bm(int n, const char* preset) {

    double (*A)[n][n]; A = (double(*)[n][n]) malloc(n*n*sizeof(double));
    double (*B)[n][n]; B = (double(*)[n][n]) malloc(n*n*sizeof(double));
    double (*C)[n][n]; C = (double(*)[n][n]) malloc(n*n*sizeof(double));
    

    dphpc_time3(
        init_array(n, *A, *B, *C),
        example(n, *A, *B, *C),
        preset
    );

    free((void*)C);
    free((void*)A);
}

#ifndef MAIN_HANDLED
int main(int argc, char** argv)
{   

    run_bm(100, "S");
    run_bm(400, "M");
    run_bm(800, "L");
    run_bm(1600, "paper");

    return 0;
}
#endif
