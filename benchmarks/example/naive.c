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

int is_valid() {

    int n = 50;

    double (*A)[n][n]; A = (double(*)[n][n]) malloc(n*n*sizeof(double));
    double (*B)[n][n]; B = (double(*)[n][n]) malloc(n*n*sizeof(double));
    double (*C)[n][n]; C = (double(*)[n][n]) malloc(n*n*sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*A)[i][j] = 1.0;
            (*B)[i][j] = 1.0;
            (*C)[i][j] = 0.0;
        }
    }

    example(n, *A, *B, *C);

    free((void*)B);
    free((void*)A);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((*C)[i][j] != n) {

                free((void*)C);
                return 0;
            }
        }
    }

    free((void*)C);
    return 1;

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
    free((void*)B);
    free((void*)A);
}

#include "_main.h"
