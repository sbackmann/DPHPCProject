#include "../../timing/dphpc_timing.h"



void init_array(int n, int m, double *alpha, double *beta,
    double C[n][n], double A[n][m])
{
    *alpha = 1.5;
    *beta = 1.2;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i][j] = (double) ((i*j+1)%n) / n;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = (double) ((i*j+2)%m) / m;
        }
    }
    
}

void kernel_syrk(
    int n, int m, 
    double alpha, double beta, 
    double C[n][n], double A[n][m])
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            C[i][j] = C[i][j] * beta;
        }
        for (int k = 0; k < m; k++) {
            for (int j = 0; j <= i; j++) {
                C[i][j] = C[i][j] + alpha * A[i][k] * A[j][k];
            }
        }
    }
}

void run_bm(int n, int m, const char* preset) {

    double alpha;
    double beta;
    
    double (*C)[n][n]; C = (double(*)[n][n]) malloc(n*n*sizeof(double));
    double (*A)[n][m]; A = (double(*)[n][m]) malloc(n*m*sizeof(double));

    dphpc_time3(
        init_array(n, m, &alpha, &beta, *C, *A),
        kernel_syrk(n, m, alpha, beta, *C, *A),
        preset
    );

    free((void*)C);
    free((void*)A);
}

// "S": { "M": 50, "N": 70 },
// "M": { "M": 150, "N": 200 },
// "L": { "M": 500, "N": 600 },
// "paper": { "M": 1000, "N": 1200 }


int main(int argc, char** argv)
{   
    // run_bm(5, 3, "missing");
    run_bm(70, 50, "S");
    run_bm(200, 150, "M");
    run_bm(600, 500, "L");
    run_bm(1200, 1000, "paper");

    return 0;
}
