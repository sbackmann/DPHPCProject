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

int is_valid() {
    int n = 200;
    int m = 70;

    double alpha = 3.0;
    double beta = 5.0;
    
    double (*C)[n][n]; C = (double(*)[n][n]) malloc(n*n*sizeof(double));
    double (*A)[n][m]; A = (double(*)[n][m]) malloc(n*m*sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*C)[i][j] = 1;
        }
        for (int j = 0; j < m; j++) {
            (*A)[i][j] = 1;
        }
    } 

    kernel_syrk(n, m, alpha, beta, *C, *A);
    
    free((void*)A);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j <= i && (*C)[i][j] != beta + alpha * m) {
                free((void*)C);
                printf("validation failed");
                return 0;
            }
        }
    } 
    free((void*)C);
    return 1;

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


#include "_main.h"