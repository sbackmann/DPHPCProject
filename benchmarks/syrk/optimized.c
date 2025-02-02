#include "../../timing/dphpc_timing.h"

#define get(A, ncols, r, c) A[(r)*(ncols)+(c)]

void init_array(int n, int m, double *alpha, double *beta,
    double* C, double* A)
{
    *alpha = 1.5;
    *beta = 1.2;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            get(A, m, i, j) = (double) ((i*j+1)%n) / n;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            get(C, n, i, j) = (double) ((i*j+2)%m) / m;
        }
    }
    
}

void kernel_syrk(
    int n, int m, 
    double alpha, double beta, 
    double * restrict C, const double * A)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            get(C, n, i, j) = get(C, n, i, j) * beta;
        }
        for (int k = 0; k < m; k++) {
            for (int j = 0; j <= i; j++) {
                get(C, n, i, j) = get(C, n, i, j) + alpha * get(A, m, i, k) * get(A, m, j, k);
            }
        }
    }
}



void run_bm(int n, int m, const char* preset) {

    double alpha;
    double beta;
    
    double* C = malloc(n*n*sizeof(double));
    double* A = malloc(n*m*sizeof(double));

    dphpc_time3(
        init_array(n, m, &alpha, &beta, C, A),
        kernel_syrk(n, m, alpha, beta, C, A),
        preset
    );
    // for(int i = 0; i < n; i++) {
    //     printf("%f, ", get(C, n, n-1, i));
    // }

    free((void*)C);
    free((void*)A);
}


int is_valid() {
    int n = 200;
    int m = 70;

    double alpha = 3.0;
    double beta = 5.0;
    
    double* C = malloc(n*n*sizeof(double));
    double* A = malloc(n*m*sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            get(C, n, i, j) = 1;
        }
        for (int j = 0; j < m; j++) {
            get(A, m, i, j) = 1;
        }
    } 

    kernel_syrk(n, m, alpha, beta, C, A);
    
    free((void*)A);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j <= i && get(C, n, i, j) != beta + alpha * m) {
                free((void*)C);
                printf("validation failed");
                return 0;
            }
        }
    } 

    free((void*)C);
    return 1;

}


// "S": { "M": 50, "N": 70 },
// "M": { "M": 150, "N": 200 },
// "L": { "M": 500, "N": 600 },
// "paper": { "M": 1000, "N": 1200 }


#include "_main.h"
