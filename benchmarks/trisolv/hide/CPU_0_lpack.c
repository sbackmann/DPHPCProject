#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <lapacke.h>

#include "../../timing/dphpc_timing.h"

#define DEV_MODE 1 
#define TIME_MODE 1
#define DEBUG_MODE 0

void reset() {}

void printMatrix(int n, int m, double *matrix) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.6lf ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

void initialize(int N, double *L, double *x, double *b) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            L[i * N + j] = (i + N - j + 1) * 2.0 / N;
        }
        x[i] = -999.0;
        b[i] = (double)i;
    }
}

void kernel(int N, double *L, double *x, double *b) {
    for (int i = 0; i < N; i++) {
        double dp = 0.0;
        for (int j = 0; j < i; j++) {
            dp += L[i * N + j] * x[j];
        }
        x[i] = (b[i] - dp) / L[i * N + i];
    }
}

bool is_correct(int N, double *L, double *x, double *b) {
    int pivots[N];
    int result;

    double L_linear[N * N];
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            if (i > j) {
                L[i * N + j] = 0.0;
            }
            L_linear[i + j * N] = L[i * N + j];
        }
    }

    result = LAPACKE_dgesv(LAPACK_COL_MAJOR, N, 1, L_linear, N, pivots, b, N);

    if (result == 0) {
        for (int i = 0; i < N; i++) {
            x[i] = b[i]; // solution stored in the 'b' array
        }
        return true;
    } else {
        return false;
    }
}

void print_all(int N, double *L, double *x, double *b) {
    printf("L:\n");
    printMatrix(N, N, L);

    printf("x:\n");
    printMatrix(1, N, x);

    printf("b:\n");
    printMatrix(1, N, b);
}

void validate() {
    int N = 4;
    double L[N * N];
    double x[N];
    double b[N];
    initialize(N, L, x, b);

    print_all(N, L, x, b);

    kernel(N, L, x, b);
    printf("Result (x):\n");
    printMatrix(1, N, x);

    if (!is_correct(N, L, x, b)) {
        printf("Validation failed\n");
        exit(1);
    } 
}

void run_bm(int N, const char *preset) {
    double *L = malloc(sizeof(double) * N * N);
    double *x = malloc(sizeof(double) * N);
    double *b = malloc(sizeof(double) * N);

    initialize(N, L, x, b);

    #if DEBUG_MODE
    printf("Data\n");
    printMatrix(N, N, L);

    printf("x:\n");
    printMatrix(1, N, x);

    printf("b:\n");
    printMatrix(1, N, b);
    #endif

    #if TIME_MODE
    dphpc_time3(
        reset(),
        kernel(N, L, x, b),
        preset
    );
    #else
    kernel(N, L, x, b);
    #endif

    
    #if DEBUG_MODE
    printf("Result (x):\n");
    printMatrix(1, N, x);
    #endif

    free(L);
    free(x);
    free(b);
}

int main() {
    validate();

    #if DEV_MODE
        run_bm(5, "S");
        run_bm(6, "M");
    #else
        run_bm(2000, "S");
        run_bm(5000, "M");
        run_bm(14000, "L");
        run_bm(16000, "paper");
    #endif

    return 0;
}
