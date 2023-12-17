#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "utils.h"
#include "../../timing/dphpc_timing.h"

bool VALIDATE = true;

void kernel(int N, double *L, double *x, double *b) {
    for (int i = 0; i < N; i++) {
        double dp = 0.0;
        for (int j = 0; j < i; j++) {
            dp += L[i * N + j] * x[j];
        }
        x[i] = (b[i] - dp) / L[i * N + i];
    }
}

void run_bm(int N, const char *preset) {
    double *L = malloc(sizeof(double) * N * N);
    double *x = malloc(sizeof(double) * N);
    double *b = malloc(sizeof(double) * N);

    if (VALIDATE) {
        initialize(N, L, x, b);

        printMatrix(N, N, L);

        kernel(N, L, x, b);
        if (!is_correct(N, x, preset)) {
            printf("Validation failed for preset: %s \n", preset);
            exit(1);
        } else {
            printf("Validation passed for preset: %s \n", preset);
        }
    }


    dphpc_time3(
        initialize(N, L, x, b),
        kernel(N, L, x, b),
        preset
    );
    
    free(L);
    free(x);
    free(b);
}

void simple_validate() {
    int N = 4;
    double L[N * N];
    double x[N];
    double b[N];
    initialize(N, L, x, b);
    kernel(N, L, x, b);

    printf("Result (x):\n");
    printMatrix(1, N, x);
}

int main() {
    run_bm(5, "paper");
    // run_bm(2000, "S");
    // run_bm(5000, "M");
    // run_bm(14000, "L");
    // run_bm(16000, "paper");

    return 0;
}

