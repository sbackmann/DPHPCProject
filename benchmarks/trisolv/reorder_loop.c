#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "utils.h"
#include "../../timing/dphpc_timing.h"
#include "_parameters.h"

bool VALIDATE = false;

void kernel(int N, double *L, double * restrict x, double *b) {
    x[0] = b[0] / L[0];
    for (int i = 2; i <= N; i++) {
        int d = i-1;
        double s = x[d-1];
        for (int j = i; j <= N; j++) {
            x[j-1] += s * L[(j-1)*N + d-1];
        }
        x[i-1] = (b[i-1] - x[i-1]) / L[(i-1) * N + (i-1)];
    }
}

void run_bm(int N, const char *preset) {
    double *L = malloc(sizeof(double) * N * N);
    double *x = malloc(sizeof(double) * N);
    double *b = malloc(sizeof(double) * N);

    if (VALIDATE) {
        initialize(N, L, x, b);

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
    const char *presets[] = {"S", "M", "L", "paper"};
    for (int i = 0; i < 4; i++) {
        const char* preset = presets[i];
        int n = get_params(preset)[0];
        run_bm(n, preset);
    }

    return 0;
}

