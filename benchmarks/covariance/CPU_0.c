#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

#include "../../timing/dphpc_timing.h"

bool PERFORM_VALIDATION = false;

void kernel(int M, int N, double* data, double* cov) {
    double* mean = calloc(M, sizeof(double));
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            mean[j] += data[i * M + j];
        }
    }

    for (int i = 0; i < M; i++) {
        mean[i] /= (double)N;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            data[i * M + j] -= mean[j];
        }
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = i; j < M; j++) {
            double partial_sum = 0.0;
            for (int k = 0; k < N; k++) {
                partial_sum += data[k * M + i] * data[k * M + j];
            }
            cov[i * M + j] =  partial_sum / ((double)N - 1.0);
            cov[j * M + i] = cov[i * M + j]; 
        }
    }

    free(mean);
}

void run_bm(int M, int N, const char* preset) {
    double *data = malloc(sizeof(double) * N * M);
    double *cov = malloc(sizeof(double) * M * M);

    if (PERFORM_VALIDATION) {
        printf("Running validation for preset: %s \n", preset);
        initialize(M, N, data);
        kernel(M, N, data, cov);
        if (!is_correct(M, cov, preset)) {
            printf("Validation failed for preset: %s \n", preset);
            exit(1);
        } else {
            printf("Validation passed for preset: %s \n", preset);
        }
    }

    dphpc_time3(
        initialize(M, N, data),
        kernel(M, N, data, cov),
        preset
    );

    free(data);
    free(cov);
}


int main() {
    run_bm(500, 600, "S");
    run_bm(1400, 1800, "M");
    // run_bm(3200, 4000, "L"); // too slow
    run_bm(1200, 1400, "paper");

    return 0;
}
