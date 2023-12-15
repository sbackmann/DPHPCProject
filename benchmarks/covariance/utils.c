#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void printMatrix(int n, int m, double *matrix) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.6lf ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

void initialize(int M, int N, double* data) {
    for (int i = 0; i < N; i++) {  
        for (int j = 0; j < M; j++) {  
            data[i * M + j] = (double)(i * j) / M;
        }
    }
}

bool is_correct(int M, double *cov, const char *preset) {
    char filename[100];  
    snprintf(filename, sizeof(filename), "./test_cases/%s.tsv", preset);


    double *result = (double*)malloc(sizeof(double) * M * M);
    read_values_from_tsv(filename, M * M, result);

    for (int i = 0; i < M * M; i++) {
        if (cov[i] < result[i] - 0.000001 || cov[i] > result[i] + 0.000001) {
            printf("Error: x[%d] = %.6lf != %.6lf\n", i, cov[i], result[i]);
            return false;
        }
    }

    return true;
}

void read_values_from_tsv(const char* filename, int N, double* values) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        if (fscanf(file, "%lf", &values[i]) != 1) {
            fprintf(stderr, "Error reading data from file: %s\n", filename);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}


