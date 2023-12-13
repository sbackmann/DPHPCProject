#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

#define DEV_MODE 0
#define TIME_MODE 1
#define DEBUG_MODE 0

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

void print_all(int N, double *L, double *x, double *b) {
    printf("L:\n");
    printMatrix(N, N, L);

    printf("x:\n");
    printMatrix(1, N, x);

    printf("b:\n");
    printMatrix(1, N, b);
}


bool is_correct(int N, double *x, const char *preset) {
    char filename[100];  
    snprintf(filename, sizeof(filename), "./test_cases/%s.tsv", preset);


    double *result = malloc(sizeof(double) * N);
    read_values_from_tsv(filename, N, result);

    for (int i = 0; i < N; i++) {
        if (x[i] < result[i] - 0.000001 || x[i] > result[i] + 0.000001) {
            printf("Error: x[%d] = %.6lf != %.6lf\n", i, x[i], result[i]);
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


