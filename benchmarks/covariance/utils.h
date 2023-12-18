#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>

void printMatrix(int n, int m, double *matrix);
void initialize(int M, int N, double* data);

void read_values_from_tsv(const char* filename, int N, double* values);
bool is_correct(int N, double *x, const char *preset);

#endif  // UTILS_H
