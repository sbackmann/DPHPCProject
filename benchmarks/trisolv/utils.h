#ifndef UTILS_H
#define UTILS_H

#include "_parameters.h"

#include <stdbool.h>

void printMatrix(int n, int m, double *matrix);
void initialize(int N, double *L, double *x, double *b);
void print_all(int N, double *L, double *x, double *b);

void read_values_from_tsv(const char* filename, int N, double* values);
bool is_correct(int N, double *x, const char *preset);

#endif  // UTILS_H
