#ifndef UTILS_H_
#define UTILS_H_

#include "_parameters.h"

#define N_S get_params("S")[0]
#define N_M get_params("M")[0]
#define N_L get_params("L")[0]
#define N_PAPER get_params("paper")[0]


void run_bm(int n, const char* preset, void (*kernel)(int, int*), const int ASSERT);

#endif /* UTILS_H_ */