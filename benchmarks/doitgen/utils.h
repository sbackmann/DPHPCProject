#ifndef UTILS_H_
#define UTILS_H_

#include "_parameters.h"

#define NR_S get_params("S")[0]
#define NQ_S get_params("S")[1]
#define NP_S get_params("S")[2]

#define NR_M get_params("M")[0]
#define NQ_M get_params("M")[1]
#define NP_M get_params("M")[2]

#define NR_L get_params("L")[0]
#define NQ_L get_params("L")[1]
#define NP_L get_params("L")[2]

#define NR_PAPER get_params("paper")[0]
#define NQ_PAPER get_params("paper")[1]
#define NP_PAPER get_params("paper")[2]

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void run_bm(int nr, int nq, int np, const char* preset, void (*kernel)(int, int, int, double*, double*, double*), const int ASSERT);

#endif /* UTILS_H_ */