#ifndef UTILS_H_
#define UTILS_H_

#define NR_S 60
#define NQ_S 60
#define NP_S 128

#define NR_M 110
#define NQ_M 125
#define NP_M 256

#define NR_L 220
#define NQ_L 250
#define NP_L 512

#define NR_PAPER 220
#define NQ_PAPER 250
#define NP_PAPER 270

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void run_bm(int nr, int nq, int np, const char* preset, void (*kernel)(int, int, int, double*, double*, double*), const int ASSERT);

#endif /* UTILS_H_ */