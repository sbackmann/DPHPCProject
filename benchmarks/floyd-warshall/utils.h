#ifndef UTILS_H_
#define UTILS_H_

#define N_S 200
#define N_M 400
#define N_L 850
#define N_PAPER 2800


void run_bm(int n, const char* preset, void (*kernel)(int, int*), const int ASSERT);

#endif /* UTILS_H_ */