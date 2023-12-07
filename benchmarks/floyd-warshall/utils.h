#ifndef UTILS_H_
#define UTILS_H_

void run_bm(int n, const char* preset, void (*kernel)(int, int*), const int ASSERT);

#endif /* UTILS_H_ */