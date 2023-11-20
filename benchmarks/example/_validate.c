
// use all the libraries you need to here, but keep the stuff that is timed minimal

#define MAIN_HANDLED

#ifdef VALIDATE_NAIVE
#include "naive.c"

int main() {

    int n = 30;

    double (*A)[n][n]; A = (double(*)[n][n]) malloc(n*n*sizeof(double));
    double (*B)[n][n]; B = (double(*)[n][n]) malloc(n*n*sizeof(double));
    double (*C)[n][n]; C = (double(*)[n][n]) malloc(n*n*sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            (*A)[i][j] = 1.0;
            (*B)[i][j] = 1.0;
            (*C)[i][j] = 0.0;
        }
    }

    example(n, *A, *B, *C);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((*C)[i][j] != n) {
                printf("naive.c VALIDATION FAILED\n");
                return 1;
            }
        }
    }
    printf("naive.c VALIDATION SUCCESS\n");
    return 0;
}
#endif // VALIDATE_NAIVE
