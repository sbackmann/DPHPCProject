#include <cstdlib>
#include "../include/parallel.h"

int main(int argc, char **argv) {
    parallelHandle_t handle;
    int err;
    long long I = 42;
    long long J = 42;
    long long K = 42;
    double dtr_stage = 42;
    double * __restrict__ u_pos = (double*) calloc(((I * J) * K), sizeof(double));
    double * __restrict__ u_stage = (double*) calloc(((I * J) * K), sizeof(double));
    double * __restrict__ utens = (double*) calloc(((I * J) * K), sizeof(double));
    double * __restrict__ utens_stage = (double*) calloc(((I * J) * K), sizeof(double));
    double * __restrict__ wcon = (double*) calloc(((J * K) * (I + 1)), sizeof(double));


    handle = __dace_init_parallel(I, J, K);
    __program_parallel(handle, u_pos, u_stage, utens, utens_stage, wcon, I, J, K, dtr_stage);
    err = __dace_exit_parallel(handle);

    free(u_pos);
    free(u_stage);
    free(utens);
    free(utens_stage);
    free(wcon);


    return err;
}
