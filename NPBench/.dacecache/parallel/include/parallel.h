#include <dace/dace.h>
typedef void * parallelHandle_t;
extern "C" parallelHandle_t __dace_init_parallel(long long I, long long J, long long K);
extern "C" int __dace_exit_parallel(parallelHandle_t handle);
extern "C" void __program_parallel(parallelHandle_t handle, double * __restrict__ u_pos, double * __restrict__ u_stage, double * __restrict__ utens, double * __restrict__ utens_stage, double * __restrict__ wcon, long long I, long long J, long long K, double dtr_stage);
