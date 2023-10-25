#include <dace/dace.h>
typedef void * auto_optHandle_t;
extern "C" auto_optHandle_t __dace_init_auto_opt(long long I, long long J, long long K);
extern "C" int __dace_exit_auto_opt(auto_optHandle_t handle);
extern "C" void __program_auto_opt(auto_optHandle_t handle, double * __restrict__ u_pos, double * __restrict__ u_stage, double * __restrict__ utens, double * __restrict__ utens_stage, double * __restrict__ wcon, long long I, long long J, long long K, double dtr_stage);
