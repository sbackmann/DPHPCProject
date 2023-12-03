#Flops Count for GEMM 
2*K*M*N

#C optimization (All are tested)
gemm_acc_gpu: accumulators 
gemm_unrolledx4_acc_gpu: accumulators and loop unrolling x4 
gemm_coalescing_gpu: global memory coalescing + changing the block size 


#Julia optimization (To be tested)
gemm_no_global_acc_gpu: eliminate global variables + accumulators 
gemm_unrolledx4_acc_gpu: accumulators and loop unrolling x4
gemm_unrolled4x_inbound_gpu: accumulators, loop unrolling x4, inbound macro to skip bounds checks by compiler 
gemm_coalescing_gpu: global memory coalescing + changing the block size  
To be implemented 
gemm_coalescing_unrolledx4_gpu:global memory coalescing + changing the block size + loop unrolling x4 
Maybe vectorized? 