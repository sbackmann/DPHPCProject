
naive:
	gcc naive.c -O3 -o ./a.exe -lm 
	./a.exe

optimized: 
	gcc optimized_1.c -O3 -o ./a.exe -lm 
	./a.exe

naive_gpu:
	nvcc -o ./a.exe gemm_naive_gpu.cu
	./a.exe

GPU_acc:
	nvcc -o ./a.exe GPU_acc.cu
	./a.exe

GPU_coalescing:
	nvcc -o ./a.exe GPU_coalescing.cu
	./a.exe

GPU_unrolledx4_acc:
	nvcc -o ./a.exe GPU_unrolledx4_acc.cu
	./a.exe