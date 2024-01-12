
naive:
	gcc naive.c -O3 -march=native -o ./a.exe -lm 
	./a.exe

# naive_col_major:
# 	gcc naive_col_major.c -O3 -o ./a.exe -lm 
# 	./a.exe

optimized_1: 
	gcc optimized_1.c -O3 -march=native -o ./a.exe -lm 
	./a.exe

optimized_col_major: 
	gcc optimized_col_major.c -O3 -march=native -o ./a.exe -lm 
	./a.exe
	
naive_gpu:
	nvcc -o ./a.exe gemm_naive_gpu.cu
	./a.exe

# GPU_acc:
# 	nvcc -o ./a.exe GPU_acc.cu
# 	./a.exe

GPU_coalescing:
	nvcc -o ./a.exe GPU_coalescing.cu
	./a.exe

GPU_unrolledx4:
	nvcc -o ./a.exe GPU_unrolledx4.cu
	./a.exe

GPU_col_major:
	nvcc -o ./a.exe GPU_col_major.cu
	./a.exe