
no_dead_code_elimination = -fno-dce -fno-dse -fno-tree-dce -fno-tree-dse

# every rule compiles and runs a version of the benchmark
naive:
	gcc naive.c -O3 -o ./a.exe -lm
	./a.exe

GPU_naive:
	nvcc -o ./a.exe GPU_naive.cu
	./a.exe

GPU_optimized:
	nvcc -o ./a.exe GPU_optimized.cu
	./a.exe

GPU_test:
	nvcc -o ./a.exe GPU_test.cu
	./a.exe