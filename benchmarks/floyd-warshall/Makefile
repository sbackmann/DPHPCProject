no_dead_code_elimination = -fno-dce -fno-dse -fno-tree-dce -fno-tree-dse

# every rule compiles and runs a version of the benchmark
naive:
	gcc naive.c -O3 -o ./a.exe -lm
	./a.exe

GPU_naive:
	nvcc -o ./a.exe GPU_naive.cu
	./a.exe

GPU_tmp_var:
	nvcc -o ./a.exe GPU_tmp_var.cu
	./a.exe

GPU_no_bd:
	nvcc -o ./a.exe GPU_no_bd.cu
	./a.exe