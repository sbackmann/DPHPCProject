no_dead_code_elimination = -fno-dce -fno-dse -fno-tree-dce -fno-tree-dse

# every rule compiles and runs a version of the benchmark
naive:
	gcc naive.c -O3 -o ./a.exe -lm
	./a.exe

naive_gpu:
	nvcc -o ./a.exe GPU_1_naive.cu utils.cu
	./a.exe

GPU_1_naive:
	nvcc -o ./a.exe GPU_1_naive.cu utils.cu
	./a.exe

GPU_2_tmp_var:
	nvcc -o ./a.exe GPU_2_tmpVar.cu utils.cu
	./a.exe

GPU_4_noBranch:
	nvcc -o ./a.exe GPU_4_noBranch.cu utils.cu
	./a.exe

GPU_5_noElse:
	nvcc -o ./a.exe GPU_5_noElse.cu utils.cu
	./a.exe

GPU_8_2xLoopUnroll:
	nvcc -o ./a.exe GPU_8_2xLoopUnroll.cu utils.cu
	./a.exe

GPU_8_3xLoopUnroll:
	nvcc -o ./a.exe GPU_8_3xLoopUnroll.cu utils.cu
	./a.exe

GPU_8_4xLoopUnroll:
	nvcc -o ./a.exe GPU_8_4xLoopUnroll.cu utils.cu
	./a.exe