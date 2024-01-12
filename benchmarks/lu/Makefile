
naive:
	gcc naive.c -O3 -march=native -o ./a.exe -lm 
	./a.exe


naive_gpu:
	nvcc -o ./a.exe naive_gpu.cu
	./a.exe

opt_gpu:
	nvcc -o ./a.exe opt_gpu.cu
	./a.exe

reorder_loop_gpu:
	nvcc -o ./a.exe reorder_loop_gpu.cu
	./a.exe
