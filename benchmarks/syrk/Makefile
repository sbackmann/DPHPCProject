
naive:
	gcc naive.c -O3 -march=skylake -o ./a.exe -lm 
	./a.exe

optimized:
	gcc optimized.c -O3 -march=skylake -o ./a.exe -lm 
	./a.exe

naive_gpu:
	nvcc -o ./a.exe naive.cu
	./a.exe

colmajor_gpu:
	nvcc -o ./a.exe colmajor.cu
	./a.exe

colmajor_unroll4x_gpu:
	nvcc -o ./a.exe colmajor_unroll4x.cu
	./a.exe