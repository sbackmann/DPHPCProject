
naive:
	gcc naive.c -O3 -o ./a.exe -lm 
	./a.exe

naive_colmajor:
	gcc naive_col_major.c -O3 -o ./a.exe -lm 
	./a.exe

naive_gpu:
	nvcc -o ./a.exe lu_naive_gpu.cu
	./a.exe

colmaj_gpu:
	nvcc -o ./a.exe GPU_col_major.cu
	./a.exe
