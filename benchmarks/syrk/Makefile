
naive:
	gcc straight.c -O3 -march=skylake -o ./a.exe -lm 
	./a.exe

optimized:
	gcc optimized.c -O3 -march=skylake -o ./a.exe -lm 
	./a.exe

naive_gpu:
	nvcc -o ./a.exe cuda.cu
	./a.exe