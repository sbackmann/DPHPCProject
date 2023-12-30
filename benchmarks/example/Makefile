# make sure there is at least 1line before the first rule

# every rule compiles and runs a version of the benchmark

# for plotting there has to be a c version called naive
naive:
	gcc naive.c -O3 -o ./a.exe -lm 
	./a.exe
	rm ./a.exe

# and a cuda c version called naive_gpu
naive_gpu:
	nvcc -O3 -o ./a.exe cuda.cu
	./a.exe
	rm ./a.exe
