# make sure there is at least 1 empty line before the first rule

# every rule compiles and runs a version of the benchmark

# for plotting there has to be a c version called naive
naive:
	gcc naive.c -O3 -o ./a.exe -lm 
	./a.exe

# and a cuda c version called naive_gpu
naive_gpu:
	nvcc -O3 -o ./a.exe cuda.cu
	./a.exe

# the timing business wont detect rules (or julia versions) that start with '_' 
_validate:
	gcc _validate.c -o ./a.exe -lm -DVALIDATE_NAIVE
	./a.exe
	nvcc _validate.cu -o ./a.exe -lm -DVALIDATE_CUDA
	./a.exe
