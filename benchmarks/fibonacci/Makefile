
no_dead_code_elimination = -fno-dce -fno-dse -fno-tree-dce -fno-tree-dse

# every rule compiles and runs a version of the benchmark
naive:
	gcc example1.c -O3 -o ./a.exe -lm 
	./a.exe


memoization:
	gcc example2.c -O3 -o ./a.exe -lm 
	./a.exe


# use flags to slow C down lol
dynamic:
	gcc dynamic.c -O3 $(no_dead_code_elimination) -fno-unroll-loops -o ./a.exe -lm 
	./a.exe


# the timing business wont detect rules (or julia versions) that start with '_' 
_dbg_memo:
	gcc example2.c -g -o ./a.exe -lm 
	gdb ./a.exe


cuda:
	nvcc -o ./a.exe gpu.cu
	./a.exe
