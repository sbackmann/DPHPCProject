
no_dead_code_elimination = -fno-dce -fno-dse -fno-tree-dce -fno-tree-dse

build:
	mkdir -p build

# every rule compiles and runs a version of the benchmark
naive: build
	gcc example1.c -O3 -o ./build/out.exe -lm 
	./build/out.exe

memoization: build
	gcc example2.c -O3 -o ./build/out.exe -lm 
	./build/out.exe


# use flags to slow C down lol
dynamic: build
	gcc dynamic.c -O3 $(no_dead_code_elimination) -fno-unroll-loops -o ./build/out.exe -lm 
	./build/out.exe


# the timing business wont detect rules that have _ in them 
dbg_memo: build
	gcc example2.c -g -o ./build/out.exe -lm 
	gdb ./build/out.exe