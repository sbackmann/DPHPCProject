
naive:
	gcc naive.c -O3 -o ./build/out.exe -lm 
	./build/out.exe

dynamic:
	gcc dynamic.c -O3 -o ./build/out.exe -lm 
	./build/out.exe

memoization:
	gcc memo.c -O3 -o ./build/out.exe -lm 
	./build/out.exe

# the timing business wont detect rules that have non word characters like _  
dbg_memo:
	gcc memo.c -g -o ./build/out.exe -lm 
	gdb ./build/out.exe