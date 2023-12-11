
no_dead_code_elimination = -fno-dce -fno-dse -fno-tree-dce -fno-tree-dse

# every rule compiles and runs a version of the benchmark
# naive:
#	gcc naive.c -O3 -o ./a.exe -lm
#	./a.exe

GPU_1_naive:
	nvcc -o ./a.exe GPU_1_naive.cu utils.cu
	./a.exe

GPU_2_tmpVar:
	nvcc -o ./a.exe GPU_2_tmpVar.cu utils.cu
	./a.exe

GPU_3_sumCopy:
	nvcc -o ./a.exe GPU_3_sumCopy.cu utils.cu
	./a.exe

GPU_4_reorderLoops:
	nvcc -o ./a.exe GPU_4_reorderLoops.cu utils.cu
	./a.exe

GPU_5_indexTrans:
	nvcc -o ./a.exe GPU_5_indexTrans.cu utils.cu
	./a.exe

GPU_6_C4Trans:
	nvcc -o ./a.exe GPU_6_C4Trans.cu utils.cu
	./a.exe

GPU_7_coalesced:
	nvcc -o ./a.exe GPU_7_coalesced.cu utils.cu
	./a.exe

GPU_9_2xLoopUnroll:
	nvcc -o ./a.exe GPU_9_2xLoopUnroll.cu utils.cu
	./a.exe
