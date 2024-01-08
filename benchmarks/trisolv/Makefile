
naive:
	gcc -o trisolv naive.c utils.c -lm
	./trisolv
	rm ./trisolv

naive_gpu:
	nvcc -o trisolv naive_gpu.cu utils.cu -lm 
	./trisolv
	rm ./trisolv

cc5:
	nvcc -o trisolv GPU_5.cu utils.cu -lm
	./trisolv
	rm ./trisolv

loop_kernel:
	nvcc -o trisolv loop_kernel.cu utils.cu -lm
	./trisolv
	rm ./trisolv
