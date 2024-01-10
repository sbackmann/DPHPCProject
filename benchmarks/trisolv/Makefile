
naive:
	gcc -o trisolv -Ofast  naive.c utils.c -lm
	./trisolv
	rm ./trisolv

naive_gpu:
	nvcc -o trisolv naive_gpu.cu utils.cu -lm 
	./trisolv
	rm ./trisolv

reorder_loop:
	gcc -o trisolv -Ofast reorder_loop.c utils.c -lm 
	./trisolv
	rm ./trisolv

cc5:
	nvcc -o trisolv GPU_5.cu utils.cu -lm
	./trisolv
	rm ./trisolv

reorder_loop_gpu:
	nvcc -o trisolv loop_kernel.cu utils.cu -lm
	./trisolv
	rm ./trisolv
