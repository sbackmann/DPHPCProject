c0:
	gcc -o trisolv naive.c utils.c -lm
	./trisolv
	rm ./trisolv

cc0:
	nvcc -o trisolv naive_gpu.cu utils.cu -lm 
	./trisolv
	rm ./trisolv

cc5:
	nvcc -o trisolv GPU_5.cu utils.cu -lm
	./trisolv
	rm ./trisolv

