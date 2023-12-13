ju:
	julia trisolv_gpu.jl
	#julia trisolv.jl

juserial:
	julia trisolv.jl

py:
	python trisolv.py

c0:
	gcc -o trisolv CPU_0.c utils.c -lm
	./trisolv
	rm ./trisolv

cc0:
	nvcc -o trisolv GPU_0.cu utils.c -lm
	./trisolv
	rm ./trisolv

cc5:
	nvcc -o trisolv GPU_5.cu utils.c -lm
	./trisolv
	rm ./trisolv

