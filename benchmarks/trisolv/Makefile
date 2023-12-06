ju:
	julia trisolv_gpu.jl
	#julia trisolv.jl

juserial:
	julia trisolv.jl

py:
	python trisolv.py

c:
	gcc -o trisolv trisolv.c -lm -llapacke -lblas 
	./trisolv
	rm ./trisolv

run: ju py c 
