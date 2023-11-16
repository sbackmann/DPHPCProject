ju:
	julia trisolve_gpu.jl
	#julia trisolve.jl

juserial:
	julia trisolve.jl

py:
	python trisolve.py

c:
	gcc -o trisolve trisolve.c -lm -llapacke -lblas 
	./trisolve
	rm ./trisolve

run: ju py c 
