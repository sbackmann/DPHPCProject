_ju:
	julia covariance_gpu.jl
	# julia covariance.jl

_py:
	python covariance.py

c:
	gcc -o covariance covariance.c -lm
	./covariance
	rm ./covariance

gc:
	nvcc -o covariance covariance_c_gpu.cu -lm
	./covariance
	rm ./covariance

run: ju py c 
