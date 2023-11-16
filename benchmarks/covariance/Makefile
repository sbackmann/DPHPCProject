_ju:
	julia covariance.jl

_py:
	python covariance.py

c:
	gcc -o covariance covariance.c -lm
	./covariance
	rm ./covariance

_run: _ju _py c 