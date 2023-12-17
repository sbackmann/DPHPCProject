c0:
	gcc -o covariance naive.c utils.c -lm #nvcc
	./covariance
	rm ./covariance

cc0:
	nvcc -o covariance naive_gpu.cu utils.cu
	./covariance
	rm ./covariance

cc4:
	nvcc -o covariance GPU_4_triangular.cu utils.cu
	./covariance
	rm ./covariance

