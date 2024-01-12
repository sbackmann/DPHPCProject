
naive:
	gcc -o covariance -O3 -march=native naive.c utils.c -lm
	./covariance
	rm ./covariance

naive_gpu:
	nvcc -o covariance naive_gpu.cu utils.cu
	./covariance
	rm ./covariance

cc4:
	nvcc -o covariance GPU_4_triangular.cu utils.cu
	./covariance
	rm ./covariance

