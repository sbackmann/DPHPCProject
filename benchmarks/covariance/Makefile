c0:
	gcc -o covariance CPU_0.c utils.c -lm
	./covariance
	rm ./covariance

cc0:
	nvcc -o covariance GPU_0_naive.cu utils.cu
	./covariance
	rm ./covariance

cc4:
	nvcc -o covariance GPU_4_triangular.cu utils.cu
	./covariance
	rm ./covariance

cc6:
	nvcc -o covariance cGPU_6_inbounds.cu
	./covariance
	rm ./covariance

ccc:
	nvcc -o covariance cGPU_6_inbounds.cu -O0
	./covariance
	rm ./covariance
