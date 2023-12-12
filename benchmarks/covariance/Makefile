c:
	gcc -o covariance covariance.c -lm
	./covariance
	rm ./covariance

cc:
	nvcc -o covariance cGPU_6_inbounds.cu
	./covariance
	rm ./covariance

ccc:
	nvcc -o covariance cGPU_6_inbounds.cu -O0
	./covariance
	rm ./covariance
