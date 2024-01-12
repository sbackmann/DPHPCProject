
# base cpu
base:
	gcc -O3 -march=native -o j2d base.c -lm
	./j2d
	rm ./j2d

# naive cpu
naive:
	gcc -O3 -march=native -o j2d naive.c -lm
	./j2d
	rm ./j2d

# optimised cpu, flat array
opt:
	gcc -O3 -march=native -o j2d opt.c -lm
	./j2d
	rm ./j2d

# optimised cpu, unflattened 2d array
unflat:
	gcc -O3 -march=native -o j2d opt_2darr.c -lm
	./j2d
	rm ./j2d

# naive gpu
naive_gpu:
	nvcc -o j2d naive_gpu.cu
	./j2d
	rm ./j2d

# optimised gpu
opt_gpu:
	nvcc -o j2d opt_gpu.cu
	./j2d
	rm ./j2d

# the timing business wont detect rules (or julia versions) that start with '_' 
_validate:
	gcc _validate.c -o ./j2d -lm -DVALIDATE_NAIVE
	./j2d
	rm ./j2d
	nvcc _validate.cu -o ./j2d -lm -DVALIDATE_CUDA
	./j2d
	rm ./j2d