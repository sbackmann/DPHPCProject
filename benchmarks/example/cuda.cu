
#include "../../timing/dphpc_timing.h"

#define get(A, ncols, r, c) A[(r)*(ncols)+(c)]

__global__ void kernel(int n, double *A, double *B, double *C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < n && c < n) {
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            s += get(A, n, r, i) * get(B, n, i, c);
        }
        get(C, n, r, c) += s;
    }
}

void init_array(int n, double *A, double *B, double *out)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            get(A, n, i, j) = (double) ((i+1)*(j+1)*3 % n);
            get(B, n, i, j) = (double) ((i+1)*(j+1)*7 % n);
            get(out, n, i, j) = 0.0;
        }
    }
}


void run_kernel(int n, double *A, double *B, double *C) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(n / 16 + 1, n / 16 + 1);
    kernel<<<numBlocks,threadsPerBlock>>>(n, A, B, C); 
    cudaDeviceSynchronize();
}


void run_bm(int n, const char* preset) {

    double *A = (double*) malloc(n*n*sizeof(double));
    double *B = (double*) malloc(n*n*sizeof(double));
    double *C = (double*) malloc(n*n*sizeof(double));
    
    double *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, n*n*sizeof(double));
    cudaMalloc((void**) &B_d, n*n*sizeof(double));
    cudaMalloc((void**) &C_d, n*n*sizeof(double));
    
    init_array(n, A, B, C);

    cudaMemcpy((void*) A_d, (void*) A, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) B_d, (void*) B, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) C_d, (void*) C, n*n*sizeof(double), cudaMemcpyHostToDevice);
    
    dphpc_time3(
        cudaMemcpy(C_d, C, n*n*sizeof(double), cudaMemcpyHostToDevice),
        run_kernel(n, A_d, B_d, C_d),
        preset
    );

    cudaFree((void*) A_d);
    cudaFree((void*) B_d);
    cudaFree((void*) C_d);
    
    free((void*)A);
    free((void*)B);
    free((void*)C);
    
}

#ifndef MAIN_HANDLED
int main(int argc, char** argv)
{   
    run_bm(100, "S");
    run_bm(400, "M");
    run_bm(800, "L");
    run_bm(1600,"paper");

    return 0;
}
#endif


