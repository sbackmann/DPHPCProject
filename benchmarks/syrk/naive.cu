
#include "../../timing/dphpc_timing.h"

#define get(A, ncols, r, c) A[(r)*(ncols)+(c)]

__global__ void syrk(int n, int k, double alpha, double beta, double *C, double *A) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < n && c < n && r >= c) {
        double s = 0.0;
        for (int i = 0; i < k; i++) {
            s += get(A, k, r, i) * get(A, k, c, i);
        }
        get(C, n, r, c) = beta * get(C, n, r, c) + alpha * s;
    }
}

void init_array(int n, int m, double *alpha, double *beta,
    double* C, double* A)
{
    *alpha = 1.5;
    *beta = 1.2;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            get(A, m, i, j) = (double) ((i*j+1)%n) / n;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            get(C, n, i, j) = (double) ((i*j+2)%m) / m;
        }
    }
}

void run_kernel(int n, int k, double alpha, double beta, double *C, double *A) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n-1) / 16 + 1, (n-1) / 16 + 1);
    syrk<<<numBlocks,threadsPerBlock>>>(n, k, alpha, beta, C, A); 
    cudaDeviceSynchronize();
}


void run_bm(int n, int m, const char* preset) {
    double alpha;
    double beta;
    
    double *C = (double*) malloc(n*n*sizeof(double));
    double *A = (double*) malloc(n*m*sizeof(double));

    double *C_d, *A_d;
    cudaMalloc((void**) &C_d, n*n*sizeof(double));
    cudaMalloc((void**) &A_d, n*m*sizeof(double));

    init_array(n, m, &alpha, &beta, C, A);

    cudaMemcpy((void*) C_d, (void*) C, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) A_d, (void*) A, n*m*sizeof(double), cudaMemcpyHostToDevice);

    dphpc_time3(
        cudaMemcpy(C_d, C, n*n*sizeof(double), cudaMemcpyHostToDevice);,
        run_kernel(n, m, alpha, beta, C_d, A_d),
        preset
    );

    cudaFree((void*) C_d);
    cudaFree((void*) A_d);

    free((void*)C);
    free((void*)A);
}


int is_valid() {

    int n = 200;
    int m = 70;

    double alpha = 3.0;
    double beta = 5.0;
    
    double *C = (double*) malloc(n*n*sizeof(double));
    double *A = (double*) malloc(n*m*sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            get(C, n, i, j) = 1;
        }
        for (int j = 0; j < m; j++) {
            get(A, m, i, j) = 1;
        }
    } 

    double *C_d, *A_d;
    cudaMalloc((void**) &C_d, n*n*sizeof(double));
    cudaMalloc((void**) &A_d, n*m*sizeof(double));

    cudaMemcpy((void*) C_d, (void*) C, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) A_d, (void*) A, n*m*sizeof(double), cudaMemcpyHostToDevice);

    run_kernel(n, m, alpha, beta, C_d, A_d);

    cudaMemcpy((void*) C, (void*) C_d, n*n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree((void*) C_d);
    cudaFree((void*) A_d);

    free((void*)A);

    

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j <= i && get(C, n, i, j) != beta + alpha * m) {
                free((void*)C);
                printf("validation failed");
                return 0;
            }
        }
    } 

    free((void*)C);
    return 1;
}

#include "_main.h"

