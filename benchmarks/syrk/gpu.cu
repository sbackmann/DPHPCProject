
#include "../../timing/dphpc_timing.h"

__global__ void syrk(int n, int k, double alpha, double beta, double* C[n][n], double A[n][k]) {

}


void init_array(int n, int m, double *alpha, double *beta, double C[n][n], double A[n][m])
{
    *alpha = 1.5;
    *beta = 1.2;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i][j] = (double) ((i*j+1)%n) / n;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = (double) ((i*j+2)%m) / m;
        }
    }
    
}

void run_kernel(int n, int k, double alpha, double beta, double C[n][n], double A[n][k]) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(n / 16 + 1, n / 16 + 1);
    syrk<<<numBlokcs,threadsPerBlock>>>(n, m, alpha, beta, C_d, A_d); 
    cudaDeviceSynchronize();
}


void run_bm(int n, int m, const char* preset) {
    double alpha;
    double beta;
    
    double *C = malloc(n*n*sizeof(double));
    double *A = malloc(n*m*sizeof(double));

    double *C_d, A_d;
    cudaMalloc((void**) &C_d, n*n*sizeof(double));
    cudaMalloc((void**) &A_d, n*m*sizeof(double));

    init_array(n, m, &alpha, &beta, C, A);

    cudaMemcpy(C_d, C, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A, n*m*sizeof(double), cudaMemcpyHostToDevice);

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

// "S": { "M": 50, "N": 70 },
// "M": { "M": 150, "N": 200 },
// "L": { "M": 500, "N": 600 },
// "paper": { "M": 1000, "N": 1200 }


int main(int argc, char** argv)
{   
    // run_bm(5, 3, "missing");
    run_bm(70, 50, "S");
    run_bm(200, 150, "M");
    run_bm(600, 500, "L");
    run_bm(1200, 1000, "paper");

    return 0;
}


