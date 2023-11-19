
// validation for gpu versions

#define MAIN_HANDLED


#ifdef VALIDATE_CUDA
#include "cuda.cu"

int main() {

    int n = 30;

    double *A = (double*) malloc(n*n*sizeof(double));
    double *B = (double*) malloc(n*n*sizeof(double));
    double *C = (double*) malloc(n*n*sizeof(double));
    

    double *A_d, *B_d, *C_d;
    cudaMalloc((void**) &A_d, n*n*sizeof(double));
    cudaMalloc((void**) &B_d, n*n*sizeof(double));
    cudaMalloc((void**) &C_d, n*n*sizeof(double));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            get(A, n, i, j) = 1.0;
            get(B, n, i, j) = 1.0;
            get(C, n, i, j) = 0.0;
        }
    }

    cudaMemcpy((void*) A_d, (void*) A, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) B_d, (void*) B, n*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) C_d, (void*) C, n*n*sizeof(double), cudaMemcpyHostToDevice);
    


    run_kernel(n, A_d, B_d, C_d);


    cudaMemcpy((void*) C, (void*) C_d, n*n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree((void*) A_d);
    cudaFree((void*) B_d);
    cudaFree((void*) C_d);



    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (get(C, n, i, j) != n) {
                printf("cuda.cu VALIDATION FAILED\n");
                return 1;
            }
        }
    }
    printf("cuda.cu VALIDATION SUCCESS\n");
    return 0;
}
#endif // VALIDATE_CUDA