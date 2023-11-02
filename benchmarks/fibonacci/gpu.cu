
#include "../../timing/dphpc_timing.h"

__global__ void gpufib(int n, int* out) {
    if (n <= 2) {
        out[0] = 1;
    } else {
        int sn = 1;
        int sn_1 = 1;
        for (int i = 3; i <= n; i++) {
            int tmp = sn_1;
            sn_1 = sn;
            sn = sn + tmp;
        }
        out[0] = sn;
    }
}

void run_kernel(int n, int* out) {
    gpufib<<<1, 1>>>(n, out); // measure performance of single gpu core
    cudaDeviceSynchronize(); // wait until kernel is done, otherwise just measuring how long it takes to launch the kernel
}

int main() {
    int* out;
    cudaMalloc((void**) &out, sizeof(int));


    dphpc_time(
        run_kernel(38, out);
    );
    
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    
    // cudaMemcpy(&result, out, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("result after: %d\n", result);

    dphpc_time3(
        ,
        run_kernel(1000, out),
        "S"
    );

    dphpc_time3(
        ,
        run_kernel(10000, out),
        "M"
    );

    dphpc_time3(
        ,
        run_kernel(100000, out),
        "L"
    );

    dphpc_time3(
        ,
        run_kernel(100500, out),
        "paper"
    );

    cudaFree(out);
}
