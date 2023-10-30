
#include "../../timing/dphpc_timing.h"

long fib(int n) {
    if (n <= 2) {
        return 1;
    }
    return fib(n-1) + fib(n-2);
}

int main() {

    // use the dphpc_time macro to collect measurements
    dphpc_time(
        fib(38)
    );
}