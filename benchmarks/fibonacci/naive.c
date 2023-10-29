
#include "../../timing/dphpc_timing.h"

long fib(int n) {
    if (n <= 2) {
        return 1;
    }
    return fib(n-1) + fib(n-2);
}

int main() {
    dphpc_time(
        fib(35);
    );
}