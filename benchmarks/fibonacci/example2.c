
#include "../../timing/dphpc_timing.h"

#include <stdlib.h>

static long* memo; // "cache" of already computed fibonacci numbers
static int N = 1000000;

void reset() {
    for (int i=0; i<N; i++) {
        memo[i] = -1;
    }
}

long fib(int n) {
    if (n <= 2) {
        return 1;
    }
    if (memo[n] != -1) {
        return memo[n]; // return cached result
    }
    return memo[n] = fib(n-1) + fib(n-2);
}


int main() {

    memo = malloc(N*sizeof(long));

    dphpc_time2(
        reset(), // "empty the cache" before timing
        fib(38)
    );

    dphpc_time3(
        reset(), 
        fib(1000),
        "S" // specify the preset, use the same sizes as in npbench 
            // (see NPBench/bench_info/ for what sizes correspond to which preset)
    );

    dphpc_time3(
        reset(),
        fib(10000),
        "M"
    );

    dphpc_time3(
        reset(),
        fib(100000),
        "L"
    );
}