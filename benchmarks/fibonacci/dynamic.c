#include "../../timing/dphpc_timing.h"



long fib(int n) {
    if (n <= 2) {
        return 1;
    }
    long sn = 1;
    long sn_1 = 1;
    for (int i = 3; i <= n; i++) {
        long tmp = sn_1;
        sn_1 = sn;
        sn = sn + tmp;
    }
    return sn;
}

int main() {

    dphpc_time3(
        , // no reset/init code in this case
        fib(1000),
        "S"
    );

    dphpc_time3(
        ,
        fib(10000),
        "M"
    );

    dphpc_time3(
        ,
        fib(100000),
        "L"
    );

    dphpc_time3(
        ,
        fib(100500),
        "paper"
    );

    // preset, if given, has to be one of these 4: "S", "M", "L", "paper"
    // versions where you use dphpc_time or dphpc_time2 (without specifying the preset) are always run
}