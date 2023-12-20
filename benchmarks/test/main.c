
#include "../../timing/dphpc_timing.h"
#include "test.h"

int main() {
    if (test() == 4) {
        dphpc_time(sleep1());
    }
}