#include <unistd.h>
#include "test.h"

int sleep1() {
    sleep(1);
    return 5;
}

int test() {
    return 4;
}
