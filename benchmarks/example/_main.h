#ifndef MAINH
#define MAINH


#include "_parameters.h"

int main()
{   
    const char *presets[] = {"S", "M", "L", "paper"};

    // every version defines is_valid() and run_bm()...
    if (is_valid()) {
        for (int i = 0; i < 4; i++) {
            const char* preset = presets[i];
            int n = get_params(preset)[0];
            run_bm(n, preset);
        }
    }

    return 0;
}

#endif