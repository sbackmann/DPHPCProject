#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static int* params = NULL;

int* get_params(const char* preset) {
    params = realloc(params, 10*sizeof(int))
    if (strcmp(preset, "S") == 0) {
        params[0] = 500;
		params[1] = 600;
		return params;
    } else if (strcmp(preset, "M") == 0) {
        params[0] = 1400;
		params[1] = 1800;
		return params;
    } else if (strcmp(preset, "L") == 0) {
        params[0] = 3200;
		params[1] = 4000;
		return params;
    } else if (strcmp(preset, "paper") == 0) {
        params[0] = 1200;
		params[1] = 1400;
		return params;
    }
    return NULL;
}

#endif 
