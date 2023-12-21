#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static int* params = NULL;

int* get_params(const char* preset) {
    params = realloc(params, 10*sizeof(int))
    if (strcmp(preset, "S") == 0) {
        params[0] = 200;
		return params;
    } else if (strcmp(preset, "M") == 0) {
        params[0] = 400;
		return params;
    } else if (strcmp(preset, "L") == 0) {
        params[0] = 850;
		return params;
    } else if (strcmp(preset, "paper") == 0) {
        params[0] = 2800;
		return params;
    }
    return NULL;
}

#endif 
