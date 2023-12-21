#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static int* params = NULL;

int* get_params(const char* preset) {
    params = realloc(params, 10*sizeof(int))
    if (strcmp(preset, "S") == 0) {
        params[0] = 50;
		params[1] = 70;
		return params;
    } else if (strcmp(preset, "M") == 0) {
        params[0] = 150;
		params[1] = 200;
		return params;
    } else if (strcmp(preset, "L") == 0) {
        params[0] = 500;
		params[1] = 600;
		return params;
    } else if (strcmp(preset, "paper") == 0) {
        params[0] = 1000;
		params[1] = 1200;
		return params;
    }
    return NULL;
}

#endif 
