#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static int* params = NULL;

int* get_params(const char* preset) {
    params = realloc(params, 10*sizeof(int))
    if (strcmp(preset, "S") == 0) {
        params[0] = 1100;
		params[1] = 1200;
		params[2] = 1000;
		return params;
    } else if (strcmp(preset, "M") == 0) {
        params[0] = 2750;
		params[1] = 3000;
		params[2] = 2500;
		return params;
    } else if (strcmp(preset, "L") == 0) {
        params[0] = 7500;
		params[1] = 8000;
		params[2] = 7000;
		return params;
    } else if (strcmp(preset, "paper") == 0) {
        params[0] = 2300;
		params[1] = 2600;
		params[2] = 2000;
		return params;
    }
    return NULL;
}

#endif 
