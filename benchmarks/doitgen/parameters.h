#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static int* params = NULL;

int* get_params(const char* preset) {
    params = realloc(params, 10*sizeof(int))
    if (strcmp(preset, "S") == 0) {
        params[0] = 128;
		params[1] = 60;
		params[2] = 60;
		return params;
    } else if (strcmp(preset, "M") == 0) {
        params[0] = 256;
		params[1] = 110;
		params[2] = 125;
		return params;
    } else if (strcmp(preset, "L") == 0) {
        params[0] = 512;
		params[1] = 220;
		params[2] = 250;
		return params;
    } else if (strcmp(preset, "paper") == 0) {
        params[0] = 270;
		params[1] = 220;
		params[2] = 250;
		return params;
    }
    return NULL;
}

#endif 
