#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static const int nr_parameters = 1;
static int* _params = NULL;

int* get_params(const char* preset) {
    _params = (int*) realloc(_params, 10*sizeof(int));
    if (strcmp(preset, "S") == 0) {
        _params[0] = 50;
		return _params;
    } else if (strcmp(preset, "M") == 0) {
        _params[0] = 100;
		return _params;
    } else if (strcmp(preset, "L") == 0) {
        _params[0] = 1000;
		return _params;
    } else if (strcmp(preset, "paper") == 0) {
        _params[0] = 220;
		return _params;
    }
    return NULL;
}

#endif 
