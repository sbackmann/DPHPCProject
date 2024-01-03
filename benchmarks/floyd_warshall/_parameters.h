#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static const int nr_parameters = 1;
static int* _params = NULL;

int* get_params(const char* preset) {
    _params = (int*) realloc(_params, 10*sizeof(int));
    if (strcmp(preset, "S") == 0) {
        _params[0] = 200;
		return _params;
    } else if (strcmp(preset, "M") == 0) {
        _params[0] = 400;
		return _params;
    } else if (strcmp(preset, "L") == 0) {
        _params[0] = 850;
		return _params;
    } else if (strcmp(preset, "paper") == 0) {
        _params[0] = 2800;
		return _params;
    }
    return NULL;
}

#endif 
