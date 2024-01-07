#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static const int nr_parameters = 2;
static int* _params = NULL;

int* get_params(const char* preset) {
    _params = (int*) realloc(_params, 10*sizeof(int));
    if (strcmp(preset, "S") == 0) {
        _params[0] = 50;
		_params[1] = 60;
		return _params;
    } else if (strcmp(preset, "M") == 0) {
        _params[0] = 1400;
		_params[1] = 1800;
		return _params;
    } else if (strcmp(preset, "L") == 0) {
        _params[0] = 3200;
		_params[1] = 4000;
		return _params;
    } else if (strcmp(preset, "paper") == 0) {
        _params[0] = 1200;
		_params[1] = 1400;
		return _params;
    }
    return NULL;
}

#endif 
