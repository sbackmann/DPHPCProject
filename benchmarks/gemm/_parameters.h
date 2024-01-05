#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static const int nr_parameters = 3;
static int* _params = NULL;

int* get_params(const char* preset) {
    _params = (int*) realloc(_params, 10*sizeof(int));
    if (strcmp(preset, "S") == 0) {
        _params[0] = 1000;
		_params[1] = 1100;
		_params[2] = 1200;
		return _params;
    } else if (strcmp(preset, "M") == 0) {
        _params[0] = 2500;
		_params[1] = 2750;
		_params[2] = 3000;
		return _params;
    } else if (strcmp(preset, "L") == 0) {
        _params[0] = 7000;
		_params[1] = 7500;
		_params[2] = 8000;
		return _params;
    } else if (strcmp(preset, "paper") == 0) {
        _params[0] = 2000;
		_params[1] = 2300;
		_params[2] = 2600;
		return _params;
    }
    return NULL;
}

#endif 
