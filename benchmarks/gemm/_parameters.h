#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static const int nr_parameters = 3;
static int* _params = NULL;

int* get_params(const char* preset) {
    _params = realloc(_params, 10*sizeof(int));
    if (strcmp(preset, "S") == 0) {
        _params[0] = 1100;
		_params[1] = 1200;
		_params[2] = 1000;
		return _params;
    } else if (strcmp(preset, "M") == 0) {
        _params[0] = 2750;
		_params[1] = 3000;
		_params[2] = 2500;
		return _params;
    } else if (strcmp(preset, "L") == 0) {
        _params[0] = 7500;
		_params[1] = 8000;
		_params[2] = 7000;
		return _params;
    } else if (strcmp(preset, "paper") == 0) {
        _params[0] = 2300;
		_params[1] = 2600;
		_params[2] = 2000;
		return _params;
    }
    return NULL;
}

#endif 
