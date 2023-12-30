#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static const int nr_parameters = 2;
static int* _params = NULL;

int* get_params(const char* preset) {
    _params = realloc(_params, 10*sizeof(int));
    if (strcmp(preset, "S") == 0) {
        _params[0] = 50;
		_params[1] = 70;
		return _params;
    } else if (strcmp(preset, "M") == 0) {
        _params[0] = 150;
		_params[1] = 200;
		return _params;
    } else if (strcmp(preset, "L") == 0) {
        _params[0] = 500;
		_params[1] = 600;
		return _params;
    } else if (strcmp(preset, "paper") == 0) {
        _params[0] = 1000;
		_params[1] = 1200;
		return _params;
    }
    return NULL;
}

#endif 
