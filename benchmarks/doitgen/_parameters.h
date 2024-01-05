#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static const int nr_parameters = 3;
static int* _params = NULL;

int* get_params(const char* preset) {
    _params = (int*) realloc(_params, 10*sizeof(int));
    if (strcmp(preset, "S") == 0) {
        _params[0] = 60;
		_params[1] = 60;
		_params[2] = 128;
		return _params;
    } else if (strcmp(preset, "M") == 0) {
        _params[0] = 110;
		_params[1] = 125;
		_params[2] = 256;
		return _params;
    } else if (strcmp(preset, "L") == 0) {
        _params[0] = 220;
		_params[1] = 250;
		_params[2] = 512;
		return _params;
    } else if (strcmp(preset, "paper") == 0) {
        _params[0] = 220;
		_params[1] = 250;
		_params[2] = 270;
		return _params;
    }
    return NULL;
}

#endif 
