#ifndef PARAMETERSH
#define PARAMETERSH

#include <stdlib.h>
#include <string.h>

static const int nr_parameters = 3;
static int* _params = NULL;

int* get_params(const char* preset) {
    _params = realloc(_params, 10*sizeof(int));
    if (strcmp(preset, "S") == 0) {
        _params[0] = 128;
		_params[1] = 60;
		_params[2] = 60;
		return _params;
    } else if (strcmp(preset, "M") == 0) {
        _params[0] = 256;
		_params[1] = 110;
		_params[2] = 125;
		return _params;
    } else if (strcmp(preset, "L") == 0) {
        _params[0] = 512;
		_params[1] = 220;
		_params[2] = 250;
		return _params;
    } else if (strcmp(preset, "paper") == 0) {
        _params[0] = 270;
		_params[1] = 220;
		_params[2] = 250;
		return _params;
    }
    return NULL;
}

#endif 
