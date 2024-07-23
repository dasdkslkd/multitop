#ifndef _ELEMENT_CUH_
#define _ELEMENT_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/script.h>
#include "../culib/gMat.cuh"
#define PI acos(-1.)
using gmatd = gpumat<double>;

void predict(const gmatd& x, gmatd& S, gmatd& dSdx, int& nel, torch::jit::Module model);

void elastisity(const gmatd& S, const gmatd& coef, gmatd& sk);

void sensitivity(const gmatd& dSdx, const gmatd& coef, gmatd& dsKdx, gmatd& temp, const int& i, const int& nel);

void filter(gmatd& v);

#endif