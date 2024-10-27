#ifndef _ELEMENT_CUH_
#define _ELEMENT_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <torch/script.h>
#include "../culib/gMat.cuh"
#define PI acos(-1.)
using gmatd = gpumat<double>;

//void predict(const gmatd& x, gmatd& S, gmatd& dSdx, int& nel, torch::jit::Module model);

void predict_py(gmatd& x, gmatd& S, gmatd& dSdx, int& nel);

void elastisity(const gmatd& S, const gmatd& coef, gmatd& sk);

void sensitivity(const gmatd& dSdx, const gmatd& coef, gmatd& dsKdx, std::vector<int32_t>& idx, const int& i, const int& nel);

void filter(gmatd& v);

void filter(double* x, int nel);

#endif