//#include "element.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../culib/gMat.cuh"
//#include "../culib/cudaCommon.cuh"
//#include "../culib/gpuVector.cuh"
//using namespace gv;

using gmatd = gpumat<double>;
gpumat<double> coef_g(576, 9);

void predict(const gmatd& x, gmatd& S, gmatd& dSdx, int n)
{

}

void elastisity(gmatd& S, gmatd& sk, int n)
{

}

void sensitivity(gmatd& dSdx, gmatd& dsKdx, int& i, int n)
{

}

void filter(gmatd& x)
{

}