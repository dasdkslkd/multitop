#include "element.h"
#include "../culib/gpuVector.cuh"
#include "device_launch_parameters.h"
using namespace gv;

__constant__ double* coef_g[576 * 9];

void uploadcoef(double* ptr)
{
	cudaMemcpyToSymbol(coef_g, ptr, 576 * 9);
	cuda_error_check;
}

void spinodal::init_gpu()
{
	cudaMalloc(&gbuf.temp, 9 * nel * sizeof(double));
	cudaMemcpy(gbuf.temp, temp, 9 * nel * sizeof(double), cudaMemcpyHostToDevice);
	uploadcoef(coef.data());
	cuda_error_check;
}

void spinodal::free_gpu()
{
	cudaFree(gbuf.temp);
	cuda_error_check;
}

void spinodal::value_gpu(const spinodal& inst)
{
	cudaFree(gbuf.temp);
	cudaMalloc(&gbuf.temp, 9 * inst.nel * sizeof(double));
	cudaMemcpy(gbuf.temp, inst.gbuf.temp, 9 * inst.nel * sizeof(double), cudaMemcpyDeviceToDevice);
	cuda_error_check;
}