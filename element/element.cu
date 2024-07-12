#include "element.h"
#include "../culib/gpuVector.cuh"
#include "device_launch_parameters.h"
using namespace gv;

__constant__ float* coef_g[576 * 9];

void uploadcoef(float* ptr)
{
	cudaMemcpyToSymbol(coef_g, ptr, 576 * 9);
	cuda_error_check;
}

void spinodal::init_gpu()
{
	cudaMalloc(&gbuf.temp, 9 * nel * sizeof(float));
	cudaMemcpy(gbuf.temp, temp, 9 * nel * sizeof(float), cudaMemcpyHostToDevice);
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
	cudaMalloc(&gbuf.temp, 9 * inst.nel * sizeof(float));
	cudaMemcpy(gbuf.temp, inst.gbuf.temp, 9 * inst.nel * sizeof(float), cudaMemcpyDeviceToDevice);
	cuda_error_check;
}