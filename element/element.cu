#include "element.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "../culib/gMat.cuh"
//#include "../culib/cudaCommon.cuh"
//#include "../culib/gpuVector.cuh"
//using namespace gv;

__constant__ double* coef_g[576 * 9];

void uploadcoef(double* ptr)
{
	cudaMemcpyToSymbol(coef_g, ptr, 576 * 9);
	cuda_error_check;
}

void spinodal::init_gpu()
{
	//cudaMalloc(&gbuf.temp, 9 * nel * sizeof(double));
	//cudaMemcpy(gbuf.temp, temp, 9 * nel * sizeof(double), cudaMemcpyHostToDevice);
	uploadcoef(coef.data());
	gbuf.temp.set_from_host(temp, 9, nel);
	cuda_error_check;
}

void spinodal::free_gpu()
{
	//cudaFree(gbuf.temp);

	cuda_error_check;
}

void spinodal::value_gpu(const spinodal& inst)
{
	//cudaFree(gbuf.temp);
	//cudaMalloc(&gbuf.temp, 9 * inst.nel * sizeof(double));
	//cudaMemcpy(gbuf.temp, inst.gbuf.temp, 9 * inst.nel * sizeof(double), cudaMemcpyDeviceToDevice);
	gbuf.temp.clear();
	gbuf.temp = inst.gbuf.temp;
	
	cuda_error_check;
}