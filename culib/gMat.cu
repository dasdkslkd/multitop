#include "gMat.cuh"

std::pair<dim3, dim3> kernel_param(int32_t len, int32_t threadperblock /*= 512*/)
{
	int32_t grid, block;
	block = threadperblock;
	grid = (len + block - 1) / block;
	return std::pair<dim3, dim3>(grid, block);
}