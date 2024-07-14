#include "gMat.cuh"

__host__ void make_kernel_param(size_t* block_num, size_t* block_size, size_t num_tasks, size_t prefer_block_size = 512)
{
	*block_size = prefer_block_size;
	*block_num = (num_tasks + prefer_block_size - 1) / prefer_block_size;
}

__host__ void make_kernel_param2d(dim3* grid, dim3* block, size_t nx, size_t ny, size_t pre_block = 32)
{
	*block = dim3(pre_block, pre_block, 0);
	*grid = dim3(std::ceil(nx / pre_block), std::ceil(ny / pre_block), 0);
}