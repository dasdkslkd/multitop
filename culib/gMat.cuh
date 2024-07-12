#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//kernel func

//helper func
__host__ void make_kernel_param(size_t* block_num, size_t* block_size, size_t num_tasks, size_t prefer_block_size)
{
	*block_size = prefer_block_size;
	*block_num = (num_tasks + prefer_block_size - 1) / prefer_block_size;
}

template<typename T>
void init_array(T* dev_array, T value, int array_size)
{
	size_t grid_dim;
	size_t block_dim;
	make_kernel_param(&grid_dim, &block_dim, array_size, 512);
	init_array_kernel << <grid_dim, block_dim >> > (dev_array, value, array_size);
	cudaDeviceSynchronize();
	cuda_error_check;
}

template<typename scalar>
class gpumat
{
//data and shape
private:
	scalar* _data;
	size_t _size;
	size_t _row;
	size_t _col;
	//v1 func v2
	template<typename Lambda>friend  void apply_vector(gpumat& v1, const gpumat& v2, Lambda func);
//helper func
public:
	typedef scalar scalar;
	
	scalar*& data() { return _data; }
	const scalar* data() const { return _data; }
	size_t size() const { return _size; }
	size_t rows() const { return _row; }
	size_t cols() const { return _col; }

	void move(scalar* ptr, size_t row, size_t col) { _data = ptr; _row = row; _col = col; }

	void clear()
	{
		if (_data == nullptr && _size == 0) { return; }
		cudaFree(_data);
		_size = 0;
		_row = 0;
		_col = 0;
		_data = nullptr;
	}
//constructor & deconstructor
public:
	gpumat(void) :_size(0), _row(0), _col(0), _data(nullptr) {}
	explicit gVector(size_t row, size_t col, Scalar default_value = 0)
	{
		_size = row * col;
		_row = row;
		_col = col;
		cudaMalloc(&_data, _size * sizeof(Scalar));
		init_array(_data, default_value, _size);
		cuda_error_check;
	}
};