#pragma once
#ifndef _GMAT_CUH_
#define _GMAT_CUH_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

#ifndef cuda_error_check
#define cuda_error_check do{ \
	auto err = cudaGetLastError(); \
	if (err != 0) { \
		printf("\x1b[31mCUDA error occured at line %d in file %s, error type %s \x1b[0m\n", __LINE__,__FILE__, cudaGetErrorName(err));\
	} \
}while(0)
#endif

template<typename scalar>
class gpumat;

//kernel func
template<typename T>
__global__ void init_array_kernel(T* array, T value, int array_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < array_size)
		array[tid] = value;
}

template<typename T, typename Tout, typename Lam>
__global__ void map_kernel(T* g_data, Tout* dst, int n, Lam func)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		dst[tid] = func(g_data[tid]);
}

template<typename T, typename Lam>
__global__ void map_kernel(T* dst, int n, Lam func)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		dst[tid] = func(tid);
}

template<typename Lam>
__global__ void map_kernel(int n, Lam func)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		func(tid);
}

template<typename scalar>
__global__ void transpose_kernel(scalar* v, int m, int n)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < n && row < m && col < row)
	{
		int xx = col * m + row;
		int yy = row * n + col;
		scalar temp = v[xx];
		v[xx] = v[yy];
		v[yy] = temp;
	}
}

template<typename scalar>
__global__ void matprod_kernel(const scalar* v1, const scalar* v2, scalar* v3, int m, int n, int l)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < l && row < m)
	{
		scalar sum = 0.;
		for (int k = 0; k < n; ++k)
		{
			sum += v1[row * n + k] * v2[k * l + col];
		}
		v3[row * l + col] = sum;
	}
}

//helper func
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

__host__ void make_kernel_param(size_t* block_num, size_t* block_size, size_t num_tasks, size_t prefer_block_size = 512);
//{
//	*block_size = prefer_block_size;
//	*block_num = (num_tasks + prefer_block_size - 1) / prefer_block_size;
//}

__host__ void make_kernel_param2d(dim3* grid, dim3* block, size_t nx, size_t ny, size_t pre_block = 32);
//{
//	*block = dim3(pre_block, pre_block, 0);
//	*grid = dim3(std::ceil(nx / pre_block), std::ceil(ny / pre_block), 0);
//}

template<typename Lambda, typename scalar>
void apply_vector(gpumat<scalar>& v1, const gpumat<scalar>& v2, Lambda func)
{
	scalar* v1data = v1.data();
	const scalar* v2data = v2.data();
	auto merge = [=] __device__(int eid) {
		v1data[eid] = func(v1data[eid], v2data[eid]);
	};
	size_t gridSize, blockSize;
	make_kernel_param(&gridSize, &blockSize, v1.size(), 512);
	map_kernel << <gridSize, blockSize >> > (v1.size(), merge);
	cudaDeviceSynchronize();
	cuda_error_check;
}

//class

template<typename scalar>
class gpumat
{
//data and shape
private:
	scalar* _data = nullptr;
	size_t _size = 0;
	size_t _row = 0;
	size_t _col = 0;
//helper func
public:
	typedef scalar scalar;
	
	//return referrance to data ptr
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

	bool isempty() const
	{
		return _data == nullptr;
	}

	bool sizefit(const gpumat& v2) const
	{
		return _row == v2.rows() && _col == v2.cols();
	}

	void resize(size_t row, size_t col)
	{
		size_t newsize = row * col;
		if (_size != newsize)
		{
			clear();
			_size = newsize;
			_row = row;
			_col = col;
			cudaMalloc(&_data, newsize * sizeof(scalar));
		}
		cuda_error_check;
	}

	void set_from_host(const scalar* host, const size_t row, const size_t col)
	{
		if (!isempty() && _size != row * col)
		{
			cudaFree(_data);
			cudaMalloc(&_data, _size * sizeof(scalar));
		}
		else if (isempty())
			cudaMalloc(&_data, row * col * sizeof(scalar));
		_row = row;
		_col = col;
		_size = row * col;
		cudaMemcpy(_data, host, _size * sizeof(scalar), cudaMemcpyHostToDevice);
		cuda_error_check;
	}

	void set_from_value(const scalar val) { init_array(data(), val, size()); }

	void set_from_value(const scalar val, size_t row, size_t col)
	{

	}

	//no mem check on host, be careful
	void download(scalar* host) const
	{
		cudaMemcpy(host, data(), _size * sizeof(scalar), cudaMemcpyDeviceToHost);
		cuda_error_check;
	}
//constructor & deconstructor
public:
	gpumat(void) :_size(0), _row(0), _col(0), _data(nullptr) {}

	explicit gpumat(size_t row, size_t col, scalar default_value = 0)
	{
		_size = row * col;
		_row = row;
		_col = col;
		cudaMalloc(&_data, _size * sizeof(Scalar));
		init_array(_data, default_value, _size);
		cuda_error_check;
	}

	virtual ~gpumat(void)
	{
		if (!isempty()) { cudaFree(_data); }
		cuda_error_check;
	}

	gpumat(const gpumat& v2)
	{
		_size = v2.size();
		_row = v2.rows();
		_col = v2.cols();
		cudaMalloc(&_data, sizeof(scalar) * _size);
		cudaMemcpy(_data, v2.data(), sizeof(scalar) * _size, cudaMemcpyDeviceToDevice);
		cuda_error_check;
	}

//math func
public:
	const gpumat& operator=(const gpumat& v2)
	{
		if (!isempty() && !sizefit(v2))
		{
			clear();
			_size = v2.size();
			_row = v2.rows();
			_col = v2.cols();
			cudaMalloc(&_data, _size * sizeof(scalar));
		}
		cudaMemcpy(_data, v2.data(), _size * sizeof(scalar), cudaMemcpyDeviceToDevice);
		cuda_error_check;
		return *this;
	}

	const gpumat& operator+=(const gpumat& v2)
	{
		if (!sizefit(v2))
		{
			printf("unmatched mat shape");
			exit(100);
		}
		apply_vector(*this, v2, [=]__device__(scalar e1, scalar e2) { return e1 + e2; });
		return *this;
	}

	const gpumat& operator-=(const gpumat& v2)
	{
		if (!sizefit(v2))
		{
			printf("unmatched mat shape");
			exit(100);
		}
		apply_vector(*this, v2, [=]__device__(scalar e1, scalar e2) { return e1 - e2; });
		return *this;
	}

	//this is element-wise product, use matprod for matrix product
	const gpumat& operator*=(const gpumat& v2)
	{
		if (!sizefit(v2))
		{
			printf("unmatched mat shape");
			exit(100);
		}
		apply_vector(*this, v2, [=]__device__(scalar e1, scalar e2) { return e1 * e2; });
		return *this;
	}

	//this is element-wise devide, no matrix devide provided
	const gpumat& operator/=(const gpumat& v2)
	{
		if (!sizefit(v2))
		{
			printf("unmatched mat shape");
			exit(100);
		}
		apply_vector(*this, v2, [=]__device__(scalar e1, scalar e2) { return e1 / e2; });
		return *this;
	}

	const gpumat& operator+=(const scalar val)
	{
		size_t grid, block;
		make_kernel_param(&grid, &block, size(), 512);
		map_kernel << <grid, block >> > (_data, size(), [=]__device__(int tid) { return _data[tid] + val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	const gpumat& operator-=(const scalar val)
	{
		size_t grid, block;
		make_kernel_param(&grid, &block, size(), 512);
		map_kernel << <grid, block >> > (_data, size(), [=]__device__(int tid) { return _data[tid] - val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	const gpumat& operator*=(const scalar val)
	{
		size_t grid, block;
		make_kernel_param(&grid, &block, size(), 512);
		map_kernel << <grid, block >> > (_data, size(), [=]__device__(int tid) { return _data[tid] * val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	const gpumat& operator/=(const scalar val)
	{
		size_t grid, block;
		make_kernel_param(&grid, &block, size(), 512);
		map_kernel << <grid, block >> > (_data, size(), [=]__device__(int tid) { return _data[tid] / val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	gpumat& transpose()
	{
		dim3 grid, block;
		make_kernel_param2d(&grid, &block, _col, _row, 32);
		transpose_kernel << <grid, block >> > (_data, _row, _col);
		return *this;
	}

	gpumat matprod(const gpumat& v2) const
	{
		if (_col != v2.rows())
		{
			printf("unmatched mat shape");
			exit(101);
		}
		gpumat rst(_row,v2.cols());
		dim3 grid, block;
		make_kernel_param2d(&grid, &block, rst.cols(), rst.rows());
		matprod_kernel << <grid, block >> > (data(), v2.data(), rst.data(), _row, _col, rst.cols());
		cudaDeviceSynchronize();
		cuda_error_check;
		return rst;
	}
};

#endif