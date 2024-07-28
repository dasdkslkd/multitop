#pragma once
#ifndef _GMAT_CUH_
#define _GMAT_CUH_
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>
#include <vector>

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
__global__ void init_array_kernel(T* array, T value, int32_t array_size)
{
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < array_size)
		array[tid] = value;
}

template<typename T, typename Tout, typename Lam>
__global__ void map_kernel(T* g_data, Tout* dst, int32_t n, Lam func)
{
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		dst[tid] = func(g_data[tid]);
}

template<typename T, typename Lam>
__global__ void map_kernel(T* dst, int32_t n, Lam func)
{
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		dst[tid] = func(tid);
}

template<typename Lam>
__global__ void map_kernel(int32_t n, Lam func)
{
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		func(tid);
}

template<typename scalar>
__global__ void transpose_inplace_kernel(scalar* v, int32_t m, int32_t n)
{
	int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < n && row < m && col < row)
	{
		int32_t xx = col * m + row;
		int32_t yy = row * n + col;
		scalar temp = v[xx];
		v[xx] = v[yy];
		v[yy] = temp;
	}
}

template<typename scalar>
__global__ void transpose_kernel(scalar* vin, scalar* vout, int32_t m, int32_t n)
{
	int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < n && row < m && col <= row)
	{
		int32_t xx = col * m + row;
		int32_t yy = row * n + col;
		vout[xx] = vin[yy];
		vout[yy] = vin[xx];
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
			sum += v1[k * m + row] * v2[col * n + k];
		}
		v3[col * m + row] = sum;
	}
}

template<typename scalar>
__global__ void spmatprodcoo_kernel(scalar* rst, const scalar* v, const int32_t* rowid, const int32_t* colid, const scalar* val, int32_t rows, int32_t cols, int32_t nnz)
{
	int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < nnz)
	{
		atomicAdd(&rst[rowid[tid]], val[tid] * v[colid[tid]]);
	}
}

template<typename scalar>
__global__ void sum_partly_kernel(const scalar* v, size_t first, size_t len, scalar* sum)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len)
		atomicAdd(sum, v[first + tid]);
}

template<typename scalar>
__global__ void set_by_index_kernel(scalar* v, const int* idx, const scalar* val, int len)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len)
		v[idx[tid]] = val[tid];
}

//helper func
template<typename T>
void init_array(T* dev_array, T value, int32_t array_size)
{
	size_t grid_dim;
	size_t block_dim;
	//make_kernel_param(&grid_dim, &block_dim, array_size, 512);
	block_dim = 512;
	grid_dim = (array_size + 512 - 1) / 512;
	init_array_kernel << <grid_dim, block_dim >> > (dev_array, value, array_size);
	cudaDeviceSynchronize();
	cuda_error_check;
}

//inline void make_kernel_param(size_t* block_num, size_t* block_size, size_t num_tasks, size_t prefer_block_size = 512)
//{
//	*block_size = prefer_block_size;
//	*block_num = (num_tasks + prefer_block_size - 1) / prefer_block_size;
//}
//
//inline void make_kernel_param2d(dim3* grid, dim3* block, size_t nx, size_t ny, size_t pre_block = 32)
//{
//	*block = dim3(pre_block, pre_block, 0);
//	*grid = dim3(std::ceil(nx / pre_block), std::ceil(ny / pre_block), 0);
//}

template<typename Lambda, typename scalar>
void apply_vector(gpumat<scalar>& v1, const gpumat<scalar>& v2, Lambda func)
{
	scalar* v1data = v1.data();
	const scalar* v2data = v2.data();
	auto merge = [=] __device__(int32_t eid) {
		v1data[eid] = func(v1data[eid], v2data[eid]);
	};
	dim3 grid, block;
	//make_kernel_param(&gridSize, &blockSize, v1.size(), 512);
	block = dim3(512, 1, 1);
	grid = dim3((v1.size() + 512 - 1) / 512, 1, 1);
	map_kernel << <grid, block >> > (v1.size(), merge);
	cudaDeviceSynchronize();
	cuda_error_check;
}

template<typename Lambda, typename scalar>
void gen_vector(const gpumat<scalar>& v1, const gpumat<scalar>& v2, gpumat<scalar>& v3, Lambda func)
{
	const scalar* v1data = v1.data();
	const scalar* v2data = v2.data();
	scalar* v3data = v3.data();
	auto merge = [=] __device__(int32_t eid)
	{
		v3data[eid] = func(v1data[eid], v2data[eid]);
	};
	//make_kernel_param(&gridSize, &blockSize, v1.size(), 512);
	dim3 block, grid;
	block = dim3(512, 1, 1);
	grid = dim3((v1.size() + 512 - 1) / 512, 1, 1);
	map_kernel << <grid, block >> > (v3.size(), merge);
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

	gpumat& resize(size_t row, size_t col)
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
		return *this;
	}

	gpumat& resize()
	{
		_row = _size;
		_col = 1;
		return *this;
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

	//copy val to [first,first+len)
	//kind only receive cudaMemcpyHostToDevice & cudaMemcpyDeviceToDevice as valid
	void set_by_index(const size_t first, const size_t len, const  scalar* val, cudaMemcpyKind kind)
	{
		cudaMemcpy(_data + first, val, len * sizeof(scalar), kind);
	}

	//copy val to some position in gpumat by idx
	//device to device only
	void set_by_index(const int* idx, const size_t len, const scalar* val)
	{
		size_t grid, block;
		block = 512;
		grid = (len + 512 - 1) / 512;
		set_by_index_kernel << <grid, block >> > (_data, idx, val, len);
		//map_kernel << <grid, block >> > (len, [=]__device__(size_t eid) { _data[idx[eid]] = val[eid]; });
		cudaDeviceSynchronize();
		cuda_error_check;
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
		cudaMalloc(&_data, _size * sizeof(scalar));
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

	gpumat(gpumat&& v2) noexcept :_data(v2._data), _size(v2.size()), _row(v2.rows()), _col(v2.cols())
	{
		v2._data = nullptr;
		v2._size = v2._row = v2._col = 0;
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
		//make_kernel_param(&grid, &block, size(), 512);
		block = 512;
		grid = (size() + 512 - 1) / 512;
		map_kernel << <grid, block >> > (_data, size(), [=]__device__(int32_t tid) { return _data[tid] + val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	const gpumat& operator-=(const scalar val)
	{
		size_t grid, block;
		//make_kernel_param(&grid, &block, size(), 512);
		block = 512;
		grid = (size() + 512 - 1) / 512;
		map_kernel << <grid, block >> > (_data, size(), [=]__device__(int32_t tid) { return _data[tid] - val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	const gpumat& operator*=(const scalar val)
	{
		size_t grid, block;
		//make_kernel_param(&grid, &block, size(), 512);
		block = 512;
		grid = (size() + 512 - 1) / 512;
		map_kernel << <grid, block >> > (_data, size(), [=]__device__(int32_t tid) { return _data[tid] * val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	const gpumat& operator/=(const scalar val)
	{
		size_t grid, block;
		//make_kernel_param(&grid, &block, size(), 512);
		block = 512;
		grid = (size() + 512 - 1) / 512;
		map_kernel << <grid, block >> > (_data, size(), [=]__device__(int32_t tid) { return _data[tid] / val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return *this;
	}

	gpumat& transpose_inplace()
	{
		dim3 grid, block;
		//make_kernel_param2d(&grid, &block, _col, _row, 32);
		block = dim3(32, 32, 1);
		grid = dim3(std::ceil(_col / 32.), std::ceil(_row / 32.), 1);
		transpose_inplace_kernel << <grid, block >> > (_data, _row, _col);
		cudaDeviceSynchronize();
		size_t temp = _row;
		_row = _col;
		_col = temp;
		cuda_error_check;
		return *this;
	}

	gpumat transpose()
	{
		dim3 grid, block;
		block = dim3(32, 32, 1);
		grid = dim3(std::ceil(_col / 32.), std::ceil(_row / 32.), 1);
		gpumat v3(_col, _row);
		transpose_kernel << <grid, block >> > (_data, v3.data(), _row, _col);
		cudaDeviceSynchronize();
		cuda_error_check;
		return v3;
	}

	gpumat matprod(const gpumat& v2) const
	{
		if (_col != v2.rows())
		{
			printf("unmatched mat shape");
			exit(101);
		}
		gpumat rst(_row, v2.cols());
		dim3 grid, block;
		//make_kernel_param2d(&grid, &block, rst.cols(), rst.rows());
		block = dim3(32, 32, 1);
		grid = dim3(std::ceil(rst.cols() / 32.), std::ceil(rst.rows() / 32.), 1);
		matprod_kernel << <grid, block >> > (data(), v2.data(), rst.data(), _row, _col, rst.cols());
		cudaDeviceSynchronize();
		cuda_error_check;
		return rst;
	}

	gpumat operator+(const gpumat& v2) const
	{
		if (!sizefit(v2))
		{
			printf("unmatched mat shape");
			exit(100);
		}
		gpumat v3(_row, _col);
		gen_vector(*this, v2, v3, [=]__device__(scalar e1, scalar e2) { return e1 + e2; });
		return v3;
	}

	gpumat operator-(const gpumat& v2) const
	{
		if (!sizefit(v2))
		{
			printf("unmatched mat shape");
			exit(100);
		}
		gpumat v3(_row, _col);
		gen_vector(*this, v2, v3, [=]__device__(scalar e1, scalar e2) { return e1 - e2; });
		return v3;
	}

	gpumat operator+(const scalar val) const
	{
		gpumat v3(_row, _col);
		size_t grid, block;
		//make_kernel_param(&grid, &block, size(), 512);
		block = 512;
		grid = (size() + 512 - 1) / 512;
		map_kernel << <grid, block >> > (_data, v3.data(), size(), [=]__device__(scalar xx) { return xx + val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return v3;
	}

	gpumat operator-(const scalar val) const
	{
		gpumat v3(_row, _col);
		size_t grid, block;
		//make_kernel_param(&grid, &block, size(), 512);
		block = 512;
		grid = (size() + 512 - 1) / 512;
		map_kernel << <grid, block >> > (_data, v3.data(), size(), [=]__device__(scalar xx) { return xx - val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return v3;
	}

	gpumat operator*(const scalar val) const
	{
		gpumat v3(_row, _col);
		size_t grid, block;
		//make_kernel_param(&grid, &block, size(), 512);
		block = 512;
		grid = (size() + 512 - 1) / 512;
		map_kernel << <grid, block >> > (_data, v3.data(), size(), [=]__device__(scalar xx) { return xx * val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return v3;
	}

	gpumat operator/(const scalar val) const
	{
		gpumat v3(_row, _col);
		size_t grid, block;
		//make_kernel_param(&grid, &block, size(), 512);
		block = 512;
		grid = (size() + 512 - 1) / 512;
		map_kernel << <grid, block >> > (_data, v3.data(), size(), [=]__device__(scalar xx) { return xx / val; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return v3;
	}

	gpumat operator()(const std::vector<int32_t>& idx)
	{
		gpumat v2(idx.size(), 1);
		size_t grid, block;
		block = 512;
		grid = (size() + 512 - 1) / 512;
		map_kernel << <grid, block >> > (_data, v2.data(), idx.size(), [=]__device__(scalar xx) { return xx; });
		cudaDeviceSynchronize();
		cuda_error_check;
		return v2;
	}

	//get a value from _data[idx] for host
	scalar get_item(const size_t idx)
	{
		scalar val = 0;
		cudaMemcpy(&val, _data + idx, sizeof(scalar), cudaMemcpyDeviceToHost);
		cuda_error_check;
		return val;
	}

	scalar sum_partly(const size_t first, const size_t len)
	{
		scalar sum = 0;
		scalar* sum_g;
		cudaMalloc(&sum_g, sizeof(scalar));
		cudaMemcpy(sum_g, &sum, sizeof(scalar), cudaMemcpyHostToDevice);

		size_t grid, block;
		block = 512;
		grid = (len + 512 - 1) / 512;
		sum_partly_kernel << <grid, block >> > (data(), first, len, sum_g);
		cudaDeviceSynchronize();
		cudaMemcpy(&sum, sum_g, sizeof(scalar), cudaMemcpyDeviceToHost);
		cuda_error_check;
		return sum;
	}
};

template<typename scalar>
//use std::move(matprod(a,b)) to avoid deep copy
gpumat<scalar> matprod(const gpumat<scalar>& v1, const gpumat<scalar>& v2)
{
	if (v1.cols() != v2.rows())
	{
		printf("unmatched mat shape");
		exit(101);
	}
	gpumat<scalar> rst(v1.rows(), v2.cols());
	dim3 grid, block;
	//make_kernel_param2d(&grid, &block, rst.cols(), rst.rows());
	block = dim3(32, 32, 1);
	grid = dim3(std::ceil(rst.cols() / 32.), std::ceil(rst.rows() / 32.), 1);
	matprod_kernel << <grid, block >> > (v1.data(), v2.data(), rst.data(), v1.rows(), v1.cols(), rst.cols());
	cudaDeviceSynchronize();
	cuda_error_check;
	return rst;
}

//coo-type sparse matrix A*v
//non unique rowid in a col supported
template<typename scalar>
gpumat<scalar> spmatprodcoo(const gpumat<scalar>& v, const gpumat<int32_t>& rowid, const gpumat<int32_t>& colid, const gpumat<scalar>& val, const int32_t rows, const int32_t cols, const int32_t nnz)
{
	if (cols != v.rows())
	{
		printf("unmatched mat shape");
		exit(301);
	}
	gpumat<scalar> rst(rows, 1);
	uint32_t grid, block;
	block = 512;
	grid = (nnz + 511) / 512;
	spmatprodcoo_kernel << <grid, block >> > (rst.data(), v.data(), rowid.data(), colid.data(), val.data(), rows, cols, nnz);
	cudaDeviceSynchronize();
	cuda_error_check;
	return rst;
}

#endif