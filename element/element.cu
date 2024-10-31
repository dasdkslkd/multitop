//#include "element.h"
#include "element.cuh"
#include "../IO/matrixIO.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
//#include <omp.h>
//#include "../culib/cudaCommon.cuh"
//#include "../culib/gpuVector.cuh"
//using namespace gv;


//gpumat<double> coef_g(576, 9);
double* x_host = nullptr;
py::scoped_interpreter guard{};

//void predict(const gmatd& x, gmatd& S, gmatd& dSdx, int& nel, torch::jit::Module model)
//{
//	if (!x_host)
//		x_host = (double*)malloc(4 * nel * sizeof(double));
//	x.download(x_host);
//	static vector<double> S_h(9 * nel);
//	static vector<double> dSdx_h(36 * nel);
////#pragma omp parallel for
//	for (int i = 0; i < nel; ++i)
//	{
//		auto input = torch::tensor({ double((x_host[i] - 0.3) / 0.4),double(x_host[i + nel] * 2 / PI),double(x_host[i + 2 * nel] * 2 / PI),double(x_host[i + 3 * nel] * 2 / PI) });
//		input.requires_grad_();
//		auto output = model({ input }).toTensor();
//		//std::cout << input << '\n';
//		//std::cout << output << '\n';
//		auto data = output.data_ptr<double>();
//		std::copy(data, data + 9, S_h.data() + 9 * i);
//		//S.set_by_index(9 * i, 9, data, cudaMemcpyHostToDevice);
//		for (int j = 0; j < 9; ++j)
//		{
//			auto xx = input.clone();
//			xx.retain_grad();
//			auto y = model({ xx }).toTensor();
//			auto t = torch::zeros({ 9 });
//			t[j] = 1;
//			y.backward(t);
//			static double ttt;
//			ttt = xx.grad()[0].item().toDouble();
//			//dSdx.set_by_index(36 * i + j, 1, &ttt, cudaMemcpyHostToDevice);
//			dSdx_h[36 * i + j] = ttt;
//			ttt = xx.grad()[1].item().toDouble();
//			//dSdx.set_by_index(36 * i + j + 9, 1, &ttt, cudaMemcpyHostToDevice);
//			dSdx_h[36 * i + j + 9] = ttt;
//			ttt = xx.grad()[2].item().toDouble();
//			//dSdx.set_by_index(36 * i + j + 18, 1, &ttt, cudaMemcpyHostToDevice);
//			dSdx_h[36 * i + j + 18] = ttt;
//			ttt = xx.grad()[3].item().toDouble();
//			//dSdx.set_by_index(36 * i + j + 27, 1, &ttt, cudaMemcpyHostToDevice);
//			dSdx_h[36 * i + j + 27] = ttt;
//		}
//	}
//	//S.set_from_value(0.);
//	S.set_by_index(0, 9 * nel, S_h.data(), cudaMemcpyHostToDevice);
//	//dSdx.set_from_value(0.);
//	dSdx.set_by_index(0, 36 * nel, dSdx_h.data(), cudaMemcpyHostToDevice);
//}

template<typename T>
py::array_t<T> _ptr_to_arrays_1d(T* data, py::ssize_t col) {
	// 创建 NumPy 数组，并指定数据指针
	return py::array_t<T>(py::buffer_info(
		data,                           // 数据指针
		sizeof(T),                     // 每个元素的大小
		py::format_descriptor<T>::format(), // 数据格式
		1,                              // 维度数量
		{ col },                       // 维度大小
		{ sizeof(T) }                  // 每个维度的步长
	));
}

void predict_py(gmatd& x, gmatd& S, gmatd& dSdx, int& nel)
{
	if (!x_host)
		x_host = (double*)malloc(4 * nel * sizeof(double));
	x.resize(nel, 4);
	x.transpose().download(x_host);
	x.resize(4 * nel, 1);
	
	py::module sys = py::module::import("sys");
	sys.attr("path").attr("append")("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop");
	//py::print(sys.attr("path"));

	auto input = _ptr_to_arrays_1d(x_host, 4 * nel);
	py::module model = py::module::import("calsds");

	py::tuple rst = model.attr("predict")(input,nel,4);
	auto y = rst[0].cast<py::array_t<double>>();
	auto J = rst[1].cast<py::array_t<double>>();

	auto bufy = y.unchecked<1>();
	auto bufJ = J.unchecked<1>();
	S.set_by_index(0, 9 * nel, bufy.data(0), cudaMemcpyHostToDevice);
	dSdx.set_by_index(0, 36 * nel, bufJ.data(0), cudaMemcpyHostToDevice);
}

void elastisity(const gmatd& S, const gmatd& coef, gmatd& sk)
{
	sk = matprod(coef, S);
}

void sensitivity(const gmatd& dSdx, const gmatd& coef, gmatd& dsKdx, std::vector<int32_t>& idx, const int& i, const int& nel)
{
	static int q;
	static int r;
	q = i / nel;
	r = i - q * nel;
	static gmatd tmp1(9, 1), tmp2(coef.rows(), 1);
	tmp1.set_by_index(0, 9, dSdx.data() + 36 * r + 9 * q, cudaMemcpyDeviceToDevice);
	tmp2 = matprod(coef, tmp1);
	dsKdx.set_from_value(0.);
	dsKdx.set_by_index(coef.rows() * r, coef.rows(), tmp2.data(), cudaMemcpyDeviceToDevice);
	for (int kk = 0; kk < coef.rows(); ++kk)
		idx[kk] = coef.rows() * r + kk;
}

void filter(gmatd& v)
{

	double* ptr = v.data();
	size_t grid, block;
	block = 512;
	grid = (v.size() + 512 - 1) / 512;
	size_t r = v.rows();
	auto merge = [=] __device__(int eid)
	{
		double rho_min = 0.3;
		double theta_min = PI / 18;
		double lam1 = 600.;
		double lam2 = 60 * 180 / PI;
		if (eid < r)
			ptr[eid] /= (1 + std::exp(-lam1 * (ptr[eid] - rho_min)));
		else
			ptr[eid] = std::max(ptr[eid], theta_min) / (1 + exp(-lam2 * (ptr[eid] - theta_min / 2)));
	};
	map_kernel << <grid, block >> > (v.size(), merge);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void filter(double* x, int nel)
{
	double rho_min = 0.3;
	double theta_min = PI / 18;
	double lam1 = 600.;
	double lam2 = 60 * 180 / PI;
	for (int i = 0; i < 4 * nel; ++i)
	{
		if (i < nel)
			x[i] /= (1 + std::exp(-lam1 * (x[i] - rho_min)));
		else
			x[i] = std::max(x[i], theta_min) / (1 + exp(-lam2 * (x[i] - theta_min / 2)));
	}
}