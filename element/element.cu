//#include "element.h"
#include "element.cuh"
#include "../IO/matrixIO.h"
//#include "../culib/cudaCommon.cuh"
//#include "../culib/gpuVector.cuh"
//using namespace gv;


//gpumat<double> coef_g(576, 9);
double* x_host = nullptr;


void predict(const gmatd& x, gmatd& S, gmatd& dSdx, int& nel, torch::jit::Module model)
{
	if (!x_host)
		x_host = (double*)malloc(4 * nel * sizeof(double));
	x.download(x_host);
	for (int i = 0; i < nel; ++i)
	{
		auto input = torch::tensor({ double((x_host[i] - 0.3) / 0.4),double(x_host[i + nel] * 2 / PI),double(x_host[i + 2 * nel] * 2 / PI),double(x_host[i + 3 * nel] * 2 / PI) });
		input.requires_grad_();
		auto output = model({ input }).toTensor();
		//std::cout << input << '\n';
		//std::cout << output << '\n';
		auto data = output.data_ptr<double>();
		S.set_by_index(9 * i, 9, data, cudaMemcpyHostToDevice);
		for (int j = 0; j < 9; ++j)
		{
			auto xx = input.clone();
			xx.retain_grad();
			auto y = model({ xx }).toTensor();
			auto t = torch::zeros({ 9 });
			t[j] = 1;
			y.backward(t);
			static double ttt;
			ttt = xx.grad()[0].item().toDouble();
			dSdx.set_by_index(36 * i + j, 1, &ttt, cudaMemcpyHostToDevice);
			ttt = xx.grad()[1].item().toDouble();
			dSdx.set_by_index(36 * i + j + 9, 1, &ttt, cudaMemcpyHostToDevice);
			ttt = xx.grad()[2].item().toDouble();
			dSdx.set_by_index(36 * i + j + 18, 1, &ttt, cudaMemcpyHostToDevice);
			ttt = xx.grad()[3].item().toDouble();
			dSdx.set_by_index(36 * i + j + 27, 1, &ttt, cudaMemcpyHostToDevice);
		}
	}
}

void elastisity(const gmatd& S, const gmatd& coef, gmatd& sk)
{
	//double* S_h = new double[S.size()];
	//S.download(S_h);
	//double* coef_test = new double[576 * 9];
	//double* sktest = new double[sk.size()];
	//coef.download(coef_test);
	//sk.download(sktest);
	//savearr("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\S.csv", S_h, S.size());
	//savearr("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\coeftest.csv", coef_test, 576 * 9);
	//savearr("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\sk-pre.csv", sktest, sk.size());
	sk = matprod(coef, S);
	//sk.download(sktest);
	//savearr("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\sk-post.csv", sktest, sk.size());
}

void sensitivity(const gmatd& dSdx, const gmatd& coef, gmatd& dsKdx, gmatd& temp, const int& i, const int& nel)
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
	//temp.set_by_index(9 * r, 9, dSdx.data() + 36 * r + 9 * q, cudaMemcpyDeviceToDevice);
	//dsKdx = std::move(matprod(coef, temp));
	//temp.set_from_value(0.);
}

void filter(gmatd& v)
{
	
	double* ptr = v.data();
	size_t grid, block;
	//make_kernel_param(&grid, &block, size(), 512);
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