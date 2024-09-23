#include "fem.cuh"
#include"../IO/matrixIO.h"
#include "cusolverDn.h"
#include <utility>
gpumat<double> F;
extern float my_erfinvf(float a);

template<typename scalar>
void printgmat(const gpumat<scalar>& v)
{
	scalar* host = new scalar[v.size()];
	v.download(host);
	for (int i = 0; i < v.size(); ++i)
		std::cout << host[i] << ' ';
	std::cout << std::endl;
}

template<typename T>
void savegmat(gpumat<T>& v, string filename)
{
	T* host = new T[v.size()];
	v.download(host);
	savearr(filename, host, v.size());
}

__global__ void gatherK_kernel(int* ik, int* jk, double* sk, int* freeidx, double* K, int nfreedofs, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && sk[freeidx[i]] != 0)
		atomicAdd(&K[ik[i] + nfreedofs * jk[i]], sk[freeidx[i]]);
}

void solvefem(vector<int>& ikfree, vector<int>& jkfree, vector<double>& sk, vector<int>& freeidx, vector<int>& freedofs, Eigen::VectorXd& F, gpumat<double>& U)
{
	static std::vector<Eigen::Triplet<double>> triplist(freeidx.size());

	for (int i = 0; i < freeidx.size(); ++i)
	{
		triplist[i] = Eigen::Triplet<double>(ikfree[i], jkfree[i], sk[freeidx[i]]);
	}

	static auto K = Eigen::SparseMatrix<double>(freedofs.size(), freedofs.size());
	K.setFromTriplets(triplist.begin(), triplist.end());
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
	cg.compute(K);
	//savevec("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\sk-g.csv", sk);
	Eigen::VectorXd utemp;
	utemp = cg.solve(F(freedofs));
	static gpumat<double> uuu;
	uuu.set_from_host(utemp.data(), freedofs.size(), 1);
	static gpumat<int> idx;
	idx.set_from_host(freedofs.data(), freedofs.size(), 1);

	//printgmat(uuu);
	//printgmat(idx);
	//std::cout << sk << std::endl;
	//for (auto v : sk)
	//	std::cout << v << '\n';


	U.set_by_index(idx.data(), freedofs.size(), uuu.data());
}

void solvefem_g(gpumat<int>& ikfree, gpumat<int>& jkfree, gpumat<double>& sk, gpumat<int>& freeidx, vector<int>& freedofs, gpumat<double>& F, gpumat<double>& U)
{
	static gpumat<double> K(freedofs.size(), freedofs.size());
	static gpumat<double> b(freedofs.size(), 1);
	static gpumat<int> idx;
	static bool dummy = (idx.set_from_host(freedofs.data(), freedofs.size(), 1), 1);
	b = F(freedofs);
	K.set_from_value(0);

	dim3 grid = 0, block = 0;
	auto pair = kernel_param(freeidx.size());
	grid = pair.first;
	block = pair.second;
	gatherK_kernel << <grid, block >> > (ikfree.data(), jkfree.data(), sk.data(), freeidx.data(), K.data(), freedofs.size(), freeidx.size());
	cudaDeviceSynchronize();
	cuda_error_check;

	cusolverDnHandle_t handle = NULL;
	cusolverDnParams_t param;
	size_t workdevice = 0, workhost = 0;
	int32_t m = freedofs.size();
	void* d_work = nullptr;
	void* h_work = nullptr;
	int* d_info = nullptr;
	cudaMalloc(&d_info, sizeof(int));
	static gpumat<int64_t> ipiv(m, 1);
	ipiv.set_from_value(0);
	cusolverDnCreate(&handle);
	cusolverDnCreateParams(&param);
	cusolverDnSetAdvOptions(param, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);
	cusolverDnXgetrf_bufferSize(handle, param, m, m, CUDA_R_64F, K.data(), m, CUDA_R_64F, &workdevice, &workhost);
	cuda_error_check;
	cudaMalloc(&d_work, workdevice);
	h_work = malloc(workhost);
	cusolverDnXgetrf(handle, param, m, m, CUDA_R_64F, K.data(), m, ipiv.data(), CUDA_R_64F, d_work, workdevice, h_work, workhost, d_info);
	cuda_error_check;
	int info = 0;
	cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	if (info < 0)
		printf("%d-th param is wrong\n", -info);

	cusolverDnXgetrs(handle, param, CUBLAS_OP_N, m, 1, CUDA_R_64F, K.data(), m, ipiv.data(), CUDA_R_64F, b.data(), m, d_info);
	cuda_error_check;

	cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	if (info < 0)
		printf("%d-th param is wrong\n", -info);

	U.set_by_index(idx.data(), freedofs.size(), b.data());
}

void computefdf(gpumat<double>& U, gpumat<double>& dSdx, gpumat<double>& dskdx, gpumat<int>& ik, gpumat<int>& jk, double& f, gpumat<double>& dfdx, gpumat<double>& x, gpumat<double>& coef, int ndofs, bool multiobj, Eigen::VectorXd& F_host)
{
	static bool dummy = (F.set_from_host(F_host.data(), F_host.size(), 1), true);
	//double* U_h = new double[U.size()];
	//U.download(U_h);
	//savearr("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\Uh.csv", U_h, U.size());
	//savemat("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\Fh.csv", F_host);
	//printgmat(matprod(U.transpose(), F));
	f = matprod(U.transpose(), F).get_item(0);
	double sum = 0;
	int nel = static_cast<int>(dfdx.size() / 4);
	static std::vector<int32_t> idx(coef.rows());
	static gpumat<int> iknz(coef.rows(), 1), jknz(coef.rows(), 1);
	static gmatd dskdxnz(coef.rows(), 1);
	for (int i = 0; i < 4 * nel; ++i)
	{
		sensitivity(dSdx, coef, dskdx, idx, i, nel);
		iknz.set_by_index(0, coef.rows(), ik.data() + idx[0], cudaMemcpyDeviceToDevice);
		jknz.set_by_index(0, coef.rows(), jk.data() + idx[0], cudaMemcpyDeviceToDevice);
		dskdxnz.set_by_index(0, coef.rows(), dskdx.data() + idx[0], cudaMemcpyDeviceToDevice);
		dfdx.set_by_index(i, 1, matprod(U.transpose() * (-1.), spmatprodcoo(U, iknz, jknz, dskdxnz, ndofs, ndofs)).data(), cudaMemcpyDeviceToDevice);
		//cout << dfdx.get_item(i) << ' ';
		//dfdx.set_by_index(i, 1, matprod(U.transpose() * (-1.), spmatprodcoo(U, ik, jk, dskdx, ndofs, ndofs)).data(), cudaMemcpyDeviceToDevice);
		//cout << dfdx.get_item(i) << endl;
		//if (i == 0)
		//{
		//	string outpath = "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\";
		//	savegmat(ik, outpath + "ikg.txt");
		//	savegmat(jk, outpath + "jkg.txt");
		//	savegmat(ik(idx), outpath + "iknz.txt");
		//	savegmat(jk(idx), outpath + "jknz.txt");
		//	savegmat(dskdx, outpath + "dskdxg.txt");
		//	savegmat(dskdx(idx), outpath + "dskdxnz.txt");
		//	savevec(outpath + "idx.txt", idx);
		//	//savegmat(U, outpath + "Ug.txt");
		//}

		if (multiobj && i < nel && x.get_item(i)>1e-3)
		{
			sum += std::exp(-std::pow(my_erfinvf(2 * x.get_item(i) - 1), 2));
			double ttt = dfdx.get_item(i) + 400 / std::sqrt(3. * PI) / nel * my_erfinvf(2 * x.get_item(i) - 1);
			dfdx.set_by_index(i, 1, &ttt, cudaMemcpyHostToDevice);
		}
	}
	f -= 200 / std::sqrt(3) / PI / nel * sum;
}

void computegdg(gpumat<double>& x, gpumat<double>& g, gpumat<double>& dgdx, const double& volfrac, const int m, const int nel)
{
	static double theta_min = PI / 18;
	double ttt = x.sum_partly(0, nel) / nel / volfrac - 1;
	g.set_by_index(m - 1, 1, &ttt, cudaMemcpyHostToDevice);
	//for (int i = 0; i < nel; ++i)
	//{
		//ttt = theta_min - x.get_item(i + nel) - x.get_item(i + 2 * nel) - x.get_item(i + 3 * nel);
		//g.set_by_index(i, 1, &ttt, cudaMemcpyHostToDevice);

		//ttt = 1e-3 - x.get_item(i + nel) * x.get_item(i + 2 * nel) * x.get_item(i + 3 * nel);
		//g.set_by_index(i + nel, 1, &ttt, cudaMemcpyHostToDevice);
		//ttt = -x.get_item(i + 2 * nel) * x.get_item(i + 3 * nel);
		//dgdx.set_by_index(m * (i + nel) + i + nel, 1, &ttt, cudaMemcpyHostToDevice);
		//ttt = -x.get_item(i + nel) * x.get_item(i + 3 * nel);
		//dgdx.set_by_index(m * (i + 2 * nel) + i + nel, 1, &ttt, cudaMemcpyHostToDevice);
		//ttt = -x.get_item(i + nel) * x.get_item(i + 2 * nel);
		//dgdx.set_by_index(m * (i + 3 * nel) + i + nel, 1, &ttt, cudaMemcpyHostToDevice);

	//}
	static bool dummy = false;
	if (!dummy)
	{
		//for (int i = 0; i < nel; ++i)
		//{
		//	ttt = -1;
		//	dgdx.set_by_index(m * (i + nel) + i, 1, &ttt, cudaMemcpyHostToDevice);
		//	dgdx.set_by_index(m * (i + 2 * nel) + i, 1, &ttt, cudaMemcpyHostToDevice);
		//	dgdx.set_by_index(m * (i + 3 * nel) + i, 1, &ttt, cudaMemcpyHostToDevice);
		//}
		for (int i = 0; i < 4 * nel; ++i)
		{
			ttt = 1. / nel / volfrac;
			double tttt = 0.;
			if (i < nel)
				dgdx.set_by_index(i * m + m - 1, 1, &ttt, cudaMemcpyHostToDevice);
			else
				dgdx.set_by_index(i * m + m - 1, 1, &tttt, cudaMemcpyHostToDevice);
		}
	}
	dummy = true;
}