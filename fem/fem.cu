#include "fem.cuh"
#include"../IO/matrixIO.h"
#include "cusolverDn.h"
#include "cusolverSp.h"
#include "cusparse_v2.h"
#include <utility>
//gpumat<double> F;
extern float my_erfinvf(float a);

extern gpumat<double> x, dfdx, g, dgdx, xmin, xmax, F, S, dSdx, sk, dskdx, U, temp, coef2, xold1g, xold2g, lowg, uppg;
extern gpumat<int> freedofs, freeidx, ik, jk, ikfree, jkfree;

gpumat<int> idxmap, ikf_squeez, jkf_squeez, permutation;
gpumat<double> skf_squeez, skf_sorted;

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

__global__ void gatherKsp_kernel(double* sk, int* freeidx, double* skf, int* map, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		atomicAdd(&skf[map[i]], sk[freeidx[i]]);
}

__global__ void permute_kernel(double* skf_sort, double* skf_sqz, int* permut, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		skf_sort[i] = skf_sqz[permut[i]];
}

__global__ void caldfdx_kernel(double* U, int* iknz, int* jknz, double* dskdxnz, double* dfdx, int nnz, int n, int nel)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	//dfdx[i] = 0;
	int q = i / nel;
	int r = i % nel;
	if (i < n && j < nnz)
	{
		atomicAdd(&dfdx[i], -U[iknz[i * nnz + j]] * U[jknz[i * nnz + j]] * dskdxnz[(4 * r + q) * nnz + j]);
	}
}

inline void hash_squeeze()
{
	vector<int> hashtable(freedofs.size() * freedofs.size(), -1);
	//idxmap.resize(ikfree.size(), 1);
	//ikf_squeez.resize(ikfree.size(), 1);
	//jkf_squeez.resize(jkfree.size(), 1);
	int* ikfree_h = new int[ikfree.size()];
	int* jkfree_h = new int[jkfree.size()];
	int* idxmap_h = new int[ikfree.size()];
	int* ikf_sqz_h = new int[ikfree.size()];
	int* jkf_sqz_h = new int[jkfree.size()];
	ikfree.download(ikfree_h);
	jkfree.download(jkfree_h);
	int n = freedofs.size();
	int count = 0;
	for (int i = 0; i < ikfree.size(); ++i)
	{
		int idx = ikfree_h[i] + n * jkfree_h[i];
		if (hashtable[idx] == -1)
		{
			hashtable[idx] = count;
			ikf_sqz_h[count] = ikfree_h[i];
			jkf_sqz_h[count] = jkfree_h[i];
			//ikf_squeez.set_by_index(count, 1, ikfree.data() + i, cudaMemcpyDeviceToDevice);
			//jkf_squeez.set_by_index(count, 1, jkfree.data() + i, cudaMemcpyDeviceToDevice);
			++count;
		}
		//idxmap.set_by_index(i, 1, &hashtable[idx], cudaMemcpyHostToDevice);
		idxmap_h[i] = hashtable[idx];
	}
	//ikf_squeez.resize(count, 1);
	//jkf_squeez.resize(count, 1);
	//skf_squeez.resize(count, 1);
	ikf_squeez.set_from_host(ikf_sqz_h, count, 1);
	jkf_squeez.set_from_host(jkf_sqz_h, count, 1);
	idxmap.set_from_host(idxmap_h, ikfree.size(), 1);
	skf_squeez.resize(count, 1);
	delete[] ikfree_h, jkfree_h, ikf_sqz_h, jkf_sqz_h, idxmap_h;
}

inline void sortcoo()
{
	void* d_buffer = nullptr;
	int nnz = ikf_squeez.size();
	skf_sorted = gpumat<double>(nnz, 1);
	permutation = gpumat<int>(nnz, 1);
	size_t bufferSize = 0;
	cusparseHandle_t handle = NULL;
	cusparseSpVecDescr_t vec_permutation;
	cusparseDnVecDescr_t vec_values;
	cusparseCreate(&handle);
	cusparseCreateSpVec(&vec_permutation, nnz, nnz, permutation.data(), skf_sorted.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cusparseCreateDnVec(&vec_values, nnz, skf_squeez.data(), CUDA_R_64F);
	cusparseXcoosort_bufferSizeExt(handle, freedofs.size(), freedofs.size(), nnz, ikf_squeez.data(), jkf_squeez.data(), &bufferSize);
	cudaMalloc(&d_buffer, bufferSize);
	cusparseCreateIdentityPermutation(handle, nnz, permutation.data());
	cusparseXcoosortByRow(handle, freedofs.size(), freedofs.size(), nnz, ikf_squeez.data(), jkf_squeez.data(), permutation.data(), d_buffer);
	cusparseGather(handle, vec_values, vec_permutation);
	cusparseDestroySpVec(vec_permutation);
	cusparseDestroyDnVec(vec_values);
	cudaFree(d_buffer);
	cusparseDestroy(handle);
	cuda_error_check;
}

void solvefem(vector<int>& ikfree, vector<int>& jkfree, vector<double>& sk, vector<int>& freeidx, vector<int>& freedofs, Eigen::VectorXd& F_h/*, gpumat<double>& U*/)
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
	utemp = cg.solve(F_h(freedofs));
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

void solvefem_g(/*gpumat<int>& ikfree, gpumat<int>& jkfree, gpumat<double>& sk, gpumat<int>& freeidx, gpumat<int>& freedofs, gpumat<double>& F, gpumat<double>& U*/)
{
	static gpumat<double> K(freedofs.size(), freedofs.size());
	static gpumat<double> b(freedofs.size(), 1);
	//static gpumat<int> idx;
	//static bool dummy = (idx.set_from_host(freedofs.data(), freedofs.size(), 1), 1);
	b = F(freedofs);
	K.set_from_value(0);

	auto [grid,block] = kernel_param(freeidx.size());
	gatherK_kernel << <grid, block >> > (ikfree.data(), jkfree.data(), sk.data(), freeidx.data(), K.data(), freedofs.size(), freeidx.size());
	cudaDeviceSynchronize();
	cuda_error_check;

	//savegmat(K, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\Kg.txt");

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

	cudaFree(d_info);
	cudaFree(d_work);
	free(h_work);
	cusolverDnDestroyParams(param);
	cusolverDnDestroy(handle);

	U.set_by_index(freedofs.data(), freedofs.size(), b.data());
	//savegmat(U, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\Ug.txt");
}

void solvefemsp_g()
{
	static gpumat<double> b(freedofs.size(), 1);
	b = F(freedofs);
	static bool dummy2 = (hash_squeeze(), 0);
	auto [grid, block] = kernel_param(freeidx.size());
	skf_squeez.set_from_value(0);
	gatherKsp_kernel << <grid, block >> > (sk.data(), freeidx.data(), skf_squeez.data(), idxmap.data(), freeidx.size());
	cudaDeviceSynchronize();
	cuda_error_check;

	////sort coo format, assume all elements in skf_squeez are nonzero
	//int* d_permutation = nullptr;
	//void* d_buffer = nullptr;
	////double* sorted_skf = nullptr;
	//int nnz = ikf_squeez.size();
	//gpumat<double> sorted_skf(nnz, 1);
	//size_t bufferSize = 0;
	//cudaMalloc((void**)&d_permutation, nnz * sizeof(int));
	////cudaMalloc((void**)&sorted_skf, nnz * sizeof(double));
	//cusparseHandle_t handle = NULL;
	//cusparseSpVecDescr_t vec_permutation;
	//cusparseDnVecDescr_t vec_values;
	//cusparseCreate(&handle);
	//cusparseCreateSpVec(&vec_permutation, nnz, nnz, d_permutation, sorted_skf.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	//cusparseCreateDnVec(&vec_values, nnz, skf_squeez.data(), CUDA_R_64F);
	////m,n are not used
	//cusparseXcoosort_bufferSizeExt(handle, freedofs.size(), freedofs.size(), nnz, ikf_squeez.data(), jkf_squeez.data(), &bufferSize);
	//cudaMalloc(&d_buffer, bufferSize);
	//cusparseCreateIdentityPermutation(handle, nnz, d_permutation);
	//cusparseXcoosortByRow(handle, freedofs.size(), freedofs.size(), nnz, ikf_squeez.data(), jkf_squeez.data(), d_permutation, d_buffer);
	//cusparseGather(handle, vec_values, vec_permutation);
	//
	//int* h_per = new int[nnz];
	//cudaMemcpy(h_per, d_permutation, nnz * sizeof(int), cudaMemcpyDeviceToHost);
	//savearr("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\permutation.txt", h_per, nnz);

	//cusparseDestroySpVec(vec_permutation);
	//cusparseDestroyDnVec(vec_values);
	//cudaFree(d_permutation);
	//cudaFree(d_buffer);

	static bool dummy3 = (sortcoo(), 0);
	//for (int i = 0; i < ikf_squeez.size(); ++i)
	//{
	//	skf_sorted.set_by_index(i, 1, skf_squeez.data() + permutation.get_item(i), cudaMemcpyDeviceToDevice);
	//}
	auto [grid2, block2] = kernel_param(ikf_squeez.size());
	permute_kernel << <grid2, block2 >> > (skf_sorted.data(), skf_squeez.data(), permutation.data(), ikf_squeez.size());


	gpumat<double> K(freedofs.size(), freedofs.size());
	//K.set_from_value(0);
	//for (int i = 0; i < ikf_squeez.size(); ++i)
	//	K.set_by_index(ikf_squeez.get_item(i) + freedofs.size() * jkf_squeez.get_item(i), 1, skf_sorted.data() + i, cudaMemcpyDeviceToDevice);
	//savegmat(K, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\Kspg1.txt");
	//savegmat(sorted_skf, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\sorted_skf1.txt");
	//savegmat(ikf_squeez, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\ikf_squeez1.txt");
	//savegmat(jkf_squeez, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\jkf_squeez1.txt");
	
	

	//coo2csr
	//int* csrRowPtr;
	cusparseHandle_t handle = NULL;
	cusparseCreate(&handle);
	gpumat<int> csrRowPtr(freedofs.size() + 1, 1);
	//cudaMalloc(&csrRowPtr, (freedofs.size() + 1) * sizeof(int));
	cusparseXcoo2csr(handle, ikf_squeez.data(), ikf_squeez.size(), freedofs.size(), csrRowPtr.data(), CUSPARSE_INDEX_BASE_ZERO);
	cusparseDestroy(handle);

	//for (int i = 0; i < ikf_squeez.size(); ++i)
	//	K.set_by_index(ikf_squeez.get_item(i) + freedofs.size() * jkf_squeez.get_item(i), 1, skf_sorted.data() + i, cudaMemcpyDeviceToDevice);
	//savegmat(K, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\Kspg2.txt");
	//savegmat(skf_sorted, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\sorted_skf2.txt");
	//savegmat(ikf_squeez, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\ikf_squeez2.txt");
	//savegmat(jkf_squeez, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\jkf_squeez2.txt");
	
	//solve
	int* singularity = (int*)malloc(sizeof(int));
	gpumat<double> x(freedofs.size(), 1);
	cusolverSpHandle_t handle2 = NULL;
	cusparseMatDescr_t descrA = NULL;
	cusolverSpCreate(&handle2);
	cusparseCreateMatDescr(&descrA);
	//cusolverSpDcsrlsvchol(handle2, freedofs.size(), nnz, descrA, skf_squeez.data(), csrRowPtr, jkf_squeez.data(), b.data(), 1e-6, 0, x.data(), singularity);
	cusolverSpDcsrlsvqr(handle2, freedofs.size(), ikf_squeez.size(), descrA, skf_sorted.data(), csrRowPtr.data(), jkf_squeez.data(), b.data(), 1e-6, 0, x.data(), singularity);
	cout << *singularity << endl;
	free(singularity);
	cusolverSpDestroy(handle2);
	cusparseDestroyMatDescr(descrA);
	U.set_by_index(freedofs.data(), freedofs.size(), x.data());
	//savegmat(U, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\Uspg.txt");
	cuda_error_check;
}

void computefdf(/*gpumat<double>& U, gpumat<double>& dSdx, gpumat<double>& dskdx, gpumat<int>& ik, gpumat<int>& jk, */double& f,/* gpumat<double>& dfdx, gpumat<double>& x, gpumat<double>& coef2, */int ndofs, bool multiobj/*, gpumat<double>& F*/)
{
	//static bool dummy = (F.set_from_host(F_host.data(), F_host.size(), 1), true);
	//double* U_h = new double[U.size()];
	//U.download(U_h);
	//savearr("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\Uh.csv", U_h, U.size());
	//savemat("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\Fh.csv", F_host);
	//printgmat(matprod(U.transpose(), F));
	f = matprod(U.transpose(), F).get_item(0);
	double sum = 0;
	int nel = static_cast<int>(dfdx.size() / 4);
	static std::vector<int32_t> idx(coef2.rows());
	static gpumat<int> iknz(coef2.rows(), dfdx.size()), jknz(coef2.rows(), dfdx.size());
	static gmatd dskdxnz(coef2.rows(), dfdx.size());
	//static gpumat<int> iknz2(coef2.rows(), 1), jknz2(coef2.rows(), 1);
	//static gmatd dskdxnz2(coef2.rows(), 1);

	static gmatd dskdx_all(coef2.rows(), dfdx.size());
	dskdx_all = matprod(coef2, dSdx);
	iknz.set_by_index(0, ik.size(), ik.data(), cudaMemcpyDeviceToDevice);
	iknz.set_by_index(ik.size(), ik.size(), ik.data(), cudaMemcpyDeviceToDevice);
	iknz.set_by_index(2 * ik.size(), ik.size(), ik.data(), cudaMemcpyDeviceToDevice);
	iknz.set_by_index(3 * ik.size(), ik.size(), ik.data(), cudaMemcpyDeviceToDevice);
	jknz.set_by_index(0, jk.size(), jk.data(), cudaMemcpyDeviceToDevice);
	jknz.set_by_index(jk.size(), jk.size(), jk.data(), cudaMemcpyDeviceToDevice);
	jknz.set_by_index(2 * jk.size(), jk.size(), jk.data(), cudaMemcpyDeviceToDevice);
	jknz.set_by_index(3 * jk.size(), jk.size(), jk.data(), cudaMemcpyDeviceToDevice);
	dim3 block(32, 32, 1);
	dim3 grid((dfdx.size() + block.x - 1) / block.x, (coef2.rows() + block.y - 1) / block.y);
	dfdx.set_from_value(0);
	caldfdx_kernel << <grid, block >> > (U.data(), iknz.data(), jknz.data(), dskdx_all.data(), dfdx.data(), coef2.rows(), dfdx.size(), nel);
	cudaDeviceSynchronize();
	cuda_error_check;

	//savegmat(dfdx, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\dfdx1.txt");

	//for (int i = 0; i < 4 * nel; ++i)
	//{
	//	sensitivity(dSdx, coef2, dskdx, idx, i, nel);
	//	iknz2.set_by_index(0, coef2.rows(), ik.data() + idx[0], cudaMemcpyDeviceToDevice);
	//	jknz2.set_by_index(0, coef2.rows(), jk.data() + idx[0], cudaMemcpyDeviceToDevice);
	//	dskdxnz2.set_by_index(0, coef2.rows(), dskdx.data() + idx[0], cudaMemcpyDeviceToDevice);
	//	dfdx.set_by_index(i, 1, matprod(U.transpose() * (-1.), spmatprodcoo(U, iknz2, jknz2, dskdxnz2, ndofs, ndofs)).data(), cudaMemcpyDeviceToDevice);
	//	//cout << dfdx.get_item(i) << ' ';
	//	//dfdx.set_by_index(i, 1, matprod(U.transpose() * (-1.), spmatprodcoo(U, ik, jk, dskdx, ndofs, ndofs)).data(), cudaMemcpyDeviceToDevice);
	//	//cout << dfdx.get_item(i) << endl;
	//	//if (i == 0)
	//	//{
	//	//	string outpath = "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\";
	//	//	savegmat(ik, outpath + "ikg.txt");
	//	//	savegmat(jk, outpath + "jkg.txt");
	//	//	savegmat(ik(idx), outpath + "iknz.txt");
	//	//	savegmat(jk(idx), outpath + "jknz.txt");
	//	//	savegmat(dskdx, outpath + "dskdxg.txt");
	//	//	savegmat(dskdx(idx), outpath + "dskdxnz.txt");
	//	//	savevec(outpath + "idx.txt", idx);
	//	//	//savegmat(U, outpath + "Ug.txt");
	//	//}

	//	if (multiobj && i < nel && x.get_item(i)>1e-3)
	//	{
	//		sum += std::exp(-std::pow(my_erfinvf(2 * x.get_item(i) - 1), 2));
	//		double ttt = dfdx.get_item(i) + 400 / std::sqrt(3. * PI) / nel * my_erfinvf(2 * x.get_item(i) - 1);
	//		dfdx.set_by_index(i, 1, &ttt, cudaMemcpyHostToDevice);
	//	}
	//}
	//savegmat(dfdx, "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\dfdx2.txt");
	f -= 200 / std::sqrt(3) / PI / nel * sum;
}

void computegdg(/*gpumat<double>& x, gpumat<double>& g, gpumat<double>& dgdx, */const double& volfrac, const int m, const int nel)
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