#ifndef _FEM_CUH_
#define _FEM_CUH_
#define EIGEN_NO_CUDA
#include "../element/element.cuh"
#include <vector>
#include<Eigen/Core>
#include<Eigen/SparseCore>
#include<Eigen/IterativeLinearSolvers>
#include<unsupported/Eigen/KroneckerProduct>
using std::vector;

void solvefem(vector<int>& ikfree, vector<int>& jkfree, vector<double>& sk, vector<int>& freeidx, vector<int>& freedofs, Eigen::VectorXd& F/*, gpumat<double>& U*/);

void solvefem_g(/*gpumat<int>& ikfree, gpumat<int>& jkfree, gpumat<double>& sk, gpumat<int>& freeidx, gpumat<int>& freedofs, gpumat<double>& F, gpumat<double>& U*/);

void solvefemsp_g();

void computefdf(/*gpumat<double>& U, gpumat<double>& dSdx, gpumat<double>& dskdx, gpumat<int>& ik, gpumat<int>& jk, */double& f, /*gpumat<double>& dfdx, gpumat<double>& x, gpumat<double>& coef, */int ndofs, bool multiobj/*, gpumat<double>& F*/);

void computegdg(/*gpumat<double>& x, gpumat<double>& g, gpumat<double>& dgdx,*/ const double& volfrac, const int m, const int nel);
#endif // !_FEM_CUH_
