#ifndef _FEM_CUH_
#define _FEM_CUH_
#define EIGEN_NO_CUDA
#include "../element/element.cuh"
#include <vector>
#include<Eigen/Core>
#include<Eigen/SparseCore>
#include<Eigen/IterativeLinearSolvers>
#include<eigen3/unsupported/Eigen/KroneckerProduct>
using std::vector;

void solvefem(vector<int>& ikfree, vector<int>& jkfree, vector<double>& sk, vector<int>& freeidx, vector<int>& freedofs, Eigen::VectorXd& F, gpumat<double>& U);

void computefdf(gpumat<double>& U, gpumat<double>& dSdx, gpumat<double>& dskdx, gpumat<int>& ik, gpumat<int>& jk, double& f, gpumat<double>& dfdx, gpumat<double>& x, gpumat<double>& coef, int ndofs, bool multiobj, Eigen::VectorXd& F_host);

void computegdg(gpumat<double>& x, gpumat<double>& g, gpumat<double>& dgdx, const double& volfrac, const int m, const int nel);
#endif // !_FEM_CUH_
