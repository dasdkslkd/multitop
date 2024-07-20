#include "../fem/fem.cuh"
//#include "mmacontext.h"
#include <torch/script.h>
extern "C"
void solve_g(
	//mma
	int m,int n,double* xval,double f,double*dfdx,double*g,double*dgdx,double*xmin,double*xmax,
	//fem
	int nelx,int nely,int nelz,double volfrac,bool multiobj,vector<double> F,vector<int> freedofs,vector<int> freeidx,double*S,double*dSdx,vector<int> ik,vector<int> jk,vector<int> ikfree,vector<int> jkfree,vector<double> sk,vector<double>dskdx,vector<double> U,
	//elem
	double* temp,torch::jit::Module model)
{
	printf("1");
}