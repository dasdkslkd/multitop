#pragma once
//#include<torch/script.h>
#include<Eigen/Core>
#include<Eigen/SparseCore>
#include<Eigen/IterativeLinearSolvers>
#include<eigen3/unsupported/Eigen/KroneckerProduct>
//#include<iostream>
//#include<algorithm>
#include "element.h"
//#define PI acos(-1.f)
//using namespace std;

class Femproblem
{
public:
	int nelx;
	int nely;
	int nelz;
	double volfrac;
	bool multiobj;
	int nel;
	int ndof;
	Eigen::VectorXd F;
	//Eigen::Vector<Eigen::Index,Eigen::Dynamic> freedofs;
	//Eigen::VectorXi freedofs;
	vector<int> freedofs;
	vector<int> freeidx;
	double* x;
	double* S;
	double* dSdx;
	Eigen::VectorXi ik;
	Eigen::VectorXi jk;
	Eigen::VectorXi ikfree;
	Eigen::VectorXi jkfree;
	Eigen::VectorXd sk;
	Eigen::VectorXd dskdx;
	Eigen::VectorXd U;
	Eigen::SparseMatrix<double> K;
	Eigen::SparseMatrix<double> dKdx;
	vector<Eigen::Triplet<double>> trip_list;
	vector<Eigen::Triplet<double>> trip_forsk;
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
	spinodal elem;

	Femproblem(int nelx, int nely, int nelz, double volfrac, bool multiobj);

	~Femproblem();

	void setforce(const Eigen::VectorXd& force)
	{
		F = force;
	}

	void setconstrain(vector<int>&& fixeddofs);

	void solvefem();

	//计算目标函数及其导数
	void computefdf(double& f, double* dfdx);
};