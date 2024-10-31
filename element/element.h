#ifndef _ELEMENT_H_
#define _ELEMENT_H_
//#include<iostream>
#include<fstream>
#include<Eigen/Core>
#include<torch/script.h>
#include "matrixIO.h"
#define PI acos(-1.)
using namespace std;

//template class gpumat<double>;

//class material_base
//{
//public:
//	static inline Eigen::MatrixXd coef;
//
//	static material_base& getinstance() { static material_base instance; return instance; }
//
//private:
//	material_base()
//	{
//		string fname = "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\coef.csv";
//		ifstream infile(fname, ios::in);
//		assert(infile.is_open());
//		auto data = vector<double>(9*576);
//		string line;
//		while (getline(infile, line))
//		{
//			stringstream ss(line);
//			string cell;
//			while (getline(ss, cell, ','))
//			{
//				double value;
//				stringstream(cell) >> value;
//				data.push_back(value);
//			}
//		}
//		infile.close();
//		double* ptr = data.data();
//		coef = Eigen::Map<Eigen::MatrixXd>(ptr, 576, 9);
//	}
//
//	~material_base() {}
//
//	material_base(const material_base& instance) = delete;
//
//	const material_base& operator=(const material_base& instance) = delete;
//};

inline Eigen::MatrixXd coef;
inline void readcoef()
{
	string fname = "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\coef.csv";
	ifstream infile(fname, ios::in);
	assert(infile.is_open());
	vector<double> data;
	data.reserve(9 * 576);
	string line;
	while (getline(infile, line))
	{
		stringstream ss(line);
		string cell;
		while (getline(ss, cell, ','))
		{
			double value;
			stringstream(cell) >> value;
			data.push_back(value);
		}
	}
	infile.close();
	double* ptr = data.data();
	coef = Eigen::Map<Eigen::MatrixXd>(ptr, 576, 9);
}

//void uploadcoef(double* ptr);

class spinodal
{
public:
	torch::jit::Module model;
	int nel;
	double* temp;

	spinodal(int nel = 0) :nel(nel)
	{
		static bool dummy = (readcoef(), true);//逗号表达式，readcoef仅调用一次
		torch::set_default_dtype(caffe2::TypeMeta::Make<double>());
		model = torch::jit::load("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\fNN_cpu_64.pt");
		temp = new double[9 * nel];
		fill(temp, temp + 9 * nel, 0.f);
	}

	~spinodal() {
		delete[] temp;
	}

	spinodal& operator=(const spinodal& inst)
	{
		model = inst.model;
		nel = inst.nel;
		delete[] temp;
		temp = new double[9 * inst.nel];
		copy(inst.temp, inst.temp + 9 * inst.nel, temp);
		return *this;
	}

	void predict(const double* x, double* S, double* dSdx);

	void elasticity(double* S, Eigen::VectorXd& sk);

	void sensitivity(double* dSdx, Eigen::VectorXd& dskdx, int &i);

	void filter(double* x);
};
#endif