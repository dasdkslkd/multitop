#pragma once
//#include<iostream>
#include<fstream>
#include<Eigen/Core>
#include<torch/script.h>
#include "matrixIO.h"
#define PI acos(-1.f)
using namespace std;

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
//		auto data = vector<float>(9*576);
//		string line;
//		while (getline(infile, line))
//		{
//			stringstream ss(line);
//			string cell;
//			while (getline(ss, cell, ','))
//			{
//				float value;
//				stringstream(cell) >> value;
//				data.push_back(value);
//			}
//		}
//		infile.close();
//		float* ptr = data.data();
//		coef = Eigen::Map<Eigen::MatrixXd>(ptr, 576, 9);
//	}
//
//	~material_base() {}
//
//	material_base(const material_base& instance) = delete;
//
//	const material_base& operator=(const material_base& instance) = delete;
//};

inline Eigen::MatrixXf coef;
inline void readcoef()
{
	string fname = "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\coef.csv";
	ifstream infile(fname, ios::in);
	assert(infile.is_open());
	vector<float> data;
	data.reserve(9 * 576);
	string line;
	while (getline(infile, line))
	{
		stringstream ss(line);
		string cell;
		while (getline(ss, cell, ','))
		{
			float value;
			stringstream(cell) >> value;
			data.push_back(value);
		}
	}
	infile.close();
	float* ptr = data.data();
	coef = Eigen::Map<Eigen::MatrixXf>(ptr, 576, 9);
}

void uploadcoef(float* ptr);

class spinodal
{
public:
	bool use_cuda;
	torch::jit::Module model;
	int nel;
	float* temp;
	struct
	{
		float* temp;
	}gbuf;


	spinodal(int nel = 0, bool use_cuda = true) :nel(nel),use_cuda(use_cuda)
	{
		static bool dummy = (readcoef(), true);//逗号表达式，readcoef仅调用一次
		model = torch::jit::load("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\model-cpu.jit");
		temp = new float[9 * nel];
		fill(temp, temp + 9 * nel, 0.f);
		if (use_cuda)
			init_gpu();
	}

	~spinodal() { 
		delete[] temp;
		if (use_cuda)
			free_gpu();
	}

	spinodal& operator=(const spinodal& inst)
	{
		model = inst.model;
		nel = inst.nel;
		delete[] temp;
		temp = new float[9 * inst.nel];
		copy(inst.temp, inst.temp + 9 * inst.nel, temp);
		if (use_cuda)
			value_gpu(inst);
		return *this;
	}

	void init_gpu();

	void free_gpu();

	void value_gpu(const spinodal& inst);

	void predict(const float* x, float* S, float* dSdx);

	void elasticity(float* S, Eigen::VectorXd& sk);

	void sensitivity(float* dSdx, Eigen::VectorXd& dskdx, int &i);

	void filter(float* x);
};