#pragma once
#include<iostream>
#include<fstream>
#include<Eigen/Core>
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

class spinodal
{
public:
	spinodal()
	{
		static bool dummy = (readcoef(), true);//逗号表达式，readcoef仅调用一次
	}
};