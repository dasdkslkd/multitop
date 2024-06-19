#include "fem.h"
#include "mmasolver.h"

int main()
{
	//Femproblem fem(2, 2, 2, 0.4, 1);
	//fem.setconstrain(vector<int>({ 3, 2, 1 }));
	//auto force = Eigen::VectorXd::Constant(fem.ndof, 1);
	//fem.setforce(force);
	//fem.solvefem();

	vector<vector<double>> a = { {1,2,3},{4,5,6},{7,8,9} };
	vector<double> b = { 1,2,3 };
	double* d = move(b.data());
	cout << d << endl;
	cout << &b << endl;
	Eigen::MatrixXd c = Eigen::Map<Eigen::Matrix<double,3,1>>(d);
	cout << &c;
	cout << c;
	return 0;
}