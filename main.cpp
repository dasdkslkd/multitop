#include "fem.h"
#include "mmasolver.h"
#include "element/element.h"

int main()
{
	//Femproblem fem(2, 2, 2, 0.4, 1);
	//fem.setconstrain(vector<int>({ 3, 2, 1 }));
	//auto force = Eigen::VectorXd::Constant(fem.ndof, 1);
	//fem.setforce(force);
	//fem.solvefem();

	readcoef();
	cout << coef;
	return 0;
}