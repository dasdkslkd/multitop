#include "fem.h"
#include "mmasolver.h"
#include "element.h"
#include "mmacontext.h"

int main()
{
	Femproblem fem(2, 2, 2, 0.4, 1);
	//fem.setconstrain(vector<int>({ 3, 2, 1 }));
	//auto force = Eigen::VectorXd::Constant(fem.ndof, 1);
	//fem.setforce(force);
	//fem.solvefem();
	spinodal ac(fem.nel);
	mmacontext bb;
	//ac.elasticity(fem.S, fem.sk);

	
	return 0;
}