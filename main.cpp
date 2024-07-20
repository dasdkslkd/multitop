#include "fem.h"
#include "mmacontext.h"

int main()
{
	int nelx = 6;
	int nely = 3;
	int nelz = 3;
	double volfrac = 0.4;
	Femproblem fem(nelx, nely, nelz, volfrac, 1);
	vector<int> cons(3 * (nely + 1) * (nelz + 1));
	for (int i = 0; i < cons.size(); ++i)
		cons[i] = i;
	fem.setconstrain(move(cons));
	Eigen::VectorXd force=Eigen::VectorXd::Constant(fem.ndof,0);
	for (int i = 0; i <= nely; ++i)
	{
		force[(nelx * (nely + 1) * (nelz + 1) + i) * 3 + 2] = -1;
	}
	fem.setforce(force);
	mmacontext mma(fem);
	mma.solve_gpu();
	return 0;
}