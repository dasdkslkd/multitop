#include "fem.h"
#include "mmasolver.h"
#include "element.h"
#include "mmacontext.h"

int main()
{
	int nelx = 6;
	int nely = 3;
	int nelz = 3;
	float volfrac = 0.4;
	Femproblem fem(nelx, nely, nelz, volfrac, 1);
	vector<int> cons(3 * (nely + 1) * (nelz + 1));
	for (int i = 0; i < cons.size(); ++i)
		cons[i] = i;
	fem.setconstrain(move(cons));
	Eigen::VectorXd force(fem.ndof);
	for (int i = 0; i < nely; ++i)
	{
		force[((nelx - 1) * nely * nelz + i) * 3 + 2] = -1;
	}
	fem.setforce(force);
	fem.elem.predict(fem.x, fem.S, fem.dSdx);
	fem.elem.elasticity(fem.S, fem.sk);
	fem.solvefem();

	//fem.setconstrain(vector<int>({ 3, 2, 1 }));
	//auto force = Eigen::VectorXd::Constant(fem.ndof, 1);
	//fem.setforce(force);
	//fem.solvefem();


	return 0;
}