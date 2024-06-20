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
	float* a=new float[10];
	for (int i = 0; i < 10; ++i)
		a[i] = 1;

	float* b=new float[5];
	for (int i = 0; i < 5; ++i)
		b[i] = 0;

	for (int i = 0; i < 10; ++i)
		cout << a[i] << endl;
	memcpy(a + 2, b, 4 * 5);
	for (int i = 0; i < 10; ++i)
		cout << a[i] << endl;

	return 0;
}