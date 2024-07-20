#ifndef _FEM_CUH_
#define _FEM_CUH_
#include "../element/element.cuh"
#include <vector>
using std::vector;

void solvefem(vector<int>& ikfree, vector<int>& jkfree, vector<double>& sk, vector<int> freeidx);

#endif // !_FEM_CUH_
