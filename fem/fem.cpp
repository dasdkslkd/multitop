#include "fem.h"
#include<unordered_set>

Femproblem::Femproblem(int nelx, int nely, int nelz, double volfrac, bool multiobj) :nelx(nelx), nely(nely), nelz(nelz), volfrac(volfrac), multiobj(multiobj)
{
	nel = nelx * nely * nelz;
	ndof = (1 + nelx) * (1 + nely) * (1 + nelz) * 3;
	Eigen::VectorXd cvec(nel);
	for (int i = 0; i < nelx; ++i)
	{
		for (int k = 0; k < nelz; ++k)
		{
			for (int j = 0; j < nely; ++j)
			{
				cvec[j + k * nely + i * nely * nelz] = (1 + j + k * (nely + 1) + i * (nely + 1) * (nelz + 1)) * 3 + 1;
			}
		}
	}
	Eigen::VectorXd idx(24);
	idx << 0, 1, 2, 3 * (nely + 1) * (nelz + 1), 3 * (nely + 1) * (nelz + 1) + 1, 3 * (nely + 1) * (nelz + 1) + 2, 3 * (nely + 1) * (nelz + 1) - 3, 3 * (nely + 1) * (nelz + 1) - 2, 3 * (nely + 1) * (nelz + 1) - 1, -3, -2, -1, 3 * (nely + 1), 3 * (nely + 1) + 1, 3 * (nely + 1) + 2, 3 * (nely + 1) * (nelz + 2), 3 * (nely + 1) * (nelz + 2) + 1, 3 * (nely + 1) * (nelz + 2) + 2, 3 * (nely + 1) * (nelz + 2) - 3, 3 * (nely + 1) * (nelz + 2) - 2, 3 * (nely + 1) * (nelz + 2) - 1, 3 * (nely + 1) - 3, 3 * (nely + 1) - 2, 3 * (nely + 1) - 1;
	Eigen::MatrixXd cmat(nel, 24);
	for (int i = 0; i < nel; ++i)
	{
		for (int j = 0; j < 24; ++j)
		{
			cmat(i, j) = cvec[i] + idx[j] - 1;
		}
	}
	ik = Eigen::kroneckerProduct(cmat, Eigen::MatrixXd::Constant(24, 1, 1)).transpose().reshaped();
	jk = Eigen::kroneckerProduct(cmat, Eigen::MatrixXd::Constant(1, 24, 1)).transpose().reshaped();
	//sk = Eigen::VectorXd(24 * 24 * nel);
	//K = Eigen::SparseMatrix<double>(ndof, ndof);
	U = Eigen::VectorXd(ndof);
	Y = Eigen::MatrixXd(nel, 9);
	F = Eigen::VectorXd(ndof);
	//trip_list.reserve(24 * 24 * nel);
	//K.reserve(Eigen::VectorXd::Constant(ndof, 192));
}

void Femproblem::setconstrain(vector<int>&& fixeddofs)
{
	int n = ndof - fixeddofs.size();
	freedofs = vector<int>(n);
	auto rule = [](int i, int j)->bool {return i < j; };
	sort(fixeddofs.data(), fixeddofs.data() + fixeddofs.size(), rule);
	for (int i = 0, idx = 0; i < n + idx; ++i)
	{
		if (idx < fixeddofs.size() && fixeddofs[idx] == i) { ++idx; continue; }
		freedofs[i - idx] = i;
	}
	unordered_set<int> smallset(freedofs.begin(),freedofs.end());
	vector<int> idx;
	idx.reserve(24 * 24 * nel);
	for (int i = 0; i < 24 * 24 * nel; ++i)
	{
		if (smallset.count(ik(i)) > 0 && smallset.count(jk(i) > 0))
		{
			idx.push_back(i);
		}
	}
	ik = ik(idx).eval();
	jk = jk(idx).eval();
	sk = Eigen::VectorXd(idx.size());
	U = Eigen::VectorXd(freedofs.size());
	trip_list.reserve(idx.size());
	K = Eigen::SparseMatrix<double>(freedofs.size(), freedofs.size());
	K.reserve(Eigen::VectorXd::Constant(freedofs.size(), 192));
}

void Femproblem::solvefem()
{
#if 1
	sk = Eigen::VectorXd::Constant(24 * 24 * nel, 1);
#endif
	for (int i = 0; i < ik.size(); ++i)
	{
		trip_list.push_back(Eigen::Triplet<double>(ik(i), jk(i), sk(i)));
	}
	K.setFromTriplets(trip_list.begin(), trip_list.end());
	cg.compute(K);
	U = cg.solve(F(freedofs));
	cout << cg.iterations() << endl << cg.error();
}