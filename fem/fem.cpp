#include "fem.h"
#include<unordered_set>

extern float my_erfinvf(float a);

Femproblem::Femproblem(int nelx, int nely, int nelz, float volfrac, bool multiobj) :nelx(nelx), nely(nely), nelz(nelz), volfrac(volfrac), multiobj(multiobj)
{
	nel = nelx * nely * nelz;
	elem = spinodal(nel);
	ndof = (1 + nelx) * (1 + nely) * (1 + nelz) * 3;
	Eigen::VectorXi cvec(nel);
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
	Eigen::VectorXi idx(24);
	idx << 0, 1, 2, 3 * (nely + 1) * (nelz + 1), 3 * (nely + 1) * (nelz + 1) + 1, 3 * (nely + 1) * (nelz + 1) + 2, 3 * (nely + 1) * (nelz + 1) - 3, 3 * (nely + 1) * (nelz + 1) - 2, 3 * (nely + 1) * (nelz + 1) - 1, -3, -2, -1, 3 * (nely + 1), 3 * (nely + 1) + 1, 3 * (nely + 1) + 2, 3 * (nely + 1) * (nelz + 2), 3 * (nely + 1) * (nelz + 2) + 1, 3 * (nely + 1) * (nelz + 2) + 2, 3 * (nely + 1) * (nelz + 2) - 3, 3 * (nely + 1) * (nelz + 2) - 2, 3 * (nely + 1) * (nelz + 2) - 1, 3 * (nely + 1) - 3, 3 * (nely + 1) - 2, 3 * (nely + 1) - 1;
	Eigen::MatrixXi cmat(nel, 24);
	for (int i = 0; i < nel; ++i)
	{
		for (int j = 0; j < 24; ++j)
		{
			cmat(i, j) = cvec[i] + idx[j] - 1;
		}
	}
	ik = Eigen::kroneckerProduct(cmat, Eigen::MatrixXi::Constant(24, 1, 1)).transpose().reshaped();
	jk = Eigen::kroneckerProduct(cmat, Eigen::MatrixXi::Constant(1, 24, 1)).transpose().reshaped();
	sk = Eigen::VectorXd::Constant(24 * 24 * nel, 0);
	dskdx = Eigen::VectorXd::Constant(24 * 24 * nel, 0);

	//K = Eigen::SparseMatrix<float>(ndof, ndof);
	U = Eigen::VectorXd(ndof);
	F = Eigen::VectorXd(ndof);
	//trip_list.resize(24 * 24 * nel);
	trip_forsk.resize(24 * 24 * nel);
	//K.reserve(Eigen::VectorXd::Constant(ndof, 192));
	x = new float[nel * 4];
	S = new float[nel * 9];
	dSdx = new float[nel * 36];
	fill(x, x + 4 * nel, 0.5f);
	fill(S, S + 9 * nel, 0.f);
	fill(dSdx, dSdx + 36 * nel, 0.f);
}

Femproblem::~Femproblem()
{
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
	unordered_set<int> smallset(freedofs.begin(), freedofs.end());

	freeidx.reserve(24 * 24 * nel);
	for (int i = 0; i < 24 * 24 * nel; ++i)
	{
		if (smallset.count(ik(i)) > 0 && smallset.count(jk(i)) > 0)
		{
			freeidx.push_back(i);
		}
	}
	//ik = ik(freeidx).eval();
	//jk = jk(freeidx).eval();
	//sk = Eigen::VectorXd(freeidx.size());
	//U = Eigen::VectorXd(freedofs.size());



	trip_list.resize(freeidx.size());
	K = Eigen::SparseMatrix<double>(freedofs.size(), freedofs.size());
	K.reserve(Eigen::VectorXd::Constant(freedofs.size(), 192));
}

void Femproblem::solvefem()
{
#if 0
	sk = Eigen::VectorXd::Constant(24 * 24 * nel, 1);
#endif
	for (int i = 0; i < freeidx.size(); ++i)
	{
		//trip_list.push_back(Eigen::Triplet<float>(ik(i), jk(i), sk(i)));
		trip_list[i] = Eigen::Triplet<double>(ik(freeidx[i]), jk(freeidx[i]), sk(freeidx[i]));
	}
	K.setFromTriplets(trip_list.begin(), trip_list.end());
	cg.compute(K);
	U(freedofs) = cg.solve(F(freedofs));
	cout << cg.iterations() << ' ' << cg.error() << endl;
}

void Femproblem::computefdf(float& f, float* dfdx)
{
	f = U.transpose() * F;

	float sum = 0;
	for (int i = 0; i < 4 * nel; ++i)
	{
		elem.sensitivity(dSdx, dskdx, i);
		for (int j = 0; j < 24 * 24 * nel; ++j)
		{
			trip_forsk[j] = Eigen::Triplet<float>(ik(j), jk(j), dskdx(j));
		}
		dKdx.setFromTriplets(trip_forsk.begin(), trip_forsk.end());
		dfdx[i] = -U.cast<float>().transpose() * dKdx * U.cast<float>();

		if (i < nel && x[i] > 1e-3f)
		{
			sum += exp(-pow(my_erfinvf(2 * x[i] - 1), 2));
			dfdx[i] += 400 / sqrtf(3.f * PI) / nel * my_erfinvf(2 * x[i] - 1);
		}
	}
	f -= 200 / sqrtf(3) / PI / nel * sum;
}