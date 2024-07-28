#include "fem.cuh"
#include"../IO/matrixIO.h"
gpumat<double> F;
extern float my_erfinvf(float a);

template<typename scalar>
void printgmat(const gpumat<scalar>& v)
{
	scalar* host = new scalar[v.size()];
	v.download(host);
	for (int i = 0; i < v.size(); ++i)
		std::cout << host[i] << ' ';
	std::cout << std::endl;
}

void solvefem(vector<int>& ikfree, vector<int>& jkfree, vector<double>& sk, vector<int>& freeidx, vector<int>& freedofs, Eigen::VectorXd& F,  gpumat<double>& U)
{
	static std::vector<Eigen::Triplet<double>> triplist(freeidx.size());
	for (int i = 0; i < freeidx.size(); ++i)
		triplist[i] = Eigen::Triplet<double>(ikfree[i], jkfree[i], sk[freeidx[i]]);
	static auto K = Eigen::SparseMatrix<double>(freedofs.size(), freedofs.size());
	K.setFromTriplets(triplist.begin(), triplist.end());
	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> cg;
	cg.compute(K);
	//savevec("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\sk-g.csv", sk);
	Eigen::VectorXd utemp;
	utemp = cg.solve(F(freedofs));
	static gpumat<double> uuu;
	uuu.set_from_host(utemp.data(), freedofs.size(), 1);
	static gpumat<int> idx;
	idx.set_from_host(freedofs.data(), freedofs.size(), 1);

	//printgmat(uuu);
	//printgmat(idx);
	//std::cout << sk << std::endl;
	//for (auto v : sk)
	//	std::cout << v << '\n';

	U.set_by_index(idx.data(), freedofs.size(), uuu.data());
}

void computefdf(gpumat<double>& U, gpumat<double>& dSdx, gpumat<double>& dskdx, gpumat<int>& ik, gpumat<int>& jk, double& f, gpumat<double>& dfdx, gpumat<double>& x, gpumat<double>& temp, gpumat<double>& coef, int ndofs, bool multiobj, Eigen::VectorXd& F_host)
{
	static bool dummy = (F.set_from_host(F_host.data(), F_host.size(), 1), true);
	//double* U_h = new double[U.size()];
	//U.download(U_h);
	//savearr("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\Uh.csv", U_h, U.size());
	//savemat("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\Fh.csv", F_host);
	//printgmat(matprod(U.transpose(), F));
	f = matprod(U.transpose(), F).get_item(0);
	double sum = 0;
	int nel = static_cast<int>(dfdx.size() / 4);
	for (int i = 0; i < 4 * nel; ++i)
	{
		sensitivity(dSdx, coef, dskdx, temp, i, nel);
		dfdx.set_by_index(i, 1, matprod(U.transpose(), spmatprodcoo(U, ik, jk, dskdx, ndofs, ndofs, ik.size())).data(), cudaMemcpyDeviceToDevice);
		if (multiobj && i < nel && x.get_item(i)>1e-3)
		{
			sum += std::exp(-std::pow(my_erfinvf(2 * x.get_item(i) - 1), 2));
			double ttt = dfdx.get_item(i) + 400 / std::sqrt(3. * PI) / nel * my_erfinvf(2 * x.get_item(i) - 1);
			dfdx.set_by_index(i, 1, &ttt, cudaMemcpyHostToDevice);
		}
	}
	f -= 200 / std::sqrt(3) / PI / nel * sum;
}

void computegdg(gpumat<double>& x, gpumat<double>& g, gpumat<double>& dgdx, const double& volfrac, const int m, const int nel)
{
	static double theta_min = PI / 18;
	double ttt = x.sum_partly(0, nel) / nel / volfrac - 1;
	g.set_by_index(m - 1, 1, &ttt, cudaMemcpyHostToDevice);
	for (int i = 0; i < nel; ++i)
	{
		ttt = theta_min - x.get_item(i + nel) - x.get_item(i + 2 * nel) - x.get_item(i + 3 * nel);
		g.set_by_index(i, 1, &ttt, cudaMemcpyHostToDevice);
		ttt = -1;
		dgdx.set_by_index(m * (i + nel) + i, 1, &ttt, cudaMemcpyHostToDevice);
		dgdx.set_by_index(m * (i + 2 * nel) + i, 1, &ttt, cudaMemcpyHostToDevice);
		dgdx.set_by_index(m * (i + 3 * nel) + i, 1, &ttt, cudaMemcpyHostToDevice);
	}
	static bool dummy = false;
	if (!dummy)
	{
		for (int i = 0; i < 4 * nel; ++i)
		{
			ttt = 1. / nel / volfrac;
			double tttt = 0.;
			if (i < nel)
				dgdx.set_by_index(i * m + m - 1, 1, &ttt, cudaMemcpyHostToDevice);
			else
				dgdx.set_by_index(i * m + m - 1, 1, &tttt, cudaMemcpyHostToDevice);
		}
	}
	dummy = true;
}