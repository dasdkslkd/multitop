#include "../fem/fem.cuh"
//#include "mmacontext.h"
#include <torch/script.h>
#include "../mma-g/mmaOpt.h"
#include "../IO/matrixIO.h"
using gmatd = gpumat<double>;

template<typename T>
void savegmat(gpumat<T>& v, string filename)
{
	T* host = new T[v.size()];
	v.download(host);
	savearr(filename, host, v.size());
}

extern void mmasub(const int m, const int n, int iter, gmatd& x, const gmatd& xmin, const gmatd& xmax, gmatd& xold1, gmatd& xold2, const gmatd& dfdx, const gmatd& g, const gmatd& dgdx, gmatd& low, gmatd& upp, const double a0, gmatd& a, const gmatd& c, const gmatd& d, const double move);

extern "C"
void solve_g(
	//mma
	int m, int n, double* x_h, double* dfdx_h, double* g_h, double* dgdx_h, double* xmin_h, double* xmax_h,
	//fem
	int nelx, int nely, int nelz, double volfrac, bool multiobj, Eigen::VectorXd & F_h, vector<int>&freedofs_h, vector<int>&freeidx_h, double* S_h, double* dSdx_h, vector<int>&ik_h, vector<int>&jk_h, vector<int>&ikfree_h, vector<int>&jkfree_h, vector<double>&sk_h, vector<double>&dskdx_h, vector<double>&U_h,
	//elem
	double* temp_h, Eigen::MatrixXd& coef_h, torch::jit::Module model)
{
	string outpath = "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\";
	//printf("1");
	double change = 1.;
	int iter = 0;
	double minf = 1e9;
	int miniter = 1;
	int maxiter = 300;
	vector<double> flist(maxiter, 0);
	double f = 1e9;
	double* xold1 = new double[n];
	double* xold2 = new double[n];
	double* low = new double[n];
	double* upp = new double[n];
	double* a = new double[m];
	double* c = new double[m];
	double* d = new double[m];
	for (int i = 0; i < m; ++i)
	{
		a[i] = 0;
		c[i] = 1000;
		d[i] = 0;
	}
	for (int i = 0; i < n; ++i)
	{
		xold1[i] = 0;
		xold2[i] = 0;
	}

	int nel = nelx * nely * nelz;
	int ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1);


	gpumat<double> x, dfdx, g, dgdx, xmin, xmax, F, S, dSdx, sk, dskdx, U, temp, coef, xold1g, xold2g, lowg, uppg;
	gpumat<int> freedofs, freeidx, ik, jk, ikfree, jkfree;

	x.set_from_host(x_h, n, 1);
	dfdx.set_from_host(dfdx_h, n, 1);
	g.set_from_host(g_h, m, 1);
	dgdx.set_from_host(dgdx_h, m, n);
	xmin.set_from_host(xmin_h, n, 1);
	xmax.set_from_host(xmax_h, n, 1);
	F.set_from_host(F_h.data(), F_h.size(), 1);
	S.set_from_host(S_h, 9, nel);
	dSdx.set_from_host(dSdx_h, 36 * nel, 1);
	sk.set_from_host(sk_h.data(), 576 * nel, 1);
	dskdx.set_from_host(dskdx_h.data(), 576 * nel, 1);
	U.set_from_host(U_h.data(), ndof, 1);
	temp.set_from_host(temp_h, 9, nel);
	coef.set_from_host(coef_h.data(), 576, 9);
	freedofs.set_from_host(freedofs_h.data(), freedofs.size(), 1);
	freeidx.set_from_host(freeidx_h.data(), freeidx_h.size(), 1);
	ik.set_from_host(ik_h.data(), ik_h.size(), 1);
	jk.set_from_host(jk_h.data(), jk_h.size(), 1);
	ikfree.set_from_host(ikfree_h.data(), ikfree_h.size(), 1);
	jkfree.set_from_host(jkfree_h.data(), jkfree_h.size(), 1);
	xold1g.set_from_host(xold1, n, 1);
	xold2g.set_from_host(xold2, n, 1);
	lowg.set_from_host(low, n, 1);
	uppg.set_from_host(upp, n, 1);

	while (change > 0.01 && iter < maxiter && iter < miniter + 50)
	{
		//auto dskdxt = readmatrix("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\dskdxg.txt");
		//auto idx = readmatrix(outpath + "idx.txt");
		//vector<int> idxi(idx.size());
		//for (int i = 0; i < idx.size(); ++i)
		//	idxi[i] = idx[i];
		//gpumat<double> dskdxg;
		//dskdxg.set_from_host(dskdxt.data(), dskdxt.size(), 1);
		//savegmat(dskdxg(idxi), "dskdxg.txt");

		double change = 0.;
		++iter;
		predict(x, S, dSdx, nel, model);
		elastisity(S, coef, sk);
		sk.download(sk_h.data());
		solvefem(ikfree_h, jkfree_h, sk_h, freeidx_h, freedofs_h, F_h, U);
		computefdf(U, dSdx, dskdx, ik, jk, f, dfdx, x, coef, ndof, multiobj, F_h);
		//dfdx.download(dfdx_h);
		computegdg(x, g, dgdx, volfrac, m, nel);
		//g.download(g_h);
		//if (iter == 1)
			//dgdx.transpose().download(dgdx_h);
		//else
			//dgdx.download(dgdx_h);
		flist[iter - 1] = f;
		if (f < minf)
		{
			miniter = iter;
			minf = f;
		}

		//cout << m << ' ' << n << ' ' << iter << endl;
		//savearr(outpath + "xin.txt", x_h, n);
		//savearr(outpath + "xmin.txt", xmin_h, n);
		//savearr(outpath + "xmax.txt", xmax_h, n);
		//savearr(outpath + "xold1.txt", xold1, n);
		//savearr(outpath + "xold2.txt", xold2, n);
		//savearr(outpath + "dfdx.txt", dfdx_h, n);
		//savearr(outpath + "g.txt", g_h, m);
		//savearr(outpath + "dgdx.txt", dgdx_h, m*n);
		//savearr(outpath + "low.txt", low, n);
		//savearr(outpath + "upp.txt", upp, n);

		//mmasub(m, n, iter, x_h, xmin_h, xmax_h, xold1, xold2, f, dfdx_h, g_h, dgdx_h, low, upp, 1, a, c, d, 0.1);
		//savearr(outpath + "x.txt", x_h, n);
		gmatd a(m, 1), c(m, 1, 1000.), d(m, 1);
		mmasub(m, n, iter, x, xmin, xmax, xold1g, xold2g, dfdx, g, dgdx, lowg, uppg, 1, a, c, d, 0.1);

		//filter(x_h, nel);
		//x.set_from_host(x_h, n, 1);
		filter(x);
		//for (int i = 0; i < n; ++i)
		//	change = std::max(change, std::abs(x_h[i] - xold1[i]));
		//printf("It:%3d Obj:%5.1f Vol:%4.3f Ch:%5.3f\n", iter, f, (g_h[m - 1] + 1) * volfrac, change);
		for (int i = 0; i < n; ++i)
			change = std::max(change, std::abs(x.get_item(i) - xold1g.get_item(i)));
		printf("It:%3d Obj:%5.1f Vol:%4.3f Ch:%5.3f\n", iter, f, (g.get_item(m - 1) + 1) * volfrac, change);
		//if (iter == 1)
			//break;
	}
	delete[] xold1, xold2, low, upp, a, c, d;
	savearr(outpath + "x.txt", x_h, n);
	savevec(outpath + "obj.txt", flist);
}