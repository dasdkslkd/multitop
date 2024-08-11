#ifndef _MMACONTEXT_H_
#define _MMACONTEXT_H_

//#include<iostream>
#include "mmasolver.h"
#include "fem.h"
#include "matrixIO.h"

extern "C"
void solve_g(
	int m, int n, double* x_h, double* dfdx_h, double* g_h, double* dgdx_h, double* xmin_h, double* xmax_h,
	int nelx, int nely, int nelz, double volfrac, bool multiobj, Eigen::VectorXd & F_h, vector<int>&freedofs_h, vector<int>&freeidx_h, double* S_h, double* dSdx_h, vector<int>&ik_h, vector<int>&jk_h, vector<int>&ikfree_h, vector<int>&jkfree_h, vector<double>&sk_h, vector<double>&dskdx_h, vector<double>&U_h,
	double* temp_h, Eigen::MatrixXd coef, torch::jit::Module model);

class mmacontext
{
public:
	int m;
	int n;
	double* xval;
	double f;
	double* dfdx;
	double* g;
	double* dgdx;//mxn,col major
	double* xmin;
	double* xmax;
	MMASolver solver;
	Femproblem* pfem;

	struct
	{
		double change;
		double minf;
		int maxiter;
		int miniter;
		double* flist;
	}logger;

	//mmacontext(int m,int n,double a,double c,double d,double*xval,double*dfdx,double*g,double* dgdx,double*xmin,double*xmax):m(m),n(n),a(a),c(c),d(d),xval(xval),dfdx(dfdx),g(g),dgdx(dgdx),xmin(xmin),xmax(xmax)
	//{
	//	MMASolver solver();
	//}

	mmacontext(Femproblem& fem, int maxiter = 300)
	{
		logger.change = 1.;
		logger.maxiter = maxiter;
		logger.miniter = 1;
		logger.minf = 1e+9;
		logger.flist = new double[logger.maxiter];
		pfem = &fem;
		m = 1 + pfem->nel;
		n = 4 * pfem->nel;
		solver = MMASolver(n, m);
		xval = pfem->x;
		g = new double[m];
		dgdx = new double[m * n];
		dfdx = new double[n];
		xmin = new double[n];
		xmax = new double[n];

		fill(g, g + m, 0.);
		fill(dgdx, dgdx + m * n, 0.);
		fill(dfdx, dfdx + n, 0.);
		fill(xmin, xmin + n, 0.);
		fill(xmax, xmax + pfem->nel, 0.7);
		fill(xmax + pfem->nel, xmax + n, 1.);
	}

	~mmacontext()
	{
		delete g;
		delete dgdx;
		delete dfdx;
		delete xmin;
		delete xmax;
		delete logger.flist;
	}

	void computegdg()
	{
		static double theta_min = PI / 18;
		g[m - 1] = accumulate(xval, xval + pfem->nel, 0.f) / pfem->nel / pfem->volfrac - 1;
		for (int i = 0; i < pfem->nel; ++i)
		{
			g[i] = theta_min - xval[i + pfem->nel] - xval[i + 2 * pfem->nel] - xval[i + 3 * pfem->nel];
			//g[i + pfem->nel] = 1e-3f - xval[i + pfem->nel] * xval[i + 2 * pfem->nel] * xval[i + 3 * pfem->nel];
			dgdx[m * (i + pfem->nel) + i] = -1;
			dgdx[m * (i + 2 * pfem->nel) + i] = -1;
			dgdx[m * (i + 3 * pfem->nel) + i] = -1;
			//dgdx[m * (i + pfem->nel) + i + pfem->nel] = -xval[i + 2 * pfem->nel] * xval[i + 3 * pfem->nel];
			//dgdx[m * (i + 2 * pfem->nel) + i + pfem->nel] = -xval[i + pfem->nel] * xval[i + 3 * pfem->nel];
			//dgdx[m * (i + 3 * pfem->nel) + i + pfem->nel] = -xval[i + pfem->nel] * xval[i + 2 * pfem->nel];
		}
		for (int i = 0; i < n; ++i)
		{
			if (i < pfem->nel)
				dgdx[i * m + m - 1] = 1. / pfem->nel / pfem->volfrac;
			else
				dgdx[i * m + m - 1] = 0.;
		}
	}

	void solve()
	{
		string outpath = "D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\output\\";

		while (logger.change > 0.01f && solver.iter < logger.maxiter && solver.iter < logger.miniter + 50)
		{
			pfem->elem.predict(xval, pfem->S, pfem->dSdx);
			//savearr("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\S.csv", pfem->S, 9 * pfem->nel);
			pfem->elem.elasticity(pfem->S, pfem->sk);
			pfem->solvefem();
			pfem->computefdf(f, dfdx);
			computegdg();
			logger.flist[solver.iter] = f;
			if (f < logger.minf)
			{
				logger.miniter = solver.iter;
				logger.minf = f;
			}
			solver.Update(xval, dfdx, g, dgdx, xmin, xmax);
			//savearr(outpath + "dfdxc.txt", dfdx, n);
			//savearr(outpath + "gc.txt", g, m);
			//savearr(outpath + "dgdxc.txt", dgdx, m * n);
			//savemat(outpath + "Uc.txt", pfem->U);
			pfem->elem.filter(xval);
			logger.change = 0;
			for (int i = 0; i < n; ++i)
				logger.change = max(logger.change, fabs(xval[i] - solver.xold1[i]));
			printf("It:%3d Obj:%5.1f Vol:%4.3f Ch:%5.3f\n", solver.iter, f, (g[m - 1] + 1) * pfem->volfrac, logger.change);
			//logger.flist[solver.iter] = f;
			if (solver.iter == 1)
				break;
		}
		savearr(outpath + "xc.txt", xval, n);
		savearr(outpath + "objc.txt", logger.flist,logger.maxiter);
	}

	void solve_gpu()
	{
		//std::vector<double> F(&pfem->F(0, 0), pfem->F.data() + pfem->F.size());
		std::vector<int> ik(&pfem->ik(0, 0), pfem->ik.data() + pfem->ik.size());
		std::vector<int> jk(&pfem->jk(0, 0), pfem->jk.data() + pfem->jk.size());
		std::vector<int> ikfree(&pfem->ikfree(0, 0), pfem->ikfree.data() + pfem->ikfree.size());
		std::vector<int> jkfree(&pfem->jkfree(0, 0), pfem->jkfree.data() + pfem->jkfree.size());
		std::vector<double> sk(&pfem->sk(0, 0), pfem->sk.data() + pfem->sk.size());
		std::vector<double> dskdx(&pfem->dskdx(0, 0), pfem->dskdx.data() + pfem->dskdx.size());
		std::vector<double> U(&pfem->U(0, 0), pfem->U.data() + pfem->U.size());

		solve_g(m, n, xval, dfdx, g, dgdx, xmin, xmax, pfem->nelx, pfem->nely, pfem->nelz, pfem->volfrac, pfem->multiobj, pfem->F, pfem->freedofs, pfem->freeidx, pfem->S, pfem->dSdx, ik, jk, ikfree, jkfree, sk, dskdx, U, pfem->elem.temp, coef, pfem->elem.model);
	}
};

#endif