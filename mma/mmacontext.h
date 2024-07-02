#pragma once
//#include<iostream>
#include "mmasolver.h"
#include "fem.h"
class mmacontext
{
public:
	int m;
	int n;
	float* xval;
	float f;
	float* dfdx;
	float* g;
	float* dgdx;//mxn,col major
	float* xmin;
	float* xmax;
	MMASolver solver;
	Femproblem* pfem;

	struct
	{
		float change;
		float minf;
		int maxiter;
		int miniter;
		float* flist;
	}logger;

	//mmacontext(int m,int n,float a,float c,float d,float*xval,float*dfdx,float*g,float* dgdx,float*xmin,float*xmax):m(m),n(n),a(a),c(c),d(d),xval(xval),dfdx(dfdx),g(g),dgdx(dgdx),xmin(xmin),xmax(xmax)
	//{
	//	MMASolver solver();
	//}

	mmacontext(Femproblem& fem, int maxiter = 300)
	{
		logger.change = 1.f;
		logger.maxiter = maxiter;
		logger.miniter = 1;
		logger.minf = 1e+9f;
		logger.flist = new float[logger.maxiter];
		pfem = &fem;
		m = 1 + 2 * pfem->nel;
		n = 4 * pfem->nel;
		solver = MMASolver(n, m);
		xval = pfem->x;
		g = new float[m];
		dgdx = new float[m * n];
		dfdx = new float[n];
		xmin = new float[n];
		xmax = new float[n];

		fill(g, g + m, 0.f);
		fill(dgdx, dgdx + m * n, 0.f);
		fill(dfdx, dfdx + n, 0.f);
		fill(xmin, xmin + n, 0.f);
		fill(xmax, xmax + pfem->nel, 0.7f);
		fill(xmax + pfem->nel, xmax + n, 1.f);
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
		static float theta_min = PI / 18;
		g[m - 1] = accumulate(xval, xval + pfem->nel, 0.f) / pfem->nel / pfem->volfrac - 1;
		for (int i = 0; i < pfem->nel; ++i)
		{
			g[i] = theta_min - xval[i + pfem->nel] - xval[i + 2 * pfem->nel] - xval[i + 3 * pfem->nel];
			g[i + pfem->nel] = 1e-3f - xval[i + pfem->nel] * xval[i + 2 * pfem->nel] * xval[i + 3 * pfem->nel];
			dgdx[m * (i + pfem->nel) + i] = -1;
			dgdx[m * (i + 2 * pfem->nel) + i] = -1;
			dgdx[m * (i + 3 * pfem->nel) + i] = -1;
			dgdx[m * (i + pfem->nel) + i + pfem->nel] = -xval[i + 2 * pfem->nel] * xval[i + 3 * pfem->nel];
			dgdx[m * (i + 2 * pfem->nel) + i + pfem->nel] = -xval[i + pfem->nel] * xval[i + 3 * pfem->nel];
			dgdx[m * (i + 3 * pfem->nel) + i + pfem->nel] = -xval[i + pfem->nel] * xval[i + 2 * pfem->nel];
		}
		for (int i = 0; i < n; ++i)
		{
			if (i < pfem->nel)
				dgdx[i * m + m - 1] = 1.f / pfem->nel / pfem->volfrac;
			else
				dgdx[i * m + m - 1] = 0.f;
		}
	}

	void solve()
	{
		while (logger.change > 0.01f && solver.iter < logger.maxiter && solver.iter < logger.miniter + 50)
		{
			pfem->elem.predict(xval, pfem->S, pfem->dSdx);
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
			logger.change = 0;
			for (int i = 0; i < n; ++i)
				logger.change = max(logger.change, fabs(xval[i] - solver.xold1[i]));
			printf("It:%3d Obj:%5.1f Vol:%4.3f Ch:%5.3f\n", solver.iter, f, (g[m - 1] + 1) * pfem->volfrac, logger.change);
			logger.flist[solver.iter] = f;
			if (solver.iter == 1)
				break;
		}
	}
};