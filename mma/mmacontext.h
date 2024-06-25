#pragma once
#include<iostream>
#include "mmasolver.h"
#include "fem.h"
class mmacontext
{
public:
	
	int m;
	int n;
	float a;
	float c;
	float d;
	float* xval;
	float f;
	float* dfdx;
	float* g;
	float* dgdx;//mxn,col major
	float* xmin;
	float* xmax;
	MMASolver* solver;

	mmacontext(int m,int n,float a,float c,float d,float*xval,float*dfdx,float*g,float* dgdx,float*xmin,float*xmax):m(m),n(n),a(a),c(c),d(d),xval(xval),dfdx(dfdx),g(g),dgdx(dgdx),xmin(xmin),xmax(xmax)
	{
		solver = new MMASolver(n, m, a, c, d);
	}

	mmacontext(Femproblem fem)
	{
		m = 1 + 2 * fem.nel;
		n = 4 * fem.nel;
		solver = new MMASolver(n, m);
		float theta_min = PI / 18;
		xval = fem.x;
		g = new float[m];
		dgdx = new float[m * n];
		dfdx = new float[n];
		xmin = new float[n];
		xmax = new float[n];

		fill(xmin, xmin + n, 0.f);
		fill(xmax, xmax + fem.nel, 0.7f);
		fill(xmax + fem.nel, xmax + n, 1.f);
		g[m - 1] = accumulate(xval, xval + fem.nel - 1, 0.f);
		for (int i = 0; i < fem.nel; ++i)
		{
			g[i] = theta_min - xval[i + fem.nel] - xval[i + 2 * fem.nel] - xval[i + 3 * fem.nel];
			g[i + fem.nel] = 1e-3f - xval[i + fem.nel] * xval[i + 2 * fem.nel] * xval[i + 3 * fem.nel];
			dgdx[m * (i + fem.nel) + i] = -1;
			dgdx[m * (i + 2 * fem.nel) + i] = -1;
			dgdx[m * (i + 3 * fem.nel) + i] = -1;
			dgdx[m * (i + fem.nel) + i + fem.nel] = -xval[i + 2 * fem.nel] * xval[i + 3 * fem.nel];
			dgdx[m * (i + 2 * fem.nel) + i + fem.nel] = -xval[i + fem.nel] * xval[i + 3 * fem.nel];
			dgdx[m * (i + 3 * fem.nel) + i + fem.nel] = -xval[i + fem.nel] * xval[i + 2 * fem.nel];
		}
		for (int i = 0; i < n; ++i)
		{
			if (i < fem.nel)
				dgdx[i * m + m - 1] = 1;
			else
				dgdx[i * m + m - 1] = 0;
		}
	}

	~mmacontext()
	{ 
		delete solver;
		delete g;
		delete dgdx;
		delete dfdx;
	}

	void computefdf(Femproblem fem)
	{
		fem.computefdf(f, dfdx);
	}
};