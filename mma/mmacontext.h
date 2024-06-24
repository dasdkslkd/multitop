#pragma once
#include<iostream>
#include "mmasolver.h"
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
	float* dgdx;
	float* xmin;
	float* xmax;
	MMASolver* solver;

	mmacontext(int m,int n,float a,float c,float d,float*xval,float*dfdx,float*g,float* dgdx,float*xmin,float*xmax):m(m),n(n),a(a),c(c),d(d),xval(xval),dfdx(dfdx),g(g),dgdx(dgdx),xmin(xmin),xmax(xmax)
	{
		solver = new MMASolver(n, m, a, c, d);
	}

	~mmacontext() { delete solver; }

	void computef()
	{

	}
};