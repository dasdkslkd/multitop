#pragma once

#ifndef __MMAOPTER_HPP
#define __MMAOPTER_HPP

#ifdef _EXPORT_MMAOPT
#define API_MMAOPT /*extern "C" __declspec(dllexport)*/
#else
#define API_MMAOPT /*extern "C" __declspec(dllimport)*/
#endif


API_MMAOPT void mmasub(int ncontrain, int nvar, int itn, double* xvar, double* xmin, double* xmax, double* xold1, double* xold2,
	double f0val, double* df0dx, double* gval, double* dgdx, double* low, double* upp,
	double a0, double* a, double* c, double* d, double move);


#endif
