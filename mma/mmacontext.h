#pragma once
#include<iostream>
typedef float Scalar;
class mmacontext
{
public:

	Scalar* xval;


	mmacontext()
	{
		xval = new Scalar[10];
		std::cout << xval[0];
	}
};