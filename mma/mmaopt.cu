#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../culib/gMat.cuh"
#include "cusolverDn.h"
using gmatd = gpumat<double>;

__global__ void calPQ_kernel(double* P, double* Q, const double* PQ, const double* ux2, const double* xl2, const int n, const int m)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (col < n)
	{
		for (int row = 0; row < m; ++row)
		{
			int eid = col * m + row;
			P[eid] += PQ[eid];
			P[eid] *= ux2[col];
			Q[eid] += PQ[eid];
			Q[eid] *= xl2[col];
		}
	}
}

__global__ void calGG_kernel(const double* P, const double* Q, double* GG, const double* ux2, const double* xl2, const int n, const int m)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (col < n)
	{
		for (int row = 0; row < m; ++row)
		{
			int eid = col * m + row;
			GG[eid] = P[eid] * ux2[col] - Q[eid] * xl2[col];
		}
	}
}

__global__ void calAlam_kernel(double* Alam, const double* GG, const double* diagxinv, const double* diaglamyi, int m, int n)
{
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	if (col < m && row < m)
	{
		double sum = 0.;
		for (int i = 0; i < n; ++i)
		{
			sum += GG[i * m + row] * GG[i * m + col] * diagxinv[i];
		}
		if (row == col)
			sum += diaglamyi[row];
		Alam[col * m + row] = sum;
	}
}

__global__ void calAlam2_kernel(double* Alam, const double* GG, const double* diagxinv, const double* diaglamyi, int m, int n)
{
	const int BlockSize = 16;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	if (col < m && row < m)
	{
		int tidx = threadIdx.x;
		int tidy = threadIdx.y;
		double sum = 0.;
		__shared__ double as[BlockSize][BlockSize];
		__shared__ double bs[BlockSize][BlockSize];
		int tilesx = (n + BlockSize - 1) / BlockSize;
		for (int i = 0; i < tilesx; ++i)
		{
			as[tidy][tidx] = GG[(i * BlockSize + tidx) * m + row];
			bs[tidy][tidx] = GG[(i * BlockSize + tidy) * m + col];
			__syncthreads();
			for (int l = 0; l < BlockSize; ++l)
				sum += as[tidy][l] * bs[l][tidx] * diagxinv[i * BlockSize + l];
			if (row = col)
				sum += diaglamyi[row];
			__syncthreads();
		}
		Alam[col * m + row] = sum;
	}
}

void calPQ(gmatd& P, gmatd& Q, const gmatd& PQ, const gmatd& ux2, const gmatd& xl2)
{
	double* Pdata = P.data();
	double* Qdata = Q.data();
	const double* PQdata = PQ.data();
	const double* udata = ux2.data();
	const double* ldata = xl2.data();
	auto [grid, block] = kernel_param(ux2.size());
	calPQ_kernel << <grid, block >> > (Pdata, Qdata, PQdata, udata, ldata, P.cols(), P.rows());
	cudaDeviceSynchronize();
	cuda_error_check;
}

void calGG(const gmatd& P, const gmatd& Q, gmatd& GG, const gmatd& uxinv2, const gmatd& xlinv2)
{
	double* Gdata = GG.data();
	const double* Pdata = P.data();
	const double* Qdata = Q.data();
	const double* udata = uxinv2.data();
	const double* ldata = xlinv2.data();
	auto [grid, block] = kernel_param(uxinv2.size());
	calGG_kernel << <grid, block >> > (Pdata, Qdata, Gdata, udata, ldata, P.cols(), P.rows());
	cudaDeviceSynchronize();
	cuda_error_check;
}

void calAlam(gmatd& Alam, const gmatd& diagxinv, const gmatd& diaglamyi, const gmatd& GG)
{
	double* Adata = Alam.data();
	const double* Gdata = GG.data();
	const double* xinv = diagxinv.data();
	const double* lamyi = diaglamyi.data();
	dim3 grid, block;
	block = dim3(16, 16, 1);
	grid = dim3(std::ceil(GG.rows() / 16.), std::ceil(GG.rows() / 16.), 1);
	calAlam_kernel << <grid, block >> > (Adata, Gdata, xinv, lamyi, GG.rows(), GG.cols());
	cudaDeviceSynchronize();
	cuda_error_check;
}

void solveAb(gmatd& AA, gmatd& b, gmatd& dlam, gmatd& dz)
{
	cusolverDnHandle_t handle = NULL;
	cusolverDnParams_t param;
	size_t workdevice = 0, workhost = 0;
	int32_t m = AA.rows();
	void* d_work = nullptr;
	void* h_work = nullptr;
	int* d_info = nullptr;
	cudaMalloc(&d_info, sizeof(int));
	static gpumat<int64_t> ipiv(m, 1);
	ipiv.set_from_value(0);
	cusolverDnCreate(&handle);
	cusolverDnCreateParams(&param);
	cusolverDnSetAdvOptions(param, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);
	cusolverDnXgetrf_bufferSize(handle, param, m, m, CUDA_R_64F, AA.data(), m, CUDA_R_64F, &workdevice, &workhost);
	cuda_error_check;
	cudaMalloc(&d_work, workdevice);
	h_work = malloc(workhost);
	cusolverDnXgetrf(handle, param, m, m, CUDA_R_64F, AA.data(), m, ipiv.data(), CUDA_R_64F, d_work, workdevice, h_work, workhost, d_info);
	cuda_error_check;

	int info = 0;
	cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	if (info < 0)
		printf("%d-th param is wrong\n", -info);

	cusolverDnXgetrs(handle, param, CUBLAS_OP_N, m, 1, CUDA_R_64F, AA.data(), m, ipiv.data(), CUDA_R_64F, b.data(), m, d_info);
	cuda_error_check;

	cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	if (info < 0)
		printf("%d-th param is wrong\n", -info);

	cudaMemcpy(dlam.data(), b.data(), (m - 1) * sizeof(double), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dz.data(), b.data() + m - 1, sizeof(double), cudaMemcpyDeviceToDevice);
	cudaFree(d_info);
	cudaFree(d_work);
	free(h_work);
	cusolverDnDestroyParams(param);
	cusolverDnDestroy(handle);
	cuda_error_check;
}

auto subsolv(const int m, const int n, const double epsimin, const gmatd& low, const gmatd& upp, const gmatd& alfa, const gmatd& beta, const gmatd& p0, const gmatd& q0, const gmatd& P, const gmatd& Q, const double a0, gmatd& a, const gmatd& b, const gmatd& c, const gmatd& d, gmatd& x/*, gmatd& ux1, gmatd& ux2, gmatd& xl1, gmatd& xl2*/)
{
	static gmatd y(m, 1, 1.), lam(m, 1, 1.), s(m, 1, 1.), z(1, 1, 1.), zet(1, 1, 1.);
	//double z = 1, zet = 1;
	x = .5 * (alfa + beta);
	y.set_from_value(1.);
	lam.set_from_value(1.);
	s.set_from_value(1.);
	z.set_from_value(1.);
	zet.set_from_value(1.);
	gmatd xsi(std::move(max(1. / (x - alfa), 1.))), eta(std::move(max(1. / (beta - x), 1.)));
	gmatd mu(std::move(max(.5 * c, 1.)));
	int itera = 0;
	double epsi = 1;
	//use static ux1,ux2,xl1,xl2 from mmasub,reuse ux1=uxinv1 xl1=xlinv1
	static gmatd ux1, ux2, xl1, xl2, uxinv1, xlinv1, ux3, xl3, uxinv2, xlinv2;
	static gmatd plam, qlam, gvec, dpsidx, GG(m, n), rex, delx, rey, dely;
	static gmatd relam, dellam, rexsi, reeta, remu, res;
	static gmatd rez(1, 1), delz(1, 1), rezet(1, 1);
	rez.set_from_value(0.);
	delz.set_from_value(0.);
	rezet.set_from_value(0.);

	static gmatd dlam(m, 1), dy, dmu, ds, dx, dxsi, deta, xold, xsiold, etaold, lamold, yold, muold, sold;
	//double dz = 0, dzet = 0, zold = 0, zetold = 0;
	static gmatd dz(1, 1), dzet(1, 1), zold(1, 1), zetold(1, 1);
	dz.set_from_value(0.);
	dzet.set_from_value(0.);
	zold.set_from_value(0.);
	zetold.set_from_value(0.);

	static gmatd diagx, diagy, diagxinv, diagyinv, diaglam, diaglamyi, GGdelxdiagx, GGxx, blam, bb, Alam(m, m), AA;

	while (epsi > epsimin)
	{
		ux1 = upp - x;
		xl1 = x - low;
		ux2 = ux1 * ux1;
		xl2 = xl1 * xl1;
		uxinv1 = 1. / ux1;
		xlinv1 = 1. / xl1;
		std::cout << "-- epsi = " << epsi;
		cuda_error_check;
		plam = p0 + matprod(P.transpose(), lam);
		qlam = q0 + matprod(Q.transpose(), lam);
		gvec = matprod(P, uxinv1) + matprod(Q, xlinv1);
		dpsidx = plam / ux2 - qlam / xl2;
		rex = dpsidx - xsi + eta;
		rey = c + d * y - mu - lam;
		rez = a0 - zet - (matprod(a.transpose(), lam)).get_item(0);
		relam = gvec - a * z.get_item(0) - y + s - b;
		rexsi = xsi * (x - alfa) - epsi;
		reeta = eta * (beta - x) - epsi;
		remu = mu * y - epsi;
		rezet = zet * z - epsi;
		res = lam * s - epsi;
		static gmatd residu;
		residu = concat(std::vector<gmatd*>({ &rex,&rey,&rez,&relam,&rexsi,&reeta,&remu,&rezet,&res }));
		double residunorm = residu.norm();
		double residumax = abs(residu).max_item();
		int ittt = 0;
		while (residumax > .9 * epsi && ittt < 200)
		{
			++ittt;
			++itera;
			ux1 = upp - x;
			xl1 = x - low;
			ux2 = ux1 * ux1;
			xl2 = xl1 * xl1;
			ux3 = ux1 * ux2;
			xl3 = xl1 * xl2;
			uxinv1 = 1. / ux1;
			xlinv1 = 1. / xl1;
			uxinv2 = 1. / ux2;
			xlinv2 = 1. / xl2;
			plam = p0 + matprod(P.transpose(), lam);
			qlam = q0 + matprod(Q.transpose(), lam);
			gvec = matprod(P, uxinv1) + matprod(Q, xlinv1);
			dpsidx = plam / ux2 - qlam / xl2;
			calGG(P, Q, GG, uxinv2, xlinv2);
			delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x);
			dely = c + d * y - lam - epsi / y;
			delz = a0 - (matprod(a.transpose(), lam)).get_item(0) - epsi / z;
			dellam = gvec - a * z.get_item(0) - y - b + epsi / lam;
			diagx = plam / ux3 + qlam / xl3;
			diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x);
			diagxinv = 1. / diagx;
			diagy = d + mu / y;
			diagyinv = 1. / diagy;
			diaglam = s / lam;
			diaglamyi = diaglam + diagyinv;
			if (m < n)
			{
				blam = dellam + dely / diagy - matprod(GG, delx / diagx);
				bb = concat(std::vector<gmatd*>({ &blam,&delz }));
				calAlam(Alam, diagxinv, diaglamyi, GG);
				static gmatd AAr1, AAr2, tmp;
				tmp = -1. * zet / z;
				AAr1 = (concat(std::vector<gmatd*>({ &Alam,&a }), 1));
				AAr2 = (concat(std::vector<gmatd*>({ &a,&tmp }))).transpose();
				AA = concat(std::vector<gmatd*>({ &AAr1,&AAr2 }));
				solveAb(AA, bb, dlam, dz);
				dx = -1. * delx / diagx - matprod(GG.transpose(), dlam) / diagx;
			}
			else {
				//use diaglamyi as inverse
				//TODO
			}
			dy = dlam / diagy - dely / diagy;
			//combine same factor???
			//dxsi = (epsi - xsi * dx) / (x - alfa) - xsi;
			//deta = (epsi + eta * dx) / (beta - x) - eta;
			//dmu = (epsi - mu * dy) / y - mu;
			//dzet = (epsi - zet * dz) / z - zet;
			//ds = (epsi - s * dlam) / lam - s;
			dxsi = epsi / (x - alfa) - (xsi * dx) / (x - alfa) - xsi;
			deta = epsi / (beta - x) + (eta * dx) / (beta - x) - eta;
			dmu = epsi / y - (mu * dy) / y - mu;
			dzet = epsi / z - zet * dz / z - zet;
			ds = epsi / lam - (s * dlam) / lam - s;
			static gmatd xx, dxx, stepxx, stepalfa, stepbeta;
			xx = concat(std::vector<gmatd*>({ &y,&z,&lam,&xsi,&eta,&mu,&zet,&s }));
			dxx = concat(std::vector<gmatd*>({ &dy,&dz,&dlam,&dxsi,&deta,&dmu,&dzet,&ds }));
			stepxx = -1.01 * dxx / xx;
			double stmxx = stepxx.max_item();
			stepalfa = -1.01 * dx / (x - alfa);
			double stmalfa = stepalfa.max_item();
			//use stepalfa as stepbeta
			stepbeta = 1.01 * dx / (beta - x);
			double stmbeta = stepbeta.max_item();
			double steg = 1. / std::max(1., std::max(stmxx, std::max(stmalfa, stmbeta)));
			xold = x;
			yold = y;
			zold = z;
			lamold = lam;
			xsiold = xsi;
			etaold = eta;
			muold = mu;
			zetold = zet;
			sold = s;
			int itto = 0;
			double resinew = 2 * residunorm;
			while (resinew > residunorm && itto < 50)
			{
				++itto;
				x = xold + steg * dx;
				y = yold + steg * dy;
				z = zold + steg * dz;
				lam = lamold + steg * dlam;
				xsi = xsiold + steg * dxsi;
				eta = etaold + steg * deta;
				mu = muold + steg * dmu;
				zet = zetold + steg * dzet;
				s = sold + steg * ds;
				ux1 = upp - x;
				xl1 = x - low;
				ux2 = ux1 * ux1;
				xl2 = xl1 * xl1;
				uxinv1 = 1. / ux1;
				xlinv1 = 1. / xl1;
				plam = p0 + matprod(P.transpose(), lam);
				qlam = q0 + matprod(Q.transpose(), lam);
				gvec = matprod(P, uxinv1) + matprod(Q, xlinv1);
				dpsidx = plam / ux2 - qlam / xl2;
				rex = dpsidx - xsi + eta;
				rey = c + d * y - mu - lam;
				rez = a0 - zet - (matprod(a.transpose(), lam)).get_item(0);
				relam = gvec - a * z.get_item(0) - y + s - b;
				rexsi = xsi * (x - alfa) - epsi;
				reeta = eta * (beta - x) - epsi;
				remu = mu * y - epsi;
				rezet = zet * z - epsi;
				res = lam * s - epsi;
				residu = concat(std::vector<gmatd*>({ &rex,&rey,&rez,&relam,&rexsi,&reeta,&remu,&rezet,&res }));
				resinew = residu.norm();
				steg /= 2;
			}
			residunorm = resinew;
			residumax = abs(residu).max_item();
			steg = 2 * steg;
		}
		std::cout << " ittt = " << ittt << std::endl;
		epsi = .1 * epsi;
	}
	//return std::make_tuple(x, y, z, lam, xsi, eta, mu, zet, s);
}

void mmasub(const int m, const int n, int iter, gmatd& x, const gmatd& xmin, const gmatd& xmax, gmatd& xold1, gmatd& xold2, const gmatd& dfdx, const gmatd& g, const gmatd& dgdx, gmatd& low, gmatd& upp, const double a0, gmatd& a, const gmatd& c, const gmatd& d, const double move)
{
	double epsimin = 1e-7;
	double raa0 = 1e-5;
	double albefa = 0.1;
	double asyinit = 0.5;
	double asyincr = 1.2;
	double asydecr = 0.7;
	static gmatd factor(n, 1);
	factor.set_from_value(1);
	static gmatd xmami;
	xmami = max(xmax - xmin, 1e-5);

	if (iter < 3)
	{
		low = x - asyinit * (xmax - xmin);
		upp = x + asyinit * (xmax - xmin);
	}
	else {
		static gmatd zzz;
		zzz = (x - xold1) * (x - xold2);
		factor.set_by_mask(zzz > 0, asyincr);
		factor.set_by_mask(zzz < 0, asydecr);
		low = x - factor * (xold1 - low);
		upp = x + factor * (upp - xold1);
		low.maximum(x - 10. * xmami);
		low.minimum(x - 0.01 * xmami);
		upp.maximum(x + 0.01 * xmami);
		upp.minimum(x + 10. * xmami);
	}
	static gmatd zzz1, zzz2, alfa, beta;
	zzz1 = low + albefa * (x - low);
	zzz2 = x - move * (xmax - xmin);
	alfa = max(zzz1, zzz2);
	alfa.maximum(xmin);
	//reuse zzz1,zzz2
	zzz1 = upp - albefa * (upp - x);
	zzz2 = x + move * (xmax - xmin);
	beta = min(zzz1, zzz2);
	beta.minimum(xmax);
	//use xmami as xmamiinv
	static gmatd xmamiinv;
	xmamiinv = 1. / xmami;
	static gmatd ux1, ux2, xl1, xl2, uxinv, xlinv;
	ux1 = upp - x;
	ux2 = ux1 * ux1;
	xl1 = x - low;
	xl2 = xl1 * xl1;
	uxinv = 1. / ux1;
	xlinv = 1. / xl1;
	static gmatd p0, q0, pq0;
	p0 = max(dfdx, 0.);
	q0 = max(-1. * dfdx, 0.);
	pq0 = 1e-3 * (p0 + q0) + raa0 * xmamiinv;
	p0 += pq0;
	p0 *= ux2;
	q0 += pq0;
	q0 *= xl2;
	static gmatd P, Q, PQ;
	P = max(dgdx, 0.);
	Q = max(-1. * dgdx, 0.);
	PQ = 1e-3 * (P + Q) + raa0 * matprod(gmatd(m, 1, 1), xmami.transpose());
	calPQ(P, Q, PQ, ux2, xl2);
	static gmatd b;
	b = matprod(P, uxinv) + matprod(Q, xlinv) - g;
	subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d, x/*, ux1, ux2, xl1, xl2*/);
	xold2 = xold1;
	xold1 = x;
}