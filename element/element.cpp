#include "element.h"

inline void data_process(const double &r, const double &t1, const double &t2, const double &t3, at::Tensor& rst)
{
	rst = move(torch::tensor({ double((r - 0.3) / 0.4),double(t1 / 90),double(t2 / 90),double(t3 / 90) }));
}

void spinodal::predict(const double* x, double* S, double* dSdx)
{
//#pragma omp parallel for
	for (int i = 0; i < nel; ++i)
	{
		at::Tensor input;
		data_process(x[i], x[i + nel] * 180 / PI, x[i + 2 * nel] * 180 / PI, x[i + 3 * nel] * 180 / PI, input);
		input.requires_grad_();
		auto output = model({ input }).toTensor();
		auto data = output.data_ptr<double>();
		memcpy(S + 9 * i, data, 9 * sizeof(double));
		for (int j = 0; j < 9; ++j)
		{
			auto xx = input.clone();
			xx.retain_grad();
			auto y = model({ xx }).toTensor();
			auto t = torch::zeros({ 9 });
			t[j] = 1;
			y.backward(t);
			dSdx[36 * i + j] = xx.grad()[0].item().toFloat();
			dSdx[36 * i + j + 9] = xx.grad()[1].item().toFloat();
			dSdx[36 * i + j + 18] = xx.grad()[2].item().toFloat();
			dSdx[36 * i + j + 27] = xx.grad()[3].item().toFloat();
		}
	}
}

void spinodal::elasticity(double* S, Eigen::VectorXd& sk)
{
	auto ss = Eigen::Map<Eigen::MatrixXd>(S, 9, nel);
	sk.setZero();
	for (int i = 0; i < 9; ++i)
		sk += (coef.col(i) * ss.row(i)).cast<double>().reshaped();
}

void spinodal::sensitivity(double* dSdx, Eigen::VectorXd& dskdx, int& i)
{
	// 0<=i<4*nel
	static int q;
	static int r;
	q = i / nel;
	r = i - q * nel;
	memcpy(temp + 9 * r, dSdx + 36 * r + 9 * q, 9 * sizeof(double));
	auto ss = Eigen::Map<Eigen::MatrixXd>(temp, 9, nel);
	dskdx.setZero();
	for (int i = 0; i < 9; ++i)
		dskdx += (coef.col(i) * ss.row(i)).cast<double>().reshaped();	
	fill(temp + 9 * r, temp + 9 * (r + 1), 0.f);
}

void spinodal::filter(double* x)
{
	static double rho_min = 0.3f;
	static double theta_min = PI / 18;
	static double lam1 = 600.f;
	static double lam2 = 60 * 180 / PI;
	for (int i = 0; i < nel; ++i)
	{
		x[i] /= (1 + exp(-lam1 * (x[i] - rho_min)));
		x[i + nel] = max(x[i + nel], theta_min) / (1 + exp(-lam2 * (x[i + nel] - theta_min / 2)));
	}

}