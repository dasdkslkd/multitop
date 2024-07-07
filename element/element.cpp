#include "element.h"

inline void data_process(const float &r, const float &t1, const float &t2, const float &t3, at::Tensor& rst)
{
	rst = move(torch::tensor({ float((r - 0.3) / 0.4),float(t1 / 90),float(t2 / 90),float(t3 / 90) }));
}

void spinodal::predict(const float* x, float* S, float* dSdx)
{
//#pragma omp parallel for
	for (int i = 0; i < nel; ++i)
	{
		at::Tensor input;
		data_process(x[i], x[i + nel] * 180 / PI, x[i + 2 * nel] * 180 / PI, x[i + 3 * nel] * 180 / PI, input);
		input.requires_grad_();
		auto output = model({ input }).toTensor();
		auto data = output.data_ptr<float>();
		memcpy(S + 9 * i, data, 9 * sizeof(float));
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

void spinodal::elasticity(float* S, Eigen::VectorXd& sk)
{
	auto ss = Eigen::Map<Eigen::MatrixXf>(S, 9, nel);
	sk.setZero();
	for (int i = 0; i < 9; ++i)
		sk += (coef.col(i) * ss.row(i)).cast<double>().reshaped();
}

void spinodal::sensitivity(float* dSdx, Eigen::VectorXd& dskdx, int& i)
{
	// 0<=i<4*nel
	static int q = i / nel;
	static int r = i - q * nel;
	memcpy(temp + 9 * r, dSdx + 36 * r + 9 * q, 9 * sizeof(float));
	auto ss = Eigen::Map<Eigen::MatrixXf>(temp, 9, nel);
	dskdx.setZero();
	for (int i = 0; i < 9; ++i)
		dskdx += (coef.col(i) * ss.row(i)).cast<double>().reshaped();	
	fill(temp + 9 * r, temp + 9 * (r + 1), 0.f);
}

void spinodal::filter(float* x)
{
	static float rho_min = 0.3f;
	static float theta_min = PI / 18;
	static float lam1 = 600.f;
	static float lam2 = 60 * 180 / PI;
	for (int i = 0; i < nel; ++i)
	{
		x[i] /= (1 + exp(-lam1 * (x[i] - rho_min)));
		x[i + nel] = max(x[i + nel], theta_min) / (1 + exp(-lam2 * (x[i + nel] - theta_min / 2)));
	}

}