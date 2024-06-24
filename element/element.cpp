#include "element.h"

inline void data_process(const float &r, const float &t1, const float &t2, const float &t3, at::Tensor& rst)
{
	rst = move(torch::tensor({ float((r - 0.3) / 0.4),float(t1 / 90),float(t2 / 90),float(t3 / 90) }));
}

void spinodal::predict(const float* x, float* S, float* dSdx)
{
	for (int i = 0; i < nel; ++i)
	{
		at::Tensor input;
		cout << &input << endl;
		data_process(x[i], x[i + nel], x[i + 2 * nel], x[i + 3 * nel], input);
		cout << &input << endl;
		input.requires_grad_();
		cout << &input << endl;
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
	for (int i = 0; i < 9; ++i)
		sk += (coef.col(i) * ss.row(i)).cast<double>().reshaped();
}

void spinodal::sensitivity(float* dSdx, Eigen::VectorXd& dskdx, int i)
{
	// 0<=i<4*nel
	int q = i / nel;
	int r = i - q * nel;
	memcpy(temp + 9 * r, dSdx + 36 * r + 9 * q, 9 * sizeof(float));
	auto ss = Eigen::Map<Eigen::MatrixXf>(temp, 9, nel);
	for (int i = 0; i < 9; ++i)
		dskdx += (coef.col(i) * ss.row(i)).cast<double>().reshaped();
	fill(temp + 9 * r, temp + 9 * (r + 1), 0.f);
}