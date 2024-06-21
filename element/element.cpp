#include "element.h"

inline void data_process(const float &r, const float &t1, const float &t2, const float &t3, at::Tensor& rst)
{
	rst = move(torch::tensor({ float((r - 0.3) / 0.4),float(t1 / 90),float(t2 / 90),float(t3 / 90) }));
}

void spinodal::predict(const float* x, float* S, float* dSdx, int nel)
{
	for (int i = 0; i < nel; ++i)
	{
		at::Tensor input;
		data_process(x[i], x[i + nel], x[i + 2 * nel], x[i + 3 * nel], input);
		input.requires_grad_();
		auto output = model({ input }).toTensor().data_ptr<float>();
		memcpy(S + 9 * i, output, 9 * sizeof(float));
		for (int j = 0; j < 9; ++j)
		{
			auto xx = input.clone();
			xx.retain_grad();
			auto y = model({ xx }).toTensor();
			auto t = torch::zeros({ 9 });
			t[j] = 1;
			y.backward(t);
			dSdx[36 * i + 4 * j] = xx.grad()[0].item().toFloat();
			dSdx[36 * i + 4 * j + 1] = xx.grad()[1].item().toFloat();
			dSdx[36 * i + 4 * j + 2] = xx.grad()[2].item().toFloat();
			dSdx[36 * i + 4 * j + 3] = xx.grad()[3].item().toFloat();
		}
	}
}

void spinodal::elasticity(const float* S, Eigen::MatrixXd& sk, int nel)
{

}

void spinodal::sensitivity(const float* dSdx, Eigen::MatrixXd& dskdx, int nel)
{

}