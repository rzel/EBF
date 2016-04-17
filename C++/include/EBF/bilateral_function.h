#pragma once
#include <Header.h>
#include<lbfgs.h>

class bilateral_function
{
protected:
	lbfgsfloatval_t *m_x;
	int N, m;
	double lambda, threshold;
public:
	MatrixXd G, big_G, Label, Lx, Ly;
public:
	bilateral_function(MatrixXf &X_train, MatrixXf &matching, double thres) : m_x(nullptr)
	{
		m = (int)X_train.cols();
		N = 3 * m + 3;
		m_x = lbfgs_malloc(N);
		threshold = thres;	
		Lx = matching.row(2).transpose().cast<double>();
		Ly = matching.row(3).transpose().cast<double>();
		lambda = 1;

		// construct G and G_big	
		G = Tools::get_GSM_fast(X_train).cast<double>();
		big_G.resize(m, N);
		big_G.block(0, 0, m, m) = G.array() * (Lx * MatrixXd::Ones(1, m)).array();
		big_G.block(0, m, m, m) = G.array() * (Ly * MatrixXd::Ones(1, m)).array();
		big_G.block(0, 2 * m, m, m) = G;
		big_G.block(0, 3 * m, m, 3) = MatrixXd::Ones(m, 3);
	}
	~bilateral_function() {
		if (m_x != nullptr) {
			lbfgs_free(m_x);
			m_x = nullptr;
		}
	}

	bool optimize(MatrixXd &w1, MatrixXd &w2) {
		double fx;

		lbfgs_parameter_t param;
		lbfgs_parameter_init(&param);


		// learning step 1
		// initialize m_x
		for (int i = 0; i < N; i++) { m_x[i] = threshold; }
		// set label
		Label = Lx;
		int ret1 = lbfgs(N, m_x, &fx, _evaluate, nullptr, this, &param);
		w1.resize(N, 1);
		memcpy(w1.data(), m_x, sizeof(double) * N);

		if (ret1!=0)
		{
			printf("learning step 1 terminated with status code = %d\n", ret1);
			printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, m_x[0], m_x[1]);
		}
	
		// learning step 2
		// initialize m_x
		for (int i = 0; i < N; i++) { m_x[i] = threshold; }
		// set label
		Label = Ly;
		int ret2 = lbfgs(N, m_x, &fx, _evaluate, nullptr, this, &param);
		w2.resize(N, 1);
		memcpy(w2.data(), m_x, sizeof(double) * N);

		if (ret2 != 0)
		{
			printf("learning step 2 terminated with status code = %d\n", ret2);
			printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, m_x[0], m_x[1]);
		}
		if (ret1 != 0 || ret2 != 0)	{ return false; }
		return true;
	}

protected:
	static lbfgsfloatval_t _evaluate(
		void *instance,
		const lbfgsfloatval_t *x,
		lbfgsfloatval_t *g,
		const int n,
		const lbfgsfloatval_t step
		)
	{
		return reinterpret_cast<bilateral_function*>(instance)->evaluate(x, g, n, step);
	}

	lbfgsfloatval_t evaluate(
		const lbfgsfloatval_t *x,
		lbfgsfloatval_t *g,
		const int n,
		const lbfgsfloatval_t step
		)
	{
		// prepare
		MatrixXd w(N, 1);	memcpy((double *)w.data(), (double *)x, sizeof(double) * N);
		MatrixXd W(m, 3);   memcpy((double *)W.data(), (double *)x, sizeof(double) * 3 * m);

		MatrixXd H = G * W;
		// cost function
		lbfgsfloatval_t cost = 0;
		MatrixXd z = Label - big_G * w;
		double regular_error = (lambda * W.transpose() * H).diagonal().sum();
		double huber_error = Tools::p_huber_cost(z, threshold);
		cost = (huber_error + regular_error) / m;
		

		// grad
		MatrixXd grad = Tools::p_huber_grad(z, threshold) * (-big_G);
		MatrixXd reguar_grad = MatrixXd::Zero(1, N);		
		MatrixXd reguarWg = 2 * lambda * H;
		memcpy(reguar_grad.data(), reguarWg.data(), 3 * m * sizeof(double));
		grad += reguar_grad;
		grad /= m;
		memcpy((double *)g, (double *)grad.data(), sizeof(double) * N);

		return cost;
	}

	static int _progress(
		void *instance,
		const lbfgsfloatval_t *x,
		const lbfgsfloatval_t *g,
		const lbfgsfloatval_t fx,
		const lbfgsfloatval_t xnorm,
		const lbfgsfloatval_t gnorm,
		const lbfgsfloatval_t step,
		int n,
		int k,
		int ls
		)
	{
		return reinterpret_cast<bilateral_function*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
	}

	int progress(
		const lbfgsfloatval_t *x,
		const lbfgsfloatval_t *g,
		const lbfgsfloatval_t fx,
		const lbfgsfloatval_t xnorm,
		const lbfgsfloatval_t gnorm,
		const lbfgsfloatval_t step,
		int n,
		int k,
		int ls
		)
	{
		printf("Iteration %d:\n", k);
		printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
		printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
		printf("\n");
		return 0;
	}


};









