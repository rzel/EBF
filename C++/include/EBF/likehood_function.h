#pragma once
#include <Header.h>
#include<lbfgs.h>

class likehood_function
{
protected:
	lbfgsfloatval_t *m_x;
	int m;
	double lambda, threshold;
public:
	MatrixXf G;
public:
	likehood_function(MatrixXf &X_train, double thres) : m_x(nullptr)
	{	
		m = (int)X_train.cols();
		m_x = lbfgs_malloc(m);
		threshold = thres;
		for (int i = 0;i < m; i++) { m_x[i] = threshold; }
		G = Tools::get_GSM_fast(X_train);
		lambda = 1;
	}
	~likehood_function() {
		if (m_x != nullptr) {
			lbfgs_free(m_x);
			m_x = nullptr;
		}
	}

	void optimize(MatrixXf &w) {
		lbfgsfloatval_t fx;

		lbfgs_parameter_t param;
		lbfgs_parameter_init(&param);
//		param.xtol = 1.0e-7;
		
		//lbfgs
		int ret = lbfgs(m, m_x, &fx, _evaluate, nullptr, this, &param);
		w.resize(m, 1);
		for (size_t i = 0; i < m; i++) { w(i) = m_x[i]; }

		if (ret != 0 && ret != -1001)
		{
			printf("L-BFGS optimization terminated with status code = %d\n", ret);
			printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, m_x[0], m_x[1]);
		}

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
		return reinterpret_cast<likehood_function*>(instance)->evaluate(x, g, n, step);
	}

	lbfgsfloatval_t evaluate(
		const lbfgsfloatval_t *x,
		lbfgsfloatval_t *g,
		const int n,
		const lbfgsfloatval_t step
		)
	{
		// prepare
		MatrixXf w(m, 1);	for (size_t i = 0; i < m; i++) { w(i) = x[i]; }
		
		MatrixXf h = G * w;
		// cost function
		lbfgsfloatval_t cost;
		MatrixXf e = 1 - h.array();
		double regular_cost = lambda * (w.transpose() * h).array().sum();
		double huber_cost = Tools::p_huber_cost(e, threshold);
		cost = (huber_cost + regular_cost) / m;
		
		// Grad
		MatrixXf huber_grad = Tools::p_huber_grad(e, threshold);
		MatrixXf grad = (2 * lambda * h.transpose() + huber_grad * (-G)) / m;
		for (size_t i = 0; i < m; i++) { g[i] = grad(i); }

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
		return reinterpret_cast<likehood_function*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
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



