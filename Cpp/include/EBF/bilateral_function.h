#pragma once
#include <Header.h>
#include<lbfgs.h>

class bilateral_function
{
protected:
	lbfgsfloatval_t *m_x;
	int N, m, n;
	double lambda, threshold;
public:
	MatrixXf G, big_G, Label, Lx, Ly, U;
public:
	bilateral_function(MatrixXf &X_train, MatrixXf &matching, double thres) : m_x(nullptr)
	{
		n = 20;
		m = (int)X_train.cols();
		N = 3 * n + 3;
		m_x = lbfgs_malloc(N);
		threshold = thres;	
		Lx = matching.row(2).transpose();
		Ly = matching.row(3).transpose();
		lambda = 1;

		// construct G and G_big	
		G = Tools::get_GSM_fast(X_train);
		G = Tools::kernel_pca(G, U, n);
		big_G.resize(m, N);
		big_G.block(0, 0, m, n) = G.array() * (Lx * MatrixXf::Ones(1, n)).array();
		big_G.block(0, n, m, n) = G.array() * (Ly * MatrixXf::Ones(1, n)).array();
		big_G.block(0, 2 * n, m, n) = G;
		big_G.block(0, 3 * n, m, 3) = MatrixXf::Ones(m, 3);
	}
	~bilateral_function() {
		if (m_x != nullptr) {
			lbfgs_free(m_x);
			m_x = nullptr;
		}
	}

	bool optimize(MatrixXf &w1, MatrixXf &w2) {
		lbfgsfloatval_t fx;

		lbfgs_parameter_t param;
		lbfgs_parameter_init(&param);


		// learning step 1
		// initialize m_x
		for (int i = 0; i < N; i++) { m_x[i] = threshold; }
		// set label
		Label = Lx;
		int ret1 = lbfgs(N, m_x, &fx, _evaluate, nullptr, this, &param);
		w1.resize(N, 1);
		for (int i = 0; i < N; i++) { w1(i) = m_x[i]; }

		if (ret1 != 0 && ret1 != -1001)
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
		for (int i = 0; i < N; i++) { w2(i) = m_x[i]; }

		if (ret2 != 0 && ret2 != -1001)
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
		MatrixXf w(N, 1);	memcpy(w.data(), x, sizeof(float) * N);
		
		// cost function
		lbfgsfloatval_t cost = 0;
		MatrixXf e = Label - big_G * w;
		double regular_error = (lambda * w.transpose() * w).sum();
		double huber_error = Tools::p_huber_cost(e, threshold);
		cost = (huber_error + regular_error) / m;
		
		// grad
		MatrixXf grad = (2 * lambda * w - Tools::p_huber_grad(e, threshold) * big_G) / m;
		memcpy(g, grad.data(), sizeof(float) * N);
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









