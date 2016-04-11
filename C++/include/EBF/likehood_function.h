#pragma once
#include <Header.h>
#include <huber_function.h>
#include <utility.h>

class likehood_function
{
protected:
	lbfgsfloatval_t *m_x;
	Mat G;
	int m;
	double lambda, threshold;
public:
	likehood_function(Mat &X_train, double thres) : m_x(nullptr) 
	{	
		m = X_train.rows; 
		m_x = lbfgs_malloc(m);
		threshold = thres;
		for (int i = 0;i < m; i++) { m_x[i] = threshold; }
		G = computerG(X_train);
		lambda = 1.0;
	}
	~likehood_function() {
		if (m_x != nullptr) {
			lbfgs_free(m_x);
			m_x = nullptr;
		}
	}

	bool optimize(vector<double> &w) {
		double fx;

		lbfgs_parameter_t param;
		lbfgs_parameter_init(&param);
//		param.max_iterations = 1000;

		//lbfgs
		int ret = lbfgs(m, m_x, &fx, _evaluate, nullptr, this, &param);

		w.resize(m);
		memcpy(w.data(), m_x, sizeof(double) * m);

		/* Report the result. */
		printf("L-BFGS optimization terminated with status code = %d\n", ret);
		printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, m_x[0], m_x[1]);

		if (ret == 0)
		{
			return true;
		}
		return false;
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
		Mat w(m, 1, CV_64FC1, (double *)x);
		Mat huber_grad(1, m, CV_64FC1);

		// cost function
		lbfgsfloatval_t cost;
		Mat z = 1 - G * w;
		Mat regular = lambda * w.t() * G * w;
		double huber_error = huber_function::getCost((double*)z.data, m, threshold);
		cost = huber_error + regular.at<double>(0, 0);
		
		// Grad
		huber_function::getGrad((double *)z.data, (double *)huber_grad.data, m, threshold);
		Mat grad = huber_grad * (-G) + 2 * lambda * w.t() * G;
		memcpy((double *)g, (double *)grad.data, sizeof(double) * m);


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



