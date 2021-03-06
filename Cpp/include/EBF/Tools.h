#pragma once
#include "math_sse2.h"
//#include "math_avx.h"
#include <Eigen/Core>
#include <Eigen/SVD>
using namespace Eigen;


class Tools{

public:
	// compute Gaussian similar matrix
	static MatrixXf get_GSM_fast(MatrixXf &X) {
		const size_t M = X.cols();
		const size_t N  = X.rows();

		MatrixXf G(M, M);
		dist_l2_matrix_sse2_f(G.data(), N, X.data(), M, NULL, 0);
		G = (-G).array().exp();
		return G;
	}

	static MatrixXf get_GSM_fast(MatrixXf &X, MatrixXf &Y) {
		const size_t M1 = X.cols();
		const size_t M2 = Y.cols();
		const size_t N  = X.rows();

		MatrixXf G(M1, M2);
		dist_l2_matrix_sse2_f(G.data(), N, X.data(), M1, Y.data(), M2);
		G = (-G).array().exp();

		return G;
	}
	static MatrixXf kernel_pca(MatrixXf &G, MatrixXf &U, int d) {
		
		if (U.rows() < 1 && U.cols() < 1)
		{
			// center G
			int m = (int)G.rows();
			MatrixXf one = MatrixXf::Ones(m, m) / m;
			MatrixXf G_center = G - one * G - G * one + one * G *one;
			JacobiSVD<MatrixXf> svd(G_center, ComputeThinU);
			U.resize(m, d);	memcpy(U.data(), svd.matrixU().data(), sizeof(float) * m *d);
			U = U.array() / (MatrixXf::Ones(m, 1) *  (U.colwise().norm())).array();
		}
		return G * U;
	}

	// Eigen for featureNormalize  row by default
	static MatrixXf featureNormalize(MatrixXf &X, MatrixXf &mu, MatrixXf &sigma, bool row = true){
		if (row == true){
			int m = (int)X.rows();
			if (mu.rows() == 0 && sigma.rows() == 0)
			{
				mu = X.array().colwise().mean();
				sigma = ((X - MatrixXf::Ones(m, 1) * mu).colwise().squaredNorm().array() / float(m)).sqrt();
			}
			return (X - MatrixXf::Ones(m, 1) * mu).array() / (MatrixXf::Ones(m, 1) * sigma).array();
		}
		else
		{
			int m = (int)X.cols();
			if (mu.rows() == 0 && sigma.rows() == 0)
			{
				mu = X.array().rowwise().mean();			
				sigma = ((X - mu * MatrixXf::Ones(1, m)).rowwise().squaredNorm().array() / m).sqrt();
			}
			return	(X - mu * MatrixXf::Ones(1, m)).array() / (sigma * MatrixXf::Ones(1, m)).array();
		}
	}
	static MatrixXd featureNormalize(MatrixXd &X, MatrixXd &mu, MatrixXd &sigma, bool row = true) {
		if (row == true) {
			int m = (int)X.rows();
			if (mu.rows() == 0 && sigma.rows() == 0)
			{
				mu = X.array().colwise().mean();
				sigma = ((X - MatrixXd::Ones(m, 1) * mu).colwise().squaredNorm().array() / m).sqrt();
			}
			return (X - MatrixXd::Ones(m, 1) * mu).array() / (MatrixXd::Ones(m, 1) * sigma).array();
		}
		else
		{
			int m = (int)X.cols();
			if (mu.rows() == 0 && sigma.rows() == 0)
			{
				mu = X.array().rowwise().mean();
				sigma = ((X - mu * MatrixXd::Ones(1, m)).rowwise().squaredNorm().array() / m).sqrt();
			}
			return	(X - mu * MatrixXd::Ones(1, m)).array() / (sigma * MatrixXd::Ones(1, m)).array();
		}
	}



	// pseudo huber cost
	static float p_huber_cost(MatrixXf &x, float threshold) {
		return threshold * threshold * (((x.array() / threshold).square() + 1).sqrt() - 1).sum();
//		return phuber_cost_avx_f(x.rows(), x.data(), threshold);
	}
	// pseudo huber grad
	static MatrixXf p_huber_grad(MatrixXf &x, float threshold) {
		return (x.array() / ((x.array() / threshold).square() + 1).sqrt()).transpose();
		
		//size_t dimension = x.size();
		//MatrixXf grad(1, dimension);
		//phuber_grad_avx_f(dimension, x.data(), grad.data(), threshold);
		//return grad;
	}


};




class huber_function {
public:
	static double cost(double * x, int length, double threshold) {
		double cost = 0;
		for (int i = 0; i < length; i++)
		{
			cost += abs(x[i]) < threshold ? pow(x[i], 2) : (2 * threshold*abs(x[i]) - pow(threshold, 2));
		}
		return cost;
	}

	static void grad(double * x, double * g, int length, double threshold) {
		for (int i = 0; i < length; i++)
		{
			if (abs(x[i]) < threshold) { g[i] = 2 * x[i]; }
			else if (x[i] > threshold) { g[i] = 2 * threshold; }
			else if (-x[i] > threshold) { g[i] = -2 * threshold; }
		}
	}
};
