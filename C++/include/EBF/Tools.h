#pragma once

#include "get_gsm_sse2_.h"


#include <Eigen/Dense>
using namespace Eigen;





class Tools{

public:
	// compute Gaussian similar matrix
	static MatrixXf get_GSM_sse2(MatrixXf &X, MatrixXf &Y = MatrixXf()) {
		const size_t M1 = X.cols();
		const size_t M2 = Y.cols();
		const size_t N = X.rows();

		MatrixXf D;
		if (M2 == 0)
		{
			D.resize(M1, M1);
			get_gsm_sse2_d(D.data(), N, X.data(), M1, NULL, 0);
		}
		else {
			D.resize(M1, M2);
			get_gsm_sse2_d(D.data(), N, X.data(), M1, Y.data(), M2);
		}
		MatrixXf G = (-D).array().exp();
		return G;
	}



	// Eigen for featureNormalize	row by default
	//
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
	static double p_huber_cost(MatrixXd &x, double threshold) {
		return pow(threshold, 2) * (((x.array() / threshold).square() + 1).sqrt() - 1).sum();
	}
	// pseudo huber grad
	static MatrixXd p_huber_grad(MatrixXd &x, double threshold) {
		return (x.array() / ((x.array() / threshold).square() + 1).sqrt()).transpose();
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