#pragma once
#include <Eigen/Dense>
using namespace Eigen;


class Tools{

public:
	// compute Gaussian similar matrix
	static MatrixXd getGSM(MatrixXd &X) {
		const int M = (int)X.cols();

		// Compute norms
		MatrixXd XX = X.colwise().squaredNorm();

		// Compute final expression
		MatrixXd D = MatrixXd::Ones(M, 1) * XX + XX.transpose() * MatrixXd::Ones(1, M) - 2 * X.transpose() * X;
		
		MatrixXd G = (-D).array().exp();

		return G;
	}
	// compute Gaussian similar matrix
	static MatrixXd getGSM(MatrixXd &X, MatrixXd &Y) {
		const int M = (int)X.cols();
		const int N = (int)Y.cols();

		// Allocate parts of the expression
		MatrixXd XX, YY, XY, D;

		// Compute norms
		XX = X.colwise().squaredNorm();
		YY = Y.colwise().squaredNorm();
		XY = 2 * X.transpose() * Y;

		// Compute final expression
		D = XX.transpose() * MatrixXd::Ones(1, N) + MatrixXd::Ones(M, 1) * YY - XY;
		MatrixXd G = (-D).array().exp();

		return G;
	}


	// Eigen for featureNormalize	row by default
	//
	static MatrixXd featureNormalize(MatrixXd &X, MatrixXd &mu, MatrixXd &sigma, bool row = true){
		if (row == true){
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
