#include "utility.h"


MatrixXd computerG(MatrixXd &X) {
	const int M = X.rows();
	const int N = X.cols();

	// Allocate parts of the expression
	MatrixXd XX, D;

	// Compute norms
	XX = X.array().square().rowwise().sum();

	// Compute final expression
	D = XX * MatrixXd::Ones(1, M) + MatrixXd::Ones(M, 1) * XX.transpose() - 2 * X * X.transpose();
	MatrixXd G = (-D).array().exp();

	return G;
}

MatrixXd computerG(MatrixXd &X, MatrixXd &Y) {
	const int M = X.rows();
	const int N = Y.rows();

	// Allocate parts of the expression
	MatrixXd XX, YY, XY, D;

	// Compute norms
	XX = X.array().square().rowwise().sum();
	YY = Y.array().square().rowwise().sum().transpose();
	XY = 2 * X * Y.transpose();

	// Compute final expression
	D = XX * MatrixXd::Ones(1, N) + MatrixXd::Ones(M, 1) * YY - XY;
	MatrixXd G = (-D).array().exp();

	return G;
}


// Eigen
void featureNormalize(MatrixXd &X, MatrixXd &mu, MatrixXd &sigma){
	int m = X.rows(), n = X.cols();
	if (mu.rows() == 0 && sigma.rows() == 0)
	{
		mu = MatrixXd::Ones(1, m) * X / m;
		sigma = ((X - MatrixXd::Ones(m, 1) * mu).array().square().colwise().sum() / m).sqrt();
	}
	X = (X - MatrixXd::Ones(m, 1) * mu).array() / (MatrixXd::Ones(m, 1) * sigma).array();
}