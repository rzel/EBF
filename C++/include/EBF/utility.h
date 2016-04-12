#pragma once
#include<Header.h>


// compute Gaussian similar matrix
MatrixXd computerG(MatrixXd &X);
MatrixXd computerG(MatrixXd &X, MatrixXd &Y);


// Eigen for featureNormalize
void featureNormalize(MatrixXd &X, MatrixXd &mu, MatrixXd &sigma);