#pragma once
#include<Header.h>

inline Mat computerG_fast(Mat &X) {
	int m = X.rows;
	Mat one = Mat::ones(m, 1, CV_64FC1);
	Mat sumX(m, 1, CV_64FC1);

	for (int i = 0; i < m; i++)
	{
		sumX.at<double>(i, 0) = pow(norm(X.row(i), NORM_L2), 2);
	}

	Mat G = -2 * X * X.t() + sumX * one.t() + one * sumX.t();
	exp(-G, G);

	return G;
}

inline Mat computerG_fast(Mat &X,  Mat &Y) {
	int m = X.rows, n = Y.rows;

	// sumX
	Mat sumX(m, 1, CV_64FC1);
	for (int i = 0; i < m; i++)
	{
		sumX.at<double>(i, 0) = pow(norm(X.row(i), NORM_L2), 2);
	}
	// sumY
	Mat sumY(n, 1, CV_64FC1);
	for (int i = 0; i < n; i++)
	{
		sumY.at<double>(i, 0) = pow(norm(Y.row(i), NORM_L2), 2);
	}

	Mat G = -2 * X * Y.t() + sumX * Mat::ones(1, n, CV_64FC1) + 
		Mat::ones(m, 1, CV_64FC1) * sumY.t();
	exp(-G, G);

	return G;
}


inline void featureNomalize(Mat &X, vector<double> &mu, vector<double> &sigma) {
	X.convertTo(X, CV_64FC1);
	if (mu.empty() && sigma.empty())
	{
		int dims = X.cols;
		mu.resize(dims);
		sigma.resize(dims);
		vector<double> tempmu, tempsigma;
		for (int i = 0; i < dims; i++)
		{
			meanStdDev(X.col(i), tempmu, tempsigma);
			mu[i] = tempmu[0];
			sigma[i] = tempsigma[0];
		}
	}

	// perform normalize
	for (int y = 0; y < X.rows; y++)
	{
		for (int x = 0; x < X.cols; x++)
		{
			X.at<double>(y, x) = (X.at<double>(y, x) - mu[x]) / sigma[x];
		}
	}
}