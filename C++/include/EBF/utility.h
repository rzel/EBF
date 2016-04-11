#pragma once
#include<Header.h>

inline Mat computerG(Mat &X) {
	Mat G(X.rows, X.rows, CV_64FC1);
	for (int y = 0; y < G.rows; y++)
	{
		for (int x = y; x < G.cols; x++)
		{
			double l2 = norm(X.row(y) - X.row(x), NORM_L2);
			G.at<double>(y, x) = exp(-pow(l2, 2));
			G.at<double>(x, y) = exp(-pow(l2, 2));
		}
	}
	return G;
}

inline Mat computerG(Mat &X, Mat &Y) {
	Mat G(X.rows, Y.rows, CV_64FC1);
	for (int y = 0; y < G.rows; y++)
	{
		for (int x = y; x < G.cols; x++)
		{
			double l2 = norm(X.row(y) - Y.row(x), NORM_L2);
			G.at<double>(y, x) = exp(-pow(l2, 2));
			G.at<double>(x, y) = exp(-pow(l2, 2));
		}
	}
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