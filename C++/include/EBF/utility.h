#pragma once
#include<Header.h>

inline Mat computerG_fast(Mat &X) {
	int m = X.rows;
	Mat one = Mat::ones(m, 1, CV_64FC1);
	// nornX
	Mat XX;
	pow(X, 2, XX);
	Mat normX = XX * Mat::ones(X.cols, 1, CV_64FC1);
	// get G
	Mat G = -2 * X * X.t() + normX * one.t() + one * normX.t();
	exp(-G, G);

	return G;
}

inline Mat computerG_fast(Mat &X,  Mat &Y) {
	int m = X.rows, n = Y.rows;

	// normX
	Mat XX;
	pow(X, 2, XX);
	Mat normX = XX * Mat::ones(X.cols, 1, CV_64FC1);

	// normY
	Mat YY;
	pow(Y, 2, YY);
	Mat normY = YY * Mat::ones(Y.cols, 1, CV_64FC1);

	// get G
	Mat G = -2 * X * Y.t() + normX * Mat::ones(1, n, CV_64FC1) + 
		Mat::ones(m, 1, CV_64FC1) * normY.t();
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