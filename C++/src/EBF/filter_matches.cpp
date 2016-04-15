#pragma once
#include <Header.h>
#include <likehood_function.h>
#include <bilateral_function.h>
#include <getInlier.h>

#define LIKEHOOD_THRESH		0.3
#define BILATERAL_THRESH	0.01
#define MINIMUM_QUERY_NUM	50


bool CampareRule(const pair<double, int>&p1, const pair<double, int>&p2) {
	return p1.first < p2.first;
}

MatrixXf filter_matches(FRAME &F1, FRAME &F2, vector<DMatch> &matches_all, vector<double> &quality)
{
	// pair<priority, index> : priority = 1.0 / quality 
	vector<pair<double, int>> PriorityIdx(matches_all.size());
	for (int i = 0; i < matches_all.size(); i++)
	{
		PriorityIdx[i] = (make_pair(1.0/quality[i], i));
	}
	std::sort(PriorityIdx.begin(), PriorityIdx.end(), CampareRule);


	// matching_all  matching  :  two points per column
	int num_query = max((int)(matches_all.size() / 100), MINIMUM_QUERY_NUM);
	Map<MatrixXf, 0, OuterStride<4> >  kp1(F1.kpts, 2, F1.num_keys);
	Map<MatrixXf, 0, OuterStride<4> >  kp2(F2.kpts, 2, F2.num_keys);

	MatrixXf  matching_all(4, matches_all.size());
	
	for (int i = 0; i < matches_all.size(); i++)
	{
		int l = matches_all[i].queryIdx;
		int r = matches_all[i].trainIdx;
		matching_all.col(i) << kp1(0, l), kp1(1, l), kp2(0, r), kp2(1, r);
	}
	// normalize the points to 0 - 1
	MatrixXf ImageSize(4, 1);	ImageSize << (float)F1.w, (float)F1.h, (float)F2.w, (float)F2.h;
	matching_all = matching_all.array() / (ImageSize * MatrixXf::Ones(1, matches_all.size())).array();

	MatrixXf matching_query(4, num_query);
	for (int i = 0; i < num_query; i++ )
	{
		matching_query.col(i) = matching_all.col(PriorityIdx[i].second);
	}

	// normalize X_query
	MatrixXf  X_query, mu, sigma;
	X_query = Tools::featureNormalize(matching_query, mu, sigma, false);
	X_query.bottomRows(2) -= X_query.topRows(2);

	// normalize X_all	
	MatrixXf X_all = Tools::featureNormalize(matching_all, mu, sigma, false);
	X_all.bottomRows(2) -= X_all.topRows(2);


	clock_t bg = clock();
	// learning likehood weight
	MatrixXd w;
	likehood_function lhf(X_query, LIKEHOOD_THRESH);
	lhf.optimize(w);
	// get inlier by likehood  must first to likehood_all, because likehood will change X_query.
	clock_t ed = clock();
	getInlier::likehood_all(w, X_all, X_query, matching_all, LIKEHOOD_THRESH);
	getInlier::likehood(w, lhf.G,X_query, matching_query, LIKEHOOD_THRESH);

	cout << "likehood time : " << ed - bg << "ms         " << matching_all.cols() << endl;


	bg = clock();
	// learning bilateral function weight
	bilateral_function blf(X_query, matching_query, BILATERAL_THRESH);
	MatrixXd w1, w2;
	blf.optimize(w1, w2);
	// get inlier by bilateral function
	getInlier::bilateral_function(w1, w2, X_all, X_query, matching_all, BILATERAL_THRESH);
	ed = clock();
	cout << "bilteral function  time : " << ed - bg << "ms         " << matching_all.cols() << endl;

	return matching_all;
}


void draw_matches(Mat &img1, Mat &img2, MatrixXf &matching) {
	
	resize(img1, img1, img1.size() / 2);
	resize(img2, img2, img2.size() / 2);
	matching = matching.array() / 2;
	
	size_t num_matching = matching.cols();

	vector<KeyPoint> kp1(num_matching), kp2(num_matching);
	vector<DMatch> matches(num_matching);

	int w1 = img1.cols, h1 = img1.rows, w2 = img2.cols, h2 = img2.rows;

	for (size_t i = 0; i < num_matching; i++)
	{
		kp1[i].pt = Point2f(matching(0, i) * w1, matching(1, i) * h1);
		kp2[i].pt = Point2f(matching(2, i) * w2, matching(3, i) * h2);
		matches[i].queryIdx = i;
		matches[i].trainIdx = i;
	}

	Mat output;
	drawMatches(img1, kp1, img2, kp2, matches, output);

	imshow("results", output);
	waitKey();
}