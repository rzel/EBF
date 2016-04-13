#pragma once
#include <Header.h>
#include <likehood_function.h>
#include <bilateral_function.h>
#include <getInlier.h>

#define INLIER_THRESH  0.2

bool CampareRule(const pair<double, int>&p1, const pair<double, int>&p2) {
	return p1.first < p2.first;
}

void filter_matches(FRAME &F1, FRAME &F2, vector<DMatch> &matches_all, vector<double> &quality)
{
	// pair<priority, index> : priority = 1.0 / quality 
	vector<pair<double, int>> PriorityIdx(matches_all.size());
	for (int i = 0; i < matches_all.size(); i++)
	{
		PriorityIdx[i] = (make_pair(1.0/quality[i], i));
	}
	std::sort(PriorityIdx.begin(), PriorityIdx.end(), CampareRule);


	//construct X_all  X_query
	int num_query = max((int)(matches_all.size() / 100), 30);
	Map<MatrixXf, 0, OuterStride<4> >  kp1(F1.kpts, 2, F1.num_keys);
	Map<MatrixXf, 0, OuterStride<4> >  kp2(F2.kpts, 2, F2.num_keys);

	MatrixXd  matching_all(4, matches_all.size());
	for (int i = 0; i < matches_all.size(); i++)
	{
		int l = matches_all[i].queryIdx;
		int r = matches_all[i].trainIdx;
		matching_all.col(i) << kp1(0, l), kp1(1, l), kp2(0, r), kp2(1, r);
	}
	// normalize the points to 0 - 1
	MatrixXd ImageSize(4, 1);	ImageSize << (double)F1.w, (double)F1.h, (double)F2.w, (double)F2.h;
	matching_all = matching_all.array() / (ImageSize * MatrixXd::Ones(1, matches_all.size())).array();

	MatrixXd matching_query(4, num_query);
	for (int i = 0; i < num_query; i++)
	{
		matching_query.col(i) = matching_all.col(PriorityIdx[i].second);
	}

	// normalize X_query
	MatrixXd  X_query, mu, sigma;
	X_query = Tools::featureNormalize(matching_query, mu, sigma, false);
	X_query.block(2, 0, 2, num_query) -= X_query.block(0, 0, 2, num_query);

	//// normalize X_all	by row		matching_all by col
	MatrixXd X_all = Tools::featureNormalize(matching_all, mu, sigma, false);
	X_all.block(2, 0, 2, matches_all.size()) -= X_all.block(0, 0, 2, matches_all.size());


	// learning likehood weight
	MatrixXd w;
	likehood_function lhf(X_query, INLIER_THRESH);
	if (!lhf.optimize(w)) {
		cout << "lbfgs error !" << endl;
	}


	// get inlier by likehood  must first to likehood_all, because likehood will change X_query.
	getInlier::likehood_all(w, X_all, X_query, matching_all, INLIER_THRESH);
	getInlier::likehood(w, lhf.G,X_query, matching_query, INLIER_THRESH);

	cout << "start bilateral function !" << endl << endl;

	// learning bilateral function weight
	double bilater_threshold = 0.01;
	bilateral_function blf(X_query, matching_query, bilater_threshold);
	MatrixXd w1, w2;
	blf.optimize(w1, w2);

	// get inlier by bilateral function
	getInlier::bilateral_function(w1, w2, X_all, X_query, matching_all, bilater_threshold);

	cout << matching_all.cols() << endl;
}


