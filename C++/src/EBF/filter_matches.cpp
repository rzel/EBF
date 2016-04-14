#pragma once
#include <Header.h>
#include <likehood_function.h>
#include <bilateral_function.h>
#include <getInlier.h>

#define LIKEHOOD_THRESH		0.2
#define BILATERAL_THRESH	0.01

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


	// matching_all  matching  :  two points per column
	int num_query = max((int)(matches_all.size() / 100), 30);
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
	for (int i = 0; i < num_query; i++)
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
	if (!lhf.optimize(w)) {
		cout << "likehood function error !" << endl;
	}
	// get inlier by likehood  must first to likehood_all, because likehood will change X_query.
	getInlier::likehood_all(w, X_all, X_query, matching_all, LIKEHOOD_THRESH);
	getInlier::likehood(w, lhf.G,X_query, matching_query, LIKEHOOD_THRESH);
	clock_t ed = clock();
	cout << "likehood time : " << ed - bg << "ms         " << matching_all.cols() << endl;


	bg = clock();
	// learning bilateral function weight
	bilateral_function blf(X_query, matching_query, BILATERAL_THRESH);
	MatrixXd w1, w2;
	if (!blf.optimize(w1, w2)) {
		cout << "bilateral function error !" << endl;
	}
	// get inlier by bilateral function
	getInlier::bilateral_function(w1, w2, X_all, X_query, matching_all, BILATERAL_THRESH);
	ed = clock();
	cout << "bilteral function  time : " << ed - bg << "ms         " << matching_all.cols() << endl;
}


