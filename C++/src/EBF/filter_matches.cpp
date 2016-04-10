#pragma once
#include <Header.h>
#include <likehood_function.h>

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
	int num_query = 50;
	Mat kp1(F1.num_keys, 4, CV_32FC1, F1.kpts);
	Mat kp2(F2.num_keys, 4, CV_32FC1, F2.kpts);
	Mat X_all(matches_all.size(), 4, CV_32FC1);
	for (int i = 0; i < matches_all.size(); i++)
	{
		int l = matches_all[i].queryIdx;
		int r = matches_all[i].trainIdx;
		
		X_all.at<float>(i, 0) = kp1.at<float>(l, 0) / F1.w;
		X_all.at<float>(i, 1) = kp1.at<float>(l, 1) / F1.h;
		X_all.at<float>(i, 2) = kp2.at<float>(r, 0) / F2.w;
		X_all.at<float>(i, 3) = kp2.at<float>(r, 1) / F2.h;
	}
 
	Mat X_query(num_query, 4, CV_32FC1);
	for (int i = 0; i < num_query; i++)
	{
		X_query.row(i) = X_all.row(PriorityIdx[i].second);
	}
	// normalize X_query
	vector<double>mu, sigma;
	featureNomalize(X_query, mu, sigma);
	X_query.colRange(2, 3) = X_query.colRange(2, 3) - X_query.colRange(0, 1);


	vector<double> w;
	likehood_function lhf(X_query, 0.1);
	if (!lhf.optimize(w)) {
		cout << "lbfgs error !" << endl;
	}

}


