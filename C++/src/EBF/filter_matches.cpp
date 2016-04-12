#pragma once
#include <Header.h>
#include <utility.h>
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
	int num_query = 100;
	Map<MatrixXf, 0, OuterStride<4> >  kp1(F1.kpts, 2, F1.num_keys);
	Map<MatrixXf, 0, OuterStride<4> >  kp2(F2.kpts, 2, F2.num_keys);

	MatrixXd  X_all(matches_all.size(), 4);
	for (int i = 0; i < matches_all.size(); i++)
	{
		int l = matches_all[i].queryIdx;
		int r = matches_all[i].trainIdx;
		X_all(i, 0) = kp1(0, l);
		X_all(i, 1) = kp1(1, l);
		X_all(i, 2) = kp2(0, r);
		X_all(i, 3) = kp2(1, r);
	}
	
	MatrixXd X_query(num_query, 4);
	for (int i = 0; i < num_query; i++)
	{
		X_query.row(i) = X_all.row(PriorityIdx[i].second);
	}

	//// normalize X_query
	MatrixXd  mu, sigma;
	featureNormalize(X_query, mu, sigma);
	X_query.block(0, 2, num_query, 2) = X_query.block(0, 2, num_query, 2) - X_query.block(0, 0, num_query, 2);

	vector<double> w;
	likehood_function lhf(X_query, 0.1);
	if (!lhf.optimize(w)) {
		cout << "lbfgs error !" << endl;
	}

}


