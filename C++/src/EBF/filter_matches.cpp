#pragma once
#include <Header.h>
#include <likehood_function.h>
#include <bilateral_function.h>
#include <getInlier.h>

#define HUBER_THRESH		0.1
#define LIKEHOOD_THRESH		0.1
#define BILATERAL_THRESH	0.01
#define SIFT_MATCH_RATIO	0.66
#define MAX_QUERY_NUMBER	1000

MatrixXf filter_matches(FRAME &F1, FRAME &F2, vector<DMatch> &matches_all, vector<double> &priority)
{
	// pair<priority, index> : priority = 1.0 / quality 
	vector<int> good_matches;	good_matches.reserve(matches_all.size());
	for (int i = 0; i < matches_all.size(); i++)
	{
		if (priority[i]  > SIFT_MATCH_RATIO) continue;	
		good_matches.push_back(i);
	}
	if (good_matches.size() > MAX_QUERY_NUMBER)
	{
		good_matches.resize(MAX_QUERY_NUMBER);
	}
	int num_query = good_matches.size();


	Map<MatrixXf, 0, OuterStride<4> >  kp1(F1.kpts, 2, F1.num_keys);
	Map<MatrixXf, 0, OuterStride<4> >  kp2(F2.kpts, 2, F2.num_keys);

	MatrixXf  matching_all(4, matches_all.size());
	
	for (int i = 0; i < matches_all.size(); i++)
	{
		int l = matches_all[i].queryIdx;
		int r = matches_all[i].trainIdx;
		float pair[4] = { kp1(0, l), kp1(1, l), kp2(0, r), kp2(1, r) };
		memcpy(matching_all.data() + 4 * i, pair, 4 * sizeof(float));
	}
	// normalize the points to 0 - 1
	MatrixXf ImageSize(4, 1);	ImageSize << (float)F1.w, (float)F1.h, (float)F2.w, (float)F2.h;
	matching_all = matching_all.array() / (ImageSize * MatrixXf::Ones(1, matches_all.size())).array();

	MatrixXf matching_query(4, num_query);
	for (int i = 0; i < num_query; i++ )
	{
		memcpy(matching_query.data() + 4 * i, matching_all.data() + 4 * good_matches[i], sizeof(float) * 4);
	}

	// normalize X_query
	MatrixXf  X_query, mu, sigma;
	X_query = Tools::featureNormalize(matching_query, mu, sigma, false);
	X_query.bottomRows(2) -= X_query.topRows(2);

	// normalize X_all	
	MatrixXf X_all = Tools::featureNormalize(matching_all, mu, sigma, false);
	X_all.bottomRows(2) -= X_all.topRows(2);


	// learning likehood weight
clock_t bg = clock();
	MatrixXf w;
	likehood_function lhf(X_query, HUBER_THRESH);
clock_t ed = clock();
	cout << "likehood SVD : " << ed - bg << "ms         "<< endl;

bg = clock();
	lhf.optimize(w);
ed = clock();
	cout << "likehood learning time : " << ed - bg << "ms         " << endl;
	// get inlier by likehood  must first to likehood_all, because likehood will change X_query.
	getInlier::likehood_all(w, lhf.U, X_all, X_query, matching_all, LIKEHOOD_THRESH);
	getInlier::likehood(w, lhf.G,X_query, matching_query, LIKEHOOD_THRESH);
	cout <<endl<< matching_all.cols() << endl;
	if (X_query.cols() < 1) { return matching_all; }


	// learning bilateral function weight
bg = clock();
	bilateral_function blf(X_query, matching_query, BILATERAL_THRESH);
ed = clock();
	cout << "likehood SVD : " << ed - bg << "ms         " << endl;
	MatrixXf w1, w2;
bg = clock();
	blf.optimize(w1, w2);
ed = clock();
	cout << "bf lerning  time : " << ed - bg << "ms         " << endl;
	// get inlier by bilateral function
bg = clock();
	getInlier::bilateral_function(w1, w2, blf.U,X_all, X_query, matching_all, BILATERAL_THRESH);
ed = clock();
	cout << "bf filter  time : " << ed - bg << "ms         " << matching_all.cols() << endl;

	return matching_all;
}


void draw_matches(Mat &img1, Mat &img2, MatrixXf &matching) {
	
	//resize(img1, img1, img1.size() / 2);
	//resize(img2, img2, img2.size() / 2);
	//matching = matching.array() / 2;
	
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