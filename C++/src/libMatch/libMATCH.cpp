#include <libMatch.h>


void cv_match(FRAME &F1, FRAME &F2, vector<DMatch>&matches_all, vector<double> &quality) {
	Mat d1(F1.num_keys, 128, CV_32FC1, F1.desp);
	Mat d2(F2.num_keys, 128, CV_32FC1, F2.desp);
	
	vector<vector<DMatch>> tempmathces;
	FlannBasedMatcher Matcher;
	Matcher.knnMatch(d1, d2, tempmathces, 2);

	quality.resize(d1.rows);
	matches_all.resize(d1.rows);
	for (int i = 0; i < tempmathces.size(); i++)
	{
		quality[i] = tempmathces[i][1].distance / tempmathces[i][0].distance;
		matches_all[i] = tempmathces[i][0];
	}
}



