#include <libMatch.h>

// base on sift match 
void flann_match1(Mat &d1, Mat &d2, vector<DMatch>&matches, vector<DMatch>&matches_all, float threshold) {
	vector<vector<DMatch>> tempmathces;
	FlannBasedMatcher Matcher;
	Matcher.knnMatch(d1, d2, tempmathces, 2);

	matches.reserve(d1.rows);
	matches_all.resize(d1.rows);
	for (int i = 0; i < tempmathces.size(); i++)
	{
		if (tempmathces[i][0].distance * threshold <= tempmathces[i][1].distance)
		{
			matches.push_back(tempmathces[i][0]);
		}
		matches_all[i] = tempmathces[i][0];
	}
}



// efficient match   matches.distance < threshold * MinDistance 
void flann_match2(Mat &d1, Mat &d2, vector<DMatch>&matches, vector<DMatch>&matches_all, float threshold) {
	FlannBasedMatcher Matcher;
	Matcher.match(d1, d2, matches_all);
	matches.reserve(d1.rows);

	double min_dist = 100;
	for (int i = 0; i < matches_all.size(); i++)
	{
		double dist = matches_all[i].distance;
		if (dist < min_dist) min_dist = dist;
	}

	for (int i = 0; i < matches_all.size(); i++)
	{
		if (matches_all[i].distance <= max(threshold * min_dist, 0.02))
		{
			matches.push_back(matches_all[i]);
		}
	}
}
