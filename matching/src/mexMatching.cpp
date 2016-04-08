#include"mex.h"
#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
using namespace cv;
using namespace std;

#define lnkLIB(name) name
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#define cvLIB(name) lnkLIB("opencv_" name CV_VERSION_ID)
#pragma comment( lib, cvLIB("core"))
#pragma comment( lib, cvLIB("features2d"))
#pragma comment( lib, cvLIB("flann"))


void match(Mat &k1, Mat &k2, vector<DMatch>&matches, vector<DMatch>&matches_all, float threshold, int flag){
	vector<vector<DMatch>> tempmathces;
    if(flag == 0){
        BFMatcher Pro;
		Pro.knnMatch(k1, k2, tempmathces, 2);
    }
    else if(flag == 1){
        FlannBasedMatcher Pro;
		Pro.knnMatch(k1, k2, tempmathces, 2);
	};
	matches.reserve(k1.rows);
	matches_all.resize(k1.rows);
	for (int i = 0; i < tempmathces.size(); i++)
	{
		if (tempmathces[i][0].distance*threshold <= tempmathces[i][1].distance)
		{
			matches.push_back(tempmathces[i][0]);
		}
		matches_all[i] = tempmathces[i][0];
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// load d1
	int rows1 = mxGetM(prhs[0]);
	int cols1 = mxGetN(prhs[0]);
	float *data1 = (float *)mxGetPr(prhs[0]);
	Mat d1(cols1, rows1, CV_32FC1, data1);

	//load d2
	int rows2 = mxGetM(prhs[1]);
	int cols2 = mxGetN(prhs[1]);
	float* data2 = (float *)mxGetPr(prhs[1]);
	Mat d2(cols2, rows2, CV_32FC1, data2);

	//match
	vector<DMatch> matches, matches_all;
	float threshold = (float)mxGetScalar(prhs[2]);
	int flag = (int)mxGetScalar(prhs[3]);
	match(d1, d2, matches, matches_all, threshold, flag);

	//output
	plhs[0] = mxCreateNumericMatrix(2, matches.size(), mxINT32_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(2, matches_all.size(), mxINT32_CLASS, mxREAL);

	//output matches
	int *m1 = (int *)mxGetPr(plhs[0]);
	for (size_t i = 0; i < matches.size(); i++)
	{
		*m1++ = matches[i].queryIdx + 1;
		*m1++ = matches[i].trainIdx + 1;
	}

	//output matches_all
	int *m2 = (int *)mxGetPr(plhs[1]);
	for (size_t i = 0; i < matches_all.size(); i++)
	{
		*m2++ = matches_all[i].queryIdx + 1;
		*m2++ = matches_all[i].trainIdx + 1;
	}
}