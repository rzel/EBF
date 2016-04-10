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


void match(Mat &k1, Mat &k2, vector<DMatch>&matches_all, vector<double>&quality){
	vector<vector<DMatch>> tempMatches;
	FlannBasedMatcher Pro;
	Pro.knnMatch(k1, k2, tempMatches, 2);
	
	quality.resize(k1.rows);
	matches_all.resize(k1.rows);
	for (int i = 0; i < matches_all.size(); i++)
	{
		quality[i] = tempMatches[i][1].distance / tempMatches[i][0].distance;
		matches_all[i] = tempMatches[i][0];
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
	vector<DMatch>  matches_all;	vector<double> quality;
	match(d1, d2, matches_all, quality);

	//output
	plhs[0] = mxCreateNumericMatrix(2, matches_all.size(), mxINT32_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(1, matches_all.size(), mxDOUBLE_CLASS, mxREAL);
	
	//output matches_all
	int *m1 = (int *)mxGetPr(plhs[0]);
	for (size_t i = 0; i < matches_all.size(); i++)
	{
		*m1++ = matches_all[i].queryIdx + 1;
		*m1++ = matches_all[i].trainIdx + 1;
	}

	//output good
	memcpy((double *)mxGetPr(plhs[1]), quality.data(), sizeof(double) *matches_all.size());
}