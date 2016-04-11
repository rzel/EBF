#pragma once

#include <opencv2/opencv.hpp>
#include<libASIFT.h>
#include<libMATCH.h>
#include<lbfgs.h>
#include<algorithm>

using namespace cv;
using namespace std;


void filter_matches(FRAME &F1, FRAME &F2, vector<DMatch> &matches_all, vector<double> &quality);


