#pragma once

#include <opencv2/opencv.hpp>
#include<algorithm>
#include<ctime>
#include "frame.h"
#include "Tools.h"


using namespace cv;
using namespace std;


MatrixXf filter_matches(FRAME &F1, FRAME &F2, vector<DMatch> &matches_all, vector<double> &quality);

void draw_matches(Mat &img1, Mat &img2, MatrixXf &matching);
