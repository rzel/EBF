#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<frame.h>
using namespace std;
using namespace cv;


// flann  quality = distance[1] / distance[0] 
void cv_match(FRAME &F1, FRAME &F2, vector<DMatch>&matches_all, vector<double> &priority);

