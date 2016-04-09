#include<vector>
#include<opencv2/opencv.hpp>
#include<opencv2/features2d.hpp>
using namespace std;
using namespace cv;

// base on sift match 
void flann_match1(Mat &d1, Mat &d2, vector<DMatch>&matches, vector<DMatch>&matches_all, float threshold);


// efficient match   matches.distance < threshold * MinDistance 
void flann_match2(Mat &d1, Mat &d2, vector<DMatch>&matches, vector<DMatch>&matches_all, float threshold);

