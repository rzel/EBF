#include <opencv2/opencv.hpp>
#include <iostream>
#include <libASIFT.h>
#include <libMATCH.h>
using namespace std;
using namespace cv;


int main(){
	Mat gray1 = imread("../../../data/image047.jpg", 0);
	Mat gray2 = imread("../../../data/image048.jpg", 0);
	if ( gray1.empty() || gray2.empty() )
	{
		cout << "read image error" << endl;
		exit(-1);
	}
	Mat gf1, gf2;
	gray1.convertTo(gf1, CV_32FC1);
	gray2.convertTo(gf2, CV_32FC1);
	
	int numTilts = 7, flag_resize = 0;
	FRAME frame1 = { (float*)gf1.data, gf1.cols, gf1.rows, numTilts, flag_resize, 0, nullptr, nullptr };	
	FRAME frame2 = { (float*)gf2.data, gf2.cols, gf2.rows, numTilts, flag_resize, 0, nullptr, nullptr };
	
	// extract feature
	extractASIFT( frame1 );
	extractASIFT( frame2 );
	
	// get descriptor 
	Mat d1(frame1.num_keys, 128, CV_32FC1, frame1.desp);
	Mat d2(frame2.num_keys, 128, CV_32FC1, frame2.desp);
	
	// match
	clock_t bg =clock();
	vector<DMatch> matches, matches_all; 
	flann_match1( d1, d2, matches, matches_all, 1.5);
	cout<<"match1 numbers : " << matches.size()<<"  "<<matches_all.size()<<endl;
	clock_t ed =clock();
	cout<<"match1 time : "<<ed-bg<<"ms"<<endl;
	
	matches.clear(); matches_all.clear();
	bg =clock();
	flann_match2( d1, d2, matches, matches_all, 2.0);
	cout<<"match2 numbers : " << matches.size()<<"  "<<matches_all.size()<<endl;
	ed =clock();
	cout<<"match2 time : "<< ed-bg<<"ms"<<endl;
	
	return 0;
}






