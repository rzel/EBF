#include<Header.h>
#include<libASIFT.h>
#include<libMATCH.h>


int main() {
	// read image
	Mat gray1 = imread("E:/Git/EBF/data/image047.jpg", 0);
	Mat gray2 = imread("E:/Git/EBF/data/image048.jpg", 0);

	// convert to float type
	Mat gf1, gf2;
	gray1.convertTo(gf1, CV_32FC1);
	gray2.convertTo(gf2, CV_32FC1);

	// construct FRAME
	int numTilts = 1, flag_resize = 0;
	FRAME F1 = { (float*)gf1.data, gf1.cols, gf1.rows, numTilts , flag_resize, 0, nullptr, nullptr };
	FRAME F2 = { (float*)gf2.data, gf2.cols, gf2.rows, numTilts , flag_resize, 0, nullptr, nullptr };
		
	// extract ASIFT feature
	extractASIFT(F1);
	extractASIFT(F2);

	// cv_match
	vector<DMatch> matches_all;
	vector<double> quality;
	cv_match(F1, F2, matches_all, quality);


	// get inlier
	clock_t bg = clock();
	filter_matches(F1, F2, matches_all, quality);
	clock_t ed = clock();
	cout << "BF time : " << ed - bg << "ms" << endl;

	


	return 0;
}

