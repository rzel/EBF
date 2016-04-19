#include<Header.h>
#include<libASIFT.h>
#include<libMATCH.h>


int main() {
	// read image
	//Mat gray1 = imread("E:/Git/EBF/data/image047.jpg", 0);
	//Mat gray2 = imread("E:/Git/EBF/data/image048.jpg", 0);

	Mat gray1 = imread("E:/Git/EBF/data/DSC00184.jpg", 0);
	Mat gray2 = imread("E:/Git/EBF/data/DSC00186.jpg", 0);

	imresize(gray1, 480);
	imresize(gray2, 480);

	//Mat gray1 = imread("E:/kitti/dataset/sequences/00/image_0/000020.png", 0);
	//Mat gray2 = imread("E:/kitti/dataset/sequences/00/image_0/000030.png", 0);
	//resize(gray1, gray1, gray1.size() / 2);
	//resize(gray2, gray2, gray2.size() / 2);

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
	vector<double> priority;
	cv_match(F1, F2, matches_all, priority);


	// get inlier
	clock_t bg = clock();
	MatrixXf matching = filter_matches(F1, F2, matches_all, priority);
	clock_t ed = clock();
	cout << "BF time : " << ed - bg << "ms" << endl;

	draw_matches(gray1, gray2, matching);


	return 0;
}

