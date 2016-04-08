// Copyright (c) 2008-2011, Guoshen Yu <yu@cmap.polytechnique.fr>
// Copyright (c) 2008-2011, Jean-Michel Morel <morel@cmla.ens-cachan.fr>
//
// WARNING: 
// This file implements an algorithm possibly linked to the patent
//
// Jean-Michel Morel and Guoshen Yu, Method and device for the invariant 
// affine recognition recognition of shapes (WO/2009/150361), patent pending. 
//
// This file is made available for the exclusive aim of serving as
// scientific tool to verify of the soundness and
// completeness of the algorithm description. Compilation,
// execution and redistribution of this file may violate exclusive
// patents rights in certain countries.
// The situation being different for every country and changing
// over time, it is your responsibility to determine which patent
// rights restrictions apply to you before you compile, use,
// modify, or redistribute this file. A patent lawyer is qualified
// to make this determination.
// If and only if they don't conflict with any patent terms, you
// can benefit from the following license terms attached to this
// file.
//
// This program is provided for scientific and educational only:
// you can use and/or modify it for these purposes, but you are
// not allowed to redistribute this work or derivative works in
// source or executable form. A license must be obtained from the
// patent right holders for any other use.
//
// 
//*----------------------------- demo_ASIFT  --------------------------------*/
// Detect corresponding points in two images with the ASIFT method. 

// Please report bugs and/or send comments to Guoshen Yu yu@cmap.polytechnique.fr
// 
// Reference: J.M. Morel and G.Yu, ASIFT: A New Framework for Fully Affine Invariant Image 
//            Comparison, SIAM Journal on Imaging Sciences, vol. 2, issue 2, pp. 438-469, 2009. 
// Reference: ASIFT online demo (You can try ASIFT with your own images online.) 
//			  http://www.ipol.im/pub/algo/my_affine_sift/
/*---------------------------------------------------------------------------*/
#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
using namespace std;

#ifdef _OPENMP
#include <omp.h>
#endif

#include "demo_lib_sift.h"
#include "io_png/io_png.h"

#include "library.h"
#include "frot.h"
#include "fproj.h"
#include "compute_asift_keypoints.h"
#include "compute_asift_matches.h"

# define IM_X 800
# define IM_Y 600

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{


	//////////////////////////////////////////////// Input
	// Read image1
	size_t w1, h1;
	w1 = mxGetM(prhs[0]);
	h1 = mxGetN(prhs[0]);
	float *iarr1 = (float*)mxGetPr(prhs[0]);
	std::vector<float> ipixels1(iarr1, iarr1 + w1 * h1);

	// Read image2
	size_t w2, h2;
	w2 = mxGetM(prhs[1]);
	h2 = mxGetN(prhs[1]);
	float * iarr2 = (float*)mxGetPr(prhs[1]);
	std::vector<float> ipixels2(iarr2, iarr2 + w2 * h2);

	///// Resize the images to area wS*hW in remaining the apsect-ratio	
	///// Resize if the resize flag is not set or if the flag is set unequal to 0
	float wS = IM_X;
	float hS = IM_Y;

	float zoom1 = 0, zoom2 = 0;
	int wS1 = 0, hS1 = 0, wS2 = 0, hS2 = 0;
	vector<float> ipixels1_zoom, ipixels2_zoom;

	int numTilts = (int)mxGetScalar(prhs[2]);
	int flag_resize = 1;
	if (nrhs == 4 )
	{
		flag_resize = (int)mxGetScalar(prhs[3]);
		
	}
	
	if ((nrhs == 4) || (flag_resize != 0))
	{
//		std::cout << "WARNING: The input images are resized to " << wS << "x" << hS << " for ASIFT. " << endl
//			<< "         But the results will be normalized to the original image size." << endl << endl;

		float InitSigma_aa = 1.6;

		float fproj_p, fproj_bg;
		char fproj_i;
		float *fproj_x4, *fproj_y4;
		int fproj_o;

		fproj_o = 3;
		fproj_p = 0;
		fproj_i = 0;
		fproj_bg = 0;
		fproj_x4 = 0;
		fproj_y4 = 0;

		float areaS = wS * hS;

		// Resize image 1 
		float area1 = w1 * h1;
		zoom1 = sqrt(area1 / areaS);

		wS1 = (int)(w1 / zoom1);
		hS1 = (int)(h1 / zoom1);

		int fproj_sx = wS1;
		int fproj_sy = hS1;

		float fproj_x1 = 0;
		float fproj_y1 = 0;
		float fproj_x2 = wS1;
		float fproj_y2 = 0;
		float fproj_x3 = 0;
		float fproj_y3 = hS1;

		/* Anti-aliasing filtering along vertical direction */
		if (zoom1 > 1)
		{
			float sigma_aa = InitSigma_aa * zoom1 / 2;
			GaussianBlur1D(ipixels1, w1, h1, sigma_aa, 1);
			GaussianBlur1D(ipixels1, w1, h1, sigma_aa, 0);
		}

		// simulate a tilt: subsample the image along the vertical axis by a factor of t.
		ipixels1_zoom.resize(wS1*hS1);
		fproj(ipixels1, ipixels1_zoom, w1, h1, &fproj_sx, &fproj_sy, &fproj_bg, &fproj_o, &fproj_p,
			&fproj_i, fproj_x1, fproj_y1, fproj_x2, fproj_y2, fproj_x3, fproj_y3, fproj_x4, fproj_y4);


		// Resize image 2 
		float area2 = w2 * h2;
		zoom2 = sqrt(area2 / areaS);

		wS2 = (int)(w2 / zoom2);
		hS2 = (int)(h2 / zoom2);

		fproj_sx = wS2;
		fproj_sy = hS2;

		fproj_x2 = wS2;
		fproj_y3 = hS2;

		/* Anti-aliasing filtering along vertical direction */
		if (zoom1 > 1)
		{
			float sigma_aa = InitSigma_aa * zoom2 / 2;
			GaussianBlur1D(ipixels2, w2, h2, sigma_aa, 1);
			GaussianBlur1D(ipixels2, w2, h2, sigma_aa, 0);
		}

		// simulate a tilt: subsample the image along the vertical axis by a factor of t.
		ipixels2_zoom.resize(wS2*hS2);
		fproj(ipixels2, ipixels2_zoom, w2, h2, &fproj_sx, &fproj_sy, &fproj_bg, &fproj_o, &fproj_p,
			&fproj_i, fproj_x1, fproj_y1, fproj_x2, fproj_y2, fproj_x3, fproj_y3, fproj_x4, fproj_y4);
	}
	else
	{
		ipixels1_zoom.resize(w1*h1);
		ipixels1_zoom = ipixels1;
		wS1 = w1;
		hS1 = h1;
		zoom1 = 1;

		ipixels2_zoom.resize(w2*h2);
		ipixels2_zoom = ipixels2;
		wS2 = w2;
		hS2 = h2;
		zoom2 = 1;
	}


	///// Compute ASIFT keypoints
	// number N of tilts to simulate t = 1, \sqrt{2}, (\sqrt{2})^2, ..., {\sqrt{2}}^(N-1)
	int num_of_tilts1 = numTilts;
	int num_of_tilts2 = numTilts;

	int verb = 0;
	// Define the SIFT parameters
	siftPar siftparameters;
	default_sift_parameters(siftparameters);

	vector< vector< keypointslist > > keys1;
	vector< vector< keypointslist > > keys2;

	vector<vector< pair<float, float> >> ttrr1;
	vector<vector< pair<float, float> >> ttrr2;

	int num_keys1 = 0, num_keys2 = 0;

//	std::cout << "Computing keypoints on the two images..." << endl;
	time_t tstart, tend;
	tstart = time(0);

	num_keys1 = compute_asift_keypoints(ipixels1_zoom, wS1, hS1, num_of_tilts1, verb, keys1, siftparameters, ttrr1);
	num_keys2 = compute_asift_keypoints(ipixels2_zoom, wS2, hS2, num_of_tilts2, verb, keys2, siftparameters, ttrr2);

	tend = time(0);
//	std::cout << "Keypoints computation accomplished in " << difftime(tend, tstart) << " seconds." << endl;


	//output f1 f2 d1 d2
	plhs[0] = mxCreateDoubleMatrix(6, num_keys1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(6, num_keys2, mxREAL);
	plhs[2] = mxCreateNumericMatrix(128, num_keys1, mxSINGLE_CLASS, mxREAL);
	plhs[3] = mxCreateNumericMatrix(128, num_keys2, mxSINGLE_CLASS, mxREAL);

	//output f1 d1
	double *frame1 = (double *)mxGetPr(plhs[0]);
	double *frame2 = (double *)mxGetPr(plhs[1]);
	float *descript1 = (float *)mxGetPr(plhs[2]);
	float *descript2 = (float *)mxGetPr(plhs[3]);
	int num = 0;
	for (int tt = 0; tt < (int)keys1.size(); tt++)
	{
		for (int rr = 0; rr < (int)keys1[tt].size(); rr++)
		{
			keypointslist::iterator ptr = keys1[tt][rr].begin();
			for (int i = 0; i < (int)keys1[tt][rr].size(); i++, ptr++)
			{
				vector<double> f1 = { zoom1*ptr->x, zoom1*ptr->y, zoom1*ptr->scale, ptr->angle, 
					double(ttrr1[tt][rr].first), double(ttrr1[tt][rr].second) };
				memcpy((double *)(frame1 + 6 * num ), f1.data(), 6 * sizeof(double));
				memcpy((float *)(descript1 + 128 * num), ptr->vec, 128 * sizeof(float));
				num++;
			}
		}
	}
	assert(num == num_keys1);

	//output f2 d2
	num = 0;
	for (int tt = 0; tt < (int)keys2.size(); tt++)
	{
		for (int rr = 0; rr < (int)keys2[tt].size(); rr++)
		{
			keypointslist::iterator ptr = keys2[tt][rr].begin();
			for (int i = 0; i < (int)keys2[tt][rr].size(); i++, ptr++)
			{
				vector<double> f2 = { zoom2*ptr->x, zoom2*ptr->y, zoom2*ptr->scale, ptr->angle, 
					double(ttrr2[tt][rr].first), double(ttrr2[tt][rr].second) };
				memcpy((double *)(frame2 + 6 * num), f2.data(), 6 * sizeof(double));
				memcpy((float *)(descript2 + 128 * num), ptr->vec, 128 * sizeof(float));
				num++;
			}
		}
	}
	assert(num == num_keys2);
}
