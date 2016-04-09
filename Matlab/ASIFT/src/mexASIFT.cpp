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

#include <omp.h>

#include "demo_lib_sift.h"
#include "library.h"
#include "frot.h"
#include "fproj.h"
#include "compute_asift_keypoints.h"


# define IM_X 800
# define IM_Y 600

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

	//////////////////////////////////////////////// Input
	// Read image
	size_t w, h;
	w = mxGetM(prhs[0]);
	h = mxGetN(prhs[0]);
	float *iarr = (float*)mxGetPr(prhs[0]);
	std::vector<float> ipixels(iarr, iarr + w * h);


	///// Resize the images to area wS*hW in remaining the apsect-ratio	
	///// Resize if the resize flag is not set or if the flag is set unequal to 0
	float wS = IM_X;
	float hS = IM_Y;

	float zoom = 0;
	vector<float> ipixels_zoom;

	int numTilts = (int)mxGetScalar(prhs[1]);
	int flag_resize = (int)mxGetScalar(prhs[2]);
					
	if (flag_resize != 0)
	{
		float InitSigma_aa = 1.6f;

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

		// Resize image  
		float area = w * h;
		zoom = sqrt(area / areaS);

		wS = (int)(w / zoom);
		hS = (int)(h / zoom);

		int fproj_sx = wS;
		int fproj_sy = hS;

		float fproj_x1 = 0;
		float fproj_y1 = 0;
		float fproj_x2 = wS;
		float fproj_y2 = 0;
		float fproj_x3 = 0;
		float fproj_y3 = hS;

		/* Anti-aliasing filtering along vertical direction */
		if (zoom > 1)
		{
			float sigma_aa = InitSigma_aa * zoom / 2;
			GaussianBlur1D(ipixels, w, h, sigma_aa, 1);
			GaussianBlur1D(ipixels, w, h, sigma_aa, 0);
		}

		// simulate a tilt: subsample the image along the vertical axis by a factor of t.
		ipixels_zoom.resize(wS*hS);
		fproj(ipixels, ipixels_zoom, w, h, &fproj_sx, &fproj_sy, &fproj_bg, &fproj_o, &fproj_p,
			&fproj_i, fproj_x1, fproj_y1, fproj_x2, fproj_y2, fproj_x3, fproj_y3, fproj_x4, fproj_y4);
	}
	else
	{
		ipixels_zoom.resize(w*h);
		ipixels_zoom = ipixels;
		wS = w;
		hS = h;
		zoom = 1;
	}


	///// Compute ASIFT keypoints
	// number N of tilts to simulate t = 1, \sqrt{2}, (\sqrt{2})^2, ..., {\sqrt{2}}^(N-1)
	int num_of_tilts = numTilts;

	int verb = 0;
	// Define the SIFT parameters
	siftPar siftparameters;
	default_sift_parameters(siftparameters);

	vector< vector< keypointslist > > keys;

	vector<vector< pair<float, float> >> ttrr;

	int num_keys;
	num_keys = compute_asift_keypoints(ipixels_zoom, wS, hS, num_of_tilts, verb, keys, siftparameters, ttrr);

	//output f d
	plhs[0] = mxCreateDoubleMatrix(6, num_keys, mxREAL);
	plhs[1] = mxCreateNumericMatrix(128, num_keys, mxSINGLE_CLASS, mxREAL);
	double *frame = (double *)mxGetPr(plhs[0]);
	float *descript = (float *)mxGetPr(plhs[1]);

	int num = 0;
	for (int tt = 0; tt < (int)keys.size(); tt++)
	{
		for (int rr = 0; rr < (int)keys[tt].size(); rr++)
		{
			keypointslist::iterator ptr = keys[tt][rr].begin();
			for (int i = 0; i < (int)keys[tt][rr].size(); i++, ptr++)
			{
				vector<double> f = { zoom*ptr->x, zoom*ptr->y, zoom*ptr->scale, ptr->angle, 
					double(ttrr[tt][rr].first), double(ttrr[tt][rr].second) };
				memcpy((double *)(frame + 6 * num ), f.data(), 6 * sizeof(double));
				memcpy((float *)(descript + 128 * num), ptr->vec, 128 * sizeof(float));
				num++;
			}
		}
	}
	assert(num == num_keys);
}
