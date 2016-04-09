
#include "libASIFT.h"

#define IM_X 800
#define IM_Y 600

void extractASIFT(FRAME &frame )
{
	// Read frame
	size_t w, h;
	w = frame.w;
	h = frame.h;
	int numTilts = frame.numTilts;
	int flag_resize = frame.flag_resize;
	float *iarr = frame.img;
	std::vector<float> ipixels( w * h );
	memcpy( ipixels.data(), iarr, sizeof(float) * w * h );


	// Resize the images to area wS*hW in remaining the apsect-ratio	
	///// Resize if the resize flag is not set or if the flag is set unequal to 0
	float wS = IM_X;
	float hS = IM_Y;

	float zoom = 0;
	vector<float> ipixels_zoom;

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

	int num_keys;
	num_keys = compute_asift_keypoints(ipixels_zoom, wS, hS, num_of_tilts, verb, keys, siftparameters);

	//output
	frame.num_keys = num_keys;
	frame.kpts = new float[ num_keys * 4 ];
	frame.desp = new float[ num_keys * 128 ];
	float *kpts = frame.kpts;
	float *desp = frame.desp;
	
	int num = 0;
	for (int tt = 0; tt < (int)keys.size(); tt++)
	{
		for (int rr = 0; rr < (int)keys[tt].size(); rr++)
		{
			keypointslist::iterator ptr = keys[tt][rr].begin();
			for (int i = 0; i < (int)keys[tt][rr].size(); i++, ptr++)
			{
				float f[4] = { zoom*ptr->x, zoom*ptr->y, zoom*ptr->scale, ptr->angle};
				memcpy((float *)(kpts + 4 * num), &f[0], 4 * sizeof(float));		
				memcpy((float *)(desp + 128 * num), ptr->vec, 128 * sizeof(float));
				num++;
			}
		}
	}
	assert(num == num_keys);
}