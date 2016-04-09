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

struct FRAME{
	float *img; size_t w; size_t h; int numTilts; int flag_resize;
	int num_keys;  float *kpts;  float *desp;
};



void extractASIFT(FRAME &frame);