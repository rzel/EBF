#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include "frame.h"
using namespace std;

#include <omp.h>
#include "demo_lib_sift.h"
#include "library.h"
#include "frot.h"
#include "fproj.h"
#include "compute_asift_keypoints.h"





void extractASIFT(FRAME &frame);