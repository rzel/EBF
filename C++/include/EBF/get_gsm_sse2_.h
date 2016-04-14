#pragma once
#include <emmintrin.h>

#define TYPE_FLOAT 


#ifdef TYPE_FLOAT
   typedef  float T;
#  define SFX f
#  define VSIZE  4
#  define VSFX   s
#  define VTYPE  __m128
#endif

#ifdef TYPE_DOUBLE
   typedef double T;
#  define SFX d
#  define VSIZE  2
#  define VSFX   d
#  define VTYPE  __m128d
#endif

#define VALIGNED(x) (! (((long long unsigned)(x)) & 0xF))

#define VL_CAT(x,y) x ## y
#define VL_XCAT(x,y) VL_CAT(x,y)

#define VMAX  VL_XCAT(_mm_max_p,     VSFX)
#define VMUL  VL_XCAT(_mm_mul_p,     VSFX)
#define VDIV  VL_XCAT(_mm_div_p,     VSFX)
#define VADD  VL_XCAT(_mm_add_p,     VSFX)
#define VSUB  VL_XCAT(_mm_sub_p,     VSFX)
#define VSTZ  VL_XCAT(_mm_setzero_p, VSFX)
#define VLD1  VL_XCAT(_mm_load1_p,   VSFX)
#define VLDU  VL_XCAT(_mm_loadu_p,   VSFX)
#define VST1  VL_XCAT(_mm_store_s,   VSFX)
#define VSET1 VL_XCAT(_mm_set_s,     VSFX)
#define VSHU  VL_XCAT(_mm_shuffle_p, VSFX)
#define VNEQ  VL_XCAT(_mm_cmpneq_p,  VSFX)
#define VAND  VL_XCAT(_mm_and_p,     VSFX)
#define VANDN VL_XCAT(_mm_andnot_p,  VSFX)
#define VST2  VL_XCAT(_mm_store_p,   VSFX)
#define VST2U VL_XCAT(_mm_storeu_p,  VSFX)





inline T
VL_XCAT(_vl_vhsum_sse2_, SFX)(VTYPE x)
{
	T acc;
#if (VSIZE == 4)
	{
		VTYPE sum;
		VTYPE shuffle;
		/* shuffle = [1 0 3 2] */
		/* sum     = [3+1 2+0 1+3 0+2] */
		/* shuffle = [2+0 3+1 0+2 1+3] */
		/* vacc    = [3+1+2+0 3+1+2+0 1+3+0+2 0+2+1+3] */
		shuffle = VSHU(x, x, _MM_SHUFFLE(1, 0, 3, 2));
		sum = VADD(x, shuffle);
		shuffle = VSHU(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
		x = VADD(sum, shuffle);
	}
#else
	{
		VTYPE shuffle;
		/* acc     = [1   0  ] */
		/* shuffle = [0   1  ] */
		/* sum     = [1+0 0+1] */
		shuffle = VSHU(x, x, _MM_SHUFFLE2(0, 1));
		x = VADD(x, shuffle);
	}
#endif
	VST1(&acc, x);
	return acc;
}



inline T
get_distance_l2_sse2_d
(size_t dimension, T const * X, T const * Y)
{
	T const * X_end = X + dimension;
	T const * X_vec_end = X_end - VSIZE + 1;
	T acc;
	VTYPE vacc = VSTZ();
	bool dataAligned = VALIGNED(X) & VALIGNED(Y);

	if (dataAligned) {
		while (X < X_vec_end) {
			VTYPE a = *(VTYPE*)X;
			VTYPE b = *(VTYPE*)Y;
			VTYPE delta = VSUB(a, b);
			VTYPE delta2 = VMUL(delta, delta);
			vacc = VADD(vacc, delta2);
			X += VSIZE;
			Y += VSIZE;
		}
	}
	else {
		while (X < X_vec_end) {
			VTYPE a = VLDU(X);
			VTYPE b = VLDU(Y);
			VTYPE delta = VSUB(a, b);
			VTYPE delta2 = VMUL(delta, delta);
			vacc = VADD(vacc, delta2);
			X += VSIZE;
			Y += VSIZE;
		}
	}

	acc = VL_XCAT(_vl_vhsum_sse2_, SFX)(vacc);

	while (X < X_end) {
		T a = *X++;
		T b = *Y++;
		T delta = a - b;
		acc += delta * delta;
	}

	return acc;
}



inline void
get_gsm_sse2_d
(T * result, size_t dimension,
	T const * X, size_t numDataX,
	T const * Y, size_t numDataY)
{
	size_t xi, yi;

	if (dimension == 0) return;
	if (numDataX == 0) return;
	assert(X);

	if (Y) {
		if (numDataY == 0) return;
		for (yi = 0; yi < numDataY; ++yi) {
			for (xi = 0; xi < numDataX; ++xi) {
				*result++ = get_distance_l2_sse2_d(dimension, X, Y);
				X += dimension;
			}
			X -= dimension * numDataX;
			Y += dimension;
		}
	}
	else {
		T * resultTransp = result;
		Y = X;
		for (yi = 0; yi < numDataX; ++yi) {
			for (xi = 0; xi <= yi; ++xi) {
				T z = get_distance_l2_sse2_d(dimension, X, Y);
				X += dimension;
				*result = z;
				*resultTransp = z;
				result += 1;
				resultTransp += numDataX;
			}
			X -= dimension * (yi + 1);
			Y += dimension;
			result += numDataX - (yi + 1);
			resultTransp += 1 - (yi + 1) * numDataX;
		}
	}
}











