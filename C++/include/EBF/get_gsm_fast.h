#pragma once


#ifdef __SSE2__
#include <emmintrin.h>
#include "fastmath.def"

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
get_gsm_fast
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
				*result++ = exp(-get_distance_l2_sse2_d(dimension, X, Y));
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
				T z = exp(-get_distance_l2_sse2_d(dimension, X, Y));
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
#endif // __SSE2__


#ifdef __AVX__
#include <immintrin.h>
#include "fastmath.def"

inline T
VL_XCAT(_vl_vhsum_avx_, SFX)(VTYPEavx x)
{
	T acc;
#if (VSIZEavx == 8)
	{
		VTYPEavx hsum = _mm256_hadd_ps(x, x);
		hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
		_mm_store_ss(&acc, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );
	}
#else
	{
		VTYPEavx hsum = _mm256_add_pd(x, _mm256_permute2f128_pd(x, x, 0x1));
		_mm_store_sd(&acc, _mm_hadd_pd( _mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum) ) );
	}
#endif
	return acc;
}

inline T
VL_XCAT(_vl_distance_l2_avx_, SFX)
(size_t dimension, T const * X, T const * Y)
{

	T const * X_end = X + dimension;
	T const * X_vec_end = X_end - VSIZEavx + 1;
	T acc;
	VTYPEavx vacc = VSTZavx();
	bool dataAligned = VALIGNEDavx(X) & VALIGNEDavx(Y);

	if (dataAligned) {
		while (X < X_vec_end) {
			VTYPEavx a = *(VTYPEavx*)X;
			VTYPEavx b = *(VTYPEavx*)Y;
			VTYPEavx delta = VSUBavx(a, b);
			VTYPEavx delta2 = VMULavx(delta, delta);
			vacc = VADDavx(vacc, delta2);
			X += VSIZEavx;
			Y += VSIZEavx;
		}
	}
	else {
		while (X < X_vec_end) {
			VTYPEavx a = VLDUavx(X);
			VTYPEavx b = VLDUavx(Y);
			VTYPEavx delta = VSUBavx(a, b);
			VTYPEavx delta2 = VMULavx(delta, delta);
			vacc = VADDavx(vacc, delta2);
			X += VSIZEavx;
			Y += VSIZEavx;
		}
	}

	acc = VL_XCAT(_vl_vhsum_avx_, SFX)(vacc);

	while (X < X_end) {
		T a = *X++;
		T b = *Y++;
		T delta = a - b;
		acc += delta * delta;
	}

	return acc;
}


inline void
get_gsm_fast
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
				*result++ = exp(-VL_XCAT(_vl_distance_l2_avx_, SFX)(dimension, X, Y));
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
				T z = exp(-VL_XCAT(_vl_distance_l2_avx_, SFX)(dimension, X, Y));
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

#endif // __AVX__

