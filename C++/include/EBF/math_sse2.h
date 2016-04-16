#pragma once
#include <emmintrin.h>


#define VSIZE_SSE2_F 4
#define VTYPE_SSE2_F __m128

#define VSIZE_SSE2_D 2
#define VTYPE_SSE2_D __m128d

typedef long long unsigned  uintptr;
#define VALIGNED_SSE2(x) (! (((uintptr)(x)) & 0xF))


// vector shuffle sum
inline float vshsum_sse2_f(VTYPE_SSE2_F x) {
	float acc;

	VTYPE_SSE2_F sum, shuffle;
	/* shuffle = [1 0 3 2] */
	/* sum     = [3+1 2+0 1+3 0+2] */
	/* shuffle = [2+0 3+1 0+2 1+3] */
	/* vacc    = [3+1+2+0 3+1+2+0 1+3+0+2 0+2+1+3] */
	shuffle = _mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
	sum		= _mm_add_ps(x, shuffle);
	shuffle = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));
	x		= _mm_add_ps(sum, shuffle);

	// store x to acc
	_mm_store_ss(&acc, x);
	return acc;
}

inline double vshsum_sse2_d(VTYPE_SSE2_D x) {
	double acc;

	VTYPE_SSE2_D  shuffle;
	/* acc     = [1   0  ] */
	/* shuffle = [0   1  ] */
	/* sum     = [1+0 0+1] */
	shuffle = _mm_shuffle_pd(x, x, _MM_SHUFFLE2(0, 1));
	x = _mm_add_pd(x, shuffle);

	// store x to acc
	_mm_store_sd(&acc, x);
	return acc;
}


inline float dot_see2_f(size_t dimension, float const*X, float const*Y) {
	float const * X_end = X + dimension;
	float const * X_vec_end = X_end - VSIZE_SSE2_F + 1;
	float acc;
	VTYPE_SSE2_F vacc = _mm_setzero_ps();
	bool dataAligned = VALIGNED_SSE2(X) & VALIGNED_SSE2(Y);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_SSE2_F a = *(VTYPE_SSE2_F*)X;
			VTYPE_SSE2_F b = *(VTYPE_SSE2_F*)Y;
			vacc = _mm_add_ps(vacc, _mm_mul_ps(a, b));
			X += VSIZE_SSE2_F;
			Y += VSIZE_SSE2_F;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_SSE2_F a = _mm_loadu_ps(X);
			VTYPE_SSE2_F b = _mm_loadu_ps(Y);
			vacc = _mm_add_ps(vacc, _mm_mul_ps(a, b));
			X += VSIZE_SSE2_F;
			Y += VSIZE_SSE2_F;
		}
	}

	acc = vshsum_sse2_f(vacc);
	
	while (X < X_end)
	{
		acc += (*X++) * (*Y++);
	}

	return acc;
}

inline double dot_see2_d(size_t dimension, double const*X, double const*Y) {
	double const * X_end = X + dimension;
	double const * X_vec_end = X_end - VSIZE_SSE2_D + 1;
	double acc;
	VTYPE_SSE2_D vacc = _mm_setzero_pd();
	bool dataAligned = VALIGNED_SSE2(X) & VALIGNED_SSE2(Y);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_SSE2_D a = *(VTYPE_SSE2_D*)X;
			VTYPE_SSE2_D b = *(VTYPE_SSE2_D*)Y;
			vacc = _mm_add_pd(vacc, _mm_mul_pd(a, b));
			X += VSIZE_SSE2_D;
			Y += VSIZE_SSE2_D;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_SSE2_D a = _mm_loadu_pd(X);
			VTYPE_SSE2_D b = _mm_loadu_pd(Y);
			vacc = _mm_add_pd(vacc, _mm_mul_pd(a, b));
			X += VSIZE_SSE2_D;
			Y += VSIZE_SSE2_D;
		}
	}

	acc = vshsum_sse2_d(vacc);

	while (X < X_end)
	{
		acc += (*X++) * (*Y++);
	}

	return acc;
}


inline float dist_l2_sse2_f(size_t dimension, float const *X, float const *Y) {
	float const * X_end = X + dimension;
	float const * X_vec_end = X_end - VSIZE_SSE2_F + 1;
	float acc;
	VTYPE_SSE2_F vacc = _mm_setzero_ps();
	int dataAligned = VALIGNED_SSE2(X) * VALIGNED_SSE2(Y);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_SSE2_F a = *(VTYPE_SSE2_F*)X;
			VTYPE_SSE2_F b = *(VTYPE_SSE2_F*)Y;
			VTYPE_SSE2_F d = _mm_sub_ps(a, b);
			vacc = _mm_add_ps(vacc, _mm_mul_ps(d, d));
			X += VSIZE_SSE2_F;
			Y += VSIZE_SSE2_F;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_SSE2_F a = _mm_loadu_ps(X);
			VTYPE_SSE2_F b = _mm_loadu_ps(Y);
			VTYPE_SSE2_F d = _mm_sub_ps(a, b);
			vacc = _mm_add_ps(vacc, _mm_mul_ps(d, d));
			X += VSIZE_SSE2_F;
			Y += VSIZE_SSE2_F;
		}
	}

	acc = vshsum_sse2_f(vacc);

	while (X < X_end)
	{
		float d = (*X++) - (*Y++);
		acc += d * d;
	}
	return acc;
}

inline double dist_l2_sse2_d(size_t dimension, double const *X, double const *Y) {
	double const * X_end = X + dimension;
	double const * X_vec_end = X_end - VSIZE_SSE2_F + 1;
	double acc;
	VTYPE_SSE2_D vacc = _mm_setzero_pd();
	int dataAligned = VALIGNED_SSE2(X) * VALIGNED_SSE2(Y);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_SSE2_D a = *(VTYPE_SSE2_D*)X;
			VTYPE_SSE2_D b = *(VTYPE_SSE2_D*)Y;
			VTYPE_SSE2_D d = _mm_sub_pd(a, b);
			vacc = _mm_add_pd(vacc, _mm_mul_pd(d, d));
			X += VSIZE_SSE2_D;
			Y += VSIZE_SSE2_D;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_SSE2_D a = _mm_loadu_pd(X);
			VTYPE_SSE2_D b = _mm_loadu_pd(Y);
			VTYPE_SSE2_D d = _mm_sub_pd(a, b);
			vacc = _mm_add_pd(vacc, _mm_mul_pd(d, d));
			X += VSIZE_SSE2_D;
			Y += VSIZE_SSE2_D;
		}
	}

	acc = vshsum_sse2_d(vacc);

	while (X < X_end)
	{
		double d = (*X++) - (*Y++);
		acc += d * d;
	}
	return acc;
}


inline void dist_l2_matrix_sse2_f
(float * result, size_t dimension,
	float const * X, size_t numDataX,
	float const * Y, size_t numDataY)
{
	size_t xi, yi;

	if (dimension == 0) return;
	if (numDataX == 0) return;
	assert(X);

	if (Y) {
		if (numDataY == 0) return;
		for (yi = 0; yi < numDataY; ++yi) {
			for (xi = 0; xi < numDataX; ++xi) {
				*result++ = dist_l2_sse2_f(dimension, X, Y);
				X += dimension;
			}
			X -= dimension * numDataX;
			Y += dimension;
		}
	}
	else {
		float * resultTransp = result;
		Y = X;
		for (yi = 0; yi < numDataX; ++yi) {
			for (xi = 0; xi <= yi; ++xi) {
				float z = dist_l2_sse2_f(dimension, X, Y);
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


inline void dist_l2_matrix_sse2_d
(double * result, size_t dimension,
	double const * X, size_t numDataX,
	double const * Y, size_t numDataY)
{
	size_t xi, yi;

	if (dimension == 0) return;
	if (numDataX == 0) return;
	assert(X);

	if (Y) {
		if (numDataY == 0) return;
		for (yi = 0; yi < numDataY; ++yi) {
			for (xi = 0; xi < numDataX; ++xi) {
				*result++ = dist_l2_sse2_d(dimension, X, Y);
				X += dimension;
			}
			X -= dimension * numDataX;
			Y += dimension;
		}
	}
	else {
		double * resultTransp = result;
		Y = X;
		for (yi = 0; yi < numDataX; ++yi) {
			for (xi = 0; xi <= yi; ++xi) {
				double z = dist_l2_sse2_d(dimension, X, Y);
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