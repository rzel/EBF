#pragma once
#include <immintrin.h>


#define VSIZE_AVX_F 8
#define VTYPE_AVX_F __m256

#define VSIZE_AVX_D 4
#define VTYPE_AVX_D __m256d

typedef long long unsigned  uintptr;
#define VALIGNEDavx(x) (! (((uintptr)(x)) & 0x1F))

inline float vshsum_avx_f(VTYPE_AVX_F x) {
	float acc;

	VTYPE_AVX_F hsum = _mm256_hadd_ps(x, x);
	hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));
	_mm_store_ss(&acc, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );
	
	return acc;
}

inline double vshsum_avx_d(VTYPE_AVX_D x) {
	double acc;

	VTYPE_AVX_D hsum = _mm256_add_pd(x, _mm256_permute2f128_pd(x, x, 0x1));
	_mm_store_sd(&acc, _mm_hadd_pd( _mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum) ) );
	
	return acc;
}


inline float dist_l2_avx_f(size_t dimension, float const * X, float const * Y) {
	float const * X_end = X + dimension;
	float const * X_vec_end = X_end - VSIZE_AVX_F + 1;
	float acc;
	VTYPE_AVX_F vacc = _mm256_setzero_ps();
	int dataAligned = VALIGNEDavx(X) & VALIGNEDavx(Y);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_F a = *(VTYPE_AVX_F*)X;
			VTYPE_AVX_F b = *(VTYPE_AVX_F*)Y;
			VTYPE_AVX_F d = _mm256_sub_ps(a, b);
			vacc = _mm256_add_ps(vacc, _mm256_mul_ps(d, d));
			X += VSIZE_AVX_F;
			Y += VSIZE_AVX_F;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_F a = _mm256_loadu_ps(X);
			VTYPE_AVX_F b = _mm256_loadu_ps(Y);
			VTYPE_AVX_F d = _mm256_sub_ps(a, b);
			vacc = _mm256_add_ps(vacc, _mm256_mul_ps(d, d));
			X += VSIZE_AVX_F;
			Y += VSIZE_AVX_F;
		}
	}

	acc = vshsum_avx_f(vacc);

	while (X < X_end)
	{
		float d = (*X++) - (*Y++);
		acc += d * d;
	}

	return acc;
}


inline double dist_l2_avx_d(size_t dimension, double const * X, double const * Y) {
	double const * X_end = X + dimension;
	double const * X_vec_end = X_end - VSIZE_AVX_D + 1;
	double acc;
	VTYPE_AVX_D vacc = _mm256_setzero_pd();
	int dataAligned = VALIGNEDavx(X) & VALIGNEDavx(Y);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_D a = *(VTYPE_AVX_D*)X;
			VTYPE_AVX_D b = *(VTYPE_AVX_D*)Y;
			VTYPE_AVX_D d = _mm256_sub_pd(a, b);
			vacc = _mm256_add_pd(vacc, _mm256_mul_pd(d, d));
			X += VSIZE_AVX_D;
			Y += VSIZE_AVX_D;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_D a = _mm256_loadu_pd(X);
			VTYPE_AVX_D b = _mm256_loadu_pd(Y);
			VTYPE_AVX_D d = _mm256_sub_pd(a, b);
			vacc = _mm256_add_pd(vacc, _mm256_mul_pd(d, d));
			X += VSIZE_AVX_D;
			Y += VSIZE_AVX_D;
		}
	}

	acc = vshsum_avx_d(vacc);

	while (X < X_end)
	{
		double d = (*X++) - (*Y++);
		acc += d * d;
	}

	return acc;
}


inline float dot_avx_f(size_t dimension, float const*X, float const*Y) {
	float const * X_end = X + dimension;
	float const * X_vec_end = X_end - VSIZE_AVX_F + 1;
	float acc;
	VTYPE_AVX_F vacc = _mm256_setzero_ps();
	bool dataAligned = VALIGNEDavx(X) & VALIGNEDavx(Y);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_F a = *(VTYPE_AVX_F*)X;
			VTYPE_AVX_F b = *(VTYPE_AVX_F*)Y;
			vacc = _mm256_add_ps(vacc, _mm256_mul_ps(a, b));
			X += VSIZE_AVX_F;
			Y += VSIZE_AVX_F;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_F a = _mm256_loadu_ps(X);
			VTYPE_AVX_F b = _mm256_loadu_ps(Y);
			vacc = _mm256_add_ps(vacc, _mm256_mul_ps(a, b));
			X += VSIZE_AVX_F;
			Y += VSIZE_AVX_F;
		}
	}

	acc = vshsum_avx_f(vacc);

	while (X < X_end)
	{
		acc += (*X++) * (*Y++);
	}

	return acc;
}

inline double dot_avx_d(size_t dimension, double const*X, double const*Y) {
	double const * X_end = X + dimension;
	double const * X_vec_end = X_end - VSIZE_AVX_D + 1;
	double acc;
	VTYPE_AVX_D vacc = _mm256_setzero_pd();
	bool dataAligned = VALIGNEDavx(X) & VALIGNEDavx(Y);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_D a = *(VTYPE_AVX_D*)X;
			VTYPE_AVX_D b = *(VTYPE_AVX_D*)Y;
			vacc = _mm256_add_pd(vacc, _mm256_mul_pd(a, b));
			X += VSIZE_AVX_D;
			Y += VSIZE_AVX_D;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_D a = _mm256_loadu_pd(X);
			VTYPE_AVX_D b = _mm256_loadu_pd(Y);
			vacc = _mm256_add_pd(vacc, _mm256_mul_pd(a, b));
			X += VSIZE_AVX_D;
			Y += VSIZE_AVX_D;
		}
	}

	acc = vshsum_avx_d(vacc);

	while (X < X_end)
	{
		acc += (*X++) * (*Y++);
	}

	return acc;
}



inline void dist_l2_matrix_avx_f
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
				*result++ = dist_l2_avx_f(dimension, X, Y);
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
				float z = dist_l2_avx_f(dimension, X, Y);
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


inline void dist_l2_matrix_avx_d
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
				*result++ = dist_l2_avx_d(dimension, X, Y);
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
				double z = dist_l2_avx_d(dimension, X, Y);
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


inline double phuber_cost_avx_d(size_t dimension, double const *X, double threshold) {
	double const * X_end = X + dimension;
	double const * X_vec_end = X_end - VSIZE_AVX_D + 1;
	double acc;
	VTYPE_AVX_D vacc = _mm256_setzero_pd();
	VTYPE_AVX_D vthres = _mm256_set_pd(threshold, threshold, threshold, threshold);
	VTYPE_AVX_D vones = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
	int dataAligned = VALIGNEDavx(X);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_D vx = *(VTYPE_AVX_D*)X;
			VTYPE_AVX_D vd = _mm256_div_pd(vx, vthres);					
			vacc = _mm256_add_pd(vacc, _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(vd, vd), vones)));
			X += VSIZE_AVX_D;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_D vx = _mm256_loadu_pd(X);
			VTYPE_AVX_D vd = _mm256_div_pd(vx, vthres);
			vacc = _mm256_add_pd(vacc, _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(vd, vd), vones)));
			X += VSIZE_AVX_D;
		}
	}

	acc = vshsum_avx_d(vacc);

	while (X < X_end)
	{
		double d = ((*X++) / threshold) ;
		acc += sqrt(d * d + 1);
	}

	return (acc - dimension) * threshold * threshold;
}

inline float phuber_cost_avx_f(size_t dimension, float const *X, float threshold) {
	float const * X_end = X + dimension;
	float const * X_vec_end = X_end - VSIZE_AVX_F + 1;
	float acc;
	VTYPE_AVX_F vacc = _mm256_setzero_ps();
	VTYPE_AVX_F vthres = _mm256_set_ps(threshold, threshold, threshold, threshold, threshold, threshold, threshold, threshold);
	VTYPE_AVX_F vones = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
	int dataAligned = VALIGNEDavx(X);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_F vx = *(VTYPE_AVX_F*)X;
			VTYPE_AVX_F vd = _mm256_div_ps(vx, vthres);
			vacc = _mm256_add_ps(vacc, _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(vd, vd), vones)));
			X += VSIZE_AVX_F;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_F vx = _mm256_loadu_ps(X);
			VTYPE_AVX_F vd = _mm256_div_ps(vx, vthres);
			vacc = _mm256_add_ps(vacc, _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(vd, vd), vones)));
			X += VSIZE_AVX_F;
		}
	}

	acc = vshsum_avx_f(vacc);

	while (X < X_end)
	{
		double d = ((*X++) / threshold);
		acc += sqrt(d * d + 1);
	}

	return (acc - dimension) * threshold * threshold;
}

inline void phuber_grad_avx_d(size_t dimension, double const *X, double *g, double threshold) {
	double const * X_end = X + dimension;
	double const * X_vec_end = X_end - VSIZE_AVX_D + 1;

	VTYPE_AVX_D vthres = _mm256_set_pd(threshold, threshold, threshold, threshold);
	VTYPE_AVX_D vones = _mm256_set_pd(1.0, 1.0, 1.0, 1.0);
	VTYPE_AVX_D vg;
	int dataAligned = VALIGNEDavx(X) & VALIGNEDavx(g);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_D vx = *(VTYPE_AVX_D*)X;
			VTYPE_AVX_D vd = _mm256_div_pd(vx, vthres);
			vg = _mm256_div_pd(vx, _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(vd, vd), vones)));
			_mm256_store_pd(g, vg);
			g += VSIZE_AVX_D;
			X += VSIZE_AVX_D;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_D vx = _mm256_loadu_pd(X);
			VTYPE_AVX_D vd = _mm256_div_pd(vx, vthres);
			vg = _mm256_div_pd(vx, _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(vd, vd), vones)));
			_mm256_storeu_pd(g, vg);
			g += VSIZE_AVX_D;
			X += VSIZE_AVX_D;
		}
	}
	while (X < X_end)
	{
		double d = (*X / threshold);
		(*g++) = *X / sqrt(d * d + 1);
		X++;
	}
}

inline void phuber_grad_avx_f(size_t dimension, float const *X, float *g, float threshold) {
	float const * X_end = X + dimension;
	float const * X_vec_end = X_end - VSIZE_AVX_F + 1;

	VTYPE_AVX_F vthres = _mm256_set_ps(threshold, threshold, threshold, threshold, threshold, threshold, threshold, threshold);
	VTYPE_AVX_F vones = _mm256_set_ps(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
	VTYPE_AVX_F vg;
	int dataAligned = VALIGNEDavx(X) & VALIGNEDavx(g);

	if (dataAligned)
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_F vx = *(VTYPE_AVX_F*)X;
			VTYPE_AVX_F vd = _mm256_div_ps(vx, vthres);
			vg = _mm256_div_ps(vx, _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(vd, vd), vones)));
			_mm256_store_ps(g, vg);
			g += VSIZE_AVX_F;
			X += VSIZE_AVX_F;
		}
	}
	else
	{
		while (X < X_vec_end)
		{
			VTYPE_AVX_F vx = _mm256_loadu_ps(X);
			VTYPE_AVX_F vd = _mm256_div_ps(vx, vthres);
			vg = _mm256_div_ps(vx, _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(vd, vd), vones)));
			_mm256_storeu_ps(g, vg);
			g += VSIZE_AVX_F;
			X += VSIZE_AVX_F;
		}
	}
	while (X < X_end)
	{
		float d = (*X / threshold);
		(*g++) = (*X++) / sqrt(d * d + 1);
	}
}

