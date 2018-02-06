#include <immintrin.h>

void AddMultipliedFMA(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b = _mm256_load_ps(src);
		auto d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst,  d);
		src += 8;
		dst += 8;

		a = _mm256_load_ps(dst);
		b = _mm256_load_ps(src);
		d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst, d);
		src += 8;
		dst += 8;
	}
	_mm256_zeroupper();
}

void AddMultipliedRestrictFMA(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b = _mm256_load_ps(src);
		auto d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst, d);
		src += 8;
		dst += 8;

		a = _mm256_load_ps(dst);
		b = _mm256_load_ps(src);
		d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst, d);
		src += 8;
		dst += 8;
	}
	_mm256_zeroupper();
}

void AddMultipliedRestrictFMA5(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b = _mm256_load_ps(src);
		auto d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst, d);

		a = _mm256_load_ps(dst + 8);
		b = _mm256_load_ps(src + 8);
		d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst + 8, d);

		a = _mm256_load_ps(dst + 16);
		b = _mm256_load_ps(src + 16);
		d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst + 16, d);

		a = _mm256_load_ps(dst + 24);
		b = _mm256_load_ps(src + 24);
		d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst + 24, d);

		a = _mm256_load_ps(dst + 32);
		b = _mm256_load_ps(src + 32);
		d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst + 32, d);

		a = _mm256_load_ps(dst + 40);
		b = _mm256_load_ps(src + 40);
		d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst + 40, d);

		a = _mm256_load_ps(dst + 48);
		b = _mm256_load_ps(src + 48);
		d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst + 48, d);

		a = _mm256_load_ps(dst + 56);
		b = _mm256_load_ps(src + 56);
		d = _mm256_fmadd_ps(b, c, a);
		_mm256_store_ps(dst + 56, d);

		dst += 64;
		src += 64;
	}
}

void AddMultipliedRestrictFMA6(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(src);
		auto a2 = _mm256_load_ps(dst + 64);
		auto b2 = _mm256_load_ps(src + 64);
		auto d1 = _mm256_fmadd_ps(b1, c, a1);
		auto d2 = _mm256_fmadd_ps(b2, c, a2);
		_mm256_store_ps(dst, d1);
		_mm256_store_ps(dst+ 64, d2);

		a1 = _mm256_load_ps(dst + 8);
		b1 = _mm256_load_ps(src + 8);
		a2 = _mm256_load_ps(dst + 72);
		b2 = _mm256_load_ps(src + 72);
		d1 = _mm256_fmadd_ps(b1, c, a1);
		d2 = _mm256_fmadd_ps(b2, c, a2);
		_mm256_store_ps(dst + 8, d1);
		_mm256_store_ps(dst + 72, d2);

		a1 = _mm256_load_ps(dst + 16);
		b1 = _mm256_load_ps(src + 16);
		a2 = _mm256_load_ps(dst + 80);
		b2 = _mm256_load_ps(src + 80);
		d1 = _mm256_fmadd_ps(b1, c, a1);
		d2 = _mm256_fmadd_ps(b2, c, a2);
		_mm256_store_ps(dst + 16, d1);
		_mm256_store_ps(dst + 80, d2);

		a1 = _mm256_load_ps(dst + 24);
		b1 = _mm256_load_ps(src + 24);
		a2 = _mm256_load_ps(dst + 88);
		b2 = _mm256_load_ps(src + 88);
		d1 = _mm256_fmadd_ps(b1, c, a1);
		d2 = _mm256_fmadd_ps(b2, c, a2);
		_mm256_store_ps(dst + 24, d1);
		_mm256_store_ps(dst + 88, d2);

		a1 = _mm256_load_ps(dst + 32);
		b1 = _mm256_load_ps(src + 32);
		a2 = _mm256_load_ps(dst + 96);
		b2 = _mm256_load_ps(src + 96);
		d1 = _mm256_fmadd_ps(b1, c, a1);
		d2 = _mm256_fmadd_ps(b2, c, a2);
		_mm256_store_ps(dst + 32, d1);
		_mm256_store_ps(dst + 96, d2);

		a1 = _mm256_load_ps(dst + 40);
		b1 = _mm256_load_ps(src + 40);
		a2 = _mm256_load_ps(dst + 104);
		b2 = _mm256_load_ps(src + 104);
		d1 = _mm256_fmadd_ps(b1, c, a1);
		d2 = _mm256_fmadd_ps(b2, c, a2);
		_mm256_store_ps(dst + 40, d1);
		_mm256_store_ps(dst + 104, d2);

		a1 = _mm256_load_ps(dst + 48);
		b1 = _mm256_load_ps(src + 48);
		a2 = _mm256_load_ps(dst + 112);
		b2 = _mm256_load_ps(src + 112);
		d1 = _mm256_fmadd_ps(b1, c, a1);
		d2 = _mm256_fmadd_ps(b2, c, a2);
		_mm256_store_ps(dst + 48, d1);
		_mm256_store_ps(dst + 112, d2);

		a1 = _mm256_load_ps(dst + 56);
		b1 = _mm256_load_ps(src + 56);
		a2 = _mm256_load_ps(dst + 120);
		b2 = _mm256_load_ps(src + 120);
		d1 = _mm256_fmadd_ps(b1, c, a1);
		d2 = _mm256_fmadd_ps(b2, c, a2);
		_mm256_store_ps(dst + 56, d1);
		_mm256_store_ps(dst + 120, d2);

		dst += 128;
		src += 128;
	}
}

void AddMultipliedX2RestrictFMA(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2])
{
	auto c1 = _mm256_set1_ps(coeffs[0]);
	auto c2 = _mm256_set1_ps(coeffs[1]);
	auto dstEnd = dst + n;
	const float* p[2] = {srcs[0], srcs[1]};
	while(dst != dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(p[0]);
		auto b2 = _mm256_load_ps(p[1]);
		a = _mm256_fmadd_ps(b1, c1, a);
		a = _mm256_fmadd_ps(b2, c2, a);
		_mm256_store_ps(dst, a);
		dst += 8;
		p[0] += 8;
		p[1] += 8;
	}
	_mm256_zeroupper();
}
