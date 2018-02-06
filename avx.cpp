#include <immintrin.h>
#include "Simd.h"

void AddMultipliedSimpleAVX(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b = _mm256_load_ps(src);
		auto d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst, d);
		src += 8;
		dst += 8;
	}
	_mm256_zeroupper();
}

void AddMultipliedAVX(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b = _mm256_load_ps(src);
		auto d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst,  d);
		src += 8;
		dst += 8;

		a = _mm256_load_ps(dst);
		b = _mm256_load_ps(src);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst, d);
		src += 8;
		dst += 8;
	}
	_mm256_zeroupper();
}

void AddMultipliedRestrictAVX(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b = _mm256_load_ps(src);
		auto d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst, d);
		src += 8;
		dst += 8;

		a = _mm256_load_ps(dst);
		b = _mm256_load_ps(src);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst, d);
		src += 8;
		dst += 8;
	}
	_mm256_zeroupper();
}

void AddMultipliedAVX2(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	//_mm_prefetch(src, );
	while(dst != dstEnd)
	{
		auto a1 = _mm256_load_ps(dst);
		auto a2 = _mm256_load_ps(dst + 8);
		auto b1 = _mm256_load_ps(src);
		auto b2 = _mm256_load_ps(src + 8);
		auto d1 = _mm256_add_ps(a1, _mm256_mul_ps(b1, c));
		auto d2 = _mm256_add_ps(a2, _mm256_mul_ps(b2, c));
		_mm256_store_ps(dst, d1);
		_mm256_store_ps(dst+8, d2);
		src += 16;
		dst += 16;
	}
	_mm256_zeroupper();
}

void AddMultipliedRestrictAVX2(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto a1 = _mm256_load_ps(dst);
		auto a2 = _mm256_load_ps(dst + 8);
		auto b1 = _mm256_load_ps(src);
		auto b2 = _mm256_load_ps(src + 8);
		auto d1 = _mm256_add_ps(a1, _mm256_mul_ps(b1, c));
		auto d2 = _mm256_add_ps(a2, _mm256_mul_ps(b2, c));
		_mm256_store_ps(dst, d1);
		_mm256_store_ps(dst+8, d2);
		src += 16;
		dst += 16;
	}
	_mm256_zeroupper();
}

void AddMultipliedAVX3(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(src);
		auto a2 = _mm256_load_ps(dst + 8);
		auto b2 = _mm256_load_ps(src + 8);
		auto a3 = _mm256_load_ps(dst + 16);
		auto b3 = _mm256_load_ps(src + 16);
		auto a4 = _mm256_load_ps(dst + 24);
		auto b4 = _mm256_load_ps(src + 24);
		auto d1 = _mm256_add_ps(a1, _mm256_mul_ps(b1, c));
		auto d2 = _mm256_add_ps(a2, _mm256_mul_ps(b2, c));
		auto d3 = _mm256_add_ps(a3, _mm256_mul_ps(b3, c));
		auto d4 = _mm256_add_ps(a4, _mm256_mul_ps(b4, c));
		_mm256_store_ps(dst, d1);
		_mm256_store_ps(dst + 8, d2);
		_mm256_store_ps(dst + 16, d3);
		_mm256_store_ps(dst + 24, d4);
		dst += 32;
		src += 32;
	}
	_mm256_zeroupper();
}

void AddMultipliedRestrictAVX3(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(src);
		auto a2 = _mm256_load_ps(dst + 8);
		auto b2 = _mm256_load_ps(src + 8);
		auto a3 = _mm256_load_ps(dst + 16);
		auto b3 = _mm256_load_ps(src + 16);
		auto a4 = _mm256_load_ps(dst + 24);
		auto b4 = _mm256_load_ps(src + 24);
		auto d1 = _mm256_add_ps(a1, _mm256_mul_ps(b1, c));
		auto d2 = _mm256_add_ps(a2, _mm256_mul_ps(b2, c));
		auto d3 = _mm256_add_ps(a3, _mm256_mul_ps(b3, c));
		auto d4 = _mm256_add_ps(a4, _mm256_mul_ps(b4, c));
		_mm256_store_ps(dst, d1);
		_mm256_store_ps(dst + 8, d2);
		_mm256_store_ps(dst + 16, d3);
		_mm256_store_ps(dst + 24, d4);
		dst += 32;
		src += 32;
	}
	_mm256_zeroupper();
}

void AddMultipliedRestrictPrefetch1AVX3(float* __restrict dst, const float* __restrict src, int n, float coeff, const float* prefetchSrc)
{
	_mm_prefetch((const char*)prefetchSrc, _MM_HINT_T0);
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto a1 = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(src);
		auto a2 = _mm256_load_ps(dst + 8);
		auto b2 = _mm256_load_ps(src + 8);
		auto a3 = _mm256_load_ps(dst + 16);
		auto b3 = _mm256_load_ps(src + 16);
		auto a4 = _mm256_load_ps(dst + 24);
		auto b4 = _mm256_load_ps(src + 24);
		auto d1 = _mm256_add_ps(a1, _mm256_mul_ps(b1, c));
		auto d2 = _mm256_add_ps(a2, _mm256_mul_ps(b2, c));
		auto d3 = _mm256_add_ps(a3, _mm256_mul_ps(b3, c));
		auto d4 = _mm256_add_ps(a4, _mm256_mul_ps(b4, c));
		_mm256_store_ps(dst, d1);
		_mm256_store_ps(dst + 8, d2);
		_mm256_store_ps(dst + 16, d3);
		_mm256_store_ps(dst + 24, d4);
		dst += 32;
		src += 32;
		prefetchSrc += 32;
	}
	_mm256_zeroupper();
}

void AddMultipliedAVX4(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(src);
		auto a2 = _mm256_load_ps(dst + 8);
		auto b2 = _mm256_load_ps(src + 8);
		auto a3 = _mm256_load_ps(dst + 16);
		auto b3 = _mm256_load_ps(src + 16);
		auto a4 = _mm256_load_ps(dst + 24);
		auto b4 = _mm256_load_ps(src + 24);
		auto a5 = _mm256_load_ps(dst + 32);
		auto b5 = _mm256_load_ps(src + 32);
		auto a6 = _mm256_load_ps(dst + 40);
		auto b6 = _mm256_load_ps(src + 40);
		auto a7 = _mm256_load_ps(dst + 48);
		auto b7 = _mm256_load_ps(src + 48);
		auto a8 = _mm256_load_ps(dst + 56);
		auto b8 = _mm256_load_ps(src + 56);
		auto d1 = _mm256_add_ps(a1, _mm256_mul_ps(b1, c));
		auto d2 = _mm256_add_ps(a2, _mm256_mul_ps(b2, c));
		auto d3 = _mm256_add_ps(a3, _mm256_mul_ps(b3, c));
		auto d4 = _mm256_add_ps(a4, _mm256_mul_ps(b4, c));
		auto d5 = _mm256_add_ps(a5, _mm256_mul_ps(b5, c));
		auto d6 = _mm256_add_ps(a6, _mm256_mul_ps(b6, c));
		auto d7 = _mm256_add_ps(a7, _mm256_mul_ps(b7, c));
		auto d8 = _mm256_add_ps(a8, _mm256_mul_ps(b8, c));
		_mm256_store_ps(dst, d1);
		_mm256_store_ps(dst + 8, d2);
		_mm256_store_ps(dst + 16, d3);
		_mm256_store_ps(dst + 24, d4);
		_mm256_store_ps(dst + 32, d5);
		_mm256_store_ps(dst + 40, d6);
		_mm256_store_ps(dst + 48, d7);
		_mm256_store_ps(dst + 56, d8);
		dst += 64;
		src += 64;
	}
	_mm256_zeroupper();
}

void AddMultipliedRestrictAVX4(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(src);
		auto a2 = _mm256_load_ps(dst + 8);
		auto b2 = _mm256_load_ps(src + 8);
		auto a3 = _mm256_load_ps(dst + 16);
		auto b3 = _mm256_load_ps(src + 16);
		auto a4 = _mm256_load_ps(dst + 24);
		auto b4 = _mm256_load_ps(src + 24);
		auto a5 = _mm256_load_ps(dst + 32);
		auto b5 = _mm256_load_ps(src + 32);
		auto a6 = _mm256_load_ps(dst + 40);
		auto b6 = _mm256_load_ps(src + 40);
		auto a7 = _mm256_load_ps(dst + 48);
		auto b7 = _mm256_load_ps(src + 48);
		auto a8 = _mm256_load_ps(dst + 56);
		auto b8 = _mm256_load_ps(src + 56);
		auto d1 = _mm256_add_ps(a1, _mm256_mul_ps(b1, c));
		auto d2 = _mm256_add_ps(a2, _mm256_mul_ps(b2, c));
		auto d3 = _mm256_add_ps(a3, _mm256_mul_ps(b3, c));
		auto d4 = _mm256_add_ps(a4, _mm256_mul_ps(b4, c));
		auto d5 = _mm256_add_ps(a5, _mm256_mul_ps(b5, c));
		auto d6 = _mm256_add_ps(a6, _mm256_mul_ps(b6, c));
		auto d7 = _mm256_add_ps(a7, _mm256_mul_ps(b7, c));
		auto d8 = _mm256_add_ps(a8, _mm256_mul_ps(b8, c));
		_mm256_store_ps(dst, d1);
		_mm256_store_ps(dst + 8, d2);
		_mm256_store_ps(dst + 16, d3);
		_mm256_store_ps(dst + 24, d4);
		_mm256_store_ps(dst + 32, d5);
		_mm256_store_ps(dst + 40, d6);
		_mm256_store_ps(dst + 48, d7);
		_mm256_store_ps(dst + 56, d8);
		dst += 64;
		src += 64;
	}
	_mm256_zeroupper();
}

void AddMultipliedAVX5(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b = _mm256_load_ps(src);
		auto d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst, d);

		a = _mm256_load_ps(dst + 8);
		b = _mm256_load_ps(src + 8);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 8, d);

		a = _mm256_load_ps(dst + 16);
		b = _mm256_load_ps(src + 16);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 16, d);

		a = _mm256_load_ps(dst + 24);
		b = _mm256_load_ps(src + 24);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 24, d);

		a = _mm256_load_ps(dst + 32);
		b = _mm256_load_ps(src + 32);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 32, d);

		a = _mm256_load_ps(dst + 40);
		b = _mm256_load_ps(src + 40);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 40, d);

		a = _mm256_load_ps(dst + 48);
		b = _mm256_load_ps(src + 48);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 48, d);

		a = _mm256_load_ps(dst + 56);
		b = _mm256_load_ps(src + 56);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 56, d);

		dst += 64;
		src += 64;
	}
	_mm256_zeroupper();
}

void AddMultipliedRestrictAVX5(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b = _mm256_load_ps(src);
		auto d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst, d);

		a = _mm256_load_ps(dst + 8);
		b = _mm256_load_ps(src + 8);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 8, d);

		a = _mm256_load_ps(dst + 16);
		b = _mm256_load_ps(src + 16);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 16, d);

		a = _mm256_load_ps(dst + 24);
		b = _mm256_load_ps(src + 24);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 24, d);

		a = _mm256_load_ps(dst + 32);
		b = _mm256_load_ps(src + 32);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 32, d);

		a = _mm256_load_ps(dst + 40);
		b = _mm256_load_ps(src + 40);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 40, d);

		a = _mm256_load_ps(dst + 48);
		b = _mm256_load_ps(src + 48);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 48, d);

		a = _mm256_load_ps(dst + 56);
		b = _mm256_load_ps(src + 56);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst + 56, d);

		dst += 64;
		src += 64;
	}
	_mm256_zeroupper();
}

void AddMultipliedRestrictPrefetch2AVX3(float* __restrict dst, const float* __restrict src, int n, float coeff, const float* prefetchSrc)
{
	auto c = _mm256_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto a1 = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(src);
		auto a2 = _mm256_load_ps(dst + 8);
		auto b2 = _mm256_load_ps(src + 8);
		auto a3 = _mm256_load_ps(dst + 16);
		auto b3 = _mm256_load_ps(src + 16);
		auto a4 = _mm256_load_ps(dst + 24);
		auto b4 = _mm256_load_ps(src + 24);
		auto d1 = _mm256_add_ps(a1, _mm256_mul_ps(b1, c));
		auto d2 = _mm256_add_ps(a2, _mm256_mul_ps(b2, c));
		auto d3 = _mm256_add_ps(a3, _mm256_mul_ps(b3, c));
		auto d4 = _mm256_add_ps(a4, _mm256_mul_ps(b4, c));
		_mm256_store_ps(dst, d1);
		_mm256_store_ps(dst + 8, d2);
		_mm256_store_ps(dst + 16, d3);
		_mm256_store_ps(dst + 24, d4);
		_mm_prefetch((const char*)prefetchSrc, _MM_HINT_T0);
		_mm_prefetch((const char*)(prefetchSrc + 16), _MM_HINT_T0);
		dst += 32;
		src += 32;
		prefetchSrc += 32;
	}
	_mm256_zeroupper();
}

/*void AddMultipliedTransposedIterationAVX(float dst[8], const float** src8s, int n, float coeff)
{
	auto c = _mm256_set1_ps(coeff);
	auto srcEnd = src + n;
	auto dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b = _mm256_load_ps(src);
		auto d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst, d);
		src += 8;
		dst += 8;

		a = _mm256_load_ps(dst);
		b = _mm256_load_ps(src);
		d = _mm256_add_ps(a, _mm256_mul_ps(b, c));
		_mm256_store_ps(dst, d);
		src += 8;
		dst += 8;
	}
	_mm256_zeroupper();
}*/

void AddMultipliedX2RestrictAVX(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2])
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
		a = _mm256_add_ps(a, _mm256_mul_ps(b1, c1));
		a = _mm256_add_ps(a, _mm256_mul_ps(b2, c2));
		_mm256_store_ps(dst, a);
		dst += 8;
		p[0] += 8;
		p[1] += 8;
	}
	_mm256_zeroupper();
}

void AddMultipliedX3RestrictAVX(float* __restrict dst, const float* __restrict srcs[3], int n, const float coeffs[3])
{
	auto c1 = _mm256_set1_ps(coeffs[0]);
	auto c2 = _mm256_set1_ps(coeffs[1]);
	auto c3 = _mm256_set1_ps(coeffs[2]);
	auto dstEnd = dst + n;
	const float* p[3] = {srcs[0], srcs[1], srcs[2]};
	while(dst != dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(p[0]);
		auto b2 = _mm256_load_ps(p[1]);
		auto b3 = _mm256_load_ps(p[2]);
		a = _mm256_add_ps(a, _mm256_mul_ps(b1, c1));
		a = _mm256_add_ps(a, _mm256_mul_ps(b2, c2));
		a = _mm256_add_ps(a, _mm256_mul_ps(b3, c3));
		_mm256_store_ps(dst, a);
		dst += 8;
		p[0] += 8;
		p[1] += 8;
		p[2] += 8;
	}
	_mm256_zeroupper();
}

void AddMultipliedX3RestrictAVXClass(float* __restrict dst, const float* __restrict srcs[3], int n, const float coeffs[3])
{
	Simd::float8 c1 = Simd::Set<Simd::float8>(coeffs[0]);
	Simd::float8 c2 = Simd::Set<Simd::float8>(coeffs[1]);
	Simd::float8 c3 = Simd::Set<Simd::float8>(coeffs[2]);
	auto dstEnd = dst + n;
	const float* p[3] = {srcs[0], srcs[1], srcs[2]};
	while(dst != dstEnd)
	{
		auto a = Simd::Load8Aligned(dst);
		auto b1 = Simd::Load8Aligned(p[0]);
		auto b2 = Simd::Load8Aligned(p[1]);
		auto b3 = Simd::Load8Aligned(p[2]);
		Simd::StoreAligned(a + b1 * c1 + b2 * c2 + b3 * c3, dst);

		dst += 8;
		p[0] += 8;
		p[1] += 8;
		p[2] += 8;
	}
	Simd::End();
}

void AddMultipliedX4RestrictAVX(float* __restrict dst, const float* __restrict srcs[4], int n, const float coeffs[4])
{
	auto c1 = _mm256_set1_ps(coeffs[0]);
	auto c2 = _mm256_set1_ps(coeffs[1]);
	auto c3 = _mm256_set1_ps(coeffs[2]);
	auto c4 = _mm256_set1_ps(coeffs[3]);
	auto dstEnd = dst + n;
	const float* p[4] = {srcs[0], srcs[1], srcs[2], srcs[3]};
	while(dst != dstEnd)
	{
		auto b1 = _mm256_load_ps(p[0]);
		auto b2 = _mm256_load_ps(p[1]);
		auto b3 = _mm256_load_ps(p[2]);
		auto b4 = _mm256_load_ps(p[3]);
		auto a = _mm256_load_ps(dst);
		b1 = _mm256_add_ps(_mm256_mul_ps(b1, c1), _mm256_mul_ps(b2, c2));
		b3 = _mm256_add_ps(_mm256_mul_ps(b3, c3), _mm256_mul_ps(b4, c4));
		a = _mm256_add_ps(a, _mm256_add_ps(b1, b3));
		p[0] += 8;
		p[1] += 8;
		p[2] += 8;
		p[3] += 8;
		_mm256_store_ps(dst, a);
		dst += 8;
	}
	_mm256_zeroupper();
}

void AddMultipliedX8RestrictAVX(float* __restrict dst, const float* __restrict srcs[8], int n, const float coeffs[8])
{
	auto c1 = _mm256_set1_ps(coeffs[0]);
	auto c2 = _mm256_set1_ps(coeffs[1]);
	auto c3 = _mm256_set1_ps(coeffs[2]);
	auto c4 = _mm256_set1_ps(coeffs[3]);
	auto c5 = _mm256_set1_ps(coeffs[4]);
	auto c6 = _mm256_set1_ps(coeffs[5]);
	auto c7 = _mm256_set1_ps(coeffs[6]);
	auto c8 = _mm256_set1_ps(coeffs[7]);
	auto dstEnd = dst + n;
	const float* p[8] = {srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7]};
	while(dst != dstEnd)
	{
		auto a = _mm256_load_ps(dst);
		auto b1 = _mm256_load_ps(p[0]);
		auto b2 = _mm256_load_ps(p[1]);
		auto b3 = _mm256_load_ps(p[2]);
		auto b4 = _mm256_load_ps(p[3]);
		a = _mm256_add_ps(a, _mm256_mul_ps(b1, c1));
		a = _mm256_add_ps(a, _mm256_mul_ps(b2, c2));
		a = _mm256_add_ps(a, _mm256_mul_ps(b3, c3));
		a = _mm256_add_ps(a, _mm256_mul_ps(b4, c4));
		b1 = _mm256_load_ps(p[4]);
		b2 = _mm256_load_ps(p[5]);
		b3 = _mm256_load_ps(p[6]);
		b4 = _mm256_load_ps(p[7]);
		a = _mm256_add_ps(a, _mm256_mul_ps(b1, c5));
		a = _mm256_add_ps(a, _mm256_mul_ps(b2, c6));
		a = _mm256_add_ps(a, _mm256_mul_ps(b3, c7));
		a = _mm256_add_ps(a, _mm256_mul_ps(b4, c8));
		_mm256_store_ps(dst, a);
		dst += 8;
		p[0] += 8;
		p[1] += 8;
		p[2] += 8;
		p[3] += 8;
		p[4] += 8;
		p[5] += 8;
		p[6] += 8;
		p[7] += 8;
	}
	_mm256_zeroupper();
}



void AddMultipliedAutoAVX(float* dst, const float* src, int n, float coeff)
{
	while(n --> 0) *dst++ += coeff * *src++;
}

void AddMultipliedUnrolledAutoAVX(float* dst, const float* src, int n, float coeff)
{
	while(n)
	{
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		n -= 8;
	}
}

void AddMultipliedUnrolledRestrictAutoAVX(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	while(n)
	{
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		*dst++ += coeff * *src++;
		n -= 8;
	}
}

void AddMultipliedUnrolled2AutoAVX(float* dst, const float* src, int n, float coeff)
{
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		dst[0] += coeff * src[0];
		dst[1] += coeff * src[1];
		dst[2] += coeff * src[2];
		dst[3] += coeff * src[3];
		dst[4] += coeff * src[4];
		dst[5] += coeff * src[5];
		dst[6] += coeff * src[6];
		dst[7] += coeff * src[7];
		dst += 8;
		src += 8;
	}
}

#include <iostream>

void AddMultipliedUnrolled2RestrictAutoAVX(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		dst[0] += coeff * src[0];
		dst[1] += coeff * src[1];
		dst[2] += coeff * src[2];
		dst[3] += coeff * src[3];
		dst[4] += coeff * src[4];
		dst[5] += coeff * src[5];
		dst[6] += coeff * src[6];
		dst[7] += coeff * src[7];
		dst += 8;
		src += 8;
	}
}

float SumAVX(const float* arr, int n)
{
	const float* arrEnd = arr + n;
	auto acc = _mm256_load_ps(arr);
	arr += 8;
	while(arr != arrEnd)
	{
		acc = _mm256_add_ps(acc, _mm256_load_ps(arr));
		arr += 8;
	}
	float temp[8];
	_mm256_storeu_ps(temp, acc);
	return temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7]; 
}

float SumAVX2(const float* arr, int n)
{
	const float* arrEnd = arr + n;
	auto acc1 = _mm256_load_ps(arr);
	auto acc2 = _mm256_load_ps(arr + 8);
	arr += 16;
	while(arr != arrEnd)
	{
		acc1 = _mm256_add_ps(acc1, _mm256_load_ps(arr));
		acc2 = _mm256_add_ps(acc2, _mm256_load_ps(arr + 8));
		arr += 16;
	}
	acc1 = _mm256_add_ps(acc1, acc2);
	float temp[8];
	_mm256_storeu_ps(temp, acc1);
	return temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
}

float SumAVX3(const float* arr, int n)
{
	const float* arrEnd = arr + n;
	auto acc1 = _mm256_load_ps(arr);
	auto acc2 = _mm256_load_ps(arr + 8);
	auto acc3 = _mm256_load_ps(arr + 16);
	auto acc4 = _mm256_load_ps(arr + 24);
	arr += 32;
	while(arr != arrEnd)
	{
		acc1 = _mm256_add_ps(acc1, _mm256_load_ps(arr));
		acc2 = _mm256_add_ps(acc2, _mm256_load_ps(arr + 8));
		acc3 = _mm256_add_ps(acc3, _mm256_load_ps(arr + 16));
		acc4 = _mm256_add_ps(acc4, _mm256_load_ps(arr + 24));
		arr += 32;
	}
	acc1 = _mm256_add_ps(acc1, acc2);
	acc3 = _mm256_add_ps(acc3, acc4);
	acc1 = _mm256_add_ps(acc1, acc3);
	float temp[8];
	_mm256_storeu_ps(temp, acc1);
	return temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
}

float SumAVX4(const float* arr, int n)
{
	const float* arr2 = arr + n/2;
	const float* arrEnd = arr + n;
	auto acc1 = _mm256_load_ps(arr);
	auto acc2 = _mm256_load_ps(arr + 8);
	auto acc3 = _mm256_load_ps(arr2);
	auto acc4 = _mm256_load_ps(arr2 + 8);
	arr += 16;
	arr2 += 16;
	while(arr2 != arrEnd)
	{
		acc1 = _mm256_add_ps(acc1, _mm256_load_ps(arr));
		acc2 = _mm256_add_ps(acc2, _mm256_load_ps(arr + 8));
		acc3 = _mm256_add_ps(acc3, _mm256_load_ps(arr2));
		acc4 = _mm256_add_ps(acc4, _mm256_load_ps(arr2 + 8));
		arr += 16;
		arr2 += 16;
	}
	acc1 = _mm256_add_ps(acc1, acc2);
	acc3 = _mm256_add_ps(acc3, acc4);
	acc1 = _mm256_add_ps(acc1, acc3);
	float temp[8];
	_mm256_storeu_ps(temp, acc1);
	return temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
}


void AddExponentialRestrictAVX(float* __restrict dst, const float* __restrict src, int n, float coeff, float ek)
{
	const float ek2 = ek*ek;
	const float ek4 = ek2*ek2;
	const auto ek_8 = _mm256_set1_ps(ek4*ek4);

	auto r1 = _mm256_set_ps(coeff*ek4*ek2*ek, coeff*ek4*ek2, coeff*ek4*ek, coeff*ek4, coeff*ek2*ek, coeff*ek2, coeff*ek, coeff);
	auto r4 = _mm256_mul_ps(ek_8, r1);
	auto ek_16 = _mm256_mul_ps(ek_8, ek_8);
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto r0 = _mm256_mul_ps(_mm256_load_ps(src), r1);
		r1 = _mm256_mul_ps(r1, ek_16);
		auto r3 = _mm256_mul_ps(_mm256_load_ps(src + 8), r4);
		_mm256_store_ps(dst, _mm256_add_ps(_mm256_load_ps(dst), r0));
		r4 = _mm256_mul_ps(r4, ek_16);
		_mm256_store_ps(dst + 8, _mm256_add_ps(_mm256_load_ps(dst + 8), r3));
		dst += 16;
		src += 16;
	}
}

void AddExponentialX2RestrictAVX(float* __restrict dst, const float* __restrict src[2], int n, const float coeff[2], float ek0, float ek1)
{
	const float ek2[2] = {ek0*ek0, ek1*ek1};
	const float ek4[2] = {ek2[0]*ek2[0], ek2[1]*ek2[1]};
	const __m256 ek_8[2] = {_mm256_set1_ps(ek4[0]*ek4[0]), _mm256_set1_ps(ek4[1]*ek4[1])};

	auto src0 = src[0];
	auto src1 = src[1];

	auto r1 = _mm256_set_ps(coeff[0]*ek4[0]*ek2[0]*ek0, coeff[0]*ek4[0]*ek2[0],
		coeff[0]*ek4[0]*ek0, coeff[0]*ek4[0], coeff[0]*ek2[0]*ek0, coeff[0]*ek2[0], coeff[0]*ek0, coeff[0]);
	auto r4 = _mm256_set_ps(coeff[1]*ek4[1]*ek2[1]*ek1, coeff[1]*ek4[1]*ek2[1],
		coeff[1]*ek4[1]*ek1, coeff[1]*ek4[1], coeff[1]*ek2[1]*ek1, coeff[1]*ek2[1], coeff[1]*ek1, coeff[1]);
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto r0 = _mm256_mul_ps(_mm256_load_ps(src0), r1);
		r1 = _mm256_mul_ps(r1, ek_8[0]);
		auto r3 = _mm256_mul_ps(_mm256_load_ps(src1), r4);
		auto a = _mm256_load_ps(dst);
		a = _mm256_add_ps(a, r0);
		r4 = _mm256_mul_ps(r4, ek_8[1]);
		a = _mm256_add_ps(a, r3);
		_mm256_store_ps(dst, a);
		dst += 8;
		src0 += 8;
		src1 += 8;
	}
}

void LinearMultiplyAddRestrictAVX(float* __restrict dst, const float* __restrict src, int n, float u, float du)
{
	auto u8 = _mm256_set_ps(u + 7*du, u + 6*du, u + 5*du, u + 4*du, u + 3*du, u + 2*du, u + du, u);
	const auto du8 = _mm256_set1_ps(8*du);
	const float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto vsrc = _mm256_load_ps(src);
		auto vdst = _mm256_load_ps(dst);
		_mm256_store_ps(dst, _mm256_add_ps(vdst, _mm256_mul_ps(vsrc, u8)));
		u8 = _mm256_add_ps(u8, du8);

		vsrc = _mm256_load_ps(src + 8);
		vdst = _mm256_load_ps(dst + 8);
		_mm256_store_ps(dst + 8, _mm256_add_ps(vdst, _mm256_mul_ps(vsrc, u8)));
		u8 = _mm256_add_ps(u8, du8);

		src += 16;
		dst += 16;
	}
}

void LinearMultiplyAddX2RestrictAVX(float* __restrict dst, const float* __restrict src[2], int n, const float u[2], float du0, float du1)
{
	auto u80 = _mm256_set_ps(u[0] + 7*du0, u[0] + 6*du0, u[0] + 5*du0, u[0] + 4*du0, u[0] + 3*du0, u[0] + 2*du0, u[0] + du0, u[0]);
	auto u81 = _mm256_set_ps(u[1] + 7*du1, u[1] + 6*du1, u[1] + 5*du1, u[1] + 4*du1, u[1] + 3*du1, u[1] + 2*du1, u[1] + du1, u[1]);
	const auto du80 = _mm256_set1_ps(8*du0);
	const auto du81 = _mm256_set1_ps(8*du1);
	const float* dstEnd = dst + n;
	const float* src0 = src[0];
	const float* src1 = src[1];
	while(dst != dstEnd)
	{
		auto vsrc0 = _mm256_load_ps(src0);
		auto vsrc1 = _mm256_load_ps(src1);
		auto vdst = _mm256_load_ps(dst);
		vsrc0 = _mm256_mul_ps(vsrc0, u80);
		vsrc1 = _mm256_mul_ps(vsrc1, u81);
		vdst = _mm256_add_ps(vdst, vsrc0);
		vdst = _mm256_add_ps(vdst, vsrc1);
		_mm256_store_ps(dst, vdst);
		u80 = _mm256_add_ps(u80, du80);
		u81 = _mm256_add_ps(u81, du81);

		src0 += 8;
		src1 += 8;
		dst += 8;
	}
}

void ExponentiateAddAVX(float* __restrict dst, const float* __restrict src, int n, float dummy)
{
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		Simd::float8 rdst = Simd::Load8Aligned(dst);
		Simd::float8 rsrc = Simd::Load8Aligned(src);
		rdst += Simd::Exp(rsrc);
		Simd::StoreAligned(rdst, dst);
		dst += 8;
		src += 8;
	}
}
