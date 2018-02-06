#include <type_traits>
#include "Simd.h"

void AddMultipliedSimpleSSE(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b = _mm_loadu_ps(src);
		auto d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst, d);
		dst += 4;
		src += 4;
	}
}

void AddMultipliedSSE(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b = _mm_loadu_ps(src);
		auto d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst, d);
		dst += 4;
		src += 4;

		a = _mm_load_ps(dst);
		b = _mm_loadu_ps(src);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst, d);
		dst += 4;
		src += 4;
	}
}

void AddMultipliedRestrictSSE(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b = _mm_loadu_ps(src);
		auto d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst, d);
		dst += 4;
		src += 4;

		a = _mm_load_ps(dst);
		b = _mm_loadu_ps(src);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst, d);
		dst += 4;
		src += 4;
	}
}

void AddMultipliedSSE2(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm_load_ps(dst);
		auto b1 = _mm_loadu_ps(src);
		auto a2 = _mm_load_ps(dst + 4);
		auto b2 = _mm_loadu_ps(src + 4);
		auto d1 = _mm_add_ps(a1, _mm_mul_ps(b1, c));
		auto d2 = _mm_add_ps(a2, _mm_mul_ps(b2, c));
		_mm_store_ps(dst, d1);
		_mm_store_ps(dst + 4, d2);
		dst += 8;
		src += 8;
	}
}

void AddMultipliedRestrictSSE2(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm_load_ps(dst);
		auto b1 = _mm_loadu_ps(src);
		auto a2 = _mm_load_ps(dst + 4);
		auto b2 = _mm_loadu_ps(src + 4);
		auto d1 = _mm_add_ps(a1, _mm_mul_ps(b1, c));
		auto d2 = _mm_add_ps(a2, _mm_mul_ps(b2, c));
		_mm_store_ps(dst, d1);
		_mm_storeu_ps(dst + 4, d2);
		dst += 8;
		src += 8;
	}
}

void AddMultipliedSSE3(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm_load_ps(dst);
		auto b1 = _mm_loadu_ps(src);
		auto a2 = _mm_load_ps(dst + 4);
		auto b2 = _mm_loadu_ps(src + 4);
		auto a3 = _mm_load_ps(dst + 8);
		auto b3 = _mm_loadu_ps(src + 8);
		auto a4 = _mm_load_ps(dst + 12);
		auto b4 = _mm_loadu_ps(src + 12);
		auto d1 = _mm_add_ps(a1, _mm_mul_ps(b1, c));
		auto d2 = _mm_add_ps(a2, _mm_mul_ps(b2, c));
		auto d3 = _mm_add_ps(a3, _mm_mul_ps(b3, c));
		auto d4 = _mm_add_ps(a4, _mm_mul_ps(b4, c));
		_mm_store_ps(dst, d1);
		_mm_store_ps(dst + 4, d2);
		_mm_store_ps(dst + 8, d3);
		_mm_store_ps(dst + 12, d4);
		dst += 16;
		src += 16;
	}
}

void AddMultipliedRestrictSSE3(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm_load_ps(dst);
		auto b1 = _mm_loadu_ps(src);
		auto a2 = _mm_load_ps(dst + 4);
		auto b2 = _mm_loadu_ps(src + 4);
		auto a3 = _mm_load_ps(dst + 8);
		auto b3 = _mm_loadu_ps(src + 8);
		auto a4 = _mm_load_ps(dst + 12);
		auto b4 = _mm_loadu_ps(src + 12);
		auto d1 = _mm_add_ps(a1, _mm_mul_ps(b1, c));
		auto d2 = _mm_add_ps(a2, _mm_mul_ps(b2, c));
		auto d3 = _mm_add_ps(a3, _mm_mul_ps(b3, c));
		auto d4 = _mm_add_ps(a4, _mm_mul_ps(b4, c));
		_mm_store_ps(dst, d1);
		_mm_store_ps(dst + 4, d2);
		_mm_store_ps(dst + 8, d3);
		_mm_store_ps(dst + 12, d4);
		dst += 16;
		src += 16;
	}
}

void AddMultipliedSSE4(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm_load_ps(dst);
		auto b1 = _mm_loadu_ps(src);
		auto a2 = _mm_load_ps(dst + 4);
		auto b2 = _mm_loadu_ps(src + 4);
		auto a3 = _mm_load_ps(dst + 8);
		auto b3 = _mm_loadu_ps(src + 8);
		auto a4 = _mm_load_ps(dst + 12);
		auto b4 = _mm_loadu_ps(src + 12);
		auto a5 = _mm_load_ps(dst + 16);
		auto b5 = _mm_loadu_ps(src + 16);
		auto a6 = _mm_load_ps(dst + 20);
		auto b6 = _mm_loadu_ps(src + 20);
		auto a7 = _mm_load_ps(dst + 24);
		auto b7 = _mm_loadu_ps(src + 24);
		auto a8 = _mm_load_ps(dst + 28);
		auto b8 = _mm_loadu_ps(src + 28);
		auto d1 = _mm_add_ps(a1, _mm_mul_ps(b1, c));
		auto d2 = _mm_add_ps(a2, _mm_mul_ps(b2, c));
		auto d3 = _mm_add_ps(a3, _mm_mul_ps(b3, c));
		auto d4 = _mm_add_ps(a4, _mm_mul_ps(b4, c));
		auto d5 = _mm_add_ps(a5, _mm_mul_ps(b5, c));
		auto d6 = _mm_add_ps(a6, _mm_mul_ps(b6, c));
		auto d7 = _mm_add_ps(a7, _mm_mul_ps(b7, c));
		auto d8 = _mm_add_ps(a8, _mm_mul_ps(b8, c));
		_mm_store_ps(dst, d1);
		_mm_store_ps(dst + 4, d2);
		_mm_store_ps(dst + 8, d3);
		_mm_store_ps(dst + 12, d4);
		_mm_store_ps(dst + 16, d5);
		_mm_store_ps(dst + 20, d6);
		_mm_store_ps(dst + 24, d7);
		_mm_store_ps(dst + 28, d8);
		dst += 32;
		src += 32;
	}
}

void AddMultipliedRestrictSSE4(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a1 = _mm_load_ps(dst);
		auto b1 = _mm_loadu_ps(src);
		auto a2 = _mm_load_ps(dst + 4);
		auto b2 = _mm_loadu_ps(src + 4);
		auto a3 = _mm_load_ps(dst + 8);
		auto b3 = _mm_loadu_ps(src + 8);
		auto a4 = _mm_load_ps(dst + 12);
		auto b4 = _mm_loadu_ps(src + 12);
		auto a5 = _mm_load_ps(dst + 16);
		auto b5 = _mm_loadu_ps(src + 16);
		auto a6 = _mm_load_ps(dst + 20);
		auto b6 = _mm_loadu_ps(src + 20);
		auto a7 = _mm_load_ps(dst + 24);
		auto b7 = _mm_loadu_ps(src + 24);
		auto a8 = _mm_load_ps(dst + 28);
		auto b8 = _mm_loadu_ps(src + 28);
		auto d1 = _mm_add_ps(a1, _mm_mul_ps(b1, c));
		auto d2 = _mm_add_ps(a2, _mm_mul_ps(b2, c));
		auto d3 = _mm_add_ps(a3, _mm_mul_ps(b3, c));
		auto d4 = _mm_add_ps(a4, _mm_mul_ps(b4, c));
		auto d5 = _mm_add_ps(a5, _mm_mul_ps(b5, c));
		auto d6 = _mm_add_ps(a6, _mm_mul_ps(b6, c));
		auto d7 = _mm_add_ps(a7, _mm_mul_ps(b7, c));
		auto d8 = _mm_add_ps(a8, _mm_mul_ps(b8, c));
		_mm_store_ps(dst, d1);
		_mm_store_ps(dst + 4, d2);
		_mm_store_ps(dst + 8, d3);
		_mm_store_ps(dst + 12, d4);
		_mm_store_ps(dst + 16, d5);
		_mm_store_ps(dst + 20, d6);
		_mm_store_ps(dst + 24, d7);
		_mm_store_ps(dst + 28, d8);
		dst += 32;
		src += 32;
	}
}

void AddMultipliedSSE5(float* dst, const float* src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b = _mm_loadu_ps(src);
		auto d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst, d);

		a = _mm_load_ps(dst + 4);
		b = _mm_loadu_ps(src + 4);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 4, d);

		a = _mm_load_ps(dst + 8);
		b = _mm_loadu_ps(src + 8);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 8, d);

		a = _mm_load_ps(dst + 12);
		b = _mm_loadu_ps(src + 12);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 12, d);

		a = _mm_load_ps(dst + 16);
		b = _mm_loadu_ps(src + 16);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 16, d);

		a = _mm_load_ps(dst + 20);
		b = _mm_loadu_ps(src + 20);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 20, d);

		a = _mm_load_ps(dst + 24);
		b = _mm_loadu_ps(src + 24);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 24, d);

		a = _mm_load_ps(dst + 28);
		b = _mm_loadu_ps(src + 28);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 28, d);

		dst += 32;
		src += 32;
	}
}

void AddMultipliedRestrictSSE5(float* __restrict dst, const float* __restrict src, int n, float coeff)
{
	auto c = _mm_set1_ps(coeff);
	auto dstEnd = dst + n;
	while(dst !=  dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b = _mm_loadu_ps(src);
		auto d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst, d);

		a = _mm_load_ps(dst + 4);
		b = _mm_loadu_ps(src + 4);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 4, d);

		a = _mm_load_ps(dst + 8);
		b = _mm_loadu_ps(src + 8);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 8, d);

		a = _mm_load_ps(dst + 12);
		b = _mm_loadu_ps(src + 12);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 12, d);

		a = _mm_load_ps(dst + 16);
		b = _mm_loadu_ps(src + 16);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 16, d);

		a = _mm_load_ps(dst + 20);
		b = _mm_loadu_ps(src + 20);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 20, d);

		a = _mm_load_ps(dst + 24);
		b = _mm_loadu_ps(src + 24);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 24, d);

		a = _mm_load_ps(dst + 28);
		b = _mm_loadu_ps(src + 28);
		d = _mm_add_ps(a, _mm_mul_ps(b, c));
		_mm_store_ps(dst + 28, d);

		dst += 32;
		src += 32;
	}
}


void AddMultipliedAutoSSE(float* dst, const float* src, int n, float coeff)
{
	while(n--> 0) *dst++ += coeff * *src++;
}

void AddMultipliedUnrolledAutoSSE(float* dst, const float* src, int n, float coeff)
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

void AddMultipliedUnrolledRestrictAutoSSE(float* __restrict dst, const float* __restrict src, int n, float coeff)
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

void AddMultipliedUnrolled2AutoSSE(float* dst, const float* src, int n, float coeff)
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

void AddMultipliedUnrolled2RestrictAutoSSE(float* __restrict dst, const float* __restrict src, int n, float coeff)
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


void AddMultipliedX2RestrictSSE(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2])
{
	auto c1 = _mm_set1_ps(coeffs[0]);
	auto c2 = _mm_set1_ps(coeffs[1]);
	auto dstEnd = dst + n;
	const float* p[2] = {srcs[0], srcs[1]};
	while(dst != dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b1 = _mm_load_ps(p[0]);
		auto b2 = _mm_load_ps(p[1]);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		_mm_store_ps(dst, a);
		dst += 4;
		p[0] += 4;
		p[1] += 4;
	}
}

void AddMultipliedX2RestrictSSE2(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2])
{
	auto c1 = _mm_set1_ps(coeffs[0]);
	auto c2 = _mm_set1_ps(coeffs[1]);
	auto dstEnd = dst + n;
	const float* p[2] = {srcs[0], srcs[1]};
	while(dst != dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b1 = _mm_load_ps(p[0]);
		auto b2 = _mm_load_ps(p[1]);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		_mm_store_ps(dst, a);

		a = _mm_load_ps(dst + 4);
		b1 = _mm_load_ps(p[0] + 4);
		b2 = _mm_load_ps(p[1] + 4);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		_mm_store_ps(dst + 4, a);

		dst += 8;
		p[0] += 8;
		p[1] += 8;
	}
}

void AddMultipliedX2RestrictSSE3(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2])
{
	auto c1 = _mm_set1_ps(coeffs[0]);
	auto c2 = _mm_set1_ps(coeffs[1]);
	auto dstEnd = dst + n;
	const float* p[2] = {srcs[0], srcs[1]};
	while(dst != dstEnd)
	{
		auto a1 = _mm_load_ps(dst);
		auto b11 = _mm_load_ps(p[0]);
		auto b12 = _mm_load_ps(p[1]);
		auto a2 = _mm_load_ps(dst + 4);
		auto b21 = _mm_load_ps(p[0] + 4);
		auto b22 = _mm_load_ps(p[1] + 4);
		a1 = _mm_add_ps(a1, _mm_mul_ps(b11, c1));
		a2 = _mm_add_ps(a2, _mm_mul_ps(b21, c1));
		a1 = _mm_add_ps(a1, _mm_mul_ps(b12, c2));
		_mm_store_ps(dst, a1);
		a2 = _mm_add_ps(a2, _mm_mul_ps(b22, c2));
		_mm_store_ps(dst + 4, a2);
		dst += 8;
		p[0] += 8;
		p[1] += 8;
	}
}

void AddMultipliedX2RestrictSSE4(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2])
{
	auto c1 = _mm_set1_ps(coeffs[0]);
	auto c2 = _mm_set1_ps(coeffs[1]);
	auto dstEnd = dst + n;
	const float* p[2] = {srcs[0], srcs[1]};
	while(dst != dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b1 = _mm_load_ps(p[0]);
		auto b2 = _mm_load_ps(p[1]);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		_mm_store_ps(dst, a);

		a = _mm_load_ps(dst + 4);
		b1 = _mm_load_ps(p[0] + 4);
		b2 = _mm_load_ps(p[1] + 4);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		_mm_store_ps(dst + 4, a);

		a = _mm_load_ps(dst + 8);
		b1 = _mm_load_ps(p[0] + 8);
		b2 = _mm_load_ps(p[1] + 8);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		_mm_store_ps(dst + 8, a);

		a = _mm_load_ps(dst + 12);
		b1 = _mm_load_ps(p[0] + 12);
		b2 = _mm_load_ps(p[1] + 12);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		_mm_store_ps(dst + 12, a);

		dst += 16;
		p[0] += 16;
		p[1] += 16;
	}
}

void AddMultipliedX3RestrictSSE(float* __restrict dst, const float* __restrict srcs[3], int n, const float coeffs[3])
{
	auto c1 = _mm_set1_ps(coeffs[0]);
	auto c2 = _mm_set1_ps(coeffs[1]);
	auto c3 = _mm_set1_ps(coeffs[2]);
	auto dstEnd = dst + n;
	const float* p[3] = {srcs[0], srcs[1], srcs[2]};
	while(dst != dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b1 = _mm_load_ps(p[0]);
		auto b2 = _mm_load_ps(p[1]);
		auto b3 = _mm_load_ps(p[2]);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		a = _mm_add_ps(a, _mm_mul_ps(b3, c3));
		_mm_store_ps(dst, a);
		dst += 4;
		p[0] += 4;
		p[1] += 4;
		p[2] += 4;
	}
}

void AddMultipliedX3RestrictSSE2(float* __restrict dst, const float* __restrict srcs[3], int n, const float coeffs[3])
{
	auto c1 = _mm_set1_ps(coeffs[0]);
	auto c2 = _mm_set1_ps(coeffs[1]);
	auto c3 = _mm_set1_ps(coeffs[2]);
	auto dstEnd = dst + n;
	const float* p[3] = {srcs[0], srcs[1], srcs[2]};
	while(dst != dstEnd)
	{
		auto a = _mm_load_ps(dst);
		auto b1 = _mm_load_ps(p[0]);
		auto b2 = _mm_load_ps(p[1]);
		auto b3 = _mm_load_ps(p[2]);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		a = _mm_add_ps(a, _mm_mul_ps(b3, c3));
		_mm_store_ps(dst, a);

		a = _mm_load_ps(dst + 4);
		b1 = _mm_load_ps(p[0] + 4);
		b2 = _mm_load_ps(p[1] + 4);
		b3 = _mm_load_ps(p[2] + 4);
		a = _mm_add_ps(a, _mm_mul_ps(b1, c1));
		a = _mm_add_ps(a, _mm_mul_ps(b2, c2));
		a = _mm_add_ps(a, _mm_mul_ps(b3, c3));
		_mm_store_ps(dst + 4, a);

		dst += 8;
		p[0] += 8;
		p[1] += 8;
		p[2] += 8;
	}
}

float SumSSE(const float* arr, int n)
{
	const float* arrEnd = arr + n;
	auto acc1 = _mm_load_ps(arr);
	auto acc2 = _mm_load_ps(arr + 4);
	auto acc3 = _mm_load_ps(arr + 8);
	auto acc4 = _mm_load_ps(arr + 12);
	arr += 16;
	while(arr != arrEnd)
	{
		acc1 = _mm_add_ps(acc1, _mm_load_ps(arr));
		acc2 = _mm_add_ps(acc2, _mm_load_ps(arr + 4));
		acc3 = _mm_add_ps(acc3, _mm_load_ps(arr + 8));
		acc4 = _mm_add_ps(acc4, _mm_load_ps(arr + 12));
		arr += 16;
	}
	acc1 = _mm_add_ps(acc1, acc2);
	acc3 = _mm_add_ps(acc3, acc4);
	acc1 = _mm_add_ps(acc1, acc3);
	float temp[4];
	_mm_storeu_ps(temp, acc1);
	return temp[0]+temp[1]+temp[2]+temp[3];
}

float SumSSE2(const float* arr, int n)
{
	const float* arr2 = arr + n/2;
	const float* arrEnd = arr + n;
	auto acc1 = _mm_load_ps(arr);
	auto acc2 = _mm_load_ps(arr + 4);
	auto acc3 = _mm_load_ps(arr2);
	auto acc4 = _mm_load_ps(arr2 + 4);
	arr += 8;
	arr2 += 8;
	while(arr2 != arrEnd)
	{
		acc1 = _mm_add_ps(acc1, _mm_load_ps(arr));
		acc2 = _mm_add_ps(acc2, _mm_load_ps(arr + 4));
		acc3 = _mm_add_ps(acc3, _mm_load_ps(arr2));
		acc4 = _mm_add_ps(acc4, _mm_load_ps(arr2 + 4));
		arr += 8;
		arr2 += 8;
	}
	acc1 = _mm_add_ps(acc1, acc2);
	acc3 = _mm_add_ps(acc3, acc4);
	acc1 = _mm_add_ps(acc1, acc3);
	float temp[4];
	_mm_storeu_ps(temp, acc1);
	return temp[0]+temp[1]+temp[2]+temp[3];
}

float SumSSE3(const float* arr, int n)
{
	const float* arr2 = arr + n/4;
	const float* arr3 = arr + n/4*2;
	const float* arr4 = arr + n/4*3;
	const float* arrEnd = arr + n;
	auto acc1 = _mm_load_ps(arr);
	auto acc2 = _mm_load_ps(arr2);
	auto acc3 = _mm_load_ps(arr3);
	auto acc4 = _mm_load_ps(arr4);
	arr += 4;
	arr2 += 4;
	arr3 += 4;
	arr4 += 4;
	while(arr4 != arrEnd)
	{
		acc1 = _mm_add_ps(acc1, _mm_load_ps(arr));
		acc2 = _mm_add_ps(acc2, _mm_load_ps(arr2));
		acc3 = _mm_add_ps(acc3, _mm_load_ps(arr3));
		acc4 = _mm_add_ps(acc4, _mm_load_ps(arr4));
		arr += 4;
		arr2 += 4;
		arr3 += 4;
		arr4 += 4;
	}
	acc1 = _mm_add_ps(acc1, acc2);
	acc3 = _mm_add_ps(acc3, acc4);
	acc1 = _mm_add_ps(acc1, acc3);
	float temp[4];
	_mm_storeu_ps(temp, acc1);
	return temp[0]+temp[1]+temp[2]+temp[3];
}

void AddExponentialRestrictSSE(float* __restrict dst, const float* __restrict src, int n, float coeff, float ek)
{
	const float ek2 = ek*ek;
	const float ek4 = ek2*ek2;
	const auto ek_4 = _mm_set1_ps(ek4);

	auto r1 = _mm_set_ps(coeff*ek2*ek, coeff*ek2, coeff*ek, coeff);
	auto ek_8 = ek_4;
	auto r4 = _mm_mul_ps(ek_8, r1);
	ek_8 = _mm_mul_ps(ek_8, ek_8);
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto r0 = _mm_mul_ps(_mm_load_ps(src), r1);
		r1 = _mm_mul_ps(r1, ek_8);
		auto r3 = _mm_mul_ps(_mm_load_ps(src+4), r4);
		_mm_store_ps(dst, _mm_add_ps(_mm_load_ps(dst), r0));
		r4 = _mm_mul_ps(r4, ek_8);
		_mm_store_ps(dst + 4, _mm_add_ps(_mm_load_ps(dst + 4), r3));
		dst += 8;
		src += 8;
	}
}

void AddExponentialX2RestrictSSE(float* __restrict dst, const float* __restrict src[2], int n, const float coeff[2], float ek0, float ek1)
{
	const float ek2[2] = {ek0*ek0, ek1*ek1};
	const __m128 ek_4[2] = {_mm_set1_ps(ek2[0]*ek2[0]), _mm_set1_ps(ek2[1]*ek2[1])};

	auto src0 = src[0];
	auto src1 = src[1];

	auto r1 = _mm_set_ps(coeff[0]*ek2[0]*ek0, coeff[0]*ek2[0], coeff[0]*ek0, coeff[0]);
	auto r4 = _mm_set_ps(coeff[1]*ek2[1]*ek1, coeff[1]*ek2[1], coeff[1]*ek1, coeff[1]);
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto r0 = _mm_mul_ps(_mm_load_ps(src0), r1);
		r1 = _mm_mul_ps(r1, ek_4[0]);
		auto r3 = _mm_mul_ps(_mm_load_ps(src1), r4);
		auto a = _mm_load_ps(dst);
		a = _mm_add_ps(a, r0);
		r4 = _mm_mul_ps(r4, ek_4[1]);
		a = _mm_add_ps(a, r3);
		_mm_store_ps(dst, a);
		dst += 4;
		src0 += 4;
		src1 += 4;
	}
}


void LinearMultiplyAddRestrictSSE(float* __restrict dst, const float* __restrict src, int n, float u, float du)
{
	auto u4 = _mm_set_ps(u + 3*du, u + 2*du, u + du, u);
	const auto du4 = _mm_set1_ps(4*du);
	const float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		auto vsrc = _mm_load_ps(src);
		auto vdst = _mm_load_ps(dst);
		_mm_store_ps(dst, _mm_add_ps(vdst, _mm_mul_ps(vsrc, u4)));
		u4 = _mm_add_ps(u4, du4);

		vsrc = _mm_load_ps(src + 4);
		vdst = _mm_load_ps(dst + 4);
		_mm_store_ps(dst + 4, _mm_add_ps(vdst, _mm_mul_ps(vsrc, u4)));
		u4 = _mm_add_ps(u4, du4);

		src += 8;
		dst += 8;
	}
}

void LinearMultiplyAddX2RestrictSSE(float* __restrict dst, const float* __restrict src[2], int n, const float u[2], float du0, float du1)
{
	auto u40 = _mm_set_ps(u[0] + 3*du0, u[0] + 2*du0, u[0] + du0, u[0]);
	auto u41 = _mm_set_ps(u[1] + 3*du1, u[1] + 2*du1, u[1] + du1, u[1]);
	const auto du40 = _mm_set1_ps(4*du0);
	const auto du41 = _mm_set1_ps(4*du1);
	const float* dstEnd = dst + n;
	const float* src0 = src[0];
	const float* src1 = src[1];
	while(dst != dstEnd)
	{
		auto vsrc0 = _mm_load_ps(src0);
		auto vsrc1 = _mm_load_ps(src1);
		auto vdst = _mm_load_ps(dst);
		vsrc0 = _mm_mul_ps(vsrc0, u40);
		vsrc1 = _mm_mul_ps(vsrc1, u41);
		vdst = _mm_add_ps(vdst, vsrc0);
		vdst = _mm_add_ps(vdst, vsrc1);
		_mm_store_ps(dst, vdst);
		u40 = _mm_add_ps(u40, du40);
		u41 = _mm_add_ps(u41, du41);

		src0 += 4;
		src1 += 4;
		dst += 4;
	}
}

void LinearMultiplyAddX2RestrictClassSSE(float* __restrict dst, const float* __restrict src[2], int n, const float u[2], float du0, float du1)
{
	Simd::float4 u40 = {u[0], u[0] + du0, u[0] + 2*du0, u[0] + 3*du0};
	Simd::float4 u41 = {u[1], u[1] + du1, u[1] + 2*du1, u[1] + 3*du1};
	const Simd::float4 du40 = Simd::Set<Simd::float4>(4*du0);
	const Simd::float4 du41 = Simd::Set<Simd::float4>(4*du1);
	const float* dstEnd = dst + n;
	const float* src0 = src[0];
	const float* src1 = src[1];
	while(dst != dstEnd)
	{
		auto vsrc0 = Simd::Load4Aligned(src0);
		auto vsrc1 = Simd::Load4Aligned(src1);
		auto vdst = Simd::Load4Aligned(dst);
		vsrc0 *= u40;
		vsrc1 *= u41;
		vdst += vsrc0 + vsrc1;
		Simd::StoreAligned(vdst, dst);
		u40 += du40;
		u41 += du41;

		src0 += 4;
		src1 += 4;
		dst += 4;
	}
}

/*void AddBoxBlurredSSE(float* dst, const float* src, int n, int len, float coeff)
{
	const float* first = src;
	if(len > n) len = n;

	float4 rsum{0,0,0,0};
	float4 rdiv{1/coeff, 2/coeff, 3/coeff, 4/coeff};
	float4 rdivInc{1/coeff, 1/coeff, 1/coeff, 1/coeff};
	int l = 0;
	while(l < len)
	{
		rsum += Simd::Load4Aligned(src + l);
		rdiv += rdivInc;
		float4 rdst = Simd::Load4Aligned(dst);
		rdst += rsum / rdiv;
		Store(rdst, dst);
		l += 4;
	}
	n -= len;

	const float c = coeff/l;
	while(n--> 0)
	{
		sum -= *first++;
		sum += *src++;
		*dst++ += c*sum;
	}
}*/


/*void process_ssse3(float *dst, const float *src, int n, float rate)
{
	static const float4 f0123 = {3, 2, 1, 0};
	static const float4 fone = {1, 1, 1, 1};
	static const int4 bcast = {0x0C0C0C0C, 0x08080808, 0x04040404, 0x00000000};
    static const int4 mask = Simd::Set<int4>(0x73727170);
    static const __m128i i8sign = _mm_set1_epi8(char(0x80));
    static const __m128i i8inc = _mm_set1_epi8(4);

    float4 coeff = f0123 * rate;
    int4 pos = TruncateToInt(coeff);
    coeff -= CastToFloat(pos);
    pos.Vec = _mm_add_epi8(_mm_shuffle_epi8((pos << 2).Vec, bcast.Vec), mask.Vec);

    float offs = 0;
	rate *= 4;
    int step = int(rate);
	rate -= step;
    for(int i = 0; i < n; i += 4)
    {
        float4 c = coeff + offs;
        int4 cmp = c >= fone;
        c -= float4(int4(fone) & cmp);
        sbyte16 p = _mm_add_epi8(pos.Vec, (i8inc & cmp).Vec);

		int4 src0 = int4(Simd::Load4Aligned(src));
		int4 src1 = int4(Simd::Load4Aligned(src + 4));
        float4 s0 = float4(_mm_shuffle_epi8(src0, p) | _mm_shuffle_epi8(src1, (p ^ i8sign))));
        p += i8inc;
        float4 s1 = float4(_mm_shuffle_epi8(src0, p) | _mm_shuffle_epi8(src1, (p ^ i8sign))));
        float4 res = s0 + (s1 - s0) * c;
        Simd::StoreAligned(res, dst);

        dst += 4;
		src += step;
		offs += rate;
		if(offs < 1) continue;
        src++;
		offs -= 1;
    }
}*/

template<int i0, int i1, int i2, int i3> forceinline Simd::float4 shuffle(const Simd::float4& s0, const Simd::float4& s2, const Simd::float4& s4)
{
	return Simd::Shuffle22<
		i0,
		(i2 < 4 && i3 == 3 && i0 == 0)? 1: i1,
		i2 >= 4? i2 - 4: (i3 >= 4? i2 - 2: ((i0 != 0 || i3 < 3)? i2: 2)),
		i2 >= 4? i3 - 4: (i3 >= 4? i3 - 2: ((i0 != 0 || i3 < 3)? i3: 3))
	>(s0, i2 >= 4? s4: (i3 >= 4? s2: s0));
}


template<int i1, int i2, int i3> std::enable_if_t<(i3  == 2)>
	process_sse_impl_shuffle(const float* src, const Simd::float4& src0, Simd::float4& src2, Simd::float4& src4)
{
	src2 = Simd::Load4(src + 2);
}

template<int i1, int i2, int i3> std::enable_if_t<(i3 < 2)>
	process_sse_impl_shuffle(const float* src, const Simd::float4& src0, Simd::float4& src2, Simd::float4& src4)
{
}

template<int i1, int i2, int i3> std::enable_if_t<(i3 >= 3)>
	process_sse_impl_shuffle(const float* src, const Simd::float4& src0, Simd::float4& src2, Simd::float4& src4)
{
	src4 = Simd::Load4(src + 4);
	src2 = Simd::Shuffle22<2, 3, 0, 1>(src0, src4);
}

template<int i1, int i2, int i3> void process_sse_impl(float* __restrict dst, const float* __restrict src, int n, float rate, int i4)
{
	static const Simd::float4 f0123 = {0, 1, 2, 3};
	static const Simd::float4 fone = {1, 1, 1, 1};

    Simd::float4 coeff = f0123 * rate;
	coeff -= Simd::float4{1, i1 + 1, i2 + 1, i3 + 1};

	Simd::float4 offs = {0, 0, 0, 0};
	Simd::float4 step = Simd::Set<Simd::float4>(4 * rate - i4);
    for(int i = 0; i < n; i += 4)
	{
		Simd::float4 c = coeff + offs;
		Simd::float4 dst0 = Simd::Load4Aligned(dst);
		Simd::float4 src0 = Simd::Load4(src);
		Simd::float4 src2, src4;
		process_sse_impl_shuffle<i1, i2, i3>(src, src0, src2, src4);

		offs += step;
		src += i4;
		if(offs[0] >= fone[0])
		{
			offs -= fone;
			src++;
		}

		Simd::float4 s0 = shuffle<0, i1 + 0, i2 + 0, i3 + 0>(src0, src2, src4);
		Simd::float4 s1 = shuffle<1, i1 + 1, i2 + 1, i3 + 1>(src0, src2, src4);
		Simd::float4 s2 = shuffle<2, i1 + 2, i2 + 2, i3 + 2>(src0, src2, src4);

		Simd::int4 cmp = c > 0;
		Simd::float4 s = Simd::float4(Simd::int4(s0) ^ ((Simd::int4(s0) ^ Simd::int4(s2)) & cmp));
		c = Simd::Max(c, -c);
		dst0 += s1 + (s - s1) * c;
		Simd::StoreAligned(dst0, dst);
		dst += 4;
	}
}


void AddInterpolatedSSE(float* __restrict dst, const float* __restrict src, int n, float rate)
{
    switch(int(12 * rate))
    {
    case 0: case 1: case 2: return process_sse_impl<0, 0, 0>(dst, src, n, rate, 0);
    case 3: return process_sse_impl<0, 0, 0>(dst, src, n, rate, 1);
    case 4: case 5: return process_sse_impl<0, 0, 1>(dst, src, n, rate, 1);
    case 6: case 7: return process_sse_impl<0, 1, 1>(dst, src, n, rate, 2);
    case 8: return process_sse_impl<0, 1, 2>(dst, src, n, rate, 2);
    case 9: case 10: case 11: return process_sse_impl<0, 1, 2>(dst, src, n, rate, 3);
    case 12: case 13: case 14: return process_sse_impl<1, 2, 3>(dst, src, n, rate, 4);
    case 15: return process_sse_impl<1, 2, 3>(dst, src, n, rate, 5);
	case 16: case 17: return process_sse_impl<1, 2, 4>(dst, src, n, rate, 5);
	case 18: case 19: return process_sse_impl<1, 3, 4>(dst, src, n, rate, 6);
    case 20: return process_sse_impl<1, 3, 5>(dst, src, n, rate, 6);
    case 21: case 22: case 23: return process_sse_impl<1, 3, 5>(dst, src, n, rate, 7);
    }
}


void ExponentiateAddSSE(float* __restrict dst, const float* __restrict src, int n, float dummy)
{
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		Simd::float4 rdst = Simd::Load4Aligned(dst);
		Simd::float4 rsrc = Simd::Load4Aligned(src);
		rdst += Simd::Exp(rsrc);
		Simd::StoreAligned(rdst, dst);
		dst += 4;
		src += 4;
	}
}
