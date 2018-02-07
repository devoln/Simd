#include <cmath>

void AddMultipliedScalar(float* dst, const float* src, int n, float coeff)
{
	while(n --> 0) *dst++ += coeff * *src++;
}

void AddMultipliedScalarUnrolled(float* dst, const float* src, int n, float coeff)
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

void AddMultipliedScalarUnrolledRestrict(float* __restrict dst, const float* __restrict src, int n, float coeff)
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

void AddMultipliedScalarUnrolled2(float* dst, const float* src, int n, float coeff)
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

void AddMultipliedScalarUnrolled2Restrict(float* __restrict dst, const float* __restrict src, int n, float coeff)
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


void AddBoxBlurredScalar(float* dst, const float* src, int n, int len, float coeff)
{
	const float* first = src;
	if(len > n) len = n;

	float sum = 0;
	int l = 0;
	while(l < len)
	{
		l++;
		sum += *src++;
		*dst++ += coeff*sum/l;
	}
	n -= len;

	const float c = coeff/l;
	while(n --> 0)
	{
		sum -= *first++;
		sum += *src++;
		*dst++ += c*sum;
	}
}

void AddInterpolatedScalar(float* __restrict dst, const float* __restrict src, int n, float rate)
{
	float offs = 0;
	int step = int(rate);
	rate -= step;
	for(int i = 0; i < n; i++)
	{
		const float a = src[0], b = src[1];
		dst[i] += a + (b - a) * offs;
		src += step;
		offs += rate;
		if(offs < 1) continue;
		src++;
		offs--;
	}
}

void ExponentiateAddScalar(float* __restrict dst, const float* __restrict src, int n, float dummy)
{
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		dst[0] += expf(src[0]);
		dst[1] += expf(src[1]);
		dst[2] += expf(src[2]);
		dst[3] += expf(src[3]);
		dst[4] += expf(src[4]);
		dst[5] += expf(src[5]);
		dst[6] += expf(src[6]);
		dst[7] += expf(src[7]);
		dst += 8;
		src += 8;
	}
}



inline float Pow2Scalar(float x) noexcept
{
	const float fractional_part = x - float(int(x) - (x < 0));

	float factor = float(-8.94283890931273951763e-03) + fractional_part * float(-1.89646052380707734290e-03);
	factor = float(-5.58662282412822480682e-02) + factor * fractional_part;
	factor = float(-2.40139721982230797126e-01) + factor * fractional_part;
	factor = float(3.06845249656632845792e-01) + factor * fractional_part;
	factor = float(1.06823753710239477000e-07) + factor * fractional_part;
	x -= factor;

	x *= float(1 << 23);
	x += float((1 << 23) * 127);

	int xi = int(x + 0.5f) - (x < 0);
	return reinterpret_cast<float&>(xi);
}

inline float ExpScalar(float x) noexcept
{return Pow2Scalar(x * float(1.442695040888963407359924681001892137426645954153));}


void Exponentiate2AddScalar(float* __restrict dst, const float* __restrict src, int n, float dummy)
{
	float* dstEnd = dst + n;
	while(dst != dstEnd)
	{
		dst[0] += ExpScalar(src[0]);
		dst[1] += ExpScalar(src[1]);
		dst[2] += ExpScalar(src[2]);
		dst[3] += ExpScalar(src[3]);
		dst[4] += ExpScalar(src[4]);
		dst[5] += ExpScalar(src[5]);
		dst[6] += ExpScalar(src[6]);
		dst[7] += ExpScalar(src[7]);
		dst += 8;
		src += 8;
	}
}
