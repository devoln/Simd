#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <Windows.h>
#include "Simd.h"
using namespace std;
using namespace chrono;


void AddMultipliedScalar(float* dst, const float* src, int n, float coeff);
void AddMultipliedScalarUnrolled(float* dst, const float* src, int n, float coeff);
void AddMultipliedScalarUnrolledRestrict(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedScalarUnrolled2(float* dst, const float* src, int n, float coeff);
void AddMultipliedScalarUnrolled2Restrict(float* __restrict dst, const float* __restrict src, int n, float coeff);
void ExponentiateAddScalar(float* __restrict dst, const float* __restrict src, int n, float dummy);
void Exponentiate2AddScalar(float* __restrict dst, const float* __restrict src, int n, float dummy);
void ExponentiateAddSSE(float* __restrict dst, const float* __restrict src, int n, float dummy);
void ExponentiateAddAVX(float* __restrict dst, const float* __restrict src, int n, float dummy);


void AddMultipliedSimpleSSE(float* dst, const float* src, int n, float coeff);
void AddMultipliedSSE(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictSSE(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedSSE2(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictSSE2(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedSSE3(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictSSE3(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedSSE4(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictSSE4(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedSSE5(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictSSE5(float* __restrict dst, const float* __restrict src, int n, float coeff);

void AddExponentialRestrictSSE(float* __restrict dst, const float* __restrict src, int n, float coeff, float ek = 0.999f);
void AddExponentialRestrictAVX(float* __restrict dst, const float* __restrict src, int n, float coeff, float ek = 0.999f);

void LinearMultiplyAddRestrictSSE(float* __restrict dst, const float* __restrict src, int n, float u, float du = 0.001f);
void LinearMultiplyAddRestrictAVX(float* __restrict dst, const float* __restrict src, int n, float u, float du = 0.001f);


void AddMultipliedAutoSSE(float* dst, const float* src, int n, float coeff);
void AddMultipliedUnrolledAutoSSE(float* dst, const float* src, int n, float coeff);
void AddMultipliedUnrolledRestrictAutoSSE(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedUnrolled2AutoSSE(float* dst, const float* src, int n, float coeff);
void AddMultipliedUnrolled2RestrictAutoSSE(float* __restrict dst, const float* __restrict src, int n, float coeff);

void AddMultipliedSimpleAVX(float* dst, const float* src, int n, float coeff);
void AddMultipliedAVX(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictAVX(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedAVX2(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictAVX2(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedAVX3(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictAVX3(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedRestrictPrefetch1AVX3(float* __restrict dst, const float* __restrict src, int n, float coeff, const float* prefetchSrc);
void AddMultipliedRestrictPrefetch2AVX3(float* __restrict dst, const float* __restrict src, int n, float coeff, const float* prefetchSrc);
void AddMultipliedAVX4(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictAVX4(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedAVX5(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictAVX5(float* __restrict dst, const float* __restrict src, int n, float coeff);

void AddMultipliedAutoAVX(float* dst, const float* src, int n, float coeff);
void AddMultipliedUnrolledAutoAVX(float* dst, const float* src, int n, float coeff);
void AddMultipliedUnrolledRestrictAutoAVX(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedUnrolled2AutoAVX(float* dst, const float* src, int n, float coeff);
void AddMultipliedUnrolled2RestrictAutoAVX(float* __restrict dst, const float* __restrict src, int n, float coeff);


void AddMultipliedFMA(float* dst, const float* src, int n, float coeff);
void AddMultipliedRestrictFMA(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedRestrictFMA5(float* __restrict dst, const float* __restrict src, int n, float coeff);
void AddMultipliedRestrictFMA6(float* __restrict dst, const float* __restrict src, int n, float coeff);


void AddMultipliedX2RestrictSSE(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2]);
void AddMultipliedX2RestrictSSE2(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2]);
void AddMultipliedX2RestrictSSE3(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2]);
void AddMultipliedX2RestrictSSE4(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2]);
void AddMultipliedX3RestrictSSE(float* __restrict dst, const float* __restrict srcs[3], int n, const float coeffs[3]);
void AddMultipliedX3RestrictSSE2(float* __restrict dst, const float* __restrict srcs[3], int n, const float coeffs[3]);
void AddMultipliedX2RestrictAVX(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2]);
void AddMultipliedX3RestrictAVX(float* __restrict dst, const float* __restrict srcs[3], int n, const float coeffs[3]);
void AddMultipliedX3RestrictAVXClass(float* __restrict dst, const float* __restrict srcs[3], int n, const float coeffs[3]);
void AddMultipliedX4RestrictAVX(float* __restrict dst, const float* __restrict srcs[4], int n, const float coeffs[4]);
void AddMultipliedX8RestrictAVX(float* __restrict dst, const float* __restrict srcs[8], int n, const float coeffs[8]);
void AddMultipliedX2RestrictFMA(float* __restrict dst, const float* __restrict srcs[2], int n, const float coeffs[2]);

void AddExponentialX2RestrictSSE(float* __restrict dst, const float* __restrict src[2], int n, const float coeff[2], float ek0 = 0.999f, float ek1 = 0.999f);
void AddExponentialX2RestrictAVX(float* __restrict dst, const float* __restrict src[2], int n, const float coeff[2], float ek0 = 0.999f, float ek1 = 0.999f);

void LinearMultiplyAddX2RestrictSSE(float* __restrict dst, const float* __restrict src[2], int n, const float u[2], float du0 = 0.001f, float du1 = 0.001f);
void LinearMultiplyAddX2RestrictClassSSE(float* __restrict dst, const float* __restrict src[2], int n, const float u[2], float du0 = 0.001f, float du1 = 0.001f);
void LinearMultiplyAddX2RestrictAVX(float* __restrict dst, const float* __restrict src[2], int n, const float u[2], float du0 = 0.001f, float du1 = 0.001f);


void AddInterpolatedScalar(float* __restrict dst, const float* __restrict src, int n, float rate = 1.213f);
void AddInterpolatedSSE(float* __restrict dst, const float* __restrict src, int n, float rate = 1.213f);

float SumAVX(const float* arr, int n);
float SumAVX2(const float* arr, int n);
float SumAVX3(const float* arr, int n);
float SumAVX4(const float* arr, int n);
float SumSSE(const float* arr, int n);
float SumSSE2(const float* arr, int n);
float SumSSE3(const float* arr, int n);

float Sum(const std::vector<float>& v)
{
	double sum = 0;
	for(float x: v) sum += x; 
	return float(sum);
}

template<size_t N> float Sum(float(&v)[N])
{
	double sum = 0;
	for(float x: v) sum += x;
	return float(sum);
}

#define TEST(f) {\
alignas(64) float dst[1024]{};\
if(size_t(dst) & 31) cout << "dst is not aligned!" << endl;\
auto t0 = high_resolution_clock::now();\
if(numThreads == 0) for(int k = 0; k < nChunks; k++)\
{\
	f(dst, srcs[k], chunkSize, coeffs[k]);\
}\
else for(int thread = 0; thread < numThreads; thread++)\
for(int k = 0; k < nChunks; k++)\
{\
	f(dst + thread * chunkSize / numThreads, srcs[k] + thread * chunkSize / numThreads, chunkSize / numThreads, coeffs[k]);\
}\
auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;\
auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);\
auto fps = 2*srcLen/(time/1000.0f)/1000000000;\
cout << left << setw(40) << #f << right << setw(6) << time << " ms\t"\
	<< gbs << " GB/s\tsum = " << Sum(dst) <<\
	"\t" << fps << " GFlops/s" << endl;\
}

#define TEST_PREFETCH(f) {\
alignas(64) float dst[1024]{};\
if(size_t(dst) & 31) cout << "dst is not aligned!" << endl;\
auto t0 = high_resolution_clock::now();\
for(int thread = 0; thread < numThreads; thread++)\
for(int k = 0; k < nChunks; k++)\
{\
	f(dst + thread * chunkSize / numThreads, srcs[k] + thread * chunkSize / numThreads, chunkSize / numThreads, coeffs[k], k + 2 < nChunks? srcs[k+2]: nullptr);\
}\
auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;\
auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);\
auto fps = 2*srcLen/(time/1000.0f)/1000000000;\
cout << left << setw(40) << #f << right << setw(6) << time << " ms\t"\
	<< gbs << " GB/s\tsum = " << Sum(dst) <<\
	"\t" << fps << " GFlops/s" << endl;\
}

#define TEST_X(f, inc) {\
alignas(64) float dst[1024]{};\
if(size_t(dst) & 31) cout << "dst is not aligned!" << endl;\
auto t0 = high_resolution_clock::now();\
for(int k = 0; k < nChunks; k += inc)\
{\
	f(dst, srcs + k, chunkSize, coeffs + k);\
}\
auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;\
auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);\
auto fps = 2*srcLen/(time/1000.0f)/1000000000;\
cout << left << setw(40) << #f << right << setw(6) << time << " ms\t"\
	<< gbs << " GB/s\tsum = " << Sum(dst) <<\
	"\t" << fps << " GFlops/s" << endl;\
}

#define TEST_REDUCE(f) {\
auto t0 = high_resolution_clock::now();\
auto result = f(srcData, srcLen);\
auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;\
auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);\
auto fps = srcLen/(time/1000.0f)/1000000000;\
cout << left << setw(40) << #f << right << setw(6) << time << " ms\t"\
	<< gbs << " GB/s\tresult = " << result <<\
	"\t" << fps << " GFlops/s" << endl;\
}

void RunTests(const float* srcs[], const float* coeffs, int chunkSize, int nChunks, int numThreads)
{
	const size_t srcLen = chunkSize * nChunks;

	cout << endl << "Scalar" << endl;
	TEST(AddMultipliedScalar);
	TEST(AddMultipliedScalarUnrolled);
	TEST(AddMultipliedScalarUnrolledRestrict);
	TEST(AddMultipliedScalarUnrolled2);
	TEST(AddMultipliedScalarUnrolled2Restrict);
	TEST(AddInterpolatedScalar);
	TEST(ExponentiateAddScalar);
	TEST(Exponentiate2AddScalar);

	cout << endl << "SSE" << endl;
	TEST(AddMultipliedSimpleSSE);
	TEST(AddMultipliedSSE);
	TEST(AddMultipliedRestrictSSE);
	TEST(AddMultipliedSSE2);
	TEST(AddMultipliedRestrictSSE2);
	TEST(AddMultipliedSSE3);
	TEST(AddMultipliedRestrictSSE3);
	TEST(AddMultipliedSSE4);
	TEST(AddMultipliedRestrictSSE4);
	TEST(AddMultipliedSSE5);
	TEST(AddMultipliedRestrictSSE5);
	TEST(AddExponentialRestrictSSE);
	TEST(LinearMultiplyAddRestrictSSE);

	TEST(AddInterpolatedSSE);

	TEST(AddMultipliedAutoSSE);
	TEST(AddMultipliedUnrolledAutoSSE);
	TEST(AddMultipliedUnrolledRestrictAutoSSE);
	TEST(AddMultipliedUnrolled2AutoSSE);
	TEST(AddMultipliedUnrolled2RestrictAutoSSE);
	TEST(ExponentiateAddSSE);

	cout << endl << "AVX" << endl;
	if(Simd::IsAvxSupported())
	{
		TEST(AddMultipliedSimpleAVX);
		TEST(AddMultipliedAVX);
		TEST(AddMultipliedRestrictAVX);
		TEST(AddMultipliedAVX2);
		TEST(AddMultipliedRestrictAVX2);
		TEST(AddMultipliedAVX3);
		TEST(AddMultipliedRestrictAVX3);
		TEST(AddMultipliedAVX4);
		TEST(AddMultipliedRestrictAVX4);
		TEST(AddMultipliedAVX5);
		TEST(AddMultipliedRestrictAVX5);
		TEST_PREFETCH(AddMultipliedRestrictPrefetch1AVX3);
		TEST_PREFETCH(AddMultipliedRestrictPrefetch2AVX3);
		TEST(AddExponentialRestrictAVX);
		TEST(LinearMultiplyAddRestrictAVX);

		TEST(AddMultipliedAutoAVX);
		TEST(AddMultipliedUnrolledAutoAVX);
		TEST(AddMultipliedUnrolledRestrictAutoAVX);
		TEST(AddMultipliedUnrolled2AutoAVX);
		TEST(AddMultipliedUnrolled2RestrictAutoAVX);
		TEST(ExponentiateAddAVX);
	}
	else cout << "Not supported" << endl;

	cout << endl << "FMA" << endl;
	if(Simd::IsAvxSupported())
	{
		TEST(AddMultipliedFMA);
		TEST(AddMultipliedRestrictFMA);
		TEST(AddMultipliedRestrictFMA5);
		TEST(AddMultipliedRestrictFMA6);
	}
	else cout << "Not supported" << endl;

	cout << endl << "Multiple sum" << endl;
	TEST_X(AddMultipliedX2RestrictSSE, 2);
	TEST_X(AddMultipliedX2RestrictSSE2, 2);
	TEST_X(AddMultipliedX2RestrictSSE3, 2);
	TEST_X(AddMultipliedX2RestrictSSE4, 2);
	TEST_X(AddMultipliedX3RestrictSSE, 3);
	TEST_X(AddMultipliedX3RestrictSSE2, 3);
	TEST_X(AddExponentialX2RestrictSSE, 2);
	TEST_X(AddExponentialX2RestrictAVX, 2);
	TEST_X(LinearMultiplyAddX2RestrictSSE, 2);
	TEST_X(LinearMultiplyAddX2RestrictClassSSE, 2);
	TEST_X(LinearMultiplyAddX2RestrictAVX, 2);
	if(Simd::IsAvxSupported())
	{
		TEST_X(AddMultipliedX2RestrictAVX, 2);
		TEST_X(AddMultipliedX3RestrictAVX, 3);
		TEST_X(AddMultipliedX3RestrictAVXClass, 3);
		TEST_X(AddMultipliedX4RestrictAVX, 4);
		TEST_X(AddMultipliedX8RestrictAVX, 8);
		TEST_X(AddMultipliedX2RestrictFMA, 2);
	}
}

void RunTestGroup(int nChunks, int chunkSize, const float* srcData, int srcLen, int numThreads)
{
	vector<float> coeffs(nChunks);
	for(float& c: coeffs) c = float(rand()) / float(RAND_MAX);

	vector<const float*> srcs(nChunks);


	cout << "--- RANDOM BLOCK READING" << endl;
	for(int i = 0; i < nChunks; i++)
		srcs[i] = srcData + unsigned(((rand() << 15) | rand()) * chunkSize) % srcLen;
	RunTests(srcs.data(), coeffs.data(), chunkSize, nChunks, numThreads);

	cout << endl << endl << "--- 128 STREAMS" << endl;
	for(int i = 128; i < nChunks; i++)
		srcs[i] = srcData + ((srcs[i & 127] - srcData) + i/128*chunkSize) % srcLen;
	RunTests(srcs.data(), coeffs.data(), chunkSize, nChunks, numThreads);

	cout << endl << endl << "--- 128 REV STREAMS" << endl;
	for(int i = 128; i < nChunks; i++)
		srcs[i] = srcData + ((srcs[(i & 128)? 127 - (i & 127): (i & 127)] - srcData) + i/128*chunkSize) % srcLen;
	RunTests(srcs.data(), coeffs.data(), chunkSize, nChunks, numThreads);

	cout << endl << endl << "--- 32 STREAMS" << endl;
	for(int i = 32; i < nChunks; i++)
		srcs[i] = srcData + ((srcs[i & 31] - srcData) + i/32*chunkSize) % srcLen;
	RunTests(srcs.data(), coeffs.data(), chunkSize, nChunks, numThreads);

	cout << endl << endl << "--- 32 REV STREAMS" << endl;
	for(int i = 32; i < nChunks; i++)
		srcs[i] = srcData + ((srcs[(i & 32)? 31 - (i & 31): (i & 31)] - srcData) + i/32*chunkSize) % srcLen;
	RunTests(srcs.data(), coeffs.data(), chunkSize, nChunks, numThreads);

	cout << endl << endl << "--- 16 STREAMS" << endl;
	for(int i = 16; i < nChunks; i++)
		srcs[i] = srcData + ((srcs[i & 15] - srcData) + i/16*chunkSize) % srcLen;
	RunTests(srcs.data(), coeffs.data(), chunkSize, nChunks, numThreads);

	cout << endl << endl << "--- SEQUENTIAL BLOCK READING" << endl;
	for(int i = 0; i < nChunks; i++)
		srcs[i] = srcData + i*chunkSize;
	RunTests(srcs.data(), coeffs.data(), chunkSize, nChunks, numThreads);

	cout << endl << endl << "--- SAME BLOCK READING" << endl;
	for(int i = 0; i < nChunks; i++)
		srcs[i] = srcData;
	RunTests(srcs.data(), coeffs.data(), chunkSize, nChunks, numThreads);
}

struct RandGen
{
	RandGen(unsigned seed = 157898685): Seed(seed) {}
	float operator()()
	{
		Seed *= 16807;
		return float(int(Seed)) * 4.6566129e-10f;
	}

	unsigned Seed;
} frand;

void PerfTests()
{
	const size_t nChunks = 300000;
	const size_t chunkSize = 1024;
	const size_t srcLen = chunkSize*nChunks;
	vector<float> src(srcLen + 16);
	float* srcData = src.data() + 16;

	for(size_t i = 0; i < srcLen; i++)
		srcData[i] = frand();
	
	cout << "------- 1 THREAD" << endl << endl;
	RunTestGroup(nChunks, chunkSize, srcData, srcLen, 1);

	cout << endl << "------- 4 THREADS" << endl << endl;
	//RunTestGroup(nChunks, chunkSize, srcData, srcLen, 2);

	vector<const float*> srcs(nChunks);
	cout << endl << endl << "--- 128 STREAMS" << endl;
	for(int i = 0; i < nChunks; i++)
		srcs[i] = srcData + unsigned(((rand() << 15) | rand()) * chunkSize) % srcLen;
	for(int i = 128; i < nChunks; i++)
		srcs[i] = srcData + ((srcs[i & 127] - srcData) + i/128*chunkSize) % srcLen;

	vector<float> coeffs(nChunks);
	for(float& c: coeffs) c = float(rand()) / float(RAND_MAX);

	alignas(64) float dst[1024]{};
	if(size_t(dst) & 31) cout << "dst is not aligned!" << endl;
	auto t0 = high_resolution_clock::now();
#pragma omp parallel for
	for(int thread = 0; thread < 4; thread++)
	{
		for(int k = 0; k < nChunks; k+=3)
		{
			const float* srcs1[] = {srcs[k] + thread * chunkSize / 4, srcs[k+1] + thread * chunkSize / 4, srcs[k+2] + thread * chunkSize / 4};
			AddMultipliedX3RestrictAVX(dst + thread * chunkSize / 4, srcs1, chunkSize / 4, coeffs.data() + k);
		}
	}
	auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;
	auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);
	auto fps = 2*srcLen/(time/1000.0f)/1000000000;
	cout << left << setw(40) << "AddMultipliedX3RestrictAVX" << right << setw(6) << time << " ms\t"
		<< gbs << " GB/s\tsum = " << Sum(dst) <<
		"\t" << fps << " GFlops/s" << endl;



	cout << endl << endl << "--- REDUCTION OPERATIONS" << endl;
	TEST_REDUCE(SumSSE);
	TEST_REDUCE(SumSSE2);
	TEST_REDUCE(SumSSE3);
	if(Simd::IsAvxSupported())
	{
		TEST_REDUCE(SumAVX);
		TEST_REDUCE(SumAVX2);
		TEST_REDUCE(SumAVX3);
		TEST_REDUCE(SumAVX4);

		{
			auto t0 = high_resolution_clock::now();
			float results[4];
			for(int i = 0; i<4; i++) results[i] = SumAVX3(srcData + srcLen/4*i, srcLen/4);
			auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;
			auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);
			auto fps = srcLen/(time/1000.0f)/1000000000;
			cout << left << setw(40) << "SumAVX3 4 parts" << right << setw(6) << time << " ms\t"
				<< gbs << " GB/s\tresult = " << Sum(results) <<
				"\t" << fps << " GFlops/s" << endl;
		}

		{
			auto t0 = high_resolution_clock::now();
			float results[4];
#pragma omp parallel for
			for(int i = 0; i<4; i++)
				results[i] = SumAVX3(srcData + srcLen/4*i, srcLen/4);
			auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;
			auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);
			auto fps = srcLen/(time/1000.0f)/1000000000;
			cout << left << setw(40) << "SumAVX3 4 parts MT" << right << setw(6) << time << " ms\t"
				<< gbs << " GB/s\tresult = " << Sum(results) <<
				"\t" << fps << " GFlops/s" << endl;
		}

		{
			auto t0 = high_resolution_clock::now();
			float results[2];
#pragma omp parallel for
			for(int i = 0; i<2; i++)
				results[i] = SumAVX3(srcData + srcLen/2*i, srcLen/2);
			auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;
			auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);
			auto fps = srcLen/(time/1000.0f)/1000000000;
			cout << left << setw(40) << "SumAVX3 2 parts MT" << right << setw(6) << time << " ms\t"
				<< gbs << " GB/s\tresult = " << Sum(results) <<
				"\t" << fps << " GFlops/s" << endl;
		}

		{
			auto t0 = high_resolution_clock::now();
			float results[2];
#pragma omp parallel for
			for(int i = 0; i<2; i++)
				results[i] = SumAVX4(srcData + srcLen/2*i, srcLen/2);
			auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;
			auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);
			auto fps = srcLen/(time/1000.0f)/1000000000;
			cout << left << setw(40) << "SumAVX4 2 parts MT" << right << setw(6) << time << " ms\t"
				<< gbs << " GB/s\tresult = " << Sum(results) <<
				"\t" << fps << " GFlops/s" << endl;
		}
	}

	{
		auto t0 = high_resolution_clock::now();
		float results[2];
#pragma omp parallel for
		for(int i = 0; i<2; i++)
			results[i] = SumSSE(srcData + srcLen/2*i, srcLen/2);
		auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;
		auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);
		auto fps = srcLen/(time/1000.0f)/1000000000;
		cout << left << setw(40) << "SumSSE 2 parts MT" << right << setw(6) << time << " ms\t"
			<< gbs << " GB/s\tresult = " << Sum(results) <<
			"\t" << fps << " GFlops/s" << endl;
	}

	{
		auto t0 = high_resolution_clock::now();
		float results[2];
#pragma omp parallel for
		for(int i = 0; i<2; i++)
			results[i] = SumSSE2(srcData + srcLen/2*i, srcLen/2);
		auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;
		auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);
		auto fps = srcLen/(time/1000.0f)/1000000000;
		cout << left << setw(40) << "SumSSE2 2 parts MT" << right << setw(6) << time << " ms\t"
			<< gbs << " GB/s\tresult = " << Sum(results) <<
			"\t" << fps << " GFlops/s" << endl;
	}

	{
		auto t0 = high_resolution_clock::now();
		float results[4];
#pragma omp parallel for
		for(int i = 0; i<4; i++)
			results[i] = SumSSE(srcData + srcLen/4*i, srcLen/4);
		auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;
		auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);
		auto fps = srcLen/(time/1000.0f)/1000000000;
		cout << left << setw(40) << "SumSSE 4 parts MT" << right << setw(6) << time << " ms\t"
			<< gbs << " GB/s\tresult = " << Sum(results) <<
			"\t" << fps << " GFlops/s" << endl;
	}

	{
		auto t0 = high_resolution_clock::now();
		float results[4];
#pragma omp parallel for
		for(int i = 0; i<4; i++)
			results[i] = SumSSE2(srcData + srcLen/4*i, srcLen/4);
		auto time = (high_resolution_clock::now() - t0).count()/100000/10.0f;
		auto gbs = srcLen/(time/1000.0f)*sizeof(float)/(1 << 30);
		auto fps = srcLen/(time/1000.0f)/1000000000;
		cout << left << setw(40) << "SumSSE2 4 parts MT" << right << setw(6) << time << " ms\t"
			<< gbs << " GB/s\tresult = " << Sum(results) <<
			"\t" << fps << " GFlops/s" << endl;
	}
}


void SimdTest();

int main()
{
	SimdTest();
	cout << endl;
	PerfTests();
}
