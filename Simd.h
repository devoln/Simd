#pragma once

#include <type_traits>
#include <cmath>

#if(defined(_M_AMD64) || defined(_M_X64) || defined(__amd64)) && !defined(__x86_64__)
#define __x86_64__ 1
#endif

#define SIMD_SSE_LEVEL_NONE 0
#define SIMD_SSE_LEVEL_SSE 1
#define SIMD_SSE_LEVEL_SSE2 2
#define SIMD_SSE_LEVEL_SSE3 3
#define SIMD_SSE_LEVEL_SSSE3 4
#define SIMD_SSE_LEVEL_SSE4_1 5
#define SIMD_SSE_LEVEL_SSE4_2 6
#define SIMD_SSE_LEVEL_AVX 7
#define SIMD_SSE_LEVEL_AVX2 8


#if !defined(SIMD_SSE_LEVEL) && (defined(__i386__) || defined(__i686__) || defined(_M_IX86) || defined(__x86_64__))
#ifdef __AVX2__
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_AVX2
#include <immintrin.h>
#elif defined(__AVX__)
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_AVX
#include <immintrin.h>
#elif defined(__SSE4_2__)
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_SSE4_2
#include <nmmintrin.h>
#elif defined(__SSE4_1__)
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_SSE4_1
#include <smmintrin.h>
#elif defined(__SSSE3__)
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_SSSE3
#include <tmmintrin.h>
#elif defined(__SSE3__)
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_SSE3
#include <pmmintrin.h>
#elif defined(__SSE2__) || defined(__x86_64__)
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_SSE2
#include <emmintrin.h>
#elif defined(__SSE__)
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_SSE
#include <xmmintrin.h>
#elif defined(_M_IX86_FP)

#if(_M_IX86_FP >= 2)
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_SSE2
#include <emmintrin.h>
#elif(_M_IX86_FP == 1)
#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_SSE
#include <xmmintrin.h>
#endif

#else
#define SIMD_SSE_LEVEL 0
#endif
#endif

#if(!defined(__FMA__) && !defined(__GNUC__) && !defined(__clang__) && defined(SIMD_SSE_LEVEL) && SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_AVX2)
//#define __FMA__ 1
#endif


#ifdef _MSC_VER
#define forceinline __forceinline
#if(defined(SIMD_SSE_LEVEL) && SIMD_SSE_LEVEL != 0)
#define SIMD_VECTORCALL __vectorcall
#else
#define SIMD_VECTORCALL
#endif
#elif defined(__GNUC__)
#define forceinline __inline__ __attribute__((always_inline))
#define SIMD_VECTORCALL
#else
#define forceinline inline
#define SIMD_VECTORCALL
#endif

namespace Simd {

template<class T> using ScalarTypeOf = typename std::remove_reference<decltype(T()[0])>::type;
template<class T> using IntAnalogOf = typename std::remove_reference<decltype(T() < T())>::type;

}

#if defined(__GNUC__) || defined(__clang__)

namespace Simd {

typedef int int4 __attribute__((vector_size(16)));
typedef float float4 __attribute__((vector_size(16)));

typedef int int8 __attribute__((vector_size(32)));
typedef float float8 __attribute__((vector_size(32)));

#define SIMD_INT4_SUPPORT
#define SIMD_FLOAT4_SUPPORT
#define SIMD_INT8_SUPPORT
#define SIMD_FLOAT8_SUPPORT

template<typename T> forceinline T Set(ScalarTypeOf<T> v) noexcept
{
	T result;
	for(int i = 0; i<sizeof(result)/sizeof(result[0]); i++) result[i] = v;
	return result;
}

forceinline int4 Load4(const int* src) {return int4{src[0], src[1], src[2], src[3]};}
forceinline int4 Load4Aligned(const int* src) {return int4{src[0], src[1], src[2], src[3]};}
forceinline float4 Load4(const float* src) {return float4{src[0], src[1], src[2], src[3]};}
forceinline float4 Load4Aligned(const float* src) {return float4{src[0], src[1], src[2], src[3]};}
forceinline int8 Load8(const int* src) {return int8{src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]};}
forceinline int8 Load8Aligned(const int* src) {return int8{src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]};}
forceinline float8 Load8(const float* src) {return float8{src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]};}
forceinline float8 Load8Aligned(const float* src) {return float8{src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7]};}
template<typename T> forceinline void Store(T v, ScalarTypeOf<T>* dst) {for(int i = 0; i<sizeof(v)/sizeof(v[0]); i++) dst[i] = v[i];}
template<typename T> forceinline void StoreAligned(T v, ScalarTypeOf<T>* dst) {for(int i = 0; i<sizeof(v)/sizeof(v[0]); i++) dst[i] = v[i];}

template<int i0, int i1, int i2, int i3, typename T> forceinline T Shuffle(T v) noexcept
{
	static_assert(
		0 <= i0 && i0 <= 3 &&
		0 <= i1 && i1 <= 3 &&
		0 <= i2 && i2 <= 3 &&
		0 <= i3 && i3 <= 3,
		"Valid range of shuffle indices is [0; 3]");
#ifdef __clang__
	return __builtin_shufflevector(v, int4{i0, i1, i2, i3});
#else
	return __builtin_shuffle(v, int4{i0,i1,i2,i3});
#endif
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, typename T> forceinline T Shuffle(const T& v) noexcept
{
	static_assert(
		0 <= i0 && i0 <= 7 &&
		0 <= i1 && i1 <= 7 &&
		0 <= i2 && i2 <= 7 &&
		0 <= i3 && i3 <= 7 &&
		0 <= i4 && i4 <= 7 &&
		0 <= i5 && i5 <= 7 &&
		0 <= i6 && i6 <= 7 &&
		0 <= i7 && i7 <= 7,
		"Valid range of shuffle indices is [0; 7]");
	return __builtin_shuffle(v, int8{i0,i1,i2,i3,i4,i5,i6,i7});
}

template<int n, typename T> forceinline std::enable_if_t<sizeof(T)/sizeof(T{}[0]) == 4, T> RotateLeft(const T& v) noexcept
{return Shuffle<n & 3, (n + 1) & 3, (n + 2) & 3, (n + 3) & 3>(v);}

template<int n, typename T> forceinline std::enable_if_t<sizeof(T)/sizeof(T{}[0]) == 8, T> RotateLeft(const T& v) noexcept
{return Shuffle<n & 7, (n + 1) & 7, (n + 2) & 7, (n + 3) & 7, (n + 4) & 7, (n + 5) & 7, (n + 6) & 7, (n + 7) & 7>(v);}

template<int n, typename T> forceinline T RotateRight(const T& v) noexcept {return RotateLeft<-n>(v);}

template<typename T> forceinline T Shuffle(T v, IntAnalogOf<T> indices) noexcept
{return __builtin_shuffle(v, indices);}

#ifdef __clang__
template<typename T> forceinline T Min(T a, T b) noexcept
{return ((a < b) & IntAnalogOf<T>(a)) | (~(a < b) & IntAnalogOf<T>(b));}

template<typename T> forceinline T Max(T a, T b) noexcept
{return ((a > b) & IntAnalogOf<T>(a)) | (~(a > b) & IntAnalogOf<T>(b));}
#else
template<typename T> forceinline T Min(T a, T b) noexcept {return a < b? a: b;}
template<typename T> forceinline T Max(T a, T b) noexcept {return a > b? a: b;}
#endif

forceinline int4 TruncateToInt(float4 v) noexcept
{return int4{int(v[0]), int(v[1]), int(v[2]), int(v[3])};}

forceinline float4 CastToFloat(int4 v) noexcept
{return float4{float(v[0]), float(v[1]), float(v[2]), float(v[3])};}

forceinline int8 TruncateToInt(float8 v) noexcept
{return int8{int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]), int(v[5]), int(v[6]), int(v[7])};}

forceinline float8 CastToFloat(int8 v) noexcept
{return float8{float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5]), float(v[6]), float(v[7])};}

template<int i0a, int i1a, int i2b, int i3b, typename T> forceinline T Shuffle22(const T& a, const T& b) noexcept
{
	static_assert(
		0 <= i0a && i0a <= 3 &&
		0 <= i1a && i1a <= 3 &&
		0 <= i2b && i2b <= 3 &&
		0 <= i3b && i3b <= 3,
		"Valid range of shuffle indices is [0; 3]");
	return T{a[i0a], a[i1a], b[i2b], b[i3b]};
}

forceinline void End() noexcept {}

forceinline int4 SIMD_VECTORCALL UnsignedRightBitShift(int4 x, int bits) noexcept
{return int4{int(unsigned(x[0]) >> bits), int(unsigned(x[1]) >> bits), int(unsigned(x[2]) >> bits), int(unsigned(x[3]) >> bits)};}

forceinline int8 SIMD_VECTORCALL UnsignedRightBitShift(int8 x, int bits) noexcept
{
	return int8{
		int(unsigned(x[0]) >> bits), int(unsigned(x[1]) >> bits), int(unsigned(x[2])) >> bits, int(unsigned(x[3]) >> bits),
		int(unsigned(x[4]) >> bits), int(unsigned(x[5]) >> bits), int(unsigned(x[6]) >> bits), int(unsigned(x[7]) >> bits)
	};
}

}

#else



#define SIMD_DEFINE_EXPRESSION_CHECKER(checker_name, expr) \
	struct D ## checker_name ## _base {\
		template<typename T> static decltype((expr), short()) func(::std::remove_reference_t<T>*);\
		template<typename T> static char func(...);\
	};\
	template<typename U> struct checker_name: D ## checker_name ## _base \
	{enum {_ = sizeof(func<U>(nullptr)) == sizeof(short)};}

namespace Simd {
SIMD_DEFINE_EXPRESSION_CHECKER(IsSimdWrapperType, T::VectorSize);
template<typename T, typename U = void> using EnableForSimdWrapper = std::enable_if_t<IsSimdWrapperType<T>::_, U>;


#ifdef SIMD_SSE_LEVEL

template<typename T> forceinline T SIMD_VECTORCALL Set(ScalarTypeOf<T> v) noexcept {return T::Set(v);}

#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_SSE2)

#define SIMD_INT4_SUPPORT

struct float4;

struct int4
{
	__m128i Vec;

	enum {VectorSize = 4};

	forceinline int4() = default;
	forceinline int4(int x, int y, int z, int w) noexcept: Vec(_mm_set_epi32(w, z, y, x)) {}

	forceinline int4 SIMD_VECTORCALL operator+(int4 rhs) const noexcept {return _mm_add_epi32(Vec, rhs.Vec);}
	forceinline int4 SIMD_VECTORCALL operator-(int4 rhs) const noexcept {return _mm_sub_epi32(Vec, rhs.Vec);}
	forceinline int4 SIMD_VECTORCALL operator*(int4 rhs) const noexcept
	{
#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_SSE4_1)
		return _mm_mullo_epi32(Vec, rhs.Vec);
#else
		__m128i tmp1 = _mm_mul_epu32(Vec, rhs.Vec);
		__m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(Vec, 4), _mm_srli_si128(rhs.Vec, 4));
		return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0)));
#endif
	}

	forceinline int4 SIMD_VECTORCALL operator/(int4 rhs) const
	{
		alignas(16) int a[4], b[4];
		_mm_store_si128(reinterpret_cast<__m128i*>(a), Vec);
		_mm_store_si128(reinterpret_cast<__m128i*>(b), rhs.Vec);
		return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]};
	}

	forceinline int4 SIMD_VECTORCALL operator&(int4 rhs) const noexcept {return _mm_and_si128(Vec, rhs.Vec);}
	forceinline int4 SIMD_VECTORCALL operator|(int4 rhs) const noexcept {return _mm_or_si128(Vec, rhs.Vec);}
	forceinline int4 SIMD_VECTORCALL operator^(int4 rhs) const noexcept {return _mm_xor_si128(Vec, rhs.Vec);}
	forceinline int4 SIMD_VECTORCALL operator~() const noexcept {return _mm_xor_si128(Vec, _mm_set1_epi32(-1));}

	forceinline int4 SIMD_VECTORCALL operator<<(int bits) const noexcept {return _mm_slli_epi32(Vec, bits);}
	forceinline int4 SIMD_VECTORCALL operator>>(int bits) const noexcept {return _mm_srai_epi32(Vec, bits);}

	forceinline int4 SIMD_VECTORCALL operator<<(int4 rhs) const noexcept
	{
#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_AVX2)
		return _mm_srlv_epi32(Vec, rhs.Vec);
#else
		alignas(16) int a[4], b[4];
		_mm_store_si128(reinterpret_cast<__m128i*>(a), Vec);
		_mm_store_si128(reinterpret_cast<__m128i*>(b), rhs.Vec);
		return {a[0] << b[0], a[1] << b[1], a[2] << b[2], a[3] << b[3]};
#endif
	}

	forceinline int4 SIMD_VECTORCALL operator>>(int4 rhs) const noexcept
	{
#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_AVX2)
		return _mm_srav_epi32(Vec, rhs.Vec);
#else
		alignas(16) int a[4], b[4];
		_mm_store_si128(reinterpret_cast<__m128i*>(a), Vec);
		_mm_store_si128(reinterpret_cast<__m128i*>(b), rhs.Vec);
		return {a[0] >> b[0], a[1] >> b[1], a[2] >> b[2], a[3] >> b[3]};
#endif
	}


	forceinline int4 SIMD_VECTORCALL operator>(int4 rhs) const noexcept {return _mm_cmpgt_epi32(Vec, rhs.Vec);}
	forceinline int4 SIMD_VECTORCALL operator<(int4 rhs) const noexcept {return _mm_cmplt_epi32(Vec, rhs.Vec);}
	forceinline int4 SIMD_VECTORCALL operator>=(int4 rhs) const noexcept {return ~operator<(rhs);}
	forceinline int4 SIMD_VECTORCALL operator<=(int4 rhs) const noexcept {return ~operator>(rhs);}
	forceinline int4 SIMD_VECTORCALL operator==(int4 rhs) const noexcept {return _mm_cmpeq_epi32(Vec, rhs.Vec);}
	forceinline int4 SIMD_VECTORCALL operator!=(int4 rhs) const noexcept {return ~operator==(rhs);}

	forceinline int operator[](int i) const
	{
		alignas(16) int arr[4];
		_mm_store_si128(reinterpret_cast<__m128i*>(arr), Vec);
		return arr[i];
	}

	forceinline int4(__m128i vec) noexcept: Vec(vec) {}
	forceinline operator __m128i() const {return Vec;}

	static forceinline int4 Set(int v) noexcept {return _mm_set1_epi32(v);}
};

forceinline int4 SIMD_VECTORCALL Load4(const int* src) {return _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));}
forceinline int4 SIMD_VECTORCALL Load4Aligned(const int* src) {return _mm_load_si128(reinterpret_cast<const __m128i*>(src));}
forceinline void SIMD_VECTORCALL Store(int4 v, int* dst) {_mm_storeu_si128(reinterpret_cast<__m128i*>(dst), v.Vec);}
forceinline void SIMD_VECTORCALL StoreAligned(int4 v, int* dst) {_mm_store_si128(reinterpret_cast<__m128i*>(dst), v.Vec);}

template<int i0, int i1, int i2, int i3> forceinline int4 SIMD_VECTORCALL Shuffle(int4 v) noexcept
{
	static_assert(
		0 <= i0 && i0 <= 3 &&
		0 <= i1 && i1 <= 3 &&
		0 <= i2 && i2 <= 3 &&
		0 <= i3 && i3 <= 3,
		"Valid range of shuffle indices is [0; 3]");
	return _mm_shuffle_epi32(v.Vec, v.Vec, _MM_SHUFFLE(i3, i2, i1, i0));
}

forceinline int4 SIMD_VECTORCALL Shuffle(int4 v, int4 indices) noexcept
{
#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_SSSE3)
	indices = indices << 2;
	indices.Vec = _mm_shuffle_epi8(indices.Vec, _mm_set_epi8(12,12,12,12,8,8,8,8,4,4,4,4,0,0,0,0));
	indices.Vec = _mm_add_epi8(indices.Vec, _mm_set_epi8(3,2,1,0,3,2,1,0,3,2,1,0,3,2,1,0));
	return _mm_shuffle_epi8(v.Vec, indices.Vec);
#else
	return {v[indices[0]], v[indices[1]], v[indices[2]], v[indices[3]]};
#endif
}

template<int n> forceinline int4 SIMD_VECTORCALL RotateLeft(int4 v) noexcept
{return Shuffle<n & 3, (n + 1) & 3, (n + 2) & 3, (n + 3) & 3>(v);}

template<int n> forceinline int4 SIMD_VECTORCALL RotateRight(int4 v) noexcept {return RotateLeft<-n>(v);}

forceinline int4 SIMD_VECTORCALL Min(int4 a, int4 b) noexcept
{
#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_SSE4_1)
	return _mm_min_epi32(a.Vec, b.Vec);
#else
	__m128i mask = _mm_cmplt_epi32(a.Vec, b.Vec);
	return _mm_or_si128(_mm_and_si128(a.Vec, mask), _mm_andnot_si128(mask, b.Vec));
#endif
}

forceinline int4 SIMD_VECTORCALL Max(int4 a, int4 b) noexcept
{
#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_SSE4_1)
	return _mm_max_epi32(a.Vec, b.Vec);
#else
	__m128i mask = _mm_cmpgt_epi32(a.Vec, b.Vec);
	return _mm_or_si128(_mm_and_si128(a.Vec, mask), _mm_andnot_si128(mask, b.Vec));
#endif
}

forceinline int SIMD_VECTORCALL HorSum(int4 v) noexcept {return v[0] + v[1] + v[2] + v[3];}

forceinline int4 SIMD_VECTORCALL UnsignedRightBitShift(int4 x, int bits) noexcept {return _mm_srli_epi32(x.Vec, bits);}

#define SIMD_FLOAT4_SUPPORT

struct float4
{
	__m128 Vec;

	enum {VectorSize = 4};

	forceinline float4() = default;
	forceinline float4(float x, float y, float z, float w) noexcept: Vec(_mm_set_ps(w, z, y, x)) {}

	forceinline float4 SIMD_VECTORCALL operator+(float4 rhs) const noexcept {return _mm_add_ps(Vec, rhs.Vec);}
	forceinline float4 SIMD_VECTORCALL operator-(float4 rhs) const noexcept {return _mm_sub_ps(Vec, rhs.Vec);}
	forceinline float4 SIMD_VECTORCALL operator*(float4 rhs) const noexcept {return _mm_mul_ps(Vec, rhs.Vec);}
	forceinline float4 SIMD_VECTORCALL operator/(float4 rhs) const {return _mm_div_ps(Vec, rhs.Vec);}

	forceinline int4 SIMD_VECTORCALL operator>(float4 rhs) const {return int4(float4(_mm_cmpgt_ps(Vec, rhs.Vec)));}
	forceinline int4 SIMD_VECTORCALL operator<(float4 rhs) const {return int4(float4(_mm_cmplt_ps(Vec, rhs.Vec)));}
	forceinline int4 SIMD_VECTORCALL operator>=(float4 rhs) const {return int4(float4(_mm_cmpge_ps(Vec, rhs.Vec)));}
	forceinline int4 SIMD_VECTORCALL operator<=(float4 rhs) const {return int4(float4(_mm_cmple_ps(Vec, rhs.Vec)));}
	forceinline int4 SIMD_VECTORCALL operator==(float4 rhs) const noexcept {return int4(float4(_mm_cmpeq_ps(Vec, rhs.Vec)));}
	forceinline int4 SIMD_VECTORCALL operator!=(float4 rhs) const noexcept {return int4(float4(_mm_cmpneq_ps(Vec, rhs.Vec)));}

	forceinline explicit float4(const int4& v) noexcept: Vec(_mm_castsi128_ps(v.Vec)) {}
	forceinline explicit operator int4() const noexcept {return _mm_castps_si128(Vec);}

	forceinline float operator[](int i) const
	{
		alignas(16) float arr[4];
		_mm_store_ps(arr, Vec);
		return arr[i];
	}

	forceinline operator __m128() const noexcept {return Vec;}
	forceinline float4(__m128 vec) noexcept: Vec(vec) {}

	static forceinline float4 Set(float v) noexcept {return _mm_set1_ps(v);}
};

forceinline float4 SIMD_VECTORCALL Load4(const float* src) noexcept {return _mm_loadu_ps(src);}
forceinline float4 SIMD_VECTORCALL Load4Aligned(const float* src) noexcept {return _mm_load_ps(src);}
forceinline void SIMD_VECTORCALL Store(float4 v, float* dst) noexcept {return _mm_storeu_ps(dst, v.Vec);}
forceinline void SIMD_VECTORCALL StoreAligned(float4 v, float* dst) noexcept {return _mm_store_ps(dst, v.Vec);}

forceinline int4 SIMD_VECTORCALL TruncateToInt(float4 v) noexcept
{return _mm_cvttps_epi32(v.Vec);}

forceinline float4 SIMD_VECTORCALL CastToFloat(int4 i4) noexcept {return _mm_cvtepi32_ps(i4.Vec);}

template<int i0a, int i1a, int i2b, int i3b> forceinline int4 SIMD_VECTORCALL Shuffle22(int4 a, int4 b) noexcept
{
	static_assert(
		0 <= i0a && i0a <= 3 &&
		0 <= i1a && i1a <= 3 &&
		0 <= i2b && i2b <= 3 &&
		0 <= i3b && i3b <= 3,
		"Valid range of shuffle indices is [0; 3]");
	return _mm_shuffle_epi32(a.Vec, b.Vec, _MM_SHUFFLE(i3b, i2b, i1a, i0a));
}

template<int i0a, int i1a, int i2b, int i3b> forceinline float4 SIMD_VECTORCALL Shuffle22(float4 a, float4 b) noexcept
{
	static_assert(
		0 <= i0a && i0a <= 3 &&
		0 <= i1a && i1a <= 3 &&
		0 <= i2b && i2b <= 3 &&
		0 <= i3b && i3b <= 3,
		"Valid range of shuffle indices is [0; 3]");
	return _mm_shuffle_ps(a.Vec, b.Vec, _MM_SHUFFLE(i3b, i2b, i1a, i0a));
}

template<int i0, int i1, int i2, int i3> forceinline float4 SIMD_VECTORCALL Shuffle(float4 v) noexcept
{
	static_assert(
		0 <= i0 && i0 <= 3 &&
		0 <= i1 && i1 <= 3 &&
		0 <= i2 && i2 <= 3 &&
		0 <= i3 && i3 <= 3,
		"Valid range of shuffle indices is [0; 3]");
	return _mm_shuffle_ps(v.Vec, v.Vec, _MM_SHUFFLE(i3, i2, i1, i0));
}

forceinline float4 SIMD_VECTORCALL Shuffle(float4 v, int4 indices) noexcept
{
#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_SSSE3)
	indices = indices << 2;
	indices.Vec = _mm_shuffle_epi8(indices.Vec, _mm_set_epi8(12,12,12,12,8,8,8,8,4,4,4,4,0,0,0,0));
	indices.Vec = _mm_add_epi8(indices.Vec, _mm_set_epi8(3,2,1,0,3,2,1,0,3,2,1,0,3,2,1,0));
	return float4(int4(_mm_shuffle_epi8(int4(v).Vec, indices.Vec)));
#else
	return {v[indices[0]], v[indices[1]], v[indices[2]], v[indices[3]]};
#endif
}

template<int n> forceinline float4 SIMD_VECTORCALL RotateLeft(float4 v) noexcept
{return Shuffle<n & 3, (n + 1) & 3, (n + 2) & 3, (n + 3) & 3>(v);}

template<int n> forceinline float4 SIMD_VECTORCALL RotateRight(float4 v) noexcept {return RotateLeft<-n>(v);}

forceinline float SIMD_VECTORCALL HorSum(float4 v) noexcept
{
#if SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_SSE3
	__m128 tmp0 = _mm_hadd_ps(v.Vec, v.Vec);
	__m128 tmp1 = _mm_hadd_ps(tmp0, tmp0);
#else
	__m128 tmp0 = _mm_add_ps(v.Vec, _mm_movehl_ps(v.Vec, v.Vec));
	__m128 tmp1 = _mm_add_ss(tmp0, _mm_shuffle_ps(tmp0, tmp0, 1));
#endif
	return _mm_cvtss_f32(tmp1);
}

forceinline float4 SIMD_VECTORCALL Max(float4 a, float4 b) noexcept {return _mm_max_ps(a.Vec, b.Vec);}
forceinline float4 SIMD_VECTORCALL Min(float4 a, float4 b) noexcept {return _mm_min_ps(a.Vec, b.Vec);}

#endif

#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_AVX2)

#define SIMD_INT8_SUPPORT

struct float8;

struct int8
{
	__m256i Vec;

	enum {VectorSize = 8};

	forceinline int8() = default;
	forceinline int8(int x1, int x2, int x3, int x4, int x5, int x6, int x7, int x8) noexcept:
		Vec(_mm256_set_epi32(x8, x7, x6, x5, x4, x3, x2, x1)) {}

	forceinline int8 SIMD_VECTORCALL operator+(int8 rhs) const noexcept {return _mm256_add_epi32(Vec, rhs.Vec);}
	forceinline int8 SIMD_VECTORCALL operator-(int8 rhs) const noexcept {return _mm256_sub_epi32(Vec, rhs.Vec);}
	forceinline int8 SIMD_VECTORCALL operator*(int8 rhs) const noexcept {return _mm256_mul_epi32(Vec, rhs.Vec);}
	forceinline int8 SIMD_VECTORCALL operator/(int8 rhs) const
	{
		alignas(32) int a[8], b[8];
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(a), Vec);
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(b), rhs.Vec);
		return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3], a[4] / b[4], a[5] / b[5], a[6] / b[6], a[7] / b[7]};
	}

	forceinline int8 SIMD_VECTORCALL operator&(int8 rhs) const noexcept {return _mm256_and_si256(Vec, rhs.Vec);}
	forceinline int8 SIMD_VECTORCALL operator|(int8 rhs) const noexcept {return _mm256_or_si256(Vec, rhs.Vec);}
	forceinline int8 SIMD_VECTORCALL operator^(int8 rhs) const noexcept {return _mm256_xor_si256(Vec, rhs.Vec);}
	forceinline int8 SIMD_VECTORCALL operator~() const noexcept {return _mm256_xor_si256(Vec, _mm256_set1_epi32(-1));}

	forceinline int8 SIMD_VECTORCALL operator<<(int bits) const noexcept {return _mm256_slli_epi32(Vec, bits);}
	forceinline int8 SIMD_VECTORCALL operator>>(int bits) const noexcept {return _mm256_srai_epi32(Vec, bits);}

	forceinline int8 SIMD_VECTORCALL operator<<(int8 rhs) const noexcept {return _mm256_srlv_epi32(Vec, rhs.Vec);}
	forceinline int8 SIMD_VECTORCALL operator>>(int8 rhs) const noexcept {return _mm256_srav_epi32(Vec, rhs.Vec);}


	forceinline int8 SIMD_VECTORCALL operator>(int8 rhs) const noexcept {return _mm256_cmpgt_epi32(Vec, rhs.Vec);}
	forceinline int8 SIMD_VECTORCALL operator<(int8 rhs) const noexcept {return rhs < *this;}
	forceinline int8 SIMD_VECTORCALL operator>=(int8 rhs) const noexcept {return ~(rhs > *this);}
	forceinline int8 SIMD_VECTORCALL operator<=(int8 rhs) const noexcept {return ~operator>(rhs);}
	forceinline int8 SIMD_VECTORCALL operator==(int8 rhs) const noexcept {return _mm256_cmpeq_epi32(Vec, rhs.Vec);}
	forceinline int8 SIMD_VECTORCALL operator!=(int8 rhs) const noexcept {return ~operator==(rhs);}

	forceinline int operator[](int i) const
	{
		alignas(32) int arr[8];
		_mm256_store_si256(reinterpret_cast<__m256i*>(arr), Vec);
		return arr[i];
	}

	forceinline int8(__m256i vec) noexcept: Vec(vec) {}
	forceinline operator __m256i() const {return Vec;}

	static forceinline int8 SIMD_VECTORCALL Set(int v) noexcept {return _mm256_set1_epi32(v);}
};

forceinline int8 SIMD_VECTORCALL Load8(const int* src) noexcept {return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));}
forceinline int8 SIMD_VECTORCALL Load8Aligned(const int* src) noexcept {return _mm256_load_si256(reinterpret_cast<const __m256i*>(src));}

forceinline void SIMD_VECTORCALL Store(int8 v, int* dst) {_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), v.Vec);}
forceinline void SIMD_VECTORCALL StoreAligned(int8 v, int* dst) {_mm256_store_si256(reinterpret_cast<__m256i*>(dst), v.Vec);}

template<int i0, int i1, int i2, int i3> forceinline int8 SIMD_VECTORCALL Shuffle(int8 v) noexcept
{
	static_assert(
		0 <= i0 && i0 <= 7 &&
		0 <= i1 && i1 <= 7 &&
		0 <= i2 && i2 <= 7 &&
		0 <= i3 && i3 <= 7 &&
		0 <= i4 && i4 <= 7 &&
		0 <= i5 && i5 <= 7 &&
		0 <= i6 && i6 <= 7 &&
		0 <= i7 && i7 <= 7,
		"Valid range of shuffle indices is [0; 7]");
	__m128i h = _mm256_extractf128_si256(v.Vec, 0);
	__m128i l = _mm256_extractf128_si256(v.Vec, 1);
	//TODO: проверить, что порядок правильный
	l = _mm_shuffle_ps(l, l, _MM_SHUFFLE(i3, i2, i1, i0));
	h = _mm_shuffle_ps(h, h, _MM_SHUFFLE(i7, i6, i5, i4));
	return _mm256_set_m128i(h, l);
}

template<int n> forceinline int8 SIMD_VECTORCALL RotateLeft(int8 v) noexcept
{return Shuffle<n & 7, (n + 1) & 7, (n + 2) & 7, (n + 3) & 7, (n + 4) & 7, (n + 5) & 7, (n + 6) & 7, (n + 7) & 7>();}

template<int n> forceinline int8 SIMD_VECTORCALL RotateRight(int8 v) noexcept {return RotateLeft<-n>(v);}

forceinline int8 SIMD_VECTORCALL Max(int8 a, int8 b) noexcept {return _mm256_max_epi32(a.Vec, b.Vec);}
forceinline int8 SIMD_VECTORCALL Min(int8 a, int8 b) noexcept {return _mm256_min_epi32(a.Vec, b.Vec);}

forceinline int8 SIMD_VECTORCALL UnsignedRightBitShift(int8 x, int bits) noexcept {return _mm256_srli_epi32(x.Vec, bits);}

#endif

#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_AVX)

#define SIMD_FLOAT8_SUPPORT

struct float8
{
	__m256 Vec;

	enum {VectorSize = 8};

	forceinline float8() = default;
	forceinline float8(float x1, float x2, float x3, float x4, float x5, float x6, float x7, float x8) noexcept:
		Vec(_mm256_set_ps(x8, x7, x6, x5, x4, x3, x2, x1)) {}

	forceinline float8 SIMD_VECTORCALL operator+(float8 rhs) const noexcept {return _mm256_add_ps(Vec, rhs.Vec);}
	forceinline float8 SIMD_VECTORCALL operator-(float8 rhs) const noexcept {return _mm256_sub_ps(Vec, rhs.Vec);}
	forceinline float8 SIMD_VECTORCALL operator*(float8 rhs) const noexcept {return _mm256_mul_ps(Vec, rhs.Vec);}
	forceinline float8 SIMD_VECTORCALL operator/(float8 rhs) const {return _mm256_div_ps(Vec, rhs.Vec);}

#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_AVX2)
	forceinline int8 SIMD_VECTORCALL operator>(float8 rhs) const {return int8(float8(_mm256_cmp_ps(Vec, rhs.Vec, _CMP_GT_OQ)));}
	forceinline int8 SIMD_VECTORCALL operator<(float8 rhs) const {return int8(float8(_mm256_cmp_ps(Vec, rhs.Vec, _CMP_LT_OQ)));}
	forceinline int8 SIMD_VECTORCALL operator>=(float8 rhs) const {return int8(float8(_mm256_cmp_ps(Vec, rhs.Vec, _CMP_GE_OQ)));}
	forceinline int8 SIMD_VECTORCALL operator<=(float8 rhs) const {return int8(float8(_mm256_cmp_ps(Vec, rhs.Vec, _CMP_LE_OQ)));}
	forceinline int8 SIMD_VECTORCALL operator==(float8 rhs) const noexcept {return int8(float8(_mm256_cmp_ps(Vec, rhs.Vec, _CMP_EQ_OQ)));}
	forceinline int8 SIMD_VECTORCALL operator!=(float8 rhs) const noexcept {return int8(float8(_mm256_cmp_ps(Vec, rhs.Vec, _CMP_NEQ_OQ)));}

	forceinline explicit float8(const int8& v): Vec(_mm256_castsi256_ps(v.Vec)) {}
	forceinline explicit operator int8() const noexcept {return _mm256_castps_si256(Vec);}
#endif

	forceinline float operator[](int i) const
	{
		alignas(32) float arr[8];
		_mm256_store_ps(arr, Vec);
		return arr[i];
	}

	forceinline operator __m256() const noexcept {return Vec;}
	forceinline float8(__m256 vec) noexcept: Vec(vec) {}

	static forceinline float8 SIMD_VECTORCALL Set(float v) noexcept {return _mm256_set1_ps(v);}
};

forceinline float8 SIMD_VECTORCALL Load8(const float* src) {return _mm256_loadu_ps(src);}
forceinline float8 SIMD_VECTORCALL Load8Aligned(const float* src) {return _mm256_load_ps(src);}
forceinline void SIMD_VECTORCALL Store(float8 v, float* dst) {_mm256_storeu_ps(dst, v.Vec);}
forceinline void SIMD_VECTORCALL StoreAligned(float8 v, float* dst) {_mm256_store_ps(dst, v.Vec);}

forceinline void End() noexcept {_mm256_zeroupper();}

forceinline float8 SIMD_VECTORCALL Max(float8 a, float8 b) noexcept {return _mm256_max_ps(a.Vec, b.Vec);}
forceinline float8 SIMD_VECTORCALL Min(float8 a, float8 b) noexcept {return _mm256_min_ps(a.Vec, b.Vec);}

forceinline float SIMD_VECTORCALL HorSum(float8 v)
{
	__m256 tmp0 = _mm256_hadd_ps(v.Vec, v.Vec);
	__m256 tmp1 = _mm256_hadd_ps(tmp0, tmp0);
	alignas(32) float arr[8];
	_mm256_store_ps(arr, tmp1);
	return arr[0] + arr[4];
}

template<int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7> forceinline
float8 SIMD_VECTORCALL Shuffle(float8 v) noexcept
{
	static_assert(
		0 <= i0 && i0 <= 7 &&
		0 <= i1 && i1 <= 7 &&
		0 <= i2 && i2 <= 7 &&
		0 <= i3 && i3 <= 7 &&
		0 <= i4 && i4 <= 7 &&
		0 <= i5 && i5 <= 7 &&
		0 <= i6 && i6 <= 7 &&
		0 <= i7 && i7 <= 7,
		"Valid range of shuffle indices is [0; 7]");
	__m128 h = _mm256_extractf128_ps(v.Vec, 0);
	__m128 l = _mm256_extractf128_ps(v.Vec, 1);
	//TODO: проверить, что порядок правильный
	l = _mm_shuffle_ps(l, l, _MM_SHUFFLE(i3, i2, i1, i0));
	h = _mm_shuffle_ps(h, h, _MM_SHUFFLE(i7, i6, i5, i4));
	return _mm256_set_m128(h, l);
}

template<int n> forceinline float8 SIMD_VECTORCALL RotateLeft(float8 v) noexcept
{return Shuffle<n & 7, (n + 1) & 7, (n + 2) & 7, (n + 3) & 7, (n + 4) & 7, (n + 5) & 7, (n + 6) & 7, (n + 7) & 7>(v);}

template<int n> forceinline float8 SIMD_VECTORCALL RotateRight(float8 v) noexcept {return RotateLeft<-n>(v);}

#endif

#if(SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_AVX2)
forceinline int8 SIMD_VECTORCALL TruncateToInt(float8 v) noexcept {return _mm256_cvttps_epi32(v.Vec);}
forceinline float8 SIMD_VECTORCALL CastToFloat(int8 v) noexcept {return _mm256_cvtepi32_ps(v.Vec);}
#endif

#endif

#if(!defined(SIMD_SSE_LEVEL) || SIMD_SSE_LEVEL < SIMD_SSE_LEVEL_AVX)
forceinline void End() noexcept {}
#endif


template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator+=(T& lhs, T rhs) noexcept {return lhs = lhs + rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator-=(T& lhs, T rhs) noexcept {return lhs = lhs - rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator*=(T& lhs, T rhs) noexcept {return lhs = lhs * rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator/=(T& lhs, T rhs) {return lhs = lhs / rhs;}

template<class T> forceinline EnableForSimdWrapper<T, T&> operator+=(T& lhs, ScalarTypeOf<T> rhs) noexcept {return lhs = lhs + rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> operator-=(T& lhs, ScalarTypeOf<T> rhs) noexcept {return lhs = lhs - rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> operator*=(T& lhs, ScalarTypeOf<T> rhs) noexcept {return lhs = lhs * rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> operator/=(T& lhs, ScalarTypeOf<T> rhs) {return lhs = lhs / rhs;}

template<class T> forceinline EnableForSimdWrapper<T, T> SIMD_VECTORCALL operator+(T lhs, ScalarTypeOf<T> rhs) noexcept {return lhs + Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, T> SIMD_VECTORCALL operator-(T lhs, ScalarTypeOf<T> rhs) noexcept {return lhs - Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, T> SIMD_VECTORCALL operator*(T lhs, ScalarTypeOf<T> rhs) noexcept {return lhs * Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, T> SIMD_VECTORCALL operator/(T lhs, ScalarTypeOf<T> rhs) {return lhs / Set<T>(rhs);}

template<class T> forceinline EnableForSimdWrapper<T, T> SIMD_VECTORCALL operator+(ScalarTypeOf<T> lhs, T rhs) noexcept {return Set<T>(lhs) + rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T> SIMD_VECTORCALL operator-(ScalarTypeOf<T> lhs, T rhs) noexcept {return Set<T>(lhs) - rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T> SIMD_VECTORCALL operator*(ScalarTypeOf<T> lhs, T rhs) noexcept {return Set<T>(lhs) * rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T> SIMD_VECTORCALL operator/(ScalarTypeOf<T> lhs, T rhs) {return Set<T>(lhs) / rhs;}

template<class T> forceinline EnableForSimdWrapper<T, T> SIMD_VECTORCALL operator-(T a) noexcept {return Set<T>(0) - a;}

template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator&(T lhs, ScalarTypeOf<T> rhs) noexcept {return lhs & Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator|(T lhs, ScalarTypeOf<T> rhs) noexcept {return lhs | Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator^(T lhs, ScalarTypeOf<T> rhs) noexcept {return lhs ^ Set<T>(rhs);}

template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator&(ScalarTypeOf<T> lhs, T rhs) noexcept {return Set<T>(lhs) & rhs;}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator|(ScalarTypeOf<T> lhs, T rhs) noexcept {return Set<T>(lhs) | rhs;}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator^(ScalarTypeOf<T> lhs, T rhs) noexcept {return Set<T>(lhs) ^ rhs;}

template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator>(T lhs, ScalarTypeOf<T> rhs) {return lhs > Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator<(T lhs, ScalarTypeOf<T> rhs) {return lhs < Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator>=(T lhs, ScalarTypeOf<T> rhs) {return lhs >= Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator<=(T lhs, ScalarTypeOf<T> rhs) {return lhs <= Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator==(T lhs, ScalarTypeOf<T> rhs) noexcept {return lhs == Set<T>(rhs);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator!=(T lhs, ScalarTypeOf<T> rhs) noexcept {return lhs != Set<T>(rhs);}

template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator>(ScalarTypeOf<T> lhs, T rhs) {return Set<T>(rhs) > lhs;}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator<(ScalarTypeOf<T> lhs, T rhs) {return Set<T>(rhs) < lhs;}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator>=(ScalarTypeOf<T> lhs, T rhs) {return Set<T>(lhs) >= rhs;}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator<=(ScalarTypeOf<T> lhs, T rhs) {return Set<T>(lhs) <= rhs;}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator==(ScalarTypeOf<T> lhs, T rhs) noexcept {return Set<T>(lhs) == rhs;}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator!=(ScalarTypeOf<T> lhs, T rhs) noexcept {return Set<T>(lhs) != rhs;}

template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator!(T lhs) noexcept {return lhs == 0;}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator&&(T lhs, T rhs) noexcept {return (lhs != 0) & (rhs != 0);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator||(T lhs, T rhs) noexcept {return (lhs != 0) | (rhs != 0);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator&&(ScalarTypeOf<T> s, T v) noexcept {return s? v != 0: Set<IntAnalogOf<T>>(0);}
template<class T> forceinline EnableForSimdWrapper<T, IntAnalogOf<T>> SIMD_VECTORCALL operator&&(T v, ScalarTypeOf<T> s) noexcept {return (v != 0) & (s? -1: 0);}

template<typename T> std::enable_if_t<T::VectorSize == 2, T> SIMD_VECTORCALL Shuffle(T v, IntAnalogOf<T> indices)
{
	return {v[indices[0]], v[indices[1]]};
}

template<typename T> std::enable_if_t<T::VectorSize == 8, T> SIMD_VECTORCALL Shuffle(T v, IntAnalogOf<T> indices)
{
	return {v[indices[0]], v[indices[1]], v[indices[2]], v[indices[3]],
		v[indices[4]], v[indices[5]], v[indices[6]], v[indices[7]]};
}

template<typename T> std::enable_if_t<T::VectorSize == 16, T> SIMD_VECTORCALL Shuffle(T v, IntAnalogOf<T> indices)
{
	return {v[indices[0]], v[indices[1]], v[indices[2]], v[indices[3]],
		v[indices[4]], v[indices[5]], v[indices[6]], v[indices[7]],
		v[indices[8]], v[indices[9]], v[indices[10]], v[indices[11]],
		v[indices[12]], v[indices[13]], v[indices[14]], v[indices[15]]};
}

template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator|=(T& lhs, T rhs) noexcept {return lhs = lhs | rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator&=(T& lhs, T rhs) noexcept {return lhs = lhs & rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator^=(T& lhs, T rhs) noexcept {return lhs = lhs ^ rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator<<=(T& lhs, T rhs) noexcept {return lhs = lhs << rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator>>=(T& lhs, T rhs) noexcept {return lhs = lhs >> rhs;}

template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator|=(T& lhs, ScalarTypeOf<T> rhs) noexcept {return lhs = lhs | rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator&=(T& lhs, ScalarTypeOf<T> rhs) noexcept {return lhs = lhs & rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator^=(T& lhs, ScalarTypeOf<T> rhs) noexcept {return lhs = lhs ^ rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator<<=(T& lhs, ScalarTypeOf<T> rhs) noexcept {return lhs = lhs << rhs;}
template<class T> forceinline EnableForSimdWrapper<T, T&> SIMD_VECTORCALL operator>>=(T& lhs, ScalarTypeOf<T> rhs) noexcept {return lhs = lhs >> rhs;}

}

#endif // !__GNUC__

namespace Simd {

template<typename T> forceinline std::enable_if_t<
	std::is_floating_point<ScalarTypeOf<std::remove_reference_t<T>>>::value,
T> SIMD_VECTORCALL Truncate(T x) noexcept
{return CastToFloat(TruncateToInt(x));}

template<typename T> forceinline std::enable_if_t<
	std::is_floating_point<ScalarTypeOf<std::remove_reference_t<T>>>::value,
T> SIMD_VECTORCALL Round(T a)
{
	return CastToFloat(RoundToInt(a));
#if 0
	const float4 vNearest2 = float4(Set<int4>(1073741823));
	const float4 aTrunc = Truncate(a);
	return aTrunc + Truncate((a - aTrunc) * vNearest2);
#endif
}


#ifdef SIMD_FLOAT4_SUPPORT

#if(defined(SIMD_SSE_LEVEL) && SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_SSE4_1)
forceinline float4 SIMD_VECTORCALL Floor(float4 x) {return _mm_floor_ps(x);}
forceinline float4 SIMD_VECTORCALL Ceil(float4 x) {return _mm_ceil_ps(x);}
#endif

#ifdef SIMD_INT4_SUPPORT

#if(defined(SIMD_SSE_LEVEL) && SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_SSE2)
forceinline int4 SIMD_VECTORCALL RoundToInt(float4 x) noexcept
{return int4(_mm_cvtps_epi32(x));}
#else
forceinline int4 SIMD_VECTORCALL RoundToInt(float4 x) noexcept
{return TruncateToInt(x + 0.5f) - ((x < 0) & 1);}
#endif

#if(!defined(SIMD_SSE_LEVEL) || SIMD_SSE_LEVEL < SIMD_SSE_LEVEL_SSE4_1)
forceinline float4 SIMD_VECTORCALL Floor(float4 x)
{
	const float4 fi = Truncate(x);
	const int4 igx = fi > x;
	const float4 j = float4(igx & int4(Set<float4>(1)));
	return fi - j;
}

forceinline float4 SIMD_VECTORCALL Ceil(float4 x)
{
	const float4 fi = Truncate(x);
	const int4 igx = fi < x;
	const float4 j = float4(igx & int4(Set<float4>(1)));
	return fi + j;
}
#endif

#endif

#endif

#ifdef SIMD_FLOAT8_SUPPORT

#if(defined(SIMD_SSE_LEVEL) && SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_AVX)

forceinline float8 SIMD_VECTORCALL Floor(float8 x) {return _mm256_floor_ps(x);}
forceinline float8 SIMD_VECTORCALL Ceil(float8 x) {return _mm256_ceil_ps(x);}

#endif

#ifdef SIMD_INT8_SUPPORT

#if(defined(SIMD_SSE_LEVEL) && SIMD_SSE_LEVEL >= SIMD_SSE_LEVEL_AVX2)
forceinline int8 SIMD_VECTORCALL RoundToInt(float8 x) noexcept {return int8(_mm256_cvtps_epi32(x));}
#else
forceinline int8 SIMD_VECTORCALL RoundToInt(float8 x) noexcept
{return TruncateToInt(x + 0.5f) - ((x < 0) & 1);}
#endif

#if(!defined(SIMD_SSE_LEVEL) || SIMD_SSE_LEVEL < SIMD_SSE_LEVEL_AVX)

forceinline float8 SIMD_VECTORCALL Floor(float8 x)
{
	const float8 fi = Truncate(x);
	const int8 igx = fi > x;
	const float8 j = float8(igx & int8(Set<float8>(1)));
	return fi - j;
}

forceinline float8 SIMD_VECTORCALL Ceil(float8 x)
{
	const float8 fi = Truncate(x);
	const int8 igx = fi < x;
	const float8 j = float8(igx & int8(Set<float8>(1)));
	return fi + j;
}

#endif

#endif

#endif


#if(defined(SIMD_SSE_LEVEL) && defined(__FMA__))

//a + b*c
forceinline float4 SIMD_VECTORCALL MultiplyAccumulate(float4 a, float4 b, float4 c) {return _mm_fmadd_ps(b, c, a);}
forceinline float8 SIMD_VECTORCALL MultiplyAccumulate(float8 a, float8 b, float8 c) {return _mm256_fmadd_ps(b, c, a);}
forceinline float4 SIMD_VECTORCALL MultiplyAccumulate(float a, float4 b, float4 c) {return MultiplyAccumulate(Set<float4>(a), b, c);}
forceinline float8 SIMD_VECTORCALL MultiplyAccumulate(float a, float8 b, float8 c) {return MultiplyAccumulate(Set<float8>(a), b, c);}
forceinline float4 SIMD_VECTORCALL MultiplyAccumulate(float4 a, float4 b, float c) {return MultiplyAccumulate(a, b, Set<float4>(c));}
forceinline float8 SIMD_VECTORCALL MultiplyAccumulate(float8 a, float8 b, float c) {return MultiplyAccumulate(a, b, Set<float8>(c));}
forceinline float4 SIMD_VECTORCALL MultiplyAccumulate(float a, float4 b, float c) {return MultiplyAccumulate(Set<float4>(a), b, Set<float4>(c));}
forceinline float8 SIMD_VECTORCALL MultiplyAccumulate(float a, float8 b, float c) {return MultiplyAccumulate(Set<float8>(a), b, Set<float8>(c));}

#else

template<typename T1, typename T2, typename T3> forceinline auto SIMD_VECTORCALL MultiplyAccumulate(T1 a, T2 b, T3 c) {return a + b*c;}

#endif

template<typename T> forceinline std::enable_if_t<
	std::is_floating_point<ScalarTypeOf<T>>::value,
T> SIMD_VECTORCALL Fract(T x) {return x - Floor(x);}

template<typename T> forceinline std::enable_if_t<
	std::is_floating_point<ScalarTypeOf<T>>::value,
T> SIMD_VECTORCALL Mod(T a, T aDiv)
{return a - Floor(a / aDiv) * aDiv;}

template<typename T> forceinline std::enable_if_t<
	std::is_same<ScalarTypeOf<T>, float>::value,
T> SIMD_VECTORCALL ModSigned(T a, T aDiv)
{return a - Truncate(a / aDiv) * aDiv;}

template<typename T> forceinline std::enable_if_t<
	std::is_same<ScalarTypeOf<T>, float>::value,
T> SIMD_VECTORCALL Abs(T v) noexcept
{return T(IntAnalogOf<T>(v) & 0x7FFFFFFF);}

template<typename T> forceinline std::enable_if_t<
	std::is_same<ScalarTypeOf<T>, float>::value,
T> SIMD_VECTORCALL Pow2(T x) noexcept
{
	const T fractional_part = Fract(x);

	T factor = MultiplyAccumulate(float(-8.94283890931273951763e-03), fractional_part, float(-1.89646052380707734290e-03));
	factor = MultiplyAccumulate(float(-5.58662282412822480682e-02), factor, fractional_part);
	factor = MultiplyAccumulate(float(-2.40139721982230797126e-01), factor, fractional_part);
	factor = MultiplyAccumulate(float(3.06845249656632845792e-01), factor, fractional_part);
	factor = MultiplyAccumulate(float(1.06823753710239477000e-07), factor, fractional_part);
	x -= factor;

	x *= float(1 << 23);
	x += float((1 << 23) * 127);

	return T(RoundToInt(x));
}

template<typename T> forceinline std::enable_if_t<
	std::is_same<ScalarTypeOf<T>, float>::value,
T> SIMD_VECTORCALL Exp(T x) noexcept
{return Pow2(x * float(1.442695040888963407359924681001892137426645954153));}

namespace detail {

// Minimax polynomial fit of log2(x)/(x - 1), for x in range [1, 2]
template<typename T, int Order> struct Log2Polynomial;
template<typename T> struct Log2Polynomial<T, 2>
{
	static forceinline T SIMD_VECTORCALL Calc(T m) noexcept
	{
		T p = MultiplyAccumulate(-1.04913055217340124191f, m, 0.204446009836232697516f);
		return MultiplyAccumulate(2.28330284476918490682f, m, p);
	}
};
template<typename T> struct Log2Polynomial<T, 3>
{
	static forceinline T SIMD_VECTORCALL Calc(T m) noexcept
	{
		T p = MultiplyAccumulate(0.688243882994381274313f, m, -0.107254423828329604454f);
		p = MultiplyAccumulate(-1.75647175389045657003f, m, p);
		return MultiplyAccumulate(2.61761038894603480148f, m, p);
	}
};
template<typename T> struct Log2Polynomial<T, 4>
{
	static forceinline T SIMD_VECTORCALL Calc(T m) noexcept
	{
		T p = MultiplyAccumulate(-0.465725644288844778798f, m, 0.0596515482674574969533f);
		p = MultiplyAccumulate(1.48116647521213171641f, m, p);
		p = MultiplyAccumulate(-2.52074962577807006663f, m, p);
		return MultiplyAccumulate(2.8882704548164776201f, m, p);
	}
};
template<typename T> struct Log2Polynomial<T, 5>
{
	static forceinline T SIMD_VECTORCALL Calc(T m) noexcept
	{
		T p = MultiplyAccumulate(3.1821337e-1f, m, -3.4436006e-2f);
		p = MultiplyAccumulate(-1.2315303f, m, p);
		p = MultiplyAccumulate(2.5988452f, m, p);
		p = MultiplyAccumulate(-3.3241990f, m, p);
		return MultiplyAccumulate(3.1157899f, m, p);
	}
};

}

template<int Order, typename T> inline std::enable_if_t<
	std::is_same<ScalarTypeOf<T>, float>::value,
T> SIMD_VECTORCALL Log2Order(T x)
{
	T one = Set<T>(1);
	T e = CastToFloat(UnsignedRightBitShift(IntAnalogOf<T>(x) & 0x7F800000, 23) - 127);
	T m = T((IntAnalogOf<T>(x) & 0x007FFFFF) | IntAnalogOf<T>(one));
	T p = detail::Log2Polynomial<T, Order>::Calc(m);
	p *= m - one; // This effectively increases the polynomial degree by one, but ensures that log2(1) == 0
	return p + e;
}

template<int Order, typename T> inline std::enable_if_t<
	std::is_same<ScalarTypeOf<T>, float>::value,
T> SIMD_VECTORCALL LogOrder(T x)
{return Log2Order<Order>(x) / float(1.442695040888963407359924681001892137426645954153);}

template<typename T> inline std::enable_if_t<
	std::is_same<ScalarTypeOf<T>, float>::value,
T> SIMD_VECTORCALL Log2(T x) {return Log2Order<5>(x);}

template<typename T> inline std::enable_if_t<
	std::is_same<ScalarTypeOf<T>, float>::value,
T> SIMD_VECTORCALL Log(T x) {return LogOrder<5>(x);}



forceinline float Pow2(float x) noexcept
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

forceinline float Exp(float x) noexcept
{return Pow2(x * float(1.442695040888963407359924681001892137426645954153));}

#if 0

EXPORT CONST float xerff_u1(float a)
{
	float s = a, t, u;
	Sleef_float2 d;

	a = fabsfk(a);
	int o0 = a < 1.1f, o1 = a < 2.4f, o2 = a < 4.0f;
	u = o0 ? (a*a) : a;

	t = o0 ? +0.7089292194e-4f : o1 ? -0.1792667899e-4f : -0.9495757695e-5f;
	
	t = mlaf(t, u, o0 ? -0.7768311189e-3f : o1 ? +0.3937633010e-3f : +0.2481465926e-3f); //t*u+(o0? ...)
	t = mlaf(t, u, o0 ? +0.5159463733e-2f : o1 ? -0.3949181177e-2f : -0.2918176819e-2f);
	t = mlaf(t, u, o0 ? -0.2683781274e-1f : o1 ? +0.2445474640e-1f : +0.2059706673e-1f);
	t = mlaf(t, u, o0 ? +0.1128318012e+0f : o1 ? -0.1070996150e+0f : -0.9901899844e-1f);
	d = dfmul_f2_f_f(t, u);
	d = dfadd2_f2_f2_f2(d, o0 ? dfx(-0.376125876000657465175213237214e+0) :
		o1 ? dfx(-0.634588905908410389971210809210e+0) :
		dfx(-0.643598050547891613081201721633e+0));
	d = dfmul_f2_f2_f(d, u);
	d = dfadd2_f2_f2_f2(d, o0 ? dfx(+0.112837916021059138255978217023e+1) :
		o1 ? dfx(-0.112879855826694507209862753992e+1) :
		dfx(-0.112461487742845562801052956293e+1));
	d = dfmul_f2_f2_f(d, a);
	d = o0 ? d : dfadd_f2_f_f2(1.0, dfneg_f2_f2(expk2f(d)));
	u = mulsignf(o2 ? (d.x + d.y) : 1, s);
	u = xisnanf(a) ? SLEEF_NANf : u;
	return u;
}

#endif

}
