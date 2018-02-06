# Simd

**Library compatible with GCC\Clang vector extensions.**

This is a small single-header library, just include `Simd.h` to use it.

It defines the following SIMD floating point types: `float4`, `int4`, `float8`, `int8` and some useful mathematical functions: `Floor`, `Round`, `Truncate`, `CastToFloat`, `TruncateToInt`, `RoundToInt`, `Exp`, `Log`, `Min`, `Max`, `HorSum` and `MultiplyAccumulate`. All types, functions and operators are defined in a Simd namespace.

Example of library usage:

```C++
Simd::float8 c1 = Simd::Set<Simd::float8>(coeffs[0]);
Simd::float8 c2 = Simd::Set<Simd::float8>(coeffs[1]);
Simd::float8 c3 = Simd::Set<Simd::float8>(coeffs[2]);
auto dstEnd = dst + n;
while(dst != dstEnd)
{
	Simd::float8 a = Simd::Load8Aligned(dst);
	Simd::float8 b1 = Simd::Load8Aligned(srcs[0]);
	Simd::float8 b2 = Simd::Load8Aligned(srcs[1]);
	Simd::float8 b3 = Simd::Load8Aligned(srcs[2]);
	Simd::StoreAligned(a + b1 * c1 + b2 * c2 + b3 * c3, dst);

	dst += 8;
	srcs[0] += 8;
	srcs[1] += 8;
	srcs[2] += 8;
}
Simd::End();
```

 The code above calculates weighted sum of three float arrays together and adds the result to dst. It assumes that all arrays are 32-byte aligned, and array size n is a multiple of 8.

 For GCC and Clang these types are just `typedef` of corresponding built-in vector types without any wrappers around them.

 For MSVC compiler that don't have support of these extensions, library implements these types as wrappers with overloaded operators `+`, `-`, `*`, `/`,`=`, `+=`, `-=`, `*=`, `/=`, `<<`, `>>`, `&`, `|`, `&&`, `||` and so on. In release builds these operators are inlined by compiler and don't affect performance. `float4` type is implemented via SSE intrinsics, `int4`  - SSE2, `float8` - AVX, `int8` - AVX2. Compile code with corresponding `/arch:(SSE|SSE2|AVX|AVX2)` key to enable them, otherwise these types will not be defined (for x86) or only `float4` and `int4` types will be defined in x64. As a alternative you can `#define SIMD_SSE_LEVEL SIMD_SSE_LEVEL_*` before including `Simd.h`. Also some functions use SSE3/SSSE3/SSE4.1/SSE4.2 for better performance when their generation is enabled with SIMD_SSE_LEVEL or /arch:AVX or /arch:AVX2 compiler key.

 The library was not tested well and probably contain bugs, so be ready to look into `Simd.h` source code to fix bugs.

 Performance tests in this project are messy and some of them have incorrect estimated value of GFlops. My future plan is to make this library a part of Intra project, so I will be able to write tests using Intra testing features.