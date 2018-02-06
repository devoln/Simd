#include "Simd.h"
#include <cmath>
#include <iostream>
using namespace std;
using namespace Simd;

ostream& operator<<(ostream& stream, const float4& v)
{
	return stream << '{' << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << "}";
}

#ifdef SIMD_FLOAT8_SUPPORT
ostream& operator<<(ostream& stream, const float8& v)
{
	return stream << '{' << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << ", " <<
		v[4] << ", " << v[5] << ", " << v[6] << ", " << v[7] << "}";
}
#endif

template<typename T> void TestVector(const T& x)
{
	cout << "x: " << x << endl;
	cout << "Pow2(x): " << Pow2(x) << endl;
	cout << "Exp(x): " << Exp(x) << endl;
	cout << "Log2(Pow2(x)): " << Log2(Pow2(x)) << endl;
	cout << "Log(Exp(x)): " << Log(Exp(x)) << endl;
	cout << "LogOrder<2>(Exp(x)): " << LogOrder<2>(Exp(x)) << endl;
	cout << "LogOrder<3>(Exp(x)): " << LogOrder<3>(Exp(x)) << endl;
	cout << "LogOrder<4>(Exp(x)): " << LogOrder<4>(Exp(x)) << endl;
	cout << "Truncate(x): " << Truncate(x) << endl;
	cout << "Floor(x): " << Floor(x) << endl;
	cout << "Ceil(x): " << Ceil(x) << endl;
	cout << "Round(x): " << Round(x) << endl;
}

bool IsAvxSupported();

void SimdTest()
{
	float4 x4 = {-5.2f, -2.8f, 1.1f, 6.7f};
	TestVector(x4);

	cout << endl;
	float4 x4_2 = {-5, -2, 1, 6};
	TestVector(x4_2);

#ifdef SIMD_FLOAT8_SUPPORT
	if(IsAvxSupported())
	{
		cout << endl;
		float8 x8 = {-5.2f, -2.8f, 1.1f, 6.7f, -5, -2, 1, 6};
		TestVector(x8);
	}
#endif
}
