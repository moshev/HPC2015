#include <iostream>
#include <chrono>
#include <cmath>
//float calcNumerator(unsigned idx, unsigned pow)
//{
//	if (idx <= 1)
//		return 1.f;
//	else
//		return calcNumerator(idx / 2, pow - 1) + (idx % 2) * (2 << pow - 1);
//}

float derCorput(unsigned n)
{
#ifdef _WIN32
	unsigned long idxFrstSet = 0;
	//some ifdefs
    
	if (!_BitScanReverse(&idxFrstSet, n))
		return 0.f;
#else
    if (!n)
        return 0.f;
    
    unsigned long idxFrstSet = sizeof(unsigned) * 8 - __builtin_clz(n);

#endif
	unsigned denominator = 2 << idxFrstSet;
	unsigned numerator = 0;
	idxFrstSet -= 1;
	while(n > 1)
	{
		//if(n & 1u)
		//	numerator += 2 << idxFrstSet;
		numerator += (n & 1u) * (2 << idxFrstSet);
		n >>= 1;
		--idxFrstSet;
	}

	return static_cast<float>(numerator + 1) / denominator;
	//return (float)calcNumerator(n, idxFrstSet)/denominator;
}

double vdc(int n, double base = 2)
{
	double vdc = 0, denom = 1;
	while (n)
	{
		vdc += fmod(n, base) / (denom *= base);
		n /= base; // note: conversion from 'double' to 'int'
	}
	return vdc;
}

int main()
{
#if 1
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	for (auto i = 0; i < 128 * 1024 * 1024; ++i)
		if (derCorput(i) == 12312312312)
			printf("kofti");
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	start = std::chrono::system_clock::now();
	for (auto i = 0; i < 128 * 1024 * 1024; ++i)
		if (vdc(i) == 12312312312)
			printf("kofti");
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
#else
	for (auto i = 0; i < 256 * 1024 * 1024; ++i)
		if (fabsf(vdc(i) - derCorput(i)) > 1e-7)
		{
			printf("kofti");
			break;
		}
#endif

	return 0;
}