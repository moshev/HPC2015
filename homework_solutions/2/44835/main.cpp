#include <stdio.h>
#include <chrono>
void reverse(char* bytes, int chunks);
void reverse_fast(char* bytes, int chunks);
#define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
#include "simd/avx.h"

void reverse(char* bytes, int chunks)
{
	for (int i = 0; i < chunks; ++i)
	{
		for (int j = 0; j < 32; j += 1)
		{
			std::swap(bytes[i * 64 + j], bytes[i * 64 + 63 - j]);
		}
	}
}


void reverse_fast(char* bytes, int chunks)
{
	for (int i = 0; i < chunks; ++i)
	{
		embree::avxi a((int*)(bytes + (i * 64)));
		embree::avxi b((int*)(bytes + (i * 64 + 32)));
		b = embree::shuffle<3, 2, 1, 0>(embree::shuffle<1,0>(b));
		a = embree::shuffle<3, 2, 1, 0>(embree::shuffle<1, 0>(a));
		a = ((a & 0xFF000000) >> 24) | ((a & 0x00FF0000) >> 8) | ((a & 0x0000FF00) << 8) | ((a & 0x000000FF) << 24);
		b = ((b & 0xFF000000) >> 24) | ((b & 0x00FF0000) >> 8) | ((b & 0x0000FF00) << 8) | ((b & 0x000000FF) << 24);
		embree::store8i(bytes + (i * 64), b);
		embree::store8i(bytes + (i * 64 + 32), a);
	}
}