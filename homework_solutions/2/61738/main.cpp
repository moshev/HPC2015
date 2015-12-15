#include<stdio.h>
#include<xmmintrin.h>
#include <cmath>

void reverse(char* bytes, int numChunks)
{
	for (int i = 0; i < numChunks; ++i)
	{
		for (int l = i * 64, r = (i + 1) * 64 - 1; l < r; ++l, --r)
		{
			char temp = bytes[l];
			bytes[l] = bytes[r];
			bytes[r] = temp;
		}
	}
}

static void foo(float(&inout)[8]) 
{
	const size_t idx[][2] = {
		{ 0, 1 },{ 2, 3 },{ 4, 5 },{ 6, 7 },
		{ 0, 2 },{ 1, 3 },{ 4, 6 },{ 5, 7 },
		{ 1, 2 },{ 5, 6 },
		{ 0, 4 },{ 1, 5 },{ 2, 6 },{ 3, 7 },
		{ 2, 4 },{ 3, 5 },
		{ 1, 2 },{ 3, 4 },{ 5, 6 }
	};

	for (size_t i = 0; i < sizeof(idx) / sizeof(idx[0]); ++i) 
	{
		const float x = inout[idx[i][0]];
		const float y = inout[idx[i][1]];

		inout[idx[i][0]] = std::fmin(x, y);
		inout[idx[i][1]] = std::fmax(x, y);
	}
}

static void bar(float(&inout)[8]) 
{
	const size_t idx[][2] = {
		{ 0, 1 },{ 3, 2 },{ 4, 5 },{ 7, 6 },
		{ 0, 2 },{ 1, 3 },{ 6, 4 },{ 7, 5 },
		{ 0, 1 },{ 2, 3 },{ 5, 4 },{ 7, 6 },
		{ 0, 4 },{ 1, 5 },{ 2, 6 },{ 3, 7 },
		{ 0, 2 },{ 1, 3 },{ 4, 6 },{ 5, 7 },
		{ 0, 1 },{ 2, 3 },{ 4, 5 },{ 6, 7 }
	};

	size_t size = sizeof(idx) / sizeof(idx[0]);
	size_t newSize = size / 6;

	union m128 
	{
		__m128 reg;    
		float values[4];  
	};
	m128 reg1;
	m128 reg2;

	for (size_t i = 0; i < newSize; ++i)
	{
		reg1.values[i] = inout[idx[i][0]];
		reg2.values[i] = inout[idx[i][1]];
	}
	
	m128 resultMin, resultMax;

	resultMin.reg = _mm_min_ps(reg1.reg,reg2.reg);
	resultMax.reg = _mm_max_ps(reg1.reg,reg2.reg);

	for (int i = 0; i < newSize; i++)
	{
		inout[idx[i][0]] = resultMin.values[i];
		inout[idx[i][1]] = resultMax.values[i];
	}

	////////////////////////////////////////////////////////////////////////////////////
	for (size_t i = newSize,j = 0; i < newSize*2; ++i,++j)
	{
		reg1.values[j] = inout[idx[i][0]];
		reg2.values[j] = inout[idx[i][1]];
	}

	resultMin.reg = _mm_min_ps(reg1.reg, reg2.reg);
	resultMax.reg = _mm_max_ps(reg1.reg, reg2.reg);

	for (int i = newSize,j=0; i < newSize*2; ++i,++j)
	{
		inout[idx[i][0]] = resultMin.values[j];
		inout[idx[i][1]] = resultMax.values[j];
	}

	//////////////////////////////////////////////////////////////////////////////////////
	for (size_t i = newSize*2,j = 0; i < newSize*3; ++i,++j)
	{
		reg1.values[j] = inout[idx[i][0]];
		reg2.values[j] = inout[idx[i][1]];
	}

	resultMin.reg = _mm_min_ps(reg1.reg, reg2.reg);
	resultMax.reg = _mm_max_ps(reg1.reg, reg2.reg);

	for (size_t i = newSize * 2,j=0; i < newSize * 3; ++i,++j)
	{
		inout[idx[i][0]] = resultMin.values[j];
		inout[idx[i][1]] = resultMax.values[j];
	}

	////////////////////////////////////////////////////////////////////////////////////
	for (size_t i = newSize * 3,j = 0; i < newSize *4; ++i,++j)
	{
		reg1.values[j] = inout[idx[i][0]];
		reg2.values[j] = inout[idx[i][1]];
	}

	resultMin.reg = _mm_min_ps(reg1.reg, reg2.reg);
	resultMax.reg = _mm_max_ps(reg1.reg, reg2.reg);

	for (size_t i = newSize * 3,j=0; i < newSize * 4; ++i,++j)
	{
		inout[idx[i][0]] = resultMin.values[j];
		inout[idx[i][1]] = resultMax.values[j];
	}

	//////////////////////////////////////////////////////////////////////////////////////
	for (size_t i = newSize * 4,j=0; i < newSize * 5; ++i,++j)
	{
		reg1.values[j] = inout[idx[i][0]];
		reg2.values[j] = inout[idx[i][1]];
	}

	resultMin.reg = _mm_min_ps(reg1.reg, reg2.reg);
	resultMax.reg = _mm_max_ps(reg1.reg, reg2.reg);

	for (size_t i = newSize * 4,j=0; i < newSize * 5; ++i,++j)
	{
		inout[idx[i][0]] = resultMin.values[j];
		inout[idx[i][1]] = resultMax.values[j];
	}

	////////////////////////////////////////////////////////////////////////////////////
	for (size_t i = newSize * 5,j=0; i < newSize * 6; ++i,++j)
	{
		reg1.values[j] = inout[idx[i][0]];
		reg2.values[j] = inout[idx[i][1]];
	}

	resultMin.reg = _mm_min_ps(reg1.reg, reg2.reg);
	resultMax.reg = _mm_max_ps(reg1.reg, reg2.reg);

	for (size_t i = newSize * 5,j=0; i < newSize * 6; ++i,++j)
	{
		inout[idx[i][0]] = resultMin.values[j];
		inout[idx[i][1]] = resultMax.values[j];
	}
}