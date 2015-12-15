#include "immintrin.h"
#include <algorithm>

static void foo(
	float(&inout)[8]) {

	__m128& a = reinterpret_cast<__m128&>(inout[0]);
	__m128& b = reinterpret_cast<__m128&>(inout[4]);

	__m128 min = _mm_min_ps(a, b);
	__m128 max = _mm_max_ps(a, b);

	a = _mm_shuffle_ps(min, max, _MM_SHUFFLE(0, 1, 0, 1));
	b = _mm_shuffle_ps(min, max, _MM_SHUFFLE(2, 3, 2, 3));
	min = _mm_min_ps(a, b);
	max = _mm_max_ps(a, b);

	a = _mm_shuffle_ps(min, max, _MM_SHUFFLE(0, 2, 0, 2));
	b = _mm_shuffle_ps(min, max, _MM_SHUFFLE(1, 3, 1, 3));
	min = _mm_min_ps(a, b);
	max = _mm_max_ps(a, b);

	a = _mm_shuffle_ps(min, max, _MM_SHUFFLE(0, 1, 0, 1));
	b = _mm_shuffle_ps(min, max, _MM_SHUFFLE(2, 3, 2, 3));

	const size_t idx[][2] = {
		{ 1, 2 },{ 5, 6 },
		{ 2, 4 },{ 3, 5 },
		{ 1, 2 },{ 3, 4 },{ 5, 6 }
	};

	for (size_t i = 0; i < sizeof(idx) / sizeof(idx[0]); ++i) {
		const float x = inout[idx[i][0]];
		const float y = inout[idx[i][1]];

		inout[idx[i][0]] = std::min(x, y);
		inout[idx[i][1]] = std::max(x, y);
	}
}

static void bar(
	float(&inout)[8]) {

	const size_t idx[][2] = {
		{ 0, 1 },{ 3, 2 },{ 4, 5 },{ 7, 6 },
		{ 0, 2 },{ 1, 3 },{ 6, 4 },{ 7, 5 },
		{ 0, 1 },{ 2, 3 },{ 5, 4 },{ 7, 6 },
		{ 0, 4 },{ 1, 5 },{ 2, 6 },{ 3, 7 },
		{ 0, 2 },{ 1, 3 },{ 4, 6 },{ 5, 7 },
		{ 0, 1 },{ 2, 3 },{ 4, 5 },{ 6, 7 }
	};

	for (size_t i = 0; i < sizeof(idx) / sizeof(idx[0]); ++i) {
		const float x = inout[idx[i][0]];
		const float y = inout[idx[i][1]];

		inout[idx[i][0]] = std::min(x, y);
		inout[idx[i][1]] = std::max(x, y);
	}
}

void reverse(char* bytes, int numChunks)
{
	const int block_size = 64;
	const __m128i mask = _mm_setr_epi8(
		15, 14, 13, 12,
		11, 10, 9, 8,
		7, 6, 5, 4,
		3, 2, 1, 0);

	for (size_t i = 0; i < numChunks; ++i)
	{
		__m128i& a = *reinterpret_cast<__m128i*>(bytes + i * block_size);
		__m128i& b = *reinterpret_cast<__m128i*>(bytes + 16 + i * block_size);
		__m128i& c = *reinterpret_cast<__m128i*>(bytes + 32 + i * block_size);
		__m128i& d = *reinterpret_cast<__m128i*>(bytes + 48 + i * block_size);

		a = _mm_xor_si128(a, d);
		d = _mm_xor_si128(a, d);
		a = _mm_xor_si128(a, d);

		b = _mm_xor_si128(b, c);
		c = _mm_xor_si128(b, c);
		b = _mm_xor_si128(b, c);

		a = _mm_shuffle_epi8(a, mask);
		b = _mm_shuffle_epi8(b, mask);
		c = _mm_shuffle_epi8(c, mask);
		d = _mm_shuffle_epi8(d, mask);
	}
}

int main()
{
	char bytes[64];
	for (int i = 0; i < 64; ++i)
		bytes[i] = i % 64;

	reverse(bytes, 1);
}
