#include <immintrin.h>
void reverse(char* bytes, int numChunks) {
	__m128i first;
	__m128i second;
	__m128i third;
	__m128i fourth;
	for (auto i = 0; i < numChunks; ++i) {
		first = _mm_load_si128((__m128i*)(bytes+i*64));
		second = _mm_load_si128((__m128i*)(bytes + i*64 + 16));
		third = _mm_load_si128((__m128i*)(bytes + i*64 + 32));
		third = _mm_shuffle_epi32(third, 0x1b);
		fourth = _mm_load_si128((__m128i*)(bytes + i*64 + 48));
		fourth = _mm_shuffle_epi32(fourth, 0x1b);
		first = _mm_xor_si128(first, fourth);
		fourth = _mm_xor_si128(first, fourth);
		first = _mm_xor_si128(first, fourth);

		second = _mm_xor_si128(first, fourth);
		third = _mm_xor_si128(first, fourth);
		second = _mm_xor_si128(first, fourth);
	}
}

int main() {
	char bytes[256];
	for (int i = 0; i < 256; ++i)bytes[i] = i;
	reverse(bytes, 4);
	return 0;
}