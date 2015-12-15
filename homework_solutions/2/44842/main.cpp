#include <immintrin.h>

void reverse(char* bytes, int numChunks) {

	__m128i chunk1;		//4 * 32bit = 16 bytes
	__m128i chunk2;
	__m128i chunk3;
	__m128i chunk4;

	const __m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

	int chunkIdx;

	for (int i = 0; i < numChunks; ++i) {
		chunkIdx = i * 64;

		chunk1 = _mm_loadu_si128((const __m128i*) &bytes[chunkIdx]);
		chunk2 = _mm_loadu_si128((const __m128i*) &bytes[chunkIdx + 16]);
		chunk3 = _mm_loadu_si128((const __m128i*) &bytes[chunkIdx + 32]);
		chunk4 = _mm_loadu_si128((const __m128i*) &bytes[chunkIdx + 48]);

		chunk1 = _mm_shuffle_epi8(chunk1, mask);
		chunk2 = _mm_shuffle_epi8(chunk2, mask);
		chunk3 = _mm_shuffle_epi8(chunk3, mask);
		chunk4 = _mm_shuffle_epi8(chunk4, mask);

		_mm_storeu_si128((__m128i*)&bytes[chunkIdx], chunk4);
		_mm_storeu_si128((__m128i*)&bytes[chunkIdx + 16], chunk3);
		_mm_storeu_si128((__m128i*)&bytes[chunkIdx + 32], chunk2);
		_mm_storeu_si128((__m128i*)&bytes[chunkIdx + 48], chunk1);
	}
}

int main() {

	#define SIZE 6400000

	char* bytes = new char[SIZE];
	for (int i = 0; i < SIZE; ++i) {
		bytes[i] = i;
	}

	reverse(bytes, SIZE / 64);

	delete[] bytes;

	return 0;
}
