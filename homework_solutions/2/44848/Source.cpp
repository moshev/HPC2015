#include "immintrin.h"

using namespace std;

const int chunkSize = 64;
const __m128i revMask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

void reverse(char* bytes, int numChunks) {
	__m128i moveLoc;

	for (int c = 0; c < numChunks; ++c) {
		__m128i * const lower_lo = reinterpret_cast<__m128i *>(bytes + chunkSize * c + 0);
		__m128i * const lower_hi = reinterpret_cast<__m128i *>(bytes + chunkSize * c + 16);

		__m128i * const higher_lo = lower_lo + 2;
		__m128i * const higher_hi = lower_hi + 2;

		// outer parts
		moveLoc = _mm_load_si128(lower_lo);
		_mm_store_si128(lower_lo, _mm_shuffle_epi8(_mm_load_si128(higher_hi), revMask));
		_mm_store_si128(higher_hi, _mm_shuffle_epi8(moveLoc, revMask));

		// inner parts
		moveLoc = _mm_load_si128(lower_hi);
		_mm_store_si128(lower_hi, _mm_shuffle_epi8(_mm_load_si128(higher_lo), revMask));
		_mm_store_si128(higher_lo, _mm_shuffle_epi8(moveLoc, revMask));
	}
}

int main() {
}


