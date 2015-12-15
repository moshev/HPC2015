#include <cstdio>
#include <immintrin.h>

void reverse(char* bytes, int numChunks)
{
    __m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i v_lo, v_hi;
    char *startChunk, *endChunk, *lo, *hi;
    const int chunkSize = 64;

    if(numChunks <= 0) {
        return;
    }

    for(int i = 0; i < numChunks; ++i) {
        startChunk = bytes + chunkSize*i;
        endChunk = startChunk + chunkSize - 1;
        for(int j = 0; j < 2; ++j) {
            lo = startChunk + 16*j;
            hi = endChunk - 16*(j+1) + 1;
            v_lo = _mm_loadu_si128((__m128i *) lo);
            v_hi = _mm_loadu_si128((__m128i *) hi);

            v_lo = _mm_shuffle_epi8(v_lo, mask);
            v_hi = _mm_shuffle_epi8(v_hi, mask);

            _mm_store_si128((__m128i *) hi, v_lo);
            _mm_store_si128((__m128i *) lo, v_hi);
        }
    }
}

int main()
{
    return 0;
}
