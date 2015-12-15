#include <immintrin.h>

void reverse(char* array, int chunkCount)
{
    __m128i one, two, three, four;
    static const __m128i mask = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    for(auto i = 0; i < chunkCount; ++i)
    {
        one = _mm_load_si128(reinterpret_cast<const __m128i*>(&array[(i * 64)]));
        one =_mm_shuffle_epi8 (one, mask);

        two = _mm_load_si128(reinterpret_cast<const __m128i*>(&array[(i * 64) + 16]));
        two = _mm_shuffle_epi8 (two, mask);

        three = _mm_load_si128(reinterpret_cast<const __m128i*>(&array[(i * 64) + 32]));
        three = _mm_shuffle_epi8 (three, mask);

        four = _mm_load_si128(reinterpret_cast<const __m128i*>(&array[(i * 64) + 48]));
        four = _mm_shuffle_epi8 (four, mask);

        _mm_store_si128 ((__m128i*)&array[i * 64], four);
        _mm_store_si128 ((__m128i*)&array[(i * 64) + 16], three);
        _mm_store_si128 ((__m128i*)&array[(i * 64) + 32], two);
        _mm_store_si128 ((__m128i*)&array[(i * 64) + 48], one);

    }

}

