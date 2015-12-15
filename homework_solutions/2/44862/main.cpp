#include <immintrin.h>

void reverse(char* bytes, int numChunks)
{
    const int chunkSize = 64;
    __m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8,
                                 9, 10, 11, 12, 13, 14, 15);

    __m128i arrays[4];
    for(int i = 0; i < numChunks; ++i) {
        for(int j = 0; j < 4; ++j) {
            arrays[j] = _mm_load_si128((__m128i*)&bytes[i * chunkSize + j * 16]);
        }

        for(int j = 0; j < 4; ++j) {
            _mm_store_si128((__m128i*)&bytes[i * chunkSize + j * 16], _mm_shuffle_epi8(arrays[3 - j], mask));
        }
    }
}

int main()
{

    return 0;
}
