#include <stdio.h>

#include <tmmintrin.h>
#include <vector>

void reverse(char* bytes, int numChunks)
{
    const __m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    for(int i = 0; i < numChunks; ++i) {
        __m128i left = _mm_load_si128((__m128i*)bytes);
        __m128i right = _mm_load_si128(((__m128i*)bytes) + 3);
        left = _mm_shuffle_epi8(left, mask);
        right = _mm_shuffle_epi8(right, mask);

        _mm_store_si128(((__m128i*)bytes) + 3, left);
        _mm_store_si128(((__m128i*)bytes), right);

        left = _mm_load_si128(((__m128i*)bytes) + 1);
        right = _mm_load_si128(((__m128i*)bytes) + 2);
        left = _mm_shuffle_epi8(left, mask);
        right = _mm_shuffle_epi8(right, mask);

        _mm_store_si128(((__m128i*)bytes) + 2, left);
        _mm_store_si128(((__m128i*)bytes) + 1, right);
        bytes += 64;
    }
}

int main()
{
    const int size = 1 << 10;
    static_assert(size % 64 == 0, "Loool");
    char b[size];
    for(int i = 0; i < size; ++i)
        b[i] = i % 127;

    for(int i = 0; i < size; ++i)
        printf("%d ", b[i]);
    printf("\n");
    reverse(b, size / 64);

    for(int i = 0; i < size; ++i)
        printf("%d ", b[i]);
    printf("\n");
    return 0;
}