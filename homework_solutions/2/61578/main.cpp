#include <cstdio>
#include <x86intrin.h>
#include <cstring>

#include <chrono>

const int CHUNKS = 10000000;
const bool PRINT_REVERSED = false;

const int SIZE_BYTES = CHUNKS * 64;
// Mask, signifying reversed order of the 16 Bytes of the sub-parts
const __m128i REVERSE_MASK = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

void reverse(char* bytes, int numChunks)
{
    char reversed[64];
    int subPart;

    // Each chunk is 64 Bytes = 64 x 8bit
    for (int chunk = 0; chunk < numChunks; chunk++)
    {
        // Each chunk has 4 x 128-bit sub-parts
        for (subPart = 0; subPart < 4; subPart++)
        {
            // Also reverse the order of the sub-part while storing
            _mm_storeu_si128((__m128i *)&reversed[(4 - subPart - 1) * 16],
                _mm_shuffle_epi8(_mm_loadu_si128((__m128i *)&bytes[subPart * 16 + 64 * chunk]), REVERSE_MASK));
        }

        memcpy(bytes + chunk * 64, reversed, 64);
    }
}

int main()
{
    char* bytes = new char[SIZE_BYTES];
    for (int i = 0; i < SIZE_BYTES; ++i)
    {
        bytes[i] = i;
    }

    auto beginTime = std::chrono::high_resolution_clock::now();

    reverse(bytes, (SIZE_BYTES/64));

    auto endTime = std::chrono::high_resolution_clock::now();
    auto timeNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime-beginTime).count();

    printf("\nReversed for %i ns :\n\n", timeNanos);

    if (PRINT_REVERSED) {
        printf("\nReversed bytes :\n\n");
        for (int i = 0; i < SIZE_BYTES; i+=8)
        {
            printf("%4i %4i %4i %4i %4i %4i %4i %4i\n",
                   bytes[i], bytes[i+1], bytes[i+2], bytes[i+3],
                   bytes[i+4], bytes[i+5], bytes[i+6], bytes[i+7]);
        }
    }

    return 0;
}
