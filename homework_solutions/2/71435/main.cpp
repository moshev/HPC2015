#include <immintrin.h>

// Chunk size in bytes
#define CHUNK_SIZE 64

void reverse(char* bytes, int numChunks)
{
	__m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  __m128i load, load1, load2, load3;

  for (int i = 0; i < numChunks; ++i)
  {
    load = _mm_load_si128((__m128i*)&bytes[i * CHUNK_SIZE + 0]);
    load = _mm_shuffle_epi8(load, mask);

    load1 = _mm_load_si128((__m128i*)&bytes[i * CHUNK_SIZE + 16]);
    load1 = _mm_shuffle_epi8(load1, mask);

    load2 = _mm_load_si128((__m128i*)&bytes[i * CHUNK_SIZE + 32]);
    load2 = _mm_shuffle_epi8(load2, mask);

    load3 = _mm_load_si128((__m128i*)&bytes[i * CHUNK_SIZE + 48]);
    load3 = _mm_shuffle_epi8(load3, mask);

    _mm_store_si128((__m128i*)(bytes + i * CHUNK_SIZE + 0), load3);
    _mm_store_si128((__m128i*)(bytes + i * CHUNK_SIZE + 16), load2);
    _mm_store_si128((__m128i*)(bytes + i * CHUNK_SIZE + 32), load1);
    _mm_store_si128((__m128i*)(bytes + i * CHUNK_SIZE + 48), load);
  }
}

int main()
{
  return 0;
}