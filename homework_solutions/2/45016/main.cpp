#include <tmmintrin.h>

void reverse(char* bytes, int numChunks)
{
    __m128i mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    int i;
    
    for(i = 0; i < numChunks; ++i)
    {
        //first two
        __m128i last = _mm_load_si128((__m128i *)(bytes + 48));
        
        _mm_store_si128((__m128i *)(bytes+48),
                        _mm_shuffle_epi8(_mm_load_si128((__m128i *)bytes), mask));
        
        _mm_store_si128((__m128i *)(bytes),
                        _mm_shuffle_epi8(last, mask));
        
        
        //second two
        __m128i lastS = _mm_load_si128((__m128i *)(bytes + 32));
        
        _mm_store_si128((__m128i *)(bytes+ 32),
                        _mm_shuffle_epi8(_mm_load_si128((__m128i *)(bytes+16)), mask));
        
        _mm_store_si128((__m128i *)(bytes+16),
                        _mm_shuffle_epi8(lastS, mask));
        
        
        bytes = bytes + 64;
    }
}


int main()
{
    return 0;
}