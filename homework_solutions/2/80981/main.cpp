#include<immintrin.h>
#include<cstdio>

void reverse(char* bytes, int numChunks)
{
    __m128i shufflemask = _mm_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);

    for(int i = 0; i < numChunks; ++i){
        int offset = i * 64;
        __m128i a = _mm_lddqu_si128((__m128i*)((bytes + offset)));
        __m128i b = _mm_lddqu_si128((__m128i*)((bytes + offset)+16));
        __m128i c = _mm_lddqu_si128((__m128i*)((bytes + offset)+32));
        __m128i d = _mm_lddqu_si128((__m128i*)((bytes + offset)+48));

        __m128i sa = _mm_shuffle_epi8(a, shufflemask);
        __m128i sb = _mm_shuffle_epi8(b, shufflemask);
        __m128i sc = _mm_shuffle_epi8(c, shufflemask);
        __m128i sd = _mm_shuffle_epi8(d, shufflemask);

        _mm_storeu_si128((__m128i*)((bytes + offset)), sd);
        _mm_storeu_si128((__m128i*)((bytes + offset)+16), sc);
        _mm_storeu_si128((__m128i*)((bytes + offset)+32), sb);
        _mm_storeu_si128((__m128i*)((bytes + offset)+48), sa);
    }
    return;
}

int main(int argc, char *argv[])
{
    char bytes[128];
    for (int i = 0; i < 128; ++i)
        bytes[i] = i;

    reverse(bytes, (128/64));

    printf("%i %i %i %i\n", bytes[0], bytes[1], bytes[126], bytes[127]); //63 62 65 64
    
    return 0;
}
