#include <stdio.h>

#include <emmintrin.h>
#include <vector>

float derCorput(unsigned n)
{
    float result = 0.0f;

    __m128i mask1 = _mm_setr_epi32(8, 4, 2, 1);
    __m128i mask2 = _mm_setr_epi32(16, 8, 4, 2);
    const __m128i shift = _mm_set1_epi32(4);
    const __m128i ns = _mm_set1_epi32(n);
    __m128 results = _mm_setzero_ps();

    __m128 maskPs, tmp;
    __m128i ms;

    
#define LOOP \
    maskPs = _mm_cvtepi32_ps(mask2);                                                 \
    ms = _mm_cmpgt_epi32(_mm_and_si128(ns, mask1), _mm_setzero_si128());            \
    tmp = _mm_castsi128_ps(_mm_and_si128(ms, _mm_castps_si128(_mm_rcp_ps(maskPs)))); \
    tmp = _mm_sub_ps(_mm_add_ps(tmp, tmp), _mm_mul_ps(_mm_mul_ps(tmp, tmp), maskPs));       \
    results = _mm_add_ps(results, tmp);                                                     \
                                                                                            \
    mask1 = _mm_slli_epi32(mask1, 4);                                                       \
    mask2 = _mm_slli_epi32(mask2, 4);


    //bits 0-3
    LOOP
    //bits 4-7
    LOOP
    //bits 8-11
    LOOP
    //bits 12-15
    LOOP
    //bits 16-19
    LOOP
    //bits 20-23
    LOOP
    //bits 24-27
    LOOP
    //bits 27-31
    LOOP

    __m128 h = _mm_add_ps(results, _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(results), _MM_SHUFFLE(1, 0, 3, 2))));
    h = _mm_add_ps(h, _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(h), _MM_SHUFFLE(2, 3, 0, 1))));

    result = _mm_cvtss_f32(h);

    return result;
}

#include <chrono>

int main()
{
    using namespace std::chrono;
    auto nums = 1 << 16;
    std::vector<float> vc;
    vc.reserve(nums);
    auto now = high_resolution_clock::now();
    for(auto i = 0; i < nums; ++i) {
        //printf("%f, ", derCorput(i));
        vc.push_back(derCorput(i));
    }
    auto time = high_resolution_clock::now() - now;
    for(auto i = 0; i < nums; ++i) {
        printf("%f, ", vc[i]);
    }
    printf("%lld ms\n", duration_cast<milliseconds>(time).count());
    printf("\n");
    return 0;
}