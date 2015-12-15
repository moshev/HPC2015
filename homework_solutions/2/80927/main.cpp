// Compile with -mavx (/arch:AVX for VS)
#include<stdio.h>
#include "immintrin.h"
#include<algorithm>

struct ReverseData
{
    union
    {
        char bytes[32];
        __m256i avx;
    };
};

inline void reverse(char* bytes, int numChunks)
{
#pragma unroll
    for (size_t i = 0; i < numChunks; ++i)
    {
        ReverseData d[2];
        d[0].avx = _mm256_setr_epi8(bytes[i * 64 + 63], bytes[i * 64 + 62], bytes[i * 64 + 61], bytes[i * 64 + 60], bytes[i * 64 + 59], bytes[i * 64 + 58], bytes[i * 64 + 57],
            bytes[i * 64 + 56], bytes[i * 64 + 55], bytes[i * 64 + 54], bytes[i * 64 + 53], bytes[i * 64 + 52], bytes[i * 64 + 51], bytes[i * 64 + 50], bytes[i * 64 + 49],
            bytes[i * 64 + 48], bytes[i * 64 + 47], bytes[i * 64 + 46], bytes[i * 64 + 45], bytes[i * 64 + 44], bytes[i * 64 + 43], bytes[i * 64 + 42], bytes[i * 64 + 41],
            bytes[i * 64 + 40], bytes[i * 64 + 39], bytes[i * 64 + 38], bytes[i * 64 + 37], bytes[i * 64 + 36], bytes[i * 64 + 35], bytes[i * 64 + 34], bytes[i * 64 + 33],
            bytes[i * 64 + 32]);
        
        d[1].avx = _mm256_setr_epi8(bytes[i * 64 + 31], bytes[i * 64 + 30], bytes[i * 64 + 29], bytes[i * 64 + 28], bytes[i * 64 + 27], bytes[i * 64 + 26], bytes[i * 64 + 25],
            bytes[i * 64 + 24], bytes[i * 64 + 23], bytes[i * 64 + 22], bytes[i * 64 + 21], bytes[i * 64 + 20], bytes[i * 64 + 19], bytes[i * 64 + 18], bytes[i * 64 + 17],
            bytes[i * 64 + 16], bytes[i * 64 + 15], bytes[i * 64 + 14], bytes[i * 64 + 13], bytes[i * 64 + 12], bytes[i * 64 + 11], bytes[i * 64 + 10], bytes[i * 64 + 9],
            bytes[i * 64 + 8], bytes[i * 64 + 7], bytes[i * 64 + 6], bytes[i * 64 + 5], bytes[i * 64 + 4], bytes[i * 64 + 3], bytes[i * 64 + 2], bytes[i * 64 + 1],
            bytes[i * 64]);

#pragma unroll
        for (size_t j = 0; j < 64; j++)
        {
            bytes[i * 64 + j] = d[j / 32].bytes[j % 32];
        }
    }
}

struct BarData
{
    union
    {
        float sort[8];
        __m256 avx;
    };
};

static void foo(
    float(&inout)[8]) {

    const size_t idx[][2] = {
        { 0, 1 },{ 2, 3 },{ 4, 5 },{ 6, 7 },
        { 0, 2 },{ 1, 3 },{ 4, 6 },{ 5, 7 },
        { 1, 2 },{ 5, 6 },
        { 0, 4 },{ 1, 5 },{ 2, 6 },{ 3, 7 },
        { 2, 4 },{ 3, 5 },
        { 1, 2 },{ 3, 4 },{ 5, 6 }
    };

    for (size_t i = 0; i < sizeof(idx) / sizeof(idx[0]); ++i) {
        const float x = inout[idx[i][0]];
        const float y = inout[idx[i][1]];

        inout[idx[i][0]] = std::min(x, y);
        inout[idx[i][1]] = std::max(x, y);
    }
}

inline static void bar(
    float(&inout)[8]) {

    const size_t idx[][2] = {
        { 0, 1 },{ 3, 2 },{ 4, 5 },{ 7, 6 },
        { 0, 2 },{ 1, 3 },{ 6, 4 },{ 7, 5 },
        { 0, 1 },{ 2, 3 },{ 5, 4 },{ 7, 6 },
        { 0, 4 },{ 1, 5 },{ 2, 6 },{ 3, 7 },
        { 0, 2 },{ 1, 3 },{ 4, 6 },{ 5, 7 },
        { 0, 1 },{ 2, 3 },{ 4, 5 },{ 6, 7 }
    };

    BarData leftData;
    BarData rightData;
    BarData minData;
    BarData maxData;


#pragma unroll
    for (size_t i = 0, size = sizeof(idx) / sizeof(idx[0]); i < size; i += 4) {
        leftData.avx = _mm256_setr_ps(inout[idx[i][0]], inout[idx[i + 1][0]], inout[idx[i + 2][0]], inout[idx[i + 3][0]], 0, 0, 0, 0);
        rightData.avx = _mm256_setr_ps(inout[idx[i][1]], inout[idx[i + 1][1]], inout[idx[i + 2][1]], inout[idx[i + 3][1]], 0, 0, 0, 0);
        minData.avx = _mm256_min_ps(leftData.avx, rightData.avx);
        maxData.avx = _mm256_max_ps(leftData.avx, rightData.avx);
        leftData = minData;
        rightData = maxData;
        inout[idx[i][0]] = leftData.sort[0];
        inout[idx[i][1]] = rightData.sort[0];
        inout[idx[i + 1][0]] = leftData.sort[1];
        inout[idx[i + 1][1]] = rightData.sort[1];
        inout[idx[i + 2][0]] = leftData.sort[2];
        inout[idx[i + 2][1]] = rightData.sort[2];
        inout[idx[i + 3][0]] = leftData.sort[3];
        inout[idx[i + 3][1]] = rightData.sort[3];
    }
}
