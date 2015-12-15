
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <smmintrin.h> // SSE4.1

static __m128 _nn_movell_ps(
                            const __m128 a,
                            const __m128 b);

static void bar( // bitonic_simd_sort
                float (& inout)[8]) {
    
    const __m128 r0_in0 = _mm_load_ps(inout + 0); // 0, 1, 2, 3
    const __m128 r0_in1 = _mm_load_ps(inout + 4); // 4, 5, 6, 7
    
    // stage 0
    const __m128 r0_A = _mm_shuffle_ps(r0_in0, r0_in1, 0xcc); // 0, 3, 4, 7
    const __m128 r0_B = _mm_shuffle_ps(r0_in0, r0_in1, 0x99); // 1, 2, 5, 6
    
    const __m128 r0_min = _mm_min_ps(r0_A, r0_B); // 0, 3, 4, 7
    const __m128 r0_max = _mm_max_ps(r0_A, r0_B); // 1, 2, 5, 6
    
    // stage 1
    __m128 r1_A = _mm_shuffle_ps(r0_max, r0_max, 0xf0); // 1, 1, 6, 6
    __m128 r1_B = _mm_shuffle_ps(r0_max, r0_max, 0xa5); // 2, 2, 5, 5
    r1_A = _mm_blend_ps(r1_A, r0_min, 0x9);      // 0, 1, 6, 7
    r1_B = _mm_blend_ps(r1_B, r0_min, 0x6);      // 2, 3, 4, 5
    
    const __m128 r1_min = _mm_min_ps(r1_A, r1_B); // 0, 1, 6, 7
    const __m128 r1_max = _mm_max_ps(r1_A, r1_B); // 2, 3, 4, 5
    
    // stage 2
    __m128 r2_A = _mm_shuffle_ps(r1_max, r1_max, 0xf0); // 2, 2, 5, 5
    __m128 r2_B = _mm_shuffle_ps(r1_min, r1_min, 0xa5); // 1, 1, 6, 6
    r2_A = _mm_blend_ps(r2_A, r1_min, 0x9);      // 0, 2, 5, 7
    r2_B = _mm_blend_ps(r2_B, r1_max, 0x6);      // 1, 3, 4, 6
    
    const __m128 r2_min = _mm_min_ps(r2_A, r2_B); // 0, 2, 5, 7
    const __m128 r2_max = _mm_max_ps(r2_A, r2_B); // 1, 3, 4, 6
    
    // stage 3
    const __m128 r3_A = _mm_unpacklo_ps(r2_min, r2_max); // 0, 1, 2, 3
    const __m128 r3_B = _mm_unpackhi_ps(r2_max, r2_min); // 4, 5, 6, 7
    
    const __m128 r3_min = _mm_min_ps(r3_A, r3_B); // 0, 1, 2, 3
    const __m128 r3_max = _mm_max_ps(r3_A, r3_B); // 4, 5, 6, 7
    
    // stage 4
    const __m128 r4_A = _mm_movelh_ps(r3_min, r3_max); // 0, 1, 4, 5
    const __m128 r4_B = _mm_movehl_ps(r3_max, r3_min); // 2, 3, 6, 7
    
    const __m128 r4_min = _mm_min_ps(r4_A, r4_B); // 0, 1, 4, 5
    const __m128 r4_max = _mm_max_ps(r4_A, r4_B); // 2, 3, 6, 7
    
    // stage 5
    const __m128 r5_a = _mm_unpacklo_ps(r4_min, r4_max); // 0, 2, 1, 3
    const __m128 r5_b = _mm_unpackhi_ps(r4_min, r4_max); // 4, 6, 5, 7
    const __m128 r5_A = _mm_movelh_ps(r5_a, r5_b); // 0, 2, 4, 6
    const __m128 r5_B = _mm_movehl_ps(r5_b, r5_a); // 1, 3, 5, 7
    
    const __m128 r5_min = _mm_min_ps(r5_A, r5_B); // 0, 2, 4, 6
    const __m128 r5_max = _mm_max_ps(r5_A, r5_B); // 1, 3, 5, 7
    
    // output
    const __m128 out0 = _mm_unpacklo_ps(r5_min, r5_max); // 0, 1, 2, 3
    const __m128 out1 = _mm_unpackhi_ps(r5_min, r5_max); // 4, 5, 6, 7
    
    _mm_store_ps(inout + 0, out0);
    _mm_store_ps(inout + 4, out1);
}

static void foo( // odd_even_simd_sort
                float (& inout)[8]) {
    
    const __m128 r0_in0 = _mm_load_ps(inout + 0); // 0, 1, 2, 3
    const __m128 r0_in1 = _mm_load_ps(inout + 4); // 4, 5, 6, 7
    
    // stage 0
    const __m128 r0_A = _mm_shuffle_ps(r0_in0, r0_in1, 0x88); // 0, 2, 4, 6
    const __m128 r0_B = _mm_shuffle_ps(r0_in0, r0_in1, 0xdd); // 1, 3, 5, 7
    
    const __m128 r0_min = _mm_min_ps(r0_A, r0_B); // 0, 2, 4, 6
    const __m128 r0_max = _mm_max_ps(r0_A, r0_B); // 1, 3, 5, 7
    
    // stage 1
    const __m128 r1_A = _mm_shuffle_ps(r0_min, r0_max, 0x88); // 0, 4, 1, 5
    const __m128 r1_B = _mm_shuffle_ps(r0_min, r0_max, 0xdd); // 2, 6, 3, 7
    
    const __m128 r1_min = _mm_min_ps(r1_A, r1_B); // 0, 4, 1, 5
    const __m128 r1_max = _mm_max_ps(r1_A, r1_B); // 2, 6, 3, 7
    
    // stage 2
    const __m128 r2_A = _mm_movehl_ps(r1_min, r1_min); // 1, 5, -, -
    const __m128 r2_B = r1_max;                        // 2, 6, -, -
    
    const __m128 r2_min = _mm_min_ps(r2_A, r2_B); // 1, 5, -, -
    const __m128 r2_max = _mm_max_ps(r2_A, r2_B); // 2, 6, -, -
    
    const __m128 r2_out0 = _mm_movelh_ps(r1_min, r2_min); // 0, 4, 1, 5
    const __m128 r2_out1 = _nn_movell_ps(r1_max, r2_max); // 2, 6, 3, 7
    
    // stage 3
    const __m128 r3_A = _mm_shuffle_ps(r2_out0, r2_out1, 0x88); // 0, 1, 2, 3
    const __m128 r3_B = _mm_shuffle_ps(r2_out0, r2_out1, 0xdd); // 4, 5, 6, 7
    
    const __m128 r3_min = _mm_min_ps(r3_A, r3_B); // 0, 1, 2, 3
    const __m128 r3_max = _mm_max_ps(r3_A, r3_B); // 4, 5, 6, 7
    
    // stage 4
    const __m128 r4_A = _mm_movehl_ps(r3_min, r3_min); // 2, 3, -, -
    const __m128 r4_B = r3_max;                        // 4, 5, -, -
    
    const __m128 r4_min = _mm_min_ps(r4_A, r4_B); // 2, 3, -, -
    const __m128 r4_max = _mm_max_ps(r4_A, r4_B); // 4, 5, -, -
    
    const __m128 r4_out0 = _mm_movelh_ps(r3_min, r4_min); // 0, 1, 2, 3
    const __m128 r4_out1 = _nn_movell_ps(r3_max, r4_max); // 4, 5, 6, 7
    
    // stage 5
    const __m128 r5_A = _mm_insert_ps(r4_out0, r4_out1, 0x60); // 0, 1, 5, 3
    __m128 r5_B = _mm_insert_ps(r4_out1, r4_out0, 0x90); // -, 2, 6, -
    r5_B = _mm_insert_ps(r5_B,    r4_out1, 0x30); // -, 2, 6, 4
    
    const __m128 r5_min = _mm_min_ps(r5_A, r5_B); // 0, 1, 5, 3
    const __m128 r5_max = _mm_max_ps(r5_A, r5_B); // -, 2, 6, 4
    
    // output
    const __m128 r5_out0 = _mm_insert_ps(r5_min,  r5_max, 0x60); // 0, 1, 2, 3
    __m128 r5_out1 = _mm_insert_ps(r4_out1, r5_max, 0xc0); // 4, -, -, 7
    r5_out1 = _mm_insert_ps(r5_out1, r5_max, 0xa0); // 4, -, 6, 7
    r5_out1 = _mm_insert_ps(r5_out1, r5_min, 0x90); // 4, 5, 6, 7
    
    _mm_store_ps(inout + 0, r5_out0);
    _mm_store_ps(inout + 4, r5_out1);
}

static void reverse(
                        char* const inout,
                        const size_t chunk_count) {
    
    const size_t count = chunk_count * 64;
    
    for (size_t i = 0; i < count; i += 64) {
        
        //_mm_prefetch(inout + i + 8 * 64, _MM_HINT_T0);
        const uint64_t byteswap[8] = {
            __builtin_bswap64(reinterpret_cast< uint64_t* >(inout + i)[0]),
            __builtin_bswap64(reinterpret_cast< uint64_t* >(inout + i)[1]),
            __builtin_bswap64(reinterpret_cast< uint64_t* >(inout + i)[2]),
            __builtin_bswap64(reinterpret_cast< uint64_t* >(inout + i)[3]),
            __builtin_bswap64(reinterpret_cast< uint64_t* >(inout + i)[4]),
            __builtin_bswap64(reinterpret_cast< uint64_t* >(inout + i)[5]),
            __builtin_bswap64(reinterpret_cast< uint64_t* >(inout + i)[6]),
            __builtin_bswap64(reinterpret_cast< uint64_t* >(inout + i)[7])
        };
        
        for (size_t j = 0; j < 8; ++j)
            reinterpret_cast< uint64_t* >(inout + i)[j] = byteswap[7 - j];
    }
}

