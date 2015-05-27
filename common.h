//
//  common.h
//  GPAPI
//
//  Created by savage309 on 6.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef GPAPI_common_h
#define GPAPI_common_h

#define RESTRICT __restrict__
#define NO_INLINE __attribute__ ((noinline))
#define INLINE __inline__ __attribute__((always_inline))

#if __cplusplus >= 201103L
#   define CPP11
#endif

#if defined(CPP11) && (defined __clang__)
#   define DISABLE_SIMD_UNROLL _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
#   define DISABLE_SIMD_UNROLL
#endif

#include <cmath>
#include <cstdlib>

inline float randomFloat() {
    return float(rand())/RAND_MAX;
}
inline int randomInt(int min, int max) {
    return min + (max - min) * randomFloat();
}


#endif
