//
//  inline.h
//  GPAPI
//
//  Created by savage309 on 6.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef GPAPI_inline_h
#define GPAPI_inline_h

#include "common.h"
#include <cmath>
#include <iostream>
#include "inline_impl.h"

namespace Inline {
    INLINE float calcInline(float f) {
        if (f > 0)
            return .5f;
        else return cos(sin(atan(f)));
    }

    size_t getTestSize() {
        return 200000000ULL;
    }
    
    void test(){
        std::cout << "Testing inlined functions ..." << std::endl;
        float res = 0.f;
        const auto TEST_SIZE = getTestSize();
        auto begin0 = getTime();
        for(size_t a = 0; a < TEST_SIZE; ++a)
            for(int b = 0; b < 5; ++b)
                res += calcInline(b);
        auto end0 = getTime();
        std::cout << "inline " << diffclock(end0, begin0) << std::endl;
        
        auto begin1 = getTime();
        for(size_t a = 0; a < TEST_SIZE; ++a)
            for(int b = 0; b < 5; ++b)
                res += calcNoInline(b);
        auto end1 = getTime();
        std::cout << "no inline " << diffclock(end1, begin1) << std::endl;
        

        std::cout << "\n **** \n\n";
    }

} //namespace Inline

#endif
