//
//  float_double.hpp
//  GPAPI
//
//  Created by savage309 on 9.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef GPAPI_float_double_hpp
#define GPAPI_float_double_hpp

#include "diffclock.h"

namespace FloatDouble {
    size_t getTestSize() {
        return 200000000;
    }
    
    NO_INLINE
    double testDouble(double f) {
        for (int i = 0; i < getTestSize(); ++i) {
            auto t = cos(f);
            if (pow(t, 3.0) < 0.5) {
                t += 5 * t;
                t -= 2 * sin(t);
            }
            f += t;
            if (f > 0.1) {
                f /= 10.0;
                f += randomFloat();
            }
            f *= 196.0;
        }
        return f;
    }
    
    NO_INLINE
    float testFloat(float f) {
        for (int i = 0; i < getTestSize(); ++i) {
            auto t = cosf(f);
            if (powf(t, 3.0f) < 0.5f) {
                t += 5 * t;
                t -= 2 * sinf(t);
            }
            f += t;
            if (f > 0.1f) {
                f /= 10.0f;
                f += randomFloat();
            }
            f *= 196.0f;
        }
        return f;

    }
    
    void test() {
        std::cout << "Testing float vs double ..." << std::endl;
        
        auto d = 0.0;
        auto time0 = getTime();
        d += testDouble(d);
        auto time1 = getTime();
        std::cout << '\t' << "Double " << diffclock(time1, time0) << std::endl;

        auto f = 0.0f;
        auto time2 = getTime();
        d += testFloat(d);
        auto time3 = getTime();
        
        std::cout << '\t' << "Float " << diffclock(time3, time2) << std::endl;
    }
}

#endif
