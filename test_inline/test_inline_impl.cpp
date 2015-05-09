//
//  inline_test.cpp
//  GPAPI
//
//  Created by savage309 on 9.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#include "test_inline_impl.h"
#include <cmath>
namespace Inline {
    float calcNoInline(float f) {
        if (f > 0)
            return .5f;
        else return cos(sin(atan(f)));

    }

}