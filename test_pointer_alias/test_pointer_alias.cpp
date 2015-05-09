//
//  pointer_aliash.cpp
//  GPAPI
//
//  Created by savage309 on 6.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#include <stdio.h>
#include "test_pointer_alias.h"
#include "diffclock.h"
#include <iostream>
namespace PointerAlias {
    void test() {
        std::cout << "Testing pointer alias ..." << std::endl;
        auto POINTER_ALIAS_TEST_SIZE = PointerAlias::getTestSize();
        auto RUN_TIMES = 5;
        std::unique_ptr<float[]> a(new float[POINTER_ALIAS_TEST_SIZE]);
        std::unique_ptr<float[]> b(new float[POINTER_ALIAS_TEST_SIZE]);
        std::unique_ptr<float[]> res(new float[POINTER_ALIAS_TEST_SIZE]);
        double res0 = 0.0;
        double res1 = 0.0;
        double res2 = 0.0;
        double res3 = 0.0;
        for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
            a[i] = i; b[i] = i * 2; res[i] = i * 3;
        }
        PointerAlias::pointerAliasSameType(a.get(), b.get(), res.get(), POINTER_ALIAS_TEST_SIZE);
        
        for (int t = 0; t < RUN_TIMES; ++t) {
            for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
                a[i] = i; b[i] = i * 2; res[i] = i * 3;
            }
            
            auto begin0 = getTime();
            
            PointerAlias::A* bInt = (PointerAlias::A*)b.get();
            PointerAlias::B* resUnsigned = (PointerAlias::B*)res.get();
            PointerAlias::pointerAliasDifferentType(a.get(), bInt, resUnsigned, POINTER_ALIAS_TEST_SIZE);
            auto end0 = getTime();
            res0 += diffclock(end0, begin0);
        }
        
        std::cout << '\t' << "Different Type Time " << res0/ RUN_TIMES << std::endl;
        
        for (int t = 0; t < RUN_TIMES; ++t) {
            for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
                a[i] = i; b[i] = i * 2; res[i] = i * 3;
            }
            auto begin1 = getTime();
            PointerAlias::pointerAliasSameType(a.get(), b.get(), res.get(), POINTER_ALIAS_TEST_SIZE);
            auto end1 = getTime();
            res1 += diffclock(end1, begin1);
        }
        
        std::cout << '\t' << "Same Type Time " << res1/ RUN_TIMES << std::endl;
        
        for (int t = 0; t < RUN_TIMES; ++t) {
            
            for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
                a[i] = i; b[i] = i * 2; res[i] = i * 3;
            }
            auto begin2 = getTime();
            PointerAlias::pointerAliasDifferentTypeNoCast(a.get(), (PointerAlias::A*)b.get(), (PointerAlias::B*)res.get(), POINTER_ALIAS_TEST_SIZE);
            auto end2 = getTime();
            res2 += diffclock(end2, begin2);
        }
        
        std::cout << '\t' << "Different Type No Cast Time " << res2/ RUN_TIMES << std::endl;
        
        for (int t = 0; t < RUN_TIMES; ++t) {
            
            for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
                a[i] = i; b[i] = i * 2; res[i] = i * 3;
            }
            auto begin3 = getTime();
            PointerAlias::pointerAliasSameTypeRestrict(a.get(), b.get(), res.get(), POINTER_ALIAS_TEST_SIZE);
            auto end3 = getTime();
            res3 += diffclock(end3, begin3);
        }
        
        std::cout << '\t' << "Different Type Restrict " << res3 / RUN_TIMES << std::endl;
        std::cout << "\n **** \n\n";
        
    }
    
} //namespace PointerAlias
