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
        auto RUN_TIMES = 2;
        std::unique_ptr<float[]> a(new float[POINTER_ALIAS_TEST_SIZE]);
        std::unique_ptr<float[]> b(new float[POINTER_ALIAS_TEST_SIZE]);
        std::unique_ptr<float[]> res(new float[POINTER_ALIAS_TEST_SIZE]);
        double res0 = 0.0;
        double res1 = 0.0;
        double res2 = 0.0;
        double res3 = 0.0;
        
        //#pragma clang loop vectorize(disable) interleave(disable)
        
        PointerAlias::pointerAliasSameType(a.get(), b.get(), res.get(), POINTER_ALIAS_TEST_SIZE);
        
        for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
            a[i] = i; b[i] = i * 2; res[i] = i * 3;
        }
        for (int t = 0; t < RUN_TIMES; ++t) {
           
            
            auto begin0 = getTime();
            
            PointerAlias::A* bInt = (PointerAlias::A*)b.get();
            PointerAlias::B* resUnsigned = (PointerAlias::B*)res.get();
            PointerAlias::pointerAliasDifferentType(a.get(), bInt, resUnsigned, POINTER_ALIAS_TEST_SIZE);
            auto end0 = getTime();
            if (t == 1)
            res0 += diffclock(end0, begin0);
        }
        
        std::cout << '\t' << "Different Type Time " << res0/ 1 << std::endl;
        for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
            a[i] = i; b[i] = i * 2; res[i] = i * 3;
        }
        for (int t = 0; t < RUN_TIMES; ++t) {
           
            auto begin1 = getTime();
            PointerAlias::pointerAliasSameType(a.get(), b.get(), res.get(), POINTER_ALIAS_TEST_SIZE);
            auto end1 = getTime();
            if (t == 1)
            res1 += diffclock(end1, begin1);
        }
        
        std::cout << '\t' << "Same Type Time " << res1/ 1 << std::endl;
        
        for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
            a[i] = i; b[i] = i * 2; res[i] = i * 3;
        }
        for (int t = 0; t < RUN_TIMES; ++t) {
            
           
            auto begin2 = getTime();
            PointerAlias::pointerAliasDifferentTypeNoCast(a.get(), (PointerAlias::A*)b.get(), (PointerAlias::B*)res.get(), POINTER_ALIAS_TEST_SIZE);
            auto end2 = getTime();
            if (t == 1)
            res2 += diffclock(end2, begin2);
        }
        
        std::cout << '\t' << "Different Type No Cast Time " << res2/ 1 << std::endl;
        
        for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
            a[i] = i; b[i] = i * 2; res[i] = i * 3;
        }
        for (int t = 0; t < RUN_TIMES; ++t) {
            auto begin3 = getTime();
            PointerAlias::pointerAliasSameTypeRestrict(a.get(), b.get(), res.get(), POINTER_ALIAS_TEST_SIZE);
            auto end3 = getTime();
            if (t == 1)
            res3 += diffclock(end3, begin3);
        }
        
        std::cout << '\t' << "Different Type Restrict " << res3 / 1 << std::endl;

#ifdef __AVX__
        double res4 = 0;
        std::unique_ptr<embree::ssef[]> sseA(new embree::ssef[getTestSize()/embree::ssef::size]);
        std::unique_ptr<embree::ssef[]> sseB(new embree::ssef[getTestSize()/embree::ssef::size]);
        std::unique_ptr<embree::ssef[]> sseRes(new embree::ssef[getTestSize()/embree::ssef::size]);
        
        for (int i = 0; i < getTestSize(); i+=embree::ssef::size) {
            sseA[i/embree::ssef::size].load(a.get()+i);
            sseB[i/embree::ssef::size].load(b.get()+i);
            sseRes[i/embree::ssef::size].load(res.get()+i);
        }
        

        for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
            a[i] = i; b[i] = i * 2; res[i] = i * 3;
        }
        
        for (int t = 0; t < RUN_TIMES; ++t) {
            auto begin4 = getTime();
            
            
            PointerAlias::pointerSIMD<embree::ssef>(sseA.get(), sseB.get(), sseRes.get(), POINTER_ALIAS_TEST_SIZE/embree::ssef::size);
            auto end4 = getTime();
            if (t == 1)
            res4 += diffclock(end4, begin4);
        }
        
        std::cout << '\t' << "SSE2  " << res4 / 1 << std::endl;
        
        double res5 = 0;
        sseA.reset();
        sseB.reset();
        sseRes.reset();
        
        
        for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
            a[i] = i; b[i] = i * 2; res[i] = i * 3;
        }
        
        std::unique_ptr<embree::avxf[]> avxA(new embree::avxf[getTestSize()/embree::avxf::size]);
        std::unique_ptr<embree::avxf[]> avxB(new embree::avxf[getTestSize()/embree::avxf::size]);
        std::unique_ptr<embree::avxf[]> avxRes(new embree::avxf[getTestSize()/embree::avxf::size]);
        
        for (int i = 0; i < getTestSize(); i+=embree::avxf::size) {
            avxA[i/embree::avxf::size].load(a.get()+i);
            avxB[i/embree::avxf::size].load(b.get()+i);
            avxRes[i/embree::avxf::size].load(res.get()+i);
        }
        
        for (int t = 0; t < RUN_TIMES; ++t) {
            auto begin5 = getTime();
            PointerAlias::pointerSIMD<embree::avxf>(avxA.get(), avxB.get(), avxRes.get(), POINTER_ALIAS_TEST_SIZE/embree::avxf::size);
            auto end5 = getTime();
            if (t == 1)
                res5 += diffclock(end5, begin5);
        }
        
        std::cout << '\t' << "AVX  " << res5 / 1 << std::endl;

        
        std::cout << "\n **** \n\n";

    }
#endif//__AVX__
    
} //namespace PointerAlias
