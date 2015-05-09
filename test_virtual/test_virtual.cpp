//
//  test_virtual.cpp
//  GPAPI
//
//  Created by savage309 on 9.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#include "test_virtual.h"
#include "handmade_virtual.h"
#include "native_virtual.h"
#include "diffclock.h"
#include <iostream>

namespace Virtual {
    size_t getTestSize() {
        return 100000000;
    }
    
    void test() {
        Virtual::init();
        
        std::cout << "Testing handmade virtual functions ...\n";
        
        size_t testSize = getTestSize();
        
        using namespace Virtual;
        
        Base** base = new Base*[testSize];
        for (int i = 0; i < testSize; ++i) {
            if (i % 2) {
                base[i] = new Derived;
            } else {
                base[i] = new Derived2;
            }
        }
        
        auto t0 = getTime();
        for (int i = 0; i < testSize; ++i) {
            base[i]->set(i * i);
            if (base[i]->get() > 5) {
                base[i]->set(base[i]->get() + 1);
            }
        }
        auto t1 = getTime();
        std::cout << '\t' << "Handmade vfuncs " << diffclock(t1, t0) << std::endl;

        NBase** nbase = new NBase*[testSize];
        for (int i = 0; i < testSize; ++i) {
            if (i % 2) {
                nbase[i] = new NDerived;
            } else {
                nbase[i] = new NDerived2;
            }
        }
        
        auto t2 = getTime();
        for (int i = 0; i < testSize; ++i) {
            nbase[i]->set(i * i);
            if (nbase[i]->get() > 5) {
                nbase[i]->set(nbase[i]->get() + 1);
            }
        }
        auto t3 = getTime();
        
        std::cout << '\t' << "Native vfuncs " << diffclock(t3, t2) << std::endl;
        std::cout << "\n **** \n\n";

    }
}