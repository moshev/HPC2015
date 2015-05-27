#pragma once

#include "diffclock.h"
#include <memory>
#include <algorithm>
#include <limits>
#include <iostream>

namespace ILP {    
    constexpr size_t getTestSize() {
        return 42; //no point in doing this because of the compiler optimizations 
    }
    
    void testDependency0(bool printResult) {
        float x = randomFloat();
        float y = randomFloat();
        float z = randomFloat();
        float q = 0;
        float a = randomFloat();
        float b = randomFloat();
        auto t0 = getTime();
        for (int i =0 ; i < getTestSize(); ++i) {
            x = y * z;
            q += z + x + x;
            x = a + b;
        }
        auto t1 = getTime();
        //
        auto t2 = getTime();
        for (int i =0 ; i < getTestSize(); ++i) {
            float x0 = y * z;
            q += z + x0 + x0;
            x = a + b;
        }
        auto t3 = getTime();
        if (printResult)
            std::cout << "\t testDependency0 " << diffclock(t1, t0) << " vs " << diffclock(t3, t2) << std::endl;
    }
    
    void testDependency1(bool printResult) {
        std::unique_ptr<float[]> a(new float[getTestSize()]);
        std::unique_ptr<float[]> b(new float[getTestSize()]);
        std::for_each(a.get(), a.get() + getTestSize(), [](float) { return randomFloat();});
        std::for_each(b.get(), b.get() + getTestSize(), [](float) { return randomFloat();});
        auto t0 = getTime();

        for (int i = 0; i < getTestSize(); ++i) {
            a[i] = b[i];
            b[i] = b[i] + randomFloat();
        }
        auto t1 = getTime();

        //
        auto t2 = getTime();
        for (int i = 0; i < getTestSize(); ++i) {
            float temp = b[i];
            a[i] = temp;
            b[i] = temp + randomFloat();
        }
        auto t3 = getTime();

        if (printResult)
            std::cout << "\t testDependency1 " << diffclock(t1, t0) << " vs " << diffclock(t3, t2) << std::endl;
        
    }
    
    void testConstants0(bool printResult) {
        float x = randomFloat();
        float y = randomFloat();
        float z = 0.f;
        auto t0 = getTime();

        for (int i = 0; i < getTestSize(); ++i) {
            z += .5f * (x + y);
            y += z * (x + y) * 42.f;
        }
        auto t1 = getTime();

        //
        auto t2 = getTime();

        float temp = x + y;
        for (int i = 0; i < getTestSize(); ++i) {
            z += .5f * temp;
            y += z * temp * 42.f;
        }
        auto t3 = getTime();

        //how does operator[] works ? Do we happen to use it in arrays ?
        if (printResult)
            std::cout << "\t testConstants0 " << diffclock(t1, t0) << " vs " << diffclock(t3, t2) << std::endl;
        
    }
    
    void testLoopBranch0(bool printResult) {
        float x = randomFloat();
        float y = randomFloat();
        float z = 0.f;
        auto t0 = getTime();

        for (int i = 0; i < getTestSize(); ++i) {
            if (x + y < 0.5f) {
                z += sinf(x + i);
            } else {
                z += cosf(y + i);
            }
        }
        auto t1 = getTime();

        //
        auto t2 = getTime();

        if (x + y < 0.5f) {
            for (int i = 0; i < getTestSize(); ++i) {
                z += sinf(x + i);
            }
        } else {
            for (int i = 0; i < getTestSize(); ++i) {
                z += cosf(y + i);
            }
        }
        auto t3 = getTime();

        if (printResult)
            std::cout << "\t testLoopBranch0 " << diffclock(t1, t0) << " vs " << diffclock(t3, t2) << std::endl;
        
    }

    void testReduction(bool printResult) {
        std::unique_ptr<int[]> arr(new int[getTestSize()]);
        std::for_each(arr.get(), arr.get() + getTestSize(), [](int){ return randomInt(0, 196); });
        auto t0 = getTime();

        int v = std::numeric_limits<int>::min();
        for (int i = 0; i < getTestSize(); ++i) {
            v = std::max(v, arr[i]);
        }
        auto t1 = getTime();

        //
        
        int v0 = std::numeric_limits<int>::min();
        int v1 = std::numeric_limits<int>::min();
        int v2 = std::numeric_limits<int>::min();
        int v3 = std::numeric_limits<int>::min();
        auto t2 = getTime();

        for (int i = 0; i < getTestSize(); i+=4) {
            v0 = std::max(v0, arr[i + 0]);
            v1 = std::max(v1, arr[i + 1]);
            v2 = std::max(v2, arr[i + 2]);
            v3 = std::max(v3, arr[i + 3]);
        }
        v = std::max(v, v0);
        v = std::max(v, v1);
        v = std::max(v, v2);
        v = std::max(v, v3);
        auto t3 = getTime();

        if (printResult)
            std::cout << "\t testReduction " << diffclock(t1, t0) << " vs " << diffclock(t3, t2) << std::endl;
        
    }
   
    void test() {
        std::cout << "Testing ILP ..." << std::endl;
        
        testDependency0(false);
        testDependency0(true);
        
        testDependency1(false);
        testDependency1(true);
        
        testConstants0(false);
        testConstants0(true);
        
        testLoopBranch0(false);
        testLoopBranch0(true);
        
        testReduction(false);
        testReduction(true);
        
        std::cout << "\n **** \n\n";
        
    }
}
