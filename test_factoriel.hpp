#pragma once

namespace Factoriel {
    
    #include <iostream>
    
    
    inline int factoriel0(int n) {
        if(n==0) return 1;
        else return factoriel0(n-1)*n;
    }
    
    inline int factoriel1(int n) {
        int r = 1;
        DISABLE_SIMD_UNROLL
        for(int k=n; k>0; --k)
            r *= k;
        
        return r;
    }
    
    
    inline int factoriel2(int n) {
        int r = 1;
        DISABLE_SIMD_UNROLL
        for(int k=1; k<=n; ++k)
            r *= k;
        
        return r;
    }
    
    constexpr int getTestSize() {
        return 100000000;
    }
    
    void test() {
        printf("Testing factoriel ...\n");
        auto s1 = getTime();
        DISABLE_SIMD_UNROLL
        for(int k=0; k<100; k++) {
            volatile int r1 = factoriel1(getTestSize());
        }
        auto e1 = getTime();
        
        auto s2 = getTime();
        DISABLE_SIMD_UNROLL
        for(int k=0; k<100; k++) {
            volatile int r2 = factoriel2(getTestSize());
        }
        
        auto e2 = getTime();
        
        
        auto s3 = getTime();
        DISABLE_SIMD_UNROLL
        for(int k=0; k<100; k++) {
            volatile int r3 = factoriel0(getTestSize());
        }
        
        auto e3 = getTime();
        
        
        std::cout << "\tfactoriel 1: " << diffclock(e1, s1) << std::endl << "\tfactoriel 2: " << diffclock(e2, s2) << std::endl << "\trecursive: "<< diffclock(e3, s3);
        std::cout << "\n **** \n\n";
    }

}