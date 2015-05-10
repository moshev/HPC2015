//
//  threads.h
//  GPAPI
//
//  Created by savage309 on 10.05.15.
//  Copyright (c) 2015 г. savage309. All rights reserved.
//

#ifndef GPAPI_threads_h
#define GPAPI_threads_h

#include <thread>
#include <iostream>
#include <functional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace Threads {
    void testHelloWorld() {
        std::function<void()> helloWorld = [](){std::cout << "Hello World";};
        std::thread t(helloWorld);
        t.join();
    }
    int getNumThreads() {
        auto hardwareThreads = std::thread::hardware_concurrency();
        return hardwareThreads !=0 ? hardwareThreads : 1;
    }
    
    template<typename Iterator, typename T>
    T testAccumulateParallel(Iterator first, Iterator last) {
        auto accumulateBlock = [](Iterator first, Iterator last, T& result) {
           result = std::accumulate(first, last, result);
        };
        
        const auto length = std::distance(first,last);
        T init = 0;
        if (length < 64)
            return std::accumulate(first, last, 0);
        
        const auto numThreads = getNumThreads();
        const auto blockSize = length/numThreads;
        std::vector<T> results(numThreads);
        std::vector<std::thread> threads(numThreads - 1);
        Iterator blockStart = first;
        
        for(int i = 0; i < (numThreads-1); ++i) {
            Iterator blockEnd = blockStart;
            std::advance(blockEnd, blockSize);
            threads[i] = std::thread(accumulateBlock,
                                    blockStart,
                                    blockEnd,
                                    std::ref(results[i]));
            blockStart = blockEnd;
        }
        accumulateBlock(blockStart, last, results[numThreads-1]);
        
        for (auto& t: threads) {
            t.join();
        }
        
        return std::accumulate(results.begin(), results.end(), init);

    }
    void test() {
        testHelloWorld();
    }
}

#endif
