#include <iostream>
#include <cmath>
#include <memory>

#include "diffclock.h"

#include <algorithm>


#include "test_cache_miss.hpp"
#include "test_pointer_alias/test_pointer_alias.h"
#include "test_SoA.hpp"
#include "test_inline/test_inline.h"
#include "test_data_oriented_design.hpp"
#include "test_float_double.hpp"

#include "test_virtual/test_virtual.h"

#include "test_ilp.h"

#include "test_threads.h"
#include "test_image.h"
//shared memory & shared nothing
//concurrency & parallelism
//subroutines & couroutines
//test_and_set
//amdahls law
//car-passenger observation
//coffman conditions
//foster methodology
//flynn taxonomy
//conditional variables / monitors
//spin lock, read-write lock, mutex, concurrent collections

int main(int argc, const char * argv[]) {
    std::cout << "Starting tests ...\n" << std::endl;

    auto t0 = getTime();
    PointerAlias::test();
    Threads::test();
    ILP::test();
    Virtual::test();
    FloatDouble::test();
    DataOrientedDesign::test();
    SoA::test();
    Inline::test();
    CacheMiss::test();
    
    auto t1 = getTime();
    
    std::cout << "Tests completed, time " << diffclock(t1, t0) << std::endl;
    
    return 0;
}


/*
 #include "simd/simd.h"
 void testSSE() {
 using namespace embree;
 size_t testSize = 1 << 15;
 
 std::unique_ptr<float[]> floats(new float[testSize]);
 
 std::generate(floats.get(),
 floats.get() + testSize,
 []{ return randomFloat();});
 
 auto sseSize = testSize/ssef::size;
 std::unique_ptr<ssef[]> sseFloats(new ssef[sseSize]);
 
 int floatIter = 0;
 for (int i = 0; i < sseSize; ++i) {
 sseFloats[i].load(floats.get() + floatIter);
 floatIter += 4;
 }
 
 auto time0 = getTime();
 std::for_each(floats.get(),
 floats.get() + testSize,
 [](float x) { return sqrtf(x); });
 auto time1 = getTime();
 std::cout << diffclock(time1, time0) << std::endl;
 auto time2 = getTime();
 std::for_each(sseFloats.get(),
 sseFloats.get() + sseSize,
 [](const ssef& f) { return sqrt(f); });
 auto time3 = getTime();
 
 std::cout << diffclock(time3, time2) << std::endl;
 
 }*/

