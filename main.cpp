#include <iostream>
#include <cmath>
#include <memory>

#include "gpapi_host.h"
#include "cache_miss.hpp"

#include "diffclock.h"

#include "pointer_alias.h"

#include "SoA.h"

namespace PointerAlias {
    void test();
} //namespace PointerAlias

int main(int argc, const char * argv[]) {
    srand (0);
    
    SoA::test();
    
    CacheMiss::test();
    
    PointerAlias::test();
    
    return 0;
    
    /*
    //query for devices
    int numDevices = 0;
    GPAPI::Result err = GPAPI::Result::Success;
    err = GPAPI::getNumDevices(GPAPI::API::Native, &numDevices);
    CHECK_ERROR(err);

    //init a device
    GPAPI::Device* device = NULL;
    err = GPAPI::newDevice(GPAPI::Native, 0, device);
    CHECK_ERROR(err);
    
    //load kernel
    GPAPI::Kernel* kernel = NULL;
    err = GPAPI::newKernel(GPAPI::Native, 0, "fooBar", kernel);
    CHECK_ERROR(err);
    
    //allocate device buffer
    GPAPI::Buffer* buffer = NULL;
    err = GPAPI::newBuffer(GPAPI::Native, 0, buffer);
    CHECK_ERROR(err);
    int* ptr = new int; *ptr = 196;
    err = buffer->alloc(ptr, sizeof(int));
    CHECK_ERROR(err);
    
    //add params to a kernel
    err = kernel->addParam(*buffer);
    CHECK_ERROR(err);

    //run kernel on a device
    device->runKernel(*kernel, GPAPI::KernelConfiguration(32, 32*128));
    err = buffer->download(ptr, sizeof(int));
    CHECK_ERROR(err);
    
    //clean up
    delete device;
    delete buffer;
    delete kernel;
    delete ptr;
    
    return 0;*/
}


namespace PointerAlias {
    void test() {
        std::cout << "Testing pointer alias ..." << std::endl;
        auto POINTER_ALIAS_TEST_SIZE = 300000000;
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
            
            auto begin0 = clock();
            
            PointerAlias::A* bInt = (PointerAlias::A*)b.get();
            PointerAlias::B* resUnsigned = (PointerAlias::B*)res.get();
            PointerAlias::pointerAliasDifferentType(a.get(), bInt, resUnsigned, POINTER_ALIAS_TEST_SIZE);
            auto end0 = clock();
            res0 += diffclock(end0, begin0);
        }
        
        std::cout << "Different Type Time " << res0/ RUN_TIMES << std::endl;
        
        for (int t = 0; t < RUN_TIMES; ++t) {
            for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
                a[i] = i; b[i] = i * 2; res[i] = i * 3;
            }
            auto begin1 = clock();
            PointerAlias::pointerAliasSameType(a.get(), b.get(), res.get(), POINTER_ALIAS_TEST_SIZE);
            auto end1 = clock();
            res1 += diffclock(end1, begin1);
        }
        
        std::cout << "Same Type Time " << res1/ RUN_TIMES << std::endl;
        
        for (int t = 0; t < RUN_TIMES; ++t) {
            
            for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
                a[i] = i; b[i] = i * 2; res[i] = i * 3;
            }
            auto begin2 = clock();
            PointerAlias::pointerAliasDifferentTypeNoCast(a.get(), (PointerAlias::A*)b.get(), (PointerAlias::B*)res.get(), POINTER_ALIAS_TEST_SIZE);
            auto end2 = clock();
            res2 += diffclock(end2, begin2);
        }
        
        std::cout << "Different Type No Cast Time " << res2/ RUN_TIMES << std::endl;
        
        for (int t = 0; t < RUN_TIMES; ++t) {
            
            for (auto i = 0; i < POINTER_ALIAS_TEST_SIZE; ++i) {
                a[i] = i; b[i] = i * 2; res[i] = i * 3;
            }
            auto begin3 = clock();
            PointerAlias::pointerAliasSameTypeRestrict(a.get(), b.get(), res.get(), POINTER_ALIAS_TEST_SIZE);
            auto end3 = clock();
            res3 += diffclock(end3, begin3);
        }
        
        std::cout << "Different Type Restrict " << res3 / RUN_TIMES << std::endl;
        std::cout << "\n **** \n\n";

    }
    
} //namespace PointerAlias


