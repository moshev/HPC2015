#pragma once

#include "stdlib.h"
#include "stdio.h"

namespace GPAPI {
    enum API {
        CUDA = 0,
        OpenCL = 1,
        Native = 2,
    };
    
    enum Result {
        Success = 0,
        InvalidContext = 1,
        OutOfMem = 2,
        RuntimeError = 3,
        InvalidDeviceIndex = 4,
        UnknownError = 999,
    };
    
#define OUT
    
    struct DeviceCaps {
        size_t maxSharedMem;
        int maxBlockSize;
        int maxThreadsSize;
        char name[128];
    };
    
    Result getDeviceCaps(const API api, int deviceIndex, OUT DeviceCaps* caps);
    
    struct KernelConfiguration {
        KernelConfiguration(int blockSize, int numThreads, size_t sharedMem = size_t(0)):blockSize(blockSize), numThreads(numThreads), sharedMem(sharedMem){}
        int blockSize;
        int numThreads;
        size_t sharedMem;
    };
    
    struct Buffer {
        virtual Result alloc(void* hostBuf, size_t numBytes) = 0;
        virtual Result freeMem() = 0;
        virtual Result upload() = 0;
        virtual Result download(void* hostBuf, size_t numBytes) = 0;
        virtual Result alloc(size_t numBytes) = 0;
        virtual ~Buffer() {}
    };
    
    struct Kernel {
        Kernel();
        virtual Result init(const char* name) = 0;
        virtual Result addParam(const Buffer& buffer) = 0;
        virtual Result freeMem() = 0;
        virtual ~Kernel() {}
    };
    
    struct Bitmap2D {
        
    };
    
    struct Bitmap3D {
        
    };
    
    struct Device {
        virtual Result runKernel(const Kernel& kernel, const KernelConfiguration& configuration) = 0;
        virtual Result freeMem() = 0;
        virtual Result getCaps(OUT DeviceCaps* caps) = 0;
        virtual ~Device() {};
    };
    
    struct Context {
        Context();
        virtual Result init(const Device& device) = 0;
        virtual Result freeMem() = 0;
        virtual Result push() = 0;
        virtual Result pop() = 0;
    };
    
    #define CHECK_ERROR(X) checkError(__FILE__, __LINE__, (X))
    
    inline void checkError(const char* file, int line, Result error) {
        if (error != Result::Success) {
            printf("Error in file %s, line %i, error #%i\n", file, line, error);
            exit(EXIT_FAILURE);
        }
    }
    
} //namespace GPAPI