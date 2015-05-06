#pragma once

#include "gpapi_interface.h"

namespace GPAPI {
    struct NativeBuffer : Buffer{
        NativeBuffer();
        virtual Result alloc(void* hostBuf, size_t numBytes);
        virtual Result freeMem();
        virtual Result upload();
        virtual Result download(void* hostBuf, size_t numBytes);
        virtual Result alloc(size_t numBytes);
        virtual ~NativeBuffer();
    private:
        void* ptr;
    };
    struct NativeDevice : Device {
        NativeDevice();
        virtual Result runKernel(const Kernel& kernel, const KernelConfiguration& configuration);
        virtual Result freeMem();
        virtual Result getCaps(OUT DeviceCaps* caps);
        virtual ~NativeDevice();
    private:
        DeviceCaps caps;
    };
}