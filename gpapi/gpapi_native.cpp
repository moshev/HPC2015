#include "gpapi_native.h"

#include <string>
#include "stdlib.h"

namespace GPAPI {
    
NativeBuffer::NativeBuffer(){}
Result NativeBuffer::alloc(void* hostBuf, size_t numBytes) {
    ptr = new char[numBytes];
    memcpy(ptr, (char*)hostBuf, numBytes);
    return Result::Success;
}
Result NativeBuffer::freeMem() {
    delete[] (char*)ptr;
    ptr = NULL;
    return Result::Success;
}
Result NativeBuffer::upload() {
    return Result::Success;
}
Result NativeBuffer::download(void* hostBuf, size_t numBytes) {
    return Result::Success;
}
Result NativeBuffer::alloc(size_t numBytes) {
    ptr = new char[numBytes];
    return Result::Success;
}
NativeBuffer::~NativeBuffer() {
    freeMem();
}
//
NativeDevice::NativeDevice() {
    strcpy(caps.name, "NativeDevice");
    caps.maxBlockSize = 1;
    caps.maxSharedMem = 0;
    caps.maxThreadsSize = 1024 * 1024;
}
Result NativeDevice::runKernel(const Kernel& kernel, const KernelConfiguration& configuration) {
    return Result::Success;
}
Result NativeDevice::freeMem() {
    return Result::Success;
}
Result NativeDevice::getCaps(OUT DeviceCaps* outCaps) {
    memcpy(outCaps, &caps, sizeof(DeviceCaps));
    return Result::Success;
}
NativeDevice::~NativeDevice() {
    freeMem();
}
    
}