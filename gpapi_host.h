#pragma once

#include "gpapi_native.h"
#include "gpapi_interface.h"

namespace GPAPI {
    
    Result newDevice(const GPAPI::API api, int index, OUT Device* device) {
        Result result = GPAPI::Result::Success;
        
        if (api == GPAPI::API::Native) {
            if (index == 0) {
                device = new NativeDevice;
            } else {
                return GPAPI::Result::InvalidDeviceIndex;
            }
        }
        
        return result;
    }
    
    Result newBuffer(const GPAPI::API api, int index, OUT Buffer* buffer) {
        Result result = GPAPI::Result::Success;
        
        if (api == GPAPI::API::Native) {
            if (index == 0) {
                buffer = new NativeBuffer;
            } else {
                return GPAPI::Result::InvalidDeviceIndex;
            }
        }
        
        return result;
    }
    Result newKernel(const API api, int index, const char* name, OUT Kernel* kernel) {
        return GPAPI::Result::Success;
    }
    Result getNumDevices(const API api, OUT int* numDevices) {
        return GPAPI::Result::Success;
    }
}