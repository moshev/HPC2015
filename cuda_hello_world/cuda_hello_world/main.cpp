//for CUDA API
#include "cuda.h"
//for NVRTC
#include "nvrtc.h"
//STD
#include <vector>
#include <string>
#include <fstream>

//misc stuff

#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))

enum LogType { LogTypeInfo = 0, LogTypeWarning, LogTypeError, LogTypeNone };

const LogType LOG_LEVEL = LogTypeInfo;

inline
void printLog(LogType priority, const char *format, ...) {
    
    if(priority < LOG_LEVEL)
        return;
    
    char s[512];
    
    time_t t = time(NULL);
    struct tm * p = localtime(&t);
    strftime(s, 512, "[%H:%M:%S] ", p);
    
    printf("%s", s);
    switch (priority) {
        case LogTypeInfo:
            printf("Info: ");
            break;
        case LogTypeWarning:
            printf("Warning: ");
            break;
        case LogTypeError:
            printf("Error: ");
            break;
        default:
            break;
    }
    
    va_list args;
    va_start(args, format);
    
    vprintf(format, args);
    
    va_end(args);
}

namespace CUDAErrorCheck {
template<typename T>
inline
void __checkError(T error, const char* file, int line) {
    if (error != 0) {
        printLog(LogTypeError, "error %i in file %s, line %i", error, file, line);
        exit(error);
    }
}
} //CUDAErrorCheck

#define CHECK_ERROR(X) CUDAErrorCheck::__checkError(X, __FILE__, __LINE__)

std::string getProgramSource(const std::string& path) {
    std::ifstream programSource(path.c_str());
    if (!programSource.good()) printLog(LogTypeError, "program source not found\n");
    return std::string((std::istreambuf_iterator <char>(programSource)), std::istreambuf_iterator <char>());
}

inline void pushContext(CUcontext context) {
    CUresult err = cuCtxPushCurrent(context);
    CHECK_ERROR(err);
}

inline void popContext(CUcontext context) {
    CUresult err = cuCtxPopCurrent(&context);
    CHECK_ERROR(err);
}

int main(int argc, const char * argv[]) {
    CUresult err = CUDA_SUCCESS;
    err = cuInit(0);
    CHECK_ERROR(err);
    int count = 0;
    err = cuDeviceGetCount(&count);
    CHECK_ERROR(err);
    
    std::vector<CUdevice> devices;
    std::vector<CUcontext> contexts;
    std::vector<CUmodule> programs;
    
    for (unsigned i = 0; i < count; ++i) {
        CUdevice device;
        err = cuDeviceGet(&device, i);
        CHECK_ERROR(err);
        
        devices.push_back(device);
        
        char* buffer[1024];
        err = cuDeviceGetName(	(char*)buffer,
                              (int)sizeof(buffer),
                              device
                              );
        CHECK_ERROR(err);
        
        printf("Found device named %s\n", (char*)buffer);
        
        CUcontext pctx;
        err = cuCtxCreate(&pctx, CU_CTX_SCHED_AUTO, device);
        CHECK_ERROR(err);
        
        contexts.push_back(pctx);
    }
    
    std::string source = getProgramSource("/Developer/git/cuda_hello_world/cuda_hello_world/kernels.cu");
    
    nvrtcResult nvRes;
    nvrtcProgram program;
    nvRes = nvrtcCreateProgram(&program, source.c_str(), "compiled_kernel", 0, NULL, NULL);
    
    const char* options[3] = {"--gpu-architecture=compute_20","--maxrregcount=64","--use_fast_math"};
    nvRes = nvrtcCompileProgram(program, 3, options);
    
    if (nvRes != NVRTC_SUCCESS) {
        size_t programLogSize;
        nvRes = nvrtcGetProgramLogSize(program, &programLogSize);
        CHECK_ERROR(nvRes);
        char* log = new char[programLogSize + 1];
        
        nvRes = nvrtcGetProgramLog(program, log);
        CHECK_ERROR(nvRes);
        printLog(LogTypeError, "%s", log);
        
        delete[] log;
        
        return -1;
    }

    size_t ptxSize;
    nvRes = nvrtcGetPTXSize(program, &ptxSize);
    CHECK_ERROR(nvRes);
    
    char* ptx = new char[ptxSize + 1];
    nvRes = nvrtcGetPTX(program, ptx);
    
    const char* TARGET_CUDA_SAVE_PTX_PATH = "/Users/savage309/Desktop/blago.txt";
    
    std::fstream ptxStream(TARGET_CUDA_SAVE_PTX_PATH, std::ios_base::trunc | std::ios_base::out);
    if (!ptxStream.good()) {
        printLog(LogTypeWarning, "Could not save PTX IR to %s\n", TARGET_CUDA_SAVE_PTX_PATH);
    } else {
        printLog(LogTypeInfo, "PTX IR saved to %s\n", TARGET_CUDA_SAVE_PTX_PATH);
    }
    ptxStream << ptx;
    
    const size_t JIT_NUM_OPTIONS = 8;
    const size_t JIT_BUFFER_SIZE_IN_BYTES = 1024;
    char logBuffer[JIT_BUFFER_SIZE_IN_BYTES];
    char errorBuffer[JIT_BUFFER_SIZE_IN_BYTES];
    
    CUjit_option jitOptions[JIT_NUM_OPTIONS];
    int optionsCounter = 0;
    jitOptions[optionsCounter++] = CU_JIT_MAX_REGISTERS;
    jitOptions[optionsCounter++] = CU_JIT_OPTIMIZATION_LEVEL;
    jitOptions[optionsCounter++] = CU_JIT_TARGET_FROM_CUCONTEXT;
    jitOptions[optionsCounter++] = CU_JIT_FALLBACK_STRATEGY;
    jitOptions[optionsCounter++] = CU_JIT_INFO_LOG_BUFFER;
    jitOptions[optionsCounter++] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    jitOptions[optionsCounter++] = CU_JIT_ERROR_LOG_BUFFER;
    jitOptions[optionsCounter++] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    void* jitValues[JIT_NUM_OPTIONS];
    const int maxRegCount = 63;
    int valuesCounter = 0;
    jitValues[valuesCounter++] = (void*)maxRegCount;
    const int optimizationLevel = 4;
    jitValues[valuesCounter++] = (void*)optimizationLevel;
    const int dummy = 0;
    jitValues[valuesCounter++] = (void*)dummy;
    const CUjit_fallback_enum fallbackStrategy = CU_PREFER_PTX;
    jitValues[valuesCounter++] = (void*)fallbackStrategy;
    jitValues[valuesCounter++] = (void*)logBuffer;
    const int logBufferSize = JIT_BUFFER_SIZE_IN_BYTES;
    jitValues[valuesCounter++] = (void*)logBufferSize;
    jitValues[valuesCounter++] = (void*)errorBuffer;
    const int errorBufferSize = JIT_BUFFER_SIZE_IN_BYTES;
    jitValues[valuesCounter++] = (void*)errorBufferSize;
    for (int i = 0; i < devices.size(); ++i) {
        CUmodule program;
        err = cuModuleLoadDataEx(&program, ptx, JIT_NUM_OPTIONS, jitOptions, jitValues);
        CHECK_ERROR(err);
        programs.push_back(program);
        printLog(LogTypeInfo, "program for device %i compiled\n", i);
    }
    nvRes = nvrtcDestroyProgram(&program);
    CHECK_ERROR(nvRes);
    delete[] ptx;
    //
    
    pushContext(contexts[0]);
    CUdeviceptr devicePtr;
    const int SIZE = 1024;
    int hostPtr[1024];
    
    err = cuMemAlloc(&devicePtr, sizeof(int) * SIZE);
    CHECK_ERROR(err);
    unsigned int globalSize = 32;
    unsigned int localSize = 32;
    CUfunction kernel;
    void *paramsPtrs[1] = {&devicePtr};
    
    const char* kernels[] = {"position196",
                            "positionBlockIdx",
                            "positionThreadIdx",
                            "positionGlobalIdx"};
    for (int i = 0; i < COUNT_OF(kernels); ++i) {
        err = cuModuleGetFunction(&kernel, programs[0], kernels[i]);
        CHECK_ERROR(err);
        
        err = cuLaunchKernel(kernel, globalSize, 1UL, 1UL, // grid size
                             localSize, 1UL, 1UL, // block size
                             0, // shared size
                             NULL, // stream
                             &paramsPtrs[0],
                             NULL
                             );
        CHECK_ERROR(err);
        
        err = cuMemcpyDtoH(hostPtr, devicePtr, sizeof(int) * SIZE);
        CHECK_ERROR(err);
        for (int i = 0; i < SIZE; ++i) {
            printf("%i, ", hostPtr[i]);
        }
        
        printf("\n\n\n");

    }
    
    popContext(contexts[0]);
    
    return 0;
}
