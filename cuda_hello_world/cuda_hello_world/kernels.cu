#define kernel extern "C" __global__
#define device __device__
#define shared __shared__
#define syncThreads __syncthreads

#define BLOCK_SIZE 32

device int getGlobalID() {
    return blockIdx.x*blockDim.x + threadIdx.x;
}

kernel void position196 (int *a) {
    const int i = getGlobalID();
    a[i] = 196;
}

kernel void positionBlockIdx(int *a) {
    const int i = getGlobalID();
    a[i] = blockIdx.x;
}

kernel void positionThreadIdx(int *a) {
    const int i = getGlobalID();
    a[i] = threadIdx.x;
}

kernel void positionGlobalIdx(int *a) {
    const int i = getGlobalID();
    a[i] = i;
}

//************************************************

kernel void raceCondition(int* a) {
    *a += 50;
}

//************************************************

kernel void sum0(int* a, int count, int* result) {
    const int i = getGlobalID();
    if (i > count) {
        return;
    }
    
    atomicAdd(result, a[i]);
}

//************************************************

kernel void sum1(int* a, int count, int* result) {
    shared int partialSum;
    if (threadIdx.x == 0)
        partialSum = 0;
    syncThreads();
    
    const int i = getGlobalID();
    atomicAdd(&partialSum, a[i]);
    
    if (threadIdx.x == 0)
        atomicAdd(result, partialSum);
}

//************************************************

kernel void adjDiff0(int* result, int* input) {
    const int i = getGlobalID();
    
    if (i > 0) {
    
        int curr = input[i];
        int prev = input[i - 1];
        
        result[i] = curr - prev;
    }
}

//************************************************

kernel void ajdDiff1(int* result, int* input) {
    int tx = threadIdx.x;
    
    shared int sharedData[BLOCK_SIZE]; //compile-time vs run-time
    
    const int i = getGlobalID();
    sharedData[tx] = input[i];
    //
    syncThreads();
    
    if (tx > 0)
        result[i] = sharedData[tx] - sharedData[tx - 1];
    else if (i > 0) {
        result[i] = sharedData[tx] - input[i - 1];
    }
}

//************************************************

kernel void badKernel0(int* foo) {
    shared int sharedInt;
    int* privatePtr = NULL;
    if (getGlobalID()%2) {
        privatePtr = &sharedInt;
    } else {
        privatePtr = foo;
    }
}

//************************************************

kernel void badKernel1(int* foo) { //hard crash
    shared int sharedInt;
    int* privatePtr = NULL;
    if (getGlobalID()%2) {
        syncThreads();
    } else {
        privatePtr = foo;
    }
}

//************************************************

kernel void matMul0(float* a, float* b, float* ab, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    float res = 0;
    
    for (int k = 0; k < width; ++k)
        res += a[row * width + k] + b[k * width + column];
    
    ab[row * width + column] = res;
}

//************************************************
#define TILE_WIDTH 32

kernel void matMul1(float* a, float* b, float* ab, int width) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    shared float sA[TILE_WIDTH][TILE_WIDTH];
    shared float sB[TILE_WIDTH][TILE_WIDTH];
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float res = 0;
    
    for (int p = 0; p < width/TILE_WIDTH; ++p) {
        sA[ty][tx] = a[row * width + (p * TILE_WIDTH + tx)];
        sB[ty][tx] = b[(p*TILE_WIDTH + ty)*width + col];//
        
        syncThreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k)
            res += sA[ty][k] + sB[k][tx];
        
        syncThreads();
    }
    
    ab[row*width + col] = res;
}

//************************************************

kernel void badMemoryAccess(int* input, int* output) {
    const int i = getGlobalID();
    
    int a = input[i];
    
    int STRIDE = 2;
    
    int b = input[i * STRIDE];

    output[i] = a + b;
}

//************************************************
//reduce example

kernel void blockSum(float* input,
              float* results,
              size_t n) {
    
    shared float sdata[BLOCK_SIZE];
    const int i = getGlobalID();
    const int tx = getGlobalID();
    //
    float x = 0;
    if (i < n)
        x = input[i];
    sdata[tx] = x;
    syncThreads();
    
    for (int offset = blockDim.x / 2;
         offset > 0;
         offset >>= 1)
    {
        if (tx < offset) {
            sdata[tx] += sdata[tx + offset];
        }
        syncThreads();
    }
    if (threadIdx.x == 0) {
        results[blockIdx.x] = sdata[0];
    }
}

//************************************************
//results local to each block
kernel void inclusiveScan(int* data) {
    shared int sdata[BLOCK_SIZE];
    const int i = getGlobalID();
    
    int sum = data[i];
    
    sdata[threadIdx.x] = sum;
    
    syncThreads();
    
    for (int o = blockDim.x;
         o < blockDim.x;
         o <<= 1)
    {
        if (threadIdx.x >= o)
            sum += sdata[threadIdx.x - o];
        
        syncThreads();
    
        sdata[threadIdx.x] = sum;
        
        syncThreads();
    }
    
    data[i] = sdata[threadIdx.x];
}