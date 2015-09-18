extern "C"
__global__ void position196 (int *a) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = 196;
}
extern "C"
__global__ void positionBlockIdx(int *a) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = blockIdx.x;
}
extern "C"
__global__ void positionThreadIdx(int *a) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = threadIdx.x;
}
extern "C"
__global__ void positionGlobalIdx(int *a) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = idx;
}