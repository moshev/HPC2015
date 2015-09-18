#GPGPU

---
```
__global__ void kernel (int *a) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = 196;
}
```
```
__global__ void kernel (int *a) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = blockIdx.x;
}
```
```
__global__ void kernel (int *a) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = threadIdx.x;
}
```

//get_global_id(0)

---

GPU:
* Can only access GPU memory
* No variable number of arguments
* No static variables
* Limited recursion
* Limited polymorphism

Function must be declared with qualifeier:
* __global__, __device__, (__host__)

---

#[DEMO OpenCL Hello World]

