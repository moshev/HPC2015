#GPGPU

Note:
code samples from cs-193-stanford

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

---

bel.red. random notes on performance considerations

---

Memory coalescing

SoA vs AoS

---

Shared memory is (sometimes) banked

If 2 or > thread access different values rom the same bank, you get a conflict

---

__ldg

textures

```
ld.global.nc
```

---

shared resoruces - ALUs (make the threads run for the roughly same amount of time)

divergence

shared memory

cuda occupancy calculator

registers

-xptxas -v

---

function inlining

__noinline__

multi kernel

---

divice & conquer (blockSize)

reduction

partition / split

expand

--

ignore

---

#reduction

sum without atamic demo

tree-data approach

```
//serial
float sum(float* data, int n) {
    float res = 0.f;
    for (int i = 0; i < n; ++i) result += data[i];
    return result;
}
```


---

Variable number of elements per thread ?

-> parial sum (scan)

2 | 1 | 0 | 3 | 2

---

| 3 | 1 | 7 | 0 | 4 | 1 | 6 | 3 

| 0 | 3 | 4 | 11| 11| 15| 16| 22

---

sort

counting sort

odd-even sort

```
while A is not sorted:
    if is_odd(i) and A[i+1] < A[i]
        swap(A[i+1], A[i]);
    barrier;
    if even(i) and A[i+1] < A[i]
        swap(A[i+1], A[i]);
    barrier
```

---

ranking sort

rank[i] = count(j < i where A[j] <= A[i]) + count (j > i where A[j] < A[i])

permute(A[i], A[rank[i]])

---

how many positions to move to the left instead

---

can be transformed to radix sort

a[i]==0, then
offset[i] = count(j < i where A[j] == 1)

a[i]==1, then
offiset[i] = count(j > i, where A[j] == 0)

permute(A[i], A[i - offset[i]))

---

how many ones/zeros to my left -> prefix sum
(for zeros invert the values)

---

