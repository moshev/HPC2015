#GPGPU Architectures

>"I saw `cout` being shifted `Hello world` times to the left and stopped right there."
— Steve Gonedes

---

## *disclaimer

* GPGPU Architectures != GPU Architectures (no rops and games)
* Theory != practice
* Lack of full documentation => speculations
* Just a mental model

---

![](./images/0.PNG)

---

![](./images/1.PNG)

---

![](./images/2.PNG)

---

![](./images/3.PNG)

---

![](./images/4.PNG)

---

![](./images/5.PNG)

---

![](./images/6.PNG)

---

![](./images/7.PNG)

---

##solutions
* Prepare fewer threads (reduce occupancy or threads in flight)
* Register spill
* __noinline__
* Code trimming

---

##divergence

```__global__
void squareEvenOdd(int* a, int count) {
int id = globalThreadIdx();
if (id >= count) return;
if (id % 2) a[id] = a[id] * a[id];
else a[id] = 0;
}
```

---

![](./images/8.PNG)

---

## IT IS LIKE HYPERTHREADING ON STEROIDS

#### optimized for throughput, not latency

---

Memory types
* Registers X1
* Shared X5
* Global X100
* Constant - broadcast
* Caches (L1, L2, NC, other)

---

### Registers

```
int foo;
float bar[196];
Widget widget
void bar(float fizzBuzz) { // <-

}
```

* Every thread has **own view**.
* Possibly the most valuable resource you have.

---

### Global 

```
__global__ int* ptr;
float bar[196]; //! <-
Widget widget;
```

* The threads has the **same view** of the global memory.
* Most of the GPGPU apps are strugling with the global memory.
* What can be done ?
* Is HBM going to help ?

---

### Constant 

```
__constant__ int c[42];
```

* CUDA
* OpenCL 1.X
* OpenCL 2.X
* Same view of own view here ?

---

### Shared 

```
__shared__ int something[42];
```

* Block of threads has same view.
* Different block has different view.
* Most of the GPGPU apps are optimizing by finding a way to exploit the shared memory.
* The idea is to cache manually what is needed in the shared memory and use the shared memory instead of the global memory.

---

#nVidia

---

### nVidia
#### compute capability 

* 1.X - Tesla - 2XX (32bit) 
* 2.X - Fermi - 4XX, 5XX
* 3.X - Kepler - 6XX, 7XX, Quadro KXXX (not all) + all new Tesla GPUs
* 5.X - Maxwell - 9XX, Quadro MXXXX

---

## FERMI

---

![](./images/9.PNG)

---

![](./images/10.PNG)

---

### FERMI

* Dynamic scheduler
* True L1 & L2 Cache
* 1 x 32
* Configurable L1 cache / Shared mem
* L2 cache
* 40nm
* Very good for GPGPU

---

## Kepler

---

![](./images/11.PNG)

---

* Static scheduler
* 6 x 32
* Bigger register file
* Non-coherent read only cache
* Configurable shared / L1
* L1 by default used for local mem only
* compiler opt in to cache global reads
* New read only / texture cache
* T __ldg(T*);
* tex1Dfetch
* const T* __restrict__
* 192 float, 4 schedulers (1x48?)
* 28nm

---

## Maxwell

---

4 x 32

![](./images/12.PNG)

---

![](./images/13.PNG)

---

![](./images/14.PNG)

---

### Maxwell

* Separate shared memory
* So, they have unified L1/Texture cache
* This can make the performance worse
* Again __ldg, tex1Dfetch, T* __restrict__
* Compiler opt in to cache global reads
​​* Spills are stored in L2
* Lower occupancy might be good
* No double precision
* 4 x 32 float
* 28nm

---

## future
* Pascal - 16nm
* Stacked Memory (HBM)
* Volta

---

# AMD

---

### GCN 

![](./images/15.PNG)

---

### GCN

* No ILP ?
* Spill to global mem only
* No max registers param
* 4 x 16 float
* 28nm
* HBM
* Native FP16
* Better FP64

---

![](./images/16.PNG)

---

![](./images/17.PNG)

---

![](./images/18.PNG)

---

![](./images/19.PNG)

bandwidth x4

power /3

price *x

capacity *y

---

![](./images/20.PNG)

---

![](./images/21.PNG)

---

#Intel HD Graphics

---

![](./images/22.PNG)

---

* Less-wider SIMD
* Bigger register file
* No dedicated memory (eDRAM)
* 1/10 of the compute power
* 2 x 4
* 14nm/finfet
* Bigget market share

---

#Imagination

---

## Power VR
![](./images/23.PNG)

---

![](./images/24.PNG)

---

![](./images/25.PNG)

---

![](./images/26.PNG)

---

## Tools

* nSight

* AMD CodeXL

* Intel Amplifier

---

![](./images/28.png)

---

![](./images/29.jpg)

---

![](./images/30.png)

---

#### Final notes

​* Make sure all caches are used (or not)
​* Reduce data dependencies
​* Have ILP
​* Reduce register usage (use -v)
​* no inline where possible
​* hide that with inline, where possible
​* code morph/trim
​* multi kernel
​* Thread divergence is A KILLER
​* Compiler params are super important
​* Function call is turbo expensive (recursion, virtual)
​* Memory access patterns 
​* Changing all the time
​* "Ако първо не успееш, опитай втори път." The Rescuers, 1977
​* Everything happens for a reason 

---

* Fermi        -   580  -    512 cores - 1581GF   - 244W - 6.4
* Kepler       -  680  -   1536 cores - 3090GF  - 195W - 15.8
* Maxwell    -  980  -   2048 cores - 4612 GF - 165W - 27.95
* AMD          - 280x  -   1792 cores - 3290GF - 190W - 17.13
* Intel          - 6200   -     384 cores -  768 GF < 47W - >16.43
​* PoweVR   - Rogue ~     192 cores ~ 115 GF < 5W   - >25