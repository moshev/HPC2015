/Developer/NVIDIA/CUDA-6.5/bin/nvcc kernels.cu -gencode arch=compute_12,code=sm_12 -cubin -o "kernels.cubin"
