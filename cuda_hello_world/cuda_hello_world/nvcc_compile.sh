/Developer/NVIDIA/CUDA-7.5/bin/nvcc kernels.cu -gencode arch=compute_30,code=sm_30 -cubin -o "kernels.cubin"
