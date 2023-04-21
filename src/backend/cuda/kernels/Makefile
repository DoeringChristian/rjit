COMPUTE_CAPABILITY=compute_70
CUDA_VER=11.8
NVCC=/usr/local/cuda-$(CUDA_VER)/bin/nvcc -m64 --ptx --expt-relaxed-constexpr -ccbin $(CC)

kernels_70.ptx: reduce.cuh scan.cuh compress.cuh mkperm.cuh misc.cuh kernels.cu
	$(NVCC) -Wno-deprecated-gpu-targets -gencode arch=compute_70,code=compute_70 kernels.cu -o kernels_70.ptx
