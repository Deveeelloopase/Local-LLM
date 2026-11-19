# Makefile for cuda_dma_pool.so (userspace shared library)
NVCC ?= nvcc
CFLAGS = -O3 -shared -Xcompiler -fPIC -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart

all: cuda_dma_pool.so

cuda_dma_pool.so: cuda_dma_pool.c
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f cuda_dma_pool.so
	