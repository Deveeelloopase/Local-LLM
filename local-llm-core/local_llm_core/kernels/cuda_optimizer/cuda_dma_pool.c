/*
 * cuda_dma_pool.c — CUDA pinned memory + device memory pool (userspace .so)
 * ========================================================================
 * Pre-allocates pinned host memory and device memory to avoid repeated
 * cudaHostAlloc() / cudaFree() during inference, reducing latency.
 *
 * Usage (Python): loaded via ctypes, provides allocate()/free()/reset().
 *
 * Build: make -f Makefile
 */

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define POOL_DEVICE_CHUNKS 16
#define POOL_HOST_CHUNKS   16
#define CHUNK_SIZE_MB      64   // 64 MiB per chunk

typedef struct {
    void   *ptr;
    size_t  size;
    int     used;
} PoolChunk;

static PoolChunk  device_pool[POOL_DEVICE_CHUNKS];
static PoolChunk  host_pool[POOL_HOST_CHUNKS];
static int        device_pool_initialized = 0;
static int        host_pool_initialized   = 0;
static int        device_pool_max_chunks  = POOL_DEVICE_CHUNKS;
static int        host_pool_max_chunks    = POOL_HOST_CHUNKS;

/* ---------------------------------------------------------------
 * Internal helpers
 * --------------------------------------------------------------- */
static void *pool_alloc(PoolChunk *pool, int max_chunks, size_t size) {
    for (int i = 0; i < max_chunks; i++) {
        if (!pool[i].used && pool[i].ptr) {
            pool[i].used = 1;
            return pool[i].ptr;
        }
    }
    return NULL;
}

static void pool_free(PoolChunk *pool, int max_chunks, void *ptr) {
    for (int i = 0; i < max_chunks; i++) {
        if (pool[i].ptr == ptr) {
            pool[i].used = 0;
            break;
        }
    }
}

static void pool_flush(PoolChunk *pool, int max_chunks, cudaError_t (*free_fn)(void*)) {
    for (int i = 0; i < max_chunks; i++) {
        if (pool[i].ptr) {
            free_fn(pool[i].ptr);
            pool[i].ptr = NULL;
            pool[i].used = 0;
        }
    }
}

/* ---------------------------------------------------------------
 * Public API (exported)
 * --------------------------------------------------------------- */

/* Initialize device memory pool (cudaMalloc) */
int cuda_pool_device_init(size_t total_bytes) {
    if (device_pool_initialized) return 0;
    cudaError_t err;
    size_t chunk_size = CHUNK_SIZE_MB * 1024 * 1024;
    int n_chunks = (total_bytes + chunk_size - 1) / chunk_size;
    if (n_chunks > POOL_DEVICE_CHUNKS) n_chunks = POOL_DEVICE_CHUNKS;

    for (int i = 0; i < n_chunks; i++) {
        err = cudaMalloc(&device_pool[i].ptr, chunk_size);
        if (err != cudaSuccess) {
            device_pool[i].ptr = NULL;
            break;
        }
        device_pool[i].size = chunk_size;
        device_pool[i].used = 0;
    }
    device_pool_initialized = 1;
    return 0;
}

/* Initialize pinned host memory pool (cudaHostAlloc) */
int cuda_pool_host_init(size_t total_bytes) {
    if (host_pool_initialized) return 0;
    cudaError_t err;
    size_t chunk_size = CHUNK_SIZE_MB * 1024 * 1024;
    int n_chunks = (total_bytes + chunk_size - 1) / chunk_size;
    if (n_chunks > POOL_HOST_CHUNKS) n_chunks = POOL_HOST_CHUNKS;

    for (int i = 0; i < n_chunks; i++) {
        err = cudaHostAlloc(&host_pool[i].ptr, chunk_size, cudaHostAllocPortable);
        if (err != cudaSuccess) {
            host_pool[i].ptr = NULL;
            break;
        }
        host_pool[i].size = chunk_size;
        host_pool[i].used = 0;
    }
    host_pool_initialized = 1;
    return 0;
}

/* Allocate a device chunk */
void *cuda_pool_device_malloc(size_t size) {
    if (!device_pool_initialized) return NULL;
    return pool_alloc(device_pool, device_pool_max_chunks, size);
}

/* Allocate a pinned host chunk */
void *cuda_pool_host_malloc(size_t size) {
    if (!host_pool_initialized) return NULL;
    return pool_alloc(host_pool, host_pool_max_chunks, size);
}

/* Free a device chunk */
void cuda_pool_device_free(void *ptr) {
    if (!device_pool_initialized) return;
    pool_free(device_pool, device_pool_max_chunks, ptr);
}

/* Free a host chunk */
void cuda_pool_host_free(void *ptr) {
    if (!host_pool_initialized) return;
    pool_free(host_pool, host_pool_max_chunks, ptr);
}

/* Destroy device pool */
void cuda_pool_device_destroy(void) {
    pool_flush(device_pool, device_pool_max_chunks, cudaFree);
    device_pool_initialized = 0;
}

/* Destroy host pool */
void cuda_pool_host_destroy(void) {
    pool_flush(host_pool, host_pool_max_chunks, cudaFreeHost);
    host_pool_initialized = 0;
}

/* ---------------------------------------------------------------
 * Convenience: auto-initialize both from environment or defaults
 * --------------------------------------------------------------- */
void cuda_pool_auto_init(size_t device_mb, size_t host_mb) {
    cuda_pool_device_init(device_mb * 1024 * 1024);
    cuda_pool_host_init(host_mb * 1024 * 1024);
}
