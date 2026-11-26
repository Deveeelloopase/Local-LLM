// hip_matmul.cpp — 1‑bit ternary matrix multiplication with HIP
// ==============================================================
// Production‑ready HIP kernel that compiles on all ROCm versions.
// Uses the triple‑chevron <<<>>> syntax for safe kernel launch
// inside an "extern C" wrapper, avoiding the hipLaunchKernelGGL
// macro which fails under "extern C".
//
// Compile with:
//   hipcc -O3 -shared -fPIC hip_matmul.cpp -o ternary_matmul.so

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>       // __half, __half2float, __float2half
#include <cstdint>

// ---------------------------------------------------------------------------
// Device helper: unpack ternary weights from byte array
// ---------------------------------------------------------------------------
static __device__ __forceinline__ int load_ternary(const int8_t* data, int idx) {
    int byte_idx = idx / 4;
    int shift = (idx % 4) * 2;
    int val = (data[byte_idx] >> shift) & 0x3;
    if (val == 0) return 0;
    return (val == 1) ? 1 : -1;
}

// ---------------------------------------------------------------------------
// GPU kernel
// ---------------------------------------------------------------------------
extern "C" __global__
void ternary_matmul_kernel(
    const __half* __restrict__ a,      // input activations [M, K]
    const int8_t* __restrict__ b,      // packed ternary weights [K/4, N]
    const float* __restrict__ scale,   // per‑column scales [N]
    __half* __restrict__ c,            // output [M, N]
    int M, int N, int K                // dimensions
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        int w = load_ternary(b, k * N + col);
        sum += __half2float(a[row * K + k]) * w;
    }
    c[row * N + col] = __float2half(sum * scale[col]);
}

// ---------------------------------------------------------------------------
// Host launch wrapper – callable from Python via ctypes
// ---------------------------------------------------------------------------
extern "C" int launch_ternary_matmul(
    const __half* a, const int8_t* b, const float* scale,
    __half* c, int M, int N, int K,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z
) {
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);

    // Triple‑chevron launch – this is the HIP‑standard way to launch kernels.
    // It works correctly inside extern "C" functions and never triggers
    // the macro expansion issues of hipLaunchKernelGGL.
    ternary_matmul_kernel<<<grid, block>>>(
        a, b, scale, c, M, N, K
    );

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return 1;
    }
    err = hipDeviceSynchronize();
    return (err == hipSuccess) ? 0 : 1;
}
