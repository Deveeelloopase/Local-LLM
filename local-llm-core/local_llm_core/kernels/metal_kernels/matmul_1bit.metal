// matmul_1bit.metal — Ternary matrix multiplication for Apple GPU
// ===============================================================

#include <metal_stdlib>
using namespace metal;

kernel void ternary_matmul(
    device const half* a [[buffer(0)]],
    device const char* b [[buffer(1)]],
    device const float* scale [[buffer(2)]],
    device half* c [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= M || gid.y >= N) return;

    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        // unpack ternary from 2-bit packed array
        uint byte_idx = (k * N + gid.y) / 4;
        uint shift = ((k * N + gid.y) % 4) * 2;
        int val = (b[byte_idx] >> shift) & 0x3;
        int w = (val == 0) ? 0 : ((val == 1) ? 1 : -1);
        sum += float(a[gid.x * K + k]) * float(w);
    }
    c[gid.x * N + gid.y] = half(sum * scale[gid.y]);
}
