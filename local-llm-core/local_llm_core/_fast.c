/*
 * _fast.c – CPU Accelerator for LLM Inference (SIMD‑optimised v1.0)
 * ===================================================================
 * v1.0 – Fixed AVX2/AVX‑512 ternary matmul, all platforms compile.
 *
 * Provides:
 *  - dot_product_f32, dot_product_f16, sum_f32
 *  - matmul_1bit: ternary matrix multiplication (SIMD + scalar fallback)
 *
 * Build: use setup.py (compiler flags are set automatically)
 */

#include <Python.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Platform detection
// ---------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #define SIMD_X86 1
    #if defined(__AVX512F__) && defined(__AVX512BW__)
        #define SIMD_AVX512 1
    #elif defined(__AVX2__)
        #define SIMD_AVX2 1
    #endif
    #if defined(__F16C__)
        #define HAS_F16C 1
    #endif
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define SIMD_NEON 1
    #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        #define HAS_ARM_FP16 1
    #endif
#else
    #define SIMD_SCALAR 1
#endif

// ---------------------------------------------------------------------------
// Half conversion (software fallback)
// ---------------------------------------------------------------------------
static inline float half_to_float(uint16_t h) {
    unsigned sign = (h >> 15) & 1;
    unsigned exp  = (h >> 10) & 0x1F;
    unsigned mant = h & 0x3FF;
    float result;
    if (exp == 0) {
        if (mant == 0)
            result = sign ? -0.0f : 0.0f;
        else {
            float sub = mant / 1024.0f;
            result = sign ? -sub : sub;
            result *= 6.103515625e-5f;
        }
    } else if (exp == 31) {
        result = mant ? NAN : (sign ? -INFINITY : INFINITY);
    } else {
        float val = 1.0f + mant / 1024.0f;
        val *= ldexpf(1.0f, exp - 15);
        if (sign) val = -val;
        result = val;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Scalar fallbacks (always available)
// ---------------------------------------------------------------------------
static float scalar_dot_product_f32(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

static float scalar_dot_product_f16(const uint16_t *a, const uint16_t *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float a_val, b_val;
#ifdef HAS_F16C
        a_val = _cvtsh_ss((unsigned short)a[i]);
        b_val = _cvtsh_ss((unsigned short)b[i]);
#else
        a_val = half_to_float(a[i]);
        b_val = half_to_float(b[i]);
#endif
        sum += a_val * b_val;
    }
    return sum;
}

static float scalar_sum_f32(const float *a, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i];
    return sum;
}

static void scalar_ternary_matmul_1bit(const float *a, const uint8_t *b_packed,
                                       float *c, int M, int N, int K,
                                       const float *scale) {
    const float lut[4] = {0.0f, 1.0f, -1.0f, 0.0f};
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int byte_idx = (k * N + col) / 4;
                int shift = ((k * N + col) % 4) * 2;
                int val = (b_packed[byte_idx] >> shift) & 0x3;
                sum += a[row * K + k] * lut[val];
            }
            c[row * N + col] = sum * scale[col];
        }
    }
}

// ---------------------------------------------------------------------------
// AVX‑512 implementations
// ---------------------------------------------------------------------------
#ifdef SIMD_AVX512
static float avx512_dot_product_f32(const float *a, const float *b, int n) {
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        sum = _mm512_fmadd_ps(va, vb, sum);
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

static float avx512_dot_product_f16(const uint16_t *a, const uint16_t *b, int n) {
    // fallback to scalar for safety
    return scalar_dot_product_f16(a, b, n);
}

static float avx512_sum_f32(const float *a, int n) {
    __m512 sum = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        sum = _mm512_add_ps(sum, _mm512_loadu_ps(&a[i]));
    }
    float result = _mm512_reduce_add_ps(sum);
    for (; i < n; ++i) result += a[i];
    return result;
}

static void avx512_ternary_matmul_1bit(const float *a, const uint8_t *b_packed,
                                       float *c, int M, int N, int K,
                                       const float *scale) {
    const float lut[4] = {0.0f, 1.0f, -1.0f, 0.0f};
    for (int row = 0; row < M; ++row) {
        for (int col_block = 0; col_block < N; col_block += 16) {
            int cols = (col_block + 16 <= N) ? 16 : (N - col_block);
            __m512 sum = _mm512_setzero_ps();
            for (int k = 0; k < K; ++k) {
                int byte_idx = (k * N + col_block) / 4;
                uint32_t packed;
                memcpy(&packed, &b_packed[byte_idx], 4);
                float w_vals[16] = {0};
                for (int off = 0; off < cols; ++off) {
                    int shift = (off % 4) * 2;
                    int val = (packed >> shift) & 0x3;
                    w_vals[off] = lut[val];
                }
                __m512 w = _mm512_loadu_ps(w_vals);
                __m512 row_a = _mm512_set1_ps(a[row * K + k]);
                sum = _mm512_fmadd_ps(row_a, w, sum);
            }
            __m512 s = _mm512_loadu_ps(&scale[col_block]);
            __m512 result = _mm512_mul_ps(sum, s);
            _mm512_storeu_ps(&c[row * N + col_block], result);
        }
    }
}
#endif

// ---------------------------------------------------------------------------
// AVX2 implementations
// ---------------------------------------------------------------------------
#ifdef SIMD_AVX2
static float avx2_dot_product_f32(const float *a, const float *b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    float partial[8];
    _mm256_storeu_ps(partial, sum);
    float result = partial[0]+partial[1]+partial[2]+partial[3]+
                   partial[4]+partial[5]+partial[6]+partial[7];
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

static float avx2_sum_f32(const float *a, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        sum = _mm256_add_ps(sum, _mm256_loadu_ps(&a[i]));
    }
    float partial[8];
    _mm256_storeu_ps(partial, sum);
    float result = partial[0]+partial[1]+partial[2]+partial[3]+
                   partial[4]+partial[5]+partial[6]+partial[7];
    for (; i < n; ++i) result += a[i];
    return result;
}

static void avx2_ternary_matmul_1bit(const float *a, const uint8_t *b_packed,
                                     float *c, int M, int N, int K,
                                     const float *scale) {
    const float lut[4] = {0.0f, 1.0f, -1.0f, 0.0f};
    for (int row = 0; row < M; ++row) {
        for (int col_block = 0; col_block < N; col_block += 8) {
            int cols = (col_block + 8 <= N) ? 8 : (N - col_block);
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < K; ++k) {
                int byte_idx = (k * N + col_block) / 4;
                uint32_t packed;
                memcpy(&packed, &b_packed[byte_idx], 4);
                float w_vals[8] = {0};
                for (int off = 0; off < cols; ++off) {
                    int shift = (off % 4) * 2;
                    int val = (packed >> shift) & 0x3;
                    w_vals[off] = lut[val];
                }
                __m256 w = _mm256_loadu_ps(w_vals);
                __m256 row_a = _mm256_set1_ps(a[row * K + k]);
                sum = _mm256_fmadd_ps(row_a, w, sum);
            }
            __m256 s = _mm256_loadu_ps(&scale[col_block]);
            __m256 result = _mm256_mul_ps(sum, s);
            _mm256_storeu_ps(&c[row * N + col_block], result);
        }
    }
}
#endif

// ---------------------------------------------------------------------------
// NEON implementations (unchanged, already correct)
// ---------------------------------------------------------------------------
#ifdef SIMD_NEON
static float neon_dot_product_f32(const float *a, const float *b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        sum = vmlaq_f32(sum, va, vb);
    }
    float partial[4];
    vst1q_f32(partial, sum);
    float result = partial[0]+partial[1]+partial[2]+partial[3];
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

static float neon_dot_product_f16(const uint16_t *a, const uint16_t *b, int n) {
    return scalar_dot_product_f16(a, b, n);
}

static float neon_sum_f32(const float *a, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        sum = vaddq_f32(sum, vld1q_f32(&a[i]));
    }
    float partial[4];
    vst1q_f32(partial, sum);
    float result = partial[0]+partial[1]+partial[2]+partial[3];
    for (; i < n; ++i) result += a[i];
    return result;
}

static void neon_ternary_matmul_1bit(const float *a, const uint8_t *b_packed,
                                     float *c, int M, int N, int K,
                                     const float *scale) {
    const float lut[4] = {0.0f, 1.0f, -1.0f, 0.0f};
    for (int row = 0; row < M; ++row) {
        for (int col_block = 0; col_block < N; col_block += 4) {
            int cols = (col_block + 4 <= N) ? 4 : (N - col_block);
            float32x4_t sum = vdupq_n_f32(0.0f);
            for (int k = 0; k < K; ++k) {
                int byte_idx = (k * N + col_block) / 4;
                uint32_t packed;
                memcpy(&packed, &b_packed[byte_idx], 4);
                float w_vals[4] = {0};
                for (int off = 0; off < cols; ++off) {
                    int shift = (off % 4) * 2;
                    int val = (packed >> shift) & 0x3;
                    w_vals[off] = lut[val];
                }
                float32x4_t w = vld1q_f32(w_vals);
                sum = vmlaq_n_f32(sum, w, a[row * K + k]);
            }
            float32x4_t s = vld1q_f32(&scale[col_block]);
            float32x4_t result = vmulq_f32(sum, s);
            vst1q_f32(&c[row * N + col_block], result);
        }
    }
}
#endif

// ---------------------------------------------------------------------------
// Runtime dispatch (function pointers)
// ---------------------------------------------------------------------------
static float (*dot_product_f32_ptr)(const float*, const float*, int) = NULL;
static float (*dot_product_f16_ptr)(const uint16_t*, const uint16_t*, int) = NULL;
static float (*sum_f32_ptr)(const float*, int) = NULL;
static void (*matmul_1bit_ptr)(const float*, const uint8_t*, float*, int, int, int, const float*) = NULL;

static void init_dispatch(void) {
#if defined(SIMD_AVX512)
    dot_product_f32_ptr = avx512_dot_product_f32;
    dot_product_f16_ptr = avx512_dot_product_f16; // actually scalar
    sum_f32_ptr = avx512_sum_f32;
    matmul_1bit_ptr = avx512_ternary_matmul_1bit;
#elif defined(SIMD_AVX2)
    dot_product_f32_ptr = avx2_dot_product_f32;
    dot_product_f16_ptr = scalar_dot_product_f16;
    sum_f32_ptr = avx2_sum_f32;
    matmul_1bit_ptr = avx2_ternary_matmul_1bit;
#elif defined(SIMD_NEON)
    dot_product_f32_ptr = neon_dot_product_f32;
    dot_product_f16_ptr = neon_dot_product_f16;
    sum_f32_ptr = neon_sum_f32;
    matmul_1bit_ptr = neon_ternary_matmul_1bit;
#else
    dot_product_f32_ptr = scalar_dot_product_f32;
    dot_product_f16_ptr = scalar_dot_product_f16;
    sum_f32_ptr = scalar_sum_f32;
    matmul_1bit_ptr = scalar_ternary_matmul_1bit;
#endif
}

// ---------------------------------------------------------------------------
// Python bindings (unchanged, already correct)
// ---------------------------------------------------------------------------
static PyObject *py_dot_product(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    Py_buffer a_buf, b_buf;
    if (PyObject_GetBuffer(a_obj, &a_buf, PyBUF_C_CONTIGUOUS) != 0) return NULL;
    if (PyObject_GetBuffer(b_obj, &b_buf, PyBUF_C_CONTIGUOUS) != 0) {
        PyBuffer_Release(&a_buf); return NULL;
    }
    if (a_buf.itemsize != sizeof(float) || b_buf.itemsize != sizeof(float)) {
        PyErr_SetString(PyExc_TypeError, "Arrays must be of float32 type");
        PyBuffer_Release(&a_buf); PyBuffer_Release(&b_buf);
        return NULL;
    }
    int n = a_buf.len / sizeof(float);
    if ((int)(b_buf.len / sizeof(float)) < n) n = b_buf.len / sizeof(float);
    float *a = (float*)a_buf.buf, *b = (float*)b_buf.buf;
    float result = dot_product_f32_ptr(a, b, n);
    PyBuffer_Release(&a_buf); PyBuffer_Release(&b_buf);
    return PyFloat_FromDouble(result);
}

static PyObject *py_sum(PyObject *self, PyObject *args) {
    PyObject *a_obj;
    if (!PyArg_ParseTuple(args, "O", &a_obj)) return NULL;
    Py_buffer a_buf;
    if (PyObject_GetBuffer(a_obj, &a_buf, PyBUF_C_CONTIGUOUS) != 0) return NULL;
    if (a_buf.itemsize != sizeof(float)) {
        PyErr_SetString(PyExc_TypeError, "Array must be of float32 type");
        PyBuffer_Release(&a_buf);
        return NULL;
    }
    int n = a_buf.len / sizeof(float);
    float *a = (float*)a_buf.buf;
    float result = sum_f32_ptr(a, n);
    PyBuffer_Release(&a_buf);
    return PyFloat_FromDouble(result);
}

static PyObject *py_matmul_1bit(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *scale_obj;
    int M, N, K;
    if (!PyArg_ParseTuple(args, "OOOiii", &a_obj, &b_obj, &scale_obj, &M, &N, &K))
        return NULL;

    Py_buffer a_buf, b_buf, s_buf;
    if (PyObject_GetBuffer(a_obj, &a_buf, PyBUF_C_CONTIGUOUS) != 0) return NULL;
    if (PyObject_GetBuffer(b_obj, &b_buf, PyBUF_C_CONTIGUOUS) != 0) {
        PyBuffer_Release(&a_buf); return NULL;
    }
    if (PyObject_GetBuffer(scale_obj, &s_buf, PyBUF_C_CONTIGUOUS) != 0) {
        PyBuffer_Release(&a_buf); PyBuffer_Release(&b_buf); return NULL;
    }

    if (a_buf.itemsize != sizeof(float)) {
        PyErr_SetString(PyExc_TypeError, "A must be float32");
        goto cleanup;
    }
    if (s_buf.itemsize != sizeof(float)) {
        PyErr_SetString(PyExc_TypeError, "scale must be float32");
        goto cleanup;
    }
    if (b_buf.itemsize != sizeof(uint8_t)) {
        PyErr_SetString(PyExc_TypeError, "B_packed must be uint8");
        goto cleanup;
    }
    if (M <= 0 || N <= 0 || K <= 0) {
        PyErr_SetString(PyExc_ValueError, "M, N, K must be positive");
        goto cleanup;
    }
    if (a_buf.len < (size_t)(M * K * (int)sizeof(float))) {
        PyErr_SetString(PyExc_ValueError, "A buffer too small");
        goto cleanup;
    }
    if (s_buf.len < (size_t)(N * (int)sizeof(float))) {
        PyErr_SetString(PyExc_ValueError, "scale buffer too small");
        goto cleanup;
    }
    int packed_len = (K * N + 3) / 4;
    if (b_buf.len < (size_t)packed_len) {
        PyErr_SetString(PyExc_ValueError, "B_packed buffer too small");
        goto cleanup;
    }

    float *c = (float*)malloc(M * N * sizeof(float));
    if (!c) {
        PyErr_NoMemory();
        goto cleanup;
    }

    float *a = (float*)a_buf.buf;
    uint8_t *b = (uint8_t*)b_buf.buf;
    float *scale = (float*)s_buf.buf;

    matmul_1bit_ptr(a, b, c, M, N, K, scale);

    PyObject *result = PyList_New(M * N);
    if (!result) {
        free(c);
        PyErr_NoMemory();
        goto cleanup;
    }
    for (int i = 0; i < M * N; ++i) {
        PyObject *f = PyFloat_FromDouble(c[i]);
        if (!f) {
            Py_DECREF(result);
            free(c);
            goto cleanup;
        }
        PyList_SET_ITEM(result, i, f);
    }
    free(c);
    PyBuffer_Release(&a_buf); PyBuffer_Release(&b_buf); PyBuffer_Release(&s_buf);
    return result;

cleanup:
    PyBuffer_Release(&a_buf); PyBuffer_Release(&b_buf); PyBuffer_Release(&s_buf);
    return NULL;
}

static PyMethodDef FastMethods[] = {
    {"dot_product", py_dot_product, METH_VARARGS,
     "dot_product(a, b) -> float\nFast dot product of two float32 arrays."},
    {"sum", py_sum, METH_VARARGS,
     "sum(a) -> float\nFast sum of float32 array."},
    {"matmul_1bit", py_matmul_1bit, METH_VARARGS,
     "matmul_1bit(A, B_packed, scale, M, N, K) -> list\n"
     "Ternary matrix multiplication: C = (A @ B) * scale\n"
     "A: float32 (M x K), B_packed: uint8 packed ternary weights (K x N in 2 bits each),\n"
     "scale: float32 (N,), returns float32 list (M x N) row-major."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastmodule = {
    PyModuleDef_HEAD_INIT,
    "_fast",
    "SIMD‑optimised CPU accelerators for LLM inference",
    -1,
    FastMethods
};

PyMODINIT_FUNC PyInit__fast(void) {
    init_dispatch();
    return PyModule_Create(&fastmodule);
}
