/*
    demo.cpp - Float-Float Arithmetic Benchmark
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This benchmark evaluates the performance of float-float (double-float) arithmetic operations
    compared to native double precision on CUDA GPUs and CPU.

    Kernel Types:
    -------------------------------------------------------------------------
    IMPORTANT: This benchmark uses TWO types of kernels:
    1. Performance kernels: Generate inputs on-the-fly, do many operations, write minimal output
       - Measure COMPUTE performance, not memory bandwidth
    2. Accuracy kernels: Read/write arrays to validate correctness
       - Ensure results are accurate vs reference

    Operations Tested:
    -------------------------------------------------------------------------
    - Addition (add)
    - Subtraction (sub)
    - Multiplication (mul)
    - Division (div)
    - Square root (sqrt)
    - Reciprocal square root (rsqrt)
    - Fused multiply-add (fma)
    - Exponential (exp)
    - Natural logarithm (log)
    - Error function (erf)
    - Complementary error function (erfc)
    - Sine (sin)
    - Cosine (cos)
    - Boys function F_0 (boys_f0) - quantum chemistry special function
    - Inverse normal CDF (normcdfinv)
    - Accumulate (acc) - optimized single-component accumulation

    Configuration:
    -------------------------------------------------------------------------
    - REPS: Number of operations per thread (default: 1024)
    - UNROLL: Loop unroll factor (default: 16)
    - THREADS_PER_BLOCK: CUDA block size (default: 256)
    - NUM_BLOCKS: Number of CUDA blocks (default: 2048)
    - NUM_ITERATIONS: Timing iterations (default: 10)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <string.h>

// FP SDK (CCCL)
#include <cuda/fpmp>
#include <cuda/fpmp_math>

using namespace cuda::experimental;

// EFP32_MP type alias
using fptype_t = fp32mp2;

// Configuration
#ifndef NUM_ITERATIONS
  #define NUM_ITERATIONS (10)
#endif

#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK (256)
#endif

#ifndef NUM_BLOCKS
  #define NUM_BLOCKS (2048)
#endif

#ifndef REPS
  #define REPS (1024)
#endif

#ifndef UNROLL
  #define UNROLL 16
#endif

#ifndef ACCURACY_LEN
  #define ACCURACY_LEN (1024 * 1024)  // 1M elements for accuracy check
#endif

#define ERR_THRESHOLD (1e-7)

// Macros for host/device compatibility
#if defined(__CUDACC__)
  #define HOST_DEVICE __host__ __device__
  #define KERNEL __global__
  #define USE_CUDA 1
#else
  #define HOST_DEVICE
  #define KERNEL
  #define USE_CUDA 0
#endif

// ====================================================================================================
// Bit manipulation to prevent compiler optimization
// ====================================================================================================

template<typename T>
HOST_DEVICE void update_bits(T a, T& r) {
    // XOR lowest bit to create dependency without changing value significantly
    uint64_t a_bits = 0, r_bits = 0;
    memcpy(&a_bits, &a, sizeof(T) < 8 ? sizeof(T) : 8);
    memcpy(&r_bits, &r, sizeof(T) < 8 ? sizeof(T) : 8);
    r_bits = r_bits ^ (a_bits & 0x1);
    memcpy(&r, &r_bits, sizeof(T) < 8 ? sizeof(T) : 8);
}

#if USE_CUDA
// Specialization for fptype_t (float-float = 8 bytes = two floats)
template <typename T>
HOST_DEVICE void update_bits_ff(T a, T& r) {
    constexpr int sz = sizeof(T);
    if constexpr (sz > 8) 
    {
        // For types > 8 bytes (e.g., double-double)
        uint64_t a_bits = 0, r_bits = 0;
        memcpy(&a_bits, &a, 8);
        memcpy(&r_bits, &r, 8);
        r_bits = r_bits ^ (a_bits & 0x0000000100000001ul);
        memcpy(&r, &r_bits, 8);
    }
    else
    {
        // For 8-byte types (float-float): XOR bit 0 and bit 32 (LSB of both floats)
        uint64_t a_bits = 0, r_bits = 0;
        memcpy(&a_bits, &a, sz);
        memcpy(&r_bits, &r, sz);
        r_bits = r_bits ^ (a_bits & 0x0000000100000001ul);
        memcpy(&r, &r_bits, sz);
    }
}
#endif

// ====================================================================================================
// PERFORMANCE KERNELS: Generate inputs, do many ops, minimal memory I/O
// ====================================================================================================

#if USE_CUDA
// Float-Float Performance Kernels
KERNEL void ff_add_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t b = fptype_t(start_val + 1.5 + (double)tid * 0.0007);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = a + b;
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_sub_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t b = fptype_t(start_val + 0.5 + (double)tid * 0.0007);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = a - b;
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_mul_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t b = fptype_t(start_val + 1.0001 + (double)tid * 0.00001);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = a * b;
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_div_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001 + 1.0);
    fptype_t b = fptype_t(start_val + 1.5 + (double)tid * 0.0007);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = a / b;
        // Update both a and b to prevent reciprocal optimization
        update_bits_ff(r, a);
        update_bits_ff(r, b);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_sqrt_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001 + 2.0);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = sqrt(a);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_rsqrt_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001 + 2.0);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = rsqrt(a);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_fma_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t b = fptype_t(start_val + 1.5 + (double)tid * 0.0007);
    fptype_t c = fptype_t(0.5);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = fma(a, b, c);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_exp_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = exp(a);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_erf_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.00001);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = erf(a);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_erfc_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = erfc(a);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_log_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = log(a);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_sin_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = sin(a);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_cos_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = cos(a);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_boys_f0_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Boys F_0 is defined for x >= 0; keep inputs strictly positive.
    fptype_t a = fptype_t(start_val + (double)tid * 0.0001);
    fptype_t r;

    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = boys_f0(a);
        update_bits_ff(r, a);
    }

    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_normcdfinv_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t a = fptype_t(start_val + (double)tid * 0.0000001);
    fptype_t r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = normcdfinv(a);
        update_bits_ff(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void ff_acc_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fptype_t acc = fptype_t(start_val + (double)tid * 0.0001);
    float c = static_cast<float>(start_val + 0.5 + (double)tid * 0.0007);
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc += c;  // Uses optimized __fpmp2_acc via operator+=
        update_bits_ff(acc, acc);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

// Double Precision Performance Kernels
KERNEL void double_add_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001;
    double b = start_val + 1.5 + (double)tid * 0.0007;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = a + b;
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_sub_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001;
    double b = start_val + 0.5 + (double)tid * 0.0007;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = a - b;
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_mul_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001;
    double b = start_val + 1.0001 + (double)tid * 0.00001;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = a * b;
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_div_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001 + 1.0;
    double b = start_val + 1.5 + (double)tid * 0.0007;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = a / b;
        // Update both a and b to prevent reciprocal optimization
        update_bits(r, a);
        update_bits(r, b);
    }

    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_sqrt_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001 + 2.0;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = sqrt(a);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_rsqrt_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001 + 2.0;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = rsqrt(a);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_fma_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001;
    double b = start_val + 1.5 + (double)tid * 0.0007;
    double c = 0.5;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = fma(a, b, c);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_exp_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = exp(a);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_erf_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.00001;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = erf(a);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_erfc_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = erfc(a);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_log_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = log(a);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_sin_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = sin(a);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_cos_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0001;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = cos(a);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_boys_f0_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Boys F_0 reference: F_0(x) = 0.5 * sqrt(pi/x) * erf(sqrt(x)) for x > 0,
    // F_0(0) = 1. Inputs kept strictly positive to skip the small-x guard.
    double a = start_val + (double)tid * 0.0001;
    double r;

    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        // No native double boys_f0 in libm; spell the closed-form scalar
        // reference inline (sqrt + sqrt + erf + div + mul) so we measure
        // the cost of the formula a fp64 compiler would generate naively.
        double sx = sqrt(a);
        r = 0.5 * sqrt(3.14159265358979323846 / a) * erf(sx);
        update_bits(r, a);
    }

    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_normcdfinv_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double a = start_val + (double)tid * 0.0000001;
    double r;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        r = normcdfinv(a);
        update_bits(r, a);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void double_acc_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double acc = start_val + (double)tid * 0.0001;
    double c = start_val + 0.5 + (double)tid * 0.0007;
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc += c;  // Simple double accumulation
        update_bits(acc, acc);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}
#endif

// ====================================================================================================
// ACCURACY KERNELS: Validate correctness on array inputs
// ====================================================================================================

#if USE_CUDA
KERNEL void ff_add_accuracy(const fptype_t* a, const fptype_t* b, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] + b[idx];
}

KERNEL void ff_sub_accuracy(const fptype_t* a, const fptype_t* b, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] - b[idx];
}

KERNEL void ff_mul_accuracy(const fptype_t* a, const fptype_t* b, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] * b[idx];
}

KERNEL void ff_div_accuracy(const fptype_t* a, const fptype_t* b, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] / b[idx];
}

KERNEL void ff_sqrt_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = sqrt(a[idx]);
}

KERNEL void ff_rsqrt_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = rsqrt(a[idx]);
}

KERNEL void ff_fma_accuracy(const fptype_t* a, const fptype_t* b, const fptype_t* c, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = fma(a[idx], b[idx], c[idx]);
}

KERNEL void ff_exp_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = exp(a[idx]);
    }
}

KERNEL void ff_erf_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = erf(a[idx]);
    }
}

KERNEL void ff_erfc_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = erfc(a[idx]);
    }
}

KERNEL void ff_log_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = log(a[idx]);
    }
}

KERNEL void ff_sin_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = sin(a[idx]);
    }
}

KERNEL void ff_cos_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = cos(a[idx]);
    }
}

KERNEL void ff_boys_f0_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = boys_f0(a[idx]);
    }
}

KERNEL void ff_normcdfinv_accuracy(const fptype_t* a, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = normcdfinv(a[idx]);
    }
}

// ACC: Accumulate single float into mp2 (uses optimized operator+=)
KERNEL void ff_acc_accuracy(const fptype_t* a, const float* b, fptype_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        fptype_t acc = a[idx];
        acc += b[idx];  // Uses optimized __fpmp2_acc via operator+=
        result[idx] = acc;
    }
}

KERNEL void double_add_accuracy(const double* a, const double* b, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] + b[idx];
}

KERNEL void double_sub_accuracy(const double* a, const double* b, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] - b[idx];
}

KERNEL void double_mul_accuracy(const double* a, const double* b, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] * b[idx];
}

KERNEL void double_div_accuracy(const double* a, const double* b, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] / b[idx];
}

KERNEL void double_sqrt_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = sqrt(a[idx]);
}

KERNEL void double_rsqrt_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = 1.0 / sqrt(a[idx]);
}

KERNEL void double_fma_accuracy(const double* a, const double* b, const double* c, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = fma(a[idx], b[idx], c[idx]);
}

KERNEL void double_exp_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = exp(a[idx]);
}

KERNEL void double_erf_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = erf(a[idx]);
}

KERNEL void double_erfc_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = erfc(a[idx]);
}

KERNEL void double_log_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = log(a[idx]);
}

KERNEL void double_sin_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = sin(a[idx]);
}

KERNEL void double_cos_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = cos(a[idx]);
}

KERNEL void double_boys_f0_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // F_0(x) = 0.5 * sqrt(pi/x) * erf(sqrt(x)) for x > 0; F_0(0) = 1.
        double x = a[idx];
        if (x < 1e-15) {
            result[idx] = 1.0;
        } else {
            double sx = sqrt(x);
            result[idx] = 0.5 * sqrt(3.14159265358979323846 / x) * erf(sx);
        }
    }
}

KERNEL void double_normcdfinv_accuracy(const double* a, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = normcdfinv(a[idx]);
}

// ACC reference: Accumulate single double into double
KERNEL void double_acc_accuracy(const double* a, const double* b, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] + b[idx];
}
#endif

// ====================================================================================================
// Timing Infrastructure
// ====================================================================================================

struct BenchmarkResult {
    const char* operation;
    double ff_time_ms;
    double double_time_ms;
    double ff_gflops;
    double double_gflops;
    double speedup;
};

#if USE_CUDA
double measure_kernel_time(void (*kernel)(double, unsigned char*, uint32_t), 
                           double start_val, unsigned char* result, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(start_val, result, 0);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(start_val, result, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / iterations;
}
#endif

void print_results(const BenchmarkResult& result) {
    printf("%-12s | %10.3f | %11.3f | %9.2f | %9.2f | %+8.2fx\n",
           result.operation,
           result.ff_time_ms,
           result.double_time_ms,
           result.ff_gflops,
           result.double_gflops,
           result.speedup);
}

void generate_gaussian_data(double* data, int n, double mean = 0.0, double stddev = 1.0, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(mean, stddev);
    
    for (int i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
}

#if USE_CUDA
int validate_accuracy() 
{
    const int n = ACCURACY_LEN;
    
    // Allocate memory
    double *h_a_dbl, *h_b_dbl, *h_res_dbl;
    fptype_t *d_a_ff, *d_b_ff, *d_res_ff;
    double *d_a_dbl, *d_b_dbl, *d_res_dbl2;
    int add_errors = 0, sub_errors = 0, mul_errors = 0, div_errors = 0, sqrt_errors = 0, rsqrt_errors = 0, fma_errors = 0, exp_errors = 0, log_errors = 0, erf_errors = 0, erfc_errors = 0, sin_errors = 0, cos_errors = 0, normcdfinv_errors = 0, acc_errors = 0, boys_f0_errors = 0;
    
    cudaMallocManaged(&h_a_dbl, n * sizeof(double));
    cudaMallocManaged(&h_b_dbl, n * sizeof(double));
    cudaMallocManaged(&h_res_dbl, n * sizeof(double));
    
    cudaMalloc(&d_a_ff, n * sizeof(fptype_t));
    cudaMalloc(&d_b_ff, n * sizeof(fptype_t));
    cudaMalloc(&d_res_ff, n * sizeof(fptype_t));
    
    cudaMalloc(&d_a_dbl, n * sizeof(double));
    cudaMalloc(&d_b_dbl, n * sizeof(double));
    cudaMalloc(&d_res_dbl2, n * sizeof(double));
    
    // Generate Gaussian random data
    printf("Generating %d Gaussian (mean=0, stddev=100) random numbers...\n", n);
    generate_gaussian_data(h_a_dbl, n, 0.0, 100.0, 42);   // seed=42
    generate_gaussian_data(h_b_dbl, n, 0.0, 100.0, 123);  // seed=123 (different!)
    
    // Convert to float-float on host
    fptype_t *h_a_ff = new fptype_t[n];
    fptype_t *h_b_ff = new fptype_t[n];
    fptype_t *h_res_ff = new fptype_t[n];
    
    for (int i = 0; i < n; i++) {
        h_a_ff[i] = fptype_t(h_a_dbl[i]);
        h_b_ff[i] = fptype_t(h_b_dbl[i]);
    }
    
    // Copy to device
    cudaMemcpy(d_a_ff, h_a_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_ff, h_b_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dbl, h_a_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_dbl, h_b_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Test ADD
    ff_add_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_b_ff, d_res_ff, n);
    double_add_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_b_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_add = 0.0, avg_err_add = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_add = fmax(max_err_add, err);
        avg_err_add += err;
    }
    avg_err_add /= n;
    if (max_err_add > ERR_THRESHOLD) add_errors++;
    
    // Test MUL
    ff_mul_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_b_ff, d_res_ff, n);
    double_mul_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_b_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_mul = 0.0, avg_err_mul = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_mul = fmax(max_err_mul, err);
        avg_err_mul += err;
    }
    avg_err_mul /= n;
    if (max_err_mul > ERR_THRESHOLD) mul_errors++;

    // Test DIV (use positive values)
    for (int i = 0; i < n; i++) {
        h_b_dbl[i] = fabs(h_b_dbl[i]) + 0.1;
        h_b_ff[i] = fptype_t(h_b_dbl[i]);
    }
    cudaMemcpy(d_b_ff, h_b_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_dbl, h_b_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    ff_div_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_b_ff, d_res_ff, n);
    double_div_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_b_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_div = 0.0, avg_err_div = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_div = fmax(max_err_div, err);
        avg_err_div += err;
    }
    avg_err_div /= n;
    if (max_err_div > ERR_THRESHOLD) div_errors++;
    
    // Test SUB (restore original b values - regenerate with different seed)
    generate_gaussian_data(h_b_dbl, n, 0.0, 100.0, 456);  // seed=456 (different from a's seed=42!)
    for (int i = 0; i < n; i++) {
        h_b_ff[i] = fptype_t(h_b_dbl[i]);
    }
    cudaMemcpy(d_b_ff, h_b_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_dbl, h_b_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    ff_sub_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_b_ff, d_res_ff, n);
    double_sub_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_b_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_sub = 0.0, avg_err_sub = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_sub = fmax(max_err_sub, err);
        avg_err_sub += err;
    }
    avg_err_sub /= n;
    if (max_err_sub > ERR_THRESHOLD) sub_errors++;
    
    // Test SQRT (use positive values)
    for (int i = 0; i < n; i++) {
        h_a_dbl[i] = fabs(h_a_dbl[i]) + 0.1;
        h_a_ff[i] = fptype_t(h_a_dbl[i]);
    }
    cudaMemcpy(d_a_ff, h_a_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dbl, h_a_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    ff_sqrt_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_res_ff, n);
    double_sqrt_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_sqrt = 0.0, avg_err_sqrt = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_sqrt = fmax(max_err_sqrt, err);
        avg_err_sqrt += err;
    }
    avg_err_sqrt /= n;
    if (max_err_sqrt > ERR_THRESHOLD) sqrt_errors++;
    
    // Test RSQRT (use positive values, already set from SQRT test)
    ff_rsqrt_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_res_ff, n);
    double_rsqrt_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_rsqrt = 0.0, avg_err_rsqrt = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_rsqrt = fmax(max_err_rsqrt, err);
        avg_err_rsqrt += err;
    }
    avg_err_rsqrt /= n;
    if (max_err_rsqrt > ERR_THRESHOLD) rsqrt_errors++;
    
    // Test FMA (need c array)
    fptype_t *h_c_ff = new fptype_t[n];
    double *h_c_dbl;
    fptype_t *d_c_ff;
    double *d_c_dbl;
    
    cudaMallocManaged(&h_c_dbl, n * sizeof(double));
    cudaMalloc(&d_c_ff, n * sizeof(fptype_t));
    cudaMalloc(&d_c_dbl, n * sizeof(double));
    
    generate_gaussian_data(h_c_dbl, n, 0.0, 1.0, 789);  // seed=789 (different from a and b!)
    for (int i = 0; i < n; i++) {
        h_c_ff[i] = fptype_t(h_c_dbl[i]);
    }
    cudaMemcpy(d_c_ff, h_c_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_dbl, h_c_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Restore a and b to original Gaussian values (with their original seeds)
    generate_gaussian_data(h_a_dbl, n, 0.0, 1.0, 42);   // seed=42 (same as original a)
    generate_gaussian_data(h_b_dbl, n, 0.0, 1.0, 123);  // seed=123 (same as original b)
    for (int i = 0; i < n; i++) {
        h_a_ff[i] = fptype_t(h_a_dbl[i]);
        h_b_ff[i] = fptype_t(h_b_dbl[i]);
    }
    cudaMemcpy(d_a_ff, h_a_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_ff, h_b_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dbl, h_a_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_dbl, h_b_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    ff_fma_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_b_ff, d_c_ff, d_res_ff, n);
    double_fma_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_b_dbl, d_c_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_fma = 0.0, avg_err_fma = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_fma = fmax(max_err_fma, err);
        avg_err_fma += err;
    }
    avg_err_fma /= n;
    if (max_err_fma > ERR_THRESHOLD) fma_errors++;
    
    // Test EXP (use smaller values to avoid overflow)
    for (int i = 0; i < n; i++) {
        h_a_dbl[i] = (h_a_dbl[i] / 1.0);  // Scale down to avoid overflow
        h_a_ff[i] = fptype_t(h_a_dbl[i]);
    }
    cudaMemcpy(d_a_ff, h_a_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dbl, h_a_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    ff_exp_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_res_ff, n);
    double_exp_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_exp = 0.0, avg_err_exp = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_exp = fmax(max_err_exp, err);
        avg_err_exp += err;
    }
    avg_err_exp /= n;
    if (max_err_exp > ERR_THRESHOLD) exp_errors++;
    
    // Test ERF (reuse same data as EXP — values already in reasonable range)
    ff_erf_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_res_ff, n);
    double_erf_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_erf = 0.0, avg_err_erf = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_erf = fmax(max_err_erf, err);
        avg_err_erf += err;
    }
    avg_err_erf /= n;
    if (max_err_erf > ERR_THRESHOLD) erf_errors++;
    
    // Test ERFC
    ff_erfc_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_res_ff, n);
    double_erfc_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_erfc = 0.0, avg_err_erfc = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
            max_err_erfc = fmax(max_err_erfc, err);
            avg_err_erfc += err;
        }
    }
    avg_err_erfc /= n;
    if (max_err_erfc > ERR_THRESHOLD) erfc_errors++;
    
    // Test SIN (reuse same data — Gaussian values work well for trig)
    ff_sin_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_res_ff, n);
    double_sin_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_sin = 0.0, avg_err_sin = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
            max_err_sin = fmax(max_err_sin, err);
            avg_err_sin += err;
        }
    }
    avg_err_sin /= n;
    if (max_err_sin > ERR_THRESHOLD) sin_errors++;
    
    // Test COS (reuse same data)
    ff_cos_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_res_ff, n);
    double_cos_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_cos = 0.0, avg_err_cos = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
            max_err_cos = fmax(max_err_cos, err);
            avg_err_cos += err;
        }
    }
    avg_err_cos /= n;
    if (max_err_cos > ERR_THRESHOLD) cos_errors++;
    
    // Test LOG (generate positive data in [1e-16, 1e16])
    for (int i = 0; i < n; i++) {
        h_a_dbl[i] = 1e-16 + (1e16 - 1e-16) * (double)i / (double)n;
        h_a_ff[i] = fptype_t(h_a_dbl[i]);
    }
    cudaMemcpy(d_a_ff, h_a_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dbl, h_a_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    ff_log_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_res_ff, n);
    double_log_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_log = 0.0, avg_err_log = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
            max_err_log = fmax(max_err_log, err);
            avg_err_log += err;
        }
    }
    avg_err_log /= n;
    if (max_err_log > ERR_THRESHOLD) log_errors++;

    // Test BOYS_F0 (positive inputs in [1e-3, ~100], reuse the [1e-16, 1e16]
    // grid from LOG would over-test the small-x guard; pick a more typical
    // quantum-chemistry range where F_0 is well-behaved and not saturated).
    for (int i = 0; i < n; i++) {
        h_a_dbl[i] = 1e-3 + (100.0 - 1e-3) * (double)i / (double)n;
        h_a_ff[i]  = fptype_t(h_a_dbl[i]);
    }
    cudaMemcpy(d_a_ff,  h_a_ff,  n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dbl, h_a_dbl, n * sizeof(double),   cudaMemcpyHostToDevice);

    ff_boys_f0_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff,  d_res_ff,    n);
    double_boys_f0_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_res_ff,  d_res_ff,    n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double),   cudaMemcpyDeviceToHost);

    double max_err_boys_f0 = 0.0, avg_err_boys_f0 = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i]) / h_res_dbl[i]);
            max_err_boys_f0  = fmax(max_err_boys_f0, err);
            avg_err_boys_f0 += err;
        }
    }
    avg_err_boys_f0 /= n;
    if (max_err_boys_f0 > ERR_THRESHOLD) boys_f0_errors++;

    // Test NORMCDFINV (generate uniform data in (1e-6, 1-1e-6))
    for (int i = 0; i < n; i++) {
        double p = 1e-6 + (1.0 - 2e-6) * (double)i / (double)n;
        h_a_dbl[i] = p;
        h_a_ff[i] = fptype_t(p);
    }
    cudaMemcpy(d_a_ff, h_a_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dbl, h_a_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    ff_normcdfinv_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_res_ff, n);
    double_normcdfinv_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_normcdfinv = 0.0, avg_err_normcdfinv = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
            max_err_normcdfinv = fmax(max_err_normcdfinv, err);
            avg_err_normcdfinv += err;
        }
    }
    avg_err_normcdfinv /= n;
    if (max_err_normcdfinv > ERR_THRESHOLD) normcdfinv_errors++;
    
    // Test ACC (accumulate single float into mp2)
    // Generate fresh Gaussian data for acc
    generate_gaussian_data(h_a_dbl, n, 0.0, 100.0, 42);   // seed=42
    for (int i = 0; i < n; i++) {
        h_a_ff[i] = fptype_t(h_a_dbl[i]);
    }
    cudaMemcpy(d_a_ff, h_a_ff, n * sizeof(fptype_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dbl, h_a_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    // Generate single-precision values to accumulate
    float *h_b_flt = new float[n];
    float *d_b_flt;
    cudaMalloc(&d_b_flt, n * sizeof(float));
    
    generate_gaussian_data(h_b_dbl, n, 0.0, 1.0, 999);  // seed=999 (different)
    for (int i = 0; i < n; i++) {
        h_b_flt[i] = static_cast<float>(h_b_dbl[i]);
        // Use the same float value (converted to double) for reference to ensure fair comparison
        h_b_dbl[i] = static_cast<double>(h_b_flt[i]);
    }
    cudaMemcpy(d_b_flt, h_b_flt, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_dbl, h_b_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    ff_acc_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_ff, d_b_flt, d_res_ff, n);
    double_acc_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_b_dbl, d_res_dbl2, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res_ff, d_res_ff, n * sizeof(fptype_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl2, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    double max_err_acc = 0.0, avg_err_acc = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_ff[i]);
        double err = fabs((ff_val - h_res_dbl[i])/h_res_dbl[i]);
        max_err_acc = fmax(max_err_acc, err);
        avg_err_acc += err;
    }
    avg_err_acc /= n;
    if (max_err_acc > ERR_THRESHOLD) acc_errors++;
    
    delete[] h_b_flt;
    cudaFree(d_b_flt);
    
    // Print results
    printf("\nAccuracy Results (vs Double Precision Reference):\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Operation    | Max Rel Error    | Avg Rel Error    | Status\n");
    printf("-------------+------------------+------------------+----------\n");
    printf("ADD          | %-16.6e | %-16.6e | %s\n", max_err_add, avg_err_add, 
           add_errors == 0 ? "OK" : "FAIL");
    printf("ACC          | %-16.6e | %-16.6e | %s\n", max_err_acc, avg_err_acc,
            acc_errors == 0 ? "OK" : "FAIL");
    printf("SUB          | %-16.6e | %-16.6e | %s\n", max_err_sub, avg_err_sub,
           sub_errors == 0 ? "OK" : "FAIL");
    printf("MUL          | %-16.6e | %-16.6e | %s\n", max_err_mul, avg_err_mul,
           mul_errors == 0 ? "OK" : "FAIL");
    printf("DIV          | %-16.6e | %-16.6e | %s\n", max_err_div, avg_err_div,
           div_errors == 0 ? "OK" : "FAIL");
    printf("SQRT         | %-16.6e | %-16.6e | %s\n", max_err_sqrt, avg_err_sqrt,
           sqrt_errors == 0 ? "OK" : "FAIL");
    printf("RSQRT        | %-16.6e | %-16.6e | %s\n", max_err_rsqrt, avg_err_rsqrt,
           rsqrt_errors == 0 ? "OK" : "FAIL");
    printf("FMA          | %-16.6e | %-16.6e | %s\n", max_err_fma, avg_err_fma,
           fma_errors == 0 ? "OK" : "FAIL");
    printf("EXP          | %-16.6e | %-16.6e | %s\n", max_err_exp, avg_err_exp,
           exp_errors == 0 ? "OK" : "FAIL");
    printf("ERF          | %-16.6e | %-16.6e | %s\n", max_err_erf, avg_err_erf,
           erf_errors == 0 ? "OK" : "FAIL");
    printf("ERFC         | %-16.6e | %-16.6e | %s\n", max_err_erfc, avg_err_erfc,
           erfc_errors == 0 ? "OK" : "FAIL");
    printf("SIN          | %-16.6e | %-16.6e | %s\n", max_err_sin, avg_err_sin,
           sin_errors == 0 ? "OK" : "FAIL");
    printf("COS          | %-16.6e | %-16.6e | %s\n", max_err_cos, avg_err_cos,
           cos_errors == 0 ? "OK" : "FAIL");
    printf("LOG          | %-16.6e | %-16.6e | %s\n", max_err_log, avg_err_log,
           log_errors == 0 ? "OK" : "FAIL");
    printf("BOYS_F0      | %-16.6e | %-16.6e | %s\n", max_err_boys_f0, avg_err_boys_f0,
           boys_f0_errors == 0 ? "OK" : "FAIL");
    printf("NORMCDFINV   | %-16.6e | %-16.6e | %s\n", max_err_normcdfinv, avg_err_normcdfinv,
           normcdfinv_errors == 0 ? "OK" : "FAIL");
    printf("--------------------------------------------------------------------------------\n\n");
    
    printf("Note: Float-float should be close to double but not identical.\n");
    
    // Cleanup
    delete[] h_a_ff;
    delete[] h_b_ff;
    delete[] h_res_ff;
    delete[] h_c_ff;
    cudaFree(h_a_dbl);
    cudaFree(h_b_dbl);
    cudaFree(h_res_dbl);
    cudaFree(h_c_dbl);
    cudaFree(d_a_ff);
    cudaFree(d_b_ff);
    cudaFree(d_res_ff);
    cudaFree(d_c_ff);
    cudaFree(d_a_dbl);
    cudaFree(d_b_dbl);
    cudaFree(d_res_dbl2);
    cudaFree(d_c_dbl);

    return add_errors + sub_errors + mul_errors + div_errors + sqrt_errors + rsqrt_errors + fma_errors + exp_errors + log_errors + erf_errors + erfc_errors + sin_errors + cos_errors + normcdfinv_errors + acc_errors + boys_f0_errors;
}
#endif

int main() 
{

#if USE_CUDA
    int cuda_dev = 0;
    int clockRate;
    // Select GPU to run on
    struct cudaDeviceProp props;
    cudaSetDevice (cuda_dev);
    cudaGetDeviceProperties (&props, cuda_dev);
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    int sm = props.multiProcessorCount;
#endif

    printf("================================================================================\n");
    printf("  FLOAT-FLOAT ARITHMETIC BENCHMARK\n");
    printf("  Comparing Float-Float vs Double Precision Performance/Accuracy\n");
    printf("================================================================================\n\n");
    
    printf("Configuration:\n");
    printf("  Repetitions/thread:   %d\n", REPS);
    printf("  Unroll factor:        %d\n", UNROLL);
    printf("  Iterations:           %d\n", NUM_ITERATIONS);
#if USE_CUDA
    printf("  Execution:            %s\n", props.name);
    printf("  Clock Rate:           %.2f MHz\n", clockRate / 1000.0);
    printf("  SMs:                  %d\n", sm);
    printf("  Blocks:               %d\n", NUM_BLOCKS);
    printf("  Threads/block:        %d\n", THREADS_PER_BLOCK);
#else
    printf("  Execution:            CPU\n");
    printf("\n  NOTE: This benchmark requires CUDA GPU for performance testing.\n");
    printf("        Skipping benchmark (build successful, no GPU available).\n\n");
    return 0;
#endif
    printf("\n");
    
#if USE_CUDA
    // Allocate minimal memory for result (never used)
    unsigned char* d_result;
    cudaMalloc(&d_result, NUM_BLOCKS * THREADS_PER_BLOCK);
    
    double start_val = 1.0;
    
    
    BenchmarkResult results[16];
    int idx = 0;
    
    // Calculate GFLOPS
    long long total_ops = (long long)NUM_BLOCKS * THREADS_PER_BLOCK * REPS;
    
    // Addition
    double ff_add_time     = measure_kernel_time(ff_add_perf, start_val, d_result, NUM_ITERATIONS);
    double double_add_time = measure_kernel_time(double_add_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"ADD", ff_add_time, double_add_time,
                      total_ops / (ff_add_time * 1e6),
                      total_ops / (double_add_time * 1e6),
                      double_add_time / ff_add_time};
                      
    // Accumulate (single float into mp2)
    double ff_acc_time     = measure_kernel_time(ff_acc_perf, start_val, d_result, NUM_ITERATIONS);
    double double_acc_time = measure_kernel_time(double_acc_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"ACC", ff_acc_time, double_acc_time,
                      total_ops / (ff_acc_time * 1e6),
                      total_ops / (double_acc_time * 1e6),
                      double_acc_time / ff_acc_time};
    
    // Subtraction
    double ff_sub_time     = measure_kernel_time(ff_sub_perf, start_val, d_result, NUM_ITERATIONS);
    double double_sub_time = measure_kernel_time(double_sub_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"SUB", ff_sub_time, double_sub_time,
                      total_ops / (ff_sub_time * 1e6),
                      total_ops / (double_sub_time * 1e6),
                      double_sub_time / ff_sub_time};
    
    // Multiplication
    double ff_mul_time = measure_kernel_time(ff_mul_perf, start_val, d_result, NUM_ITERATIONS);
    double double_mul_time = measure_kernel_time(double_mul_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"MUL", ff_mul_time, double_mul_time,
                      total_ops / (ff_mul_time * 1e6),
                      total_ops / (double_mul_time * 1e6),
                      double_mul_time / ff_mul_time};
    
    // Division
    double ff_div_time     = measure_kernel_time(ff_div_perf, start_val, d_result, NUM_ITERATIONS);
    double double_div_time = measure_kernel_time(double_div_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"DIV", ff_div_time, double_div_time,
                      total_ops / (ff_div_time * 1e6),
                      total_ops / (double_div_time * 1e6),
                      double_div_time / ff_div_time};
    
    // Square root
    double ff_sqrt_time     = measure_kernel_time(ff_sqrt_perf, start_val, d_result, NUM_ITERATIONS);
    double double_sqrt_time = measure_kernel_time(double_sqrt_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"SQRT", ff_sqrt_time, double_sqrt_time,
                      total_ops / (ff_sqrt_time * 1e6),
                      total_ops / (double_sqrt_time * 1e6),
                      double_sqrt_time / ff_sqrt_time};
    
    // Reciprocal square root
    double ff_rsqrt_time     = measure_kernel_time(ff_rsqrt_perf, start_val, d_result, NUM_ITERATIONS);
    double double_rsqrt_time = measure_kernel_time(double_rsqrt_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"RSQRT", ff_rsqrt_time, double_rsqrt_time,
                      total_ops / (ff_rsqrt_time * 1e6),
                      total_ops / (double_rsqrt_time * 1e6),
                      double_rsqrt_time / ff_rsqrt_time};
    
    // FMA
    double ff_fma_time     = measure_kernel_time(ff_fma_perf, start_val, d_result, NUM_ITERATIONS);
    double double_fma_time = measure_kernel_time(double_fma_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"FMA", ff_fma_time, double_fma_time,
                      2 * total_ops / (ff_fma_time * 1e6),  // FMA counts as 2 ops
                      2 * total_ops / (double_fma_time * 1e6),
                      double_fma_time / ff_fma_time};
    
    // Exponential
    double ff_exp_time     = measure_kernel_time(ff_exp_perf, start_val, d_result, NUM_ITERATIONS);
    double double_exp_time = measure_kernel_time(double_exp_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"EXP", ff_exp_time, double_exp_time,
                      total_ops / (ff_exp_time * 1e6),
                      total_ops / (double_exp_time * 1e6),
                      double_exp_time / ff_exp_time};
    
    // Error function (smaller start_val + stride to stay below erf saturation at ~5.92)
    double erf_start_val = 0.01;
    double ff_erf_time     = measure_kernel_time(ff_erf_perf, erf_start_val, d_result, NUM_ITERATIONS);
    double double_erf_time = measure_kernel_time(double_erf_perf, erf_start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"ERF", ff_erf_time, double_erf_time,
                      total_ops / (ff_erf_time * 1e6),
                      total_ops / (double_erf_time * 1e6),
                      double_erf_time / ff_erf_time};
    
    // Complementary error function
    double ff_erfc_time     = measure_kernel_time(ff_erfc_perf, start_val, d_result, NUM_ITERATIONS);
    double double_erfc_time = measure_kernel_time(double_erfc_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"ERFC", ff_erfc_time, double_erfc_time,
                      total_ops / (ff_erfc_time * 1e6),
                      total_ops / (double_erfc_time * 1e6),
                      double_erfc_time / ff_erfc_time};
    
    // Natural logarithm
    double ff_log_time     = measure_kernel_time(ff_log_perf, start_val, d_result, NUM_ITERATIONS);
    double double_log_time = measure_kernel_time(double_log_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"LOG", ff_log_time, double_log_time,
                      total_ops / (ff_log_time * 1e6),
                      total_ops / (double_log_time * 1e6),
                      double_log_time / ff_log_time};
    
    // Sine
    double ff_sin_time     = measure_kernel_time(ff_sin_perf, start_val, d_result, NUM_ITERATIONS);
    double double_sin_time = measure_kernel_time(double_sin_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"SIN", ff_sin_time, double_sin_time,
                      total_ops / (ff_sin_time * 1e6),
                      total_ops / (double_sin_time * 1e6),
                      double_sin_time / ff_sin_time};
    
    // Cosine
    double ff_cos_time     = measure_kernel_time(ff_cos_perf, start_val, d_result, NUM_ITERATIONS);
    double double_cos_time = measure_kernel_time(double_cos_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"COS", ff_cos_time, double_cos_time,
                      total_ops / (ff_cos_time * 1e6),
                      total_ops / (double_cos_time * 1e6),
                      double_cos_time / ff_cos_time};
    
    // Boys F_0 (start at 0.5, stride 0.0001 keeps inputs strictly positive)
    double boys_f0_start_val = 0.5;
    double ff_boys_f0_time     = measure_kernel_time(ff_boys_f0_perf,     boys_f0_start_val, d_result, NUM_ITERATIONS);
    double double_boys_f0_time = measure_kernel_time(double_boys_f0_perf, boys_f0_start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"BOYS_F0", ff_boys_f0_time, double_boys_f0_time,
                      total_ops / (ff_boys_f0_time     * 1e6),
                      total_ops / (double_boys_f0_time * 1e6),
                      double_boys_f0_time / ff_boys_f0_time};

    // Inverse normal CDF (start at 0.5, stride 0.0000001 keeps inputs in (0,1))
    double normcdfinv_start_val = 0.5;
    double ff_normcdfinv_time     = measure_kernel_time(ff_normcdfinv_perf, normcdfinv_start_val, d_result, NUM_ITERATIONS);
    double double_normcdfinv_time = measure_kernel_time(double_normcdfinv_perf, normcdfinv_start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"NORMCDFINV", ff_normcdfinv_time, double_normcdfinv_time,
                      total_ops / (ff_normcdfinv_time * 1e6),
                      total_ops / (double_normcdfinv_time * 1e6),
                      double_normcdfinv_time / ff_normcdfinv_time};
    
    // Print results table
    printf("================================================================================\n");
    printf("  BENCHMARK RESULTS\n");
    printf("================================================================================\n");
    printf("Operation    | FF Time(ms)| Dbl Time(ms)| FF GFLOPS | Dbl GFLOPS| Speedup\n");
    printf("-------------+------------+-------------+-----------+-----------+----------\n");
    
    for (int i = 0; i < idx; i++) {
        print_results(results[i]);
    }
    
    printf("================================================================================\n\n");
    
    printf("Legend:\n");
    printf("  FF       - Float-Float (double-float) arithmetic\n");
    printf("  Dbl      - Native double precision\n");
    printf("  GFLOPS   - Giga Floating-Point Operations Per Second\n");
    printf("  Speedup  - Ratio (>1.0 means FF is faster, <1.0 means slower)\n");
    printf("\n");
    
    cudaFree(d_result);
    
    // Run accuracy validation
    int accuracy_errors = validate_accuracy();
    
    printf("Benchmark completed %s.\n\n", accuracy_errors == 0 ? "SUCCESSFULLY" : "with ERRORS");
#endif
    
    return 0;
}
