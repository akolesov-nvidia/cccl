/*
    acc.cpp - Accumulate (ACC) vs Addition (ADD) Performance Benchmark
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This benchmark compares the performance of the optimized single-component accumulate operation
    (__nv_fpmp2_acc, operator+=) against using full mp2+mp2 addition when adding a single-precision
    value to a multi-precision number.

    The ACC operation is optimized for the common case of accumulating a single float/double into
    a multi-precision (hi, lo) pair. It saves ~6 floating-point operations compared to full addition
    by avoiding the 2Sum for the low parts (since the contribution has lo=0).

    Comparison:
    -------------------------------------------------------------------------
    - ACC: Uses __nv_fpmp2_acc(c, &acc_hi, &acc_lo) - optimized for single component
    - ADD: Uses __nv_fpmp2_add(acc_hi, acc_lo, c, 0.0, &res_hi, &res_lo) - treats single value as mp2

    Test Variants:
    -------------------------------------------------------------------------
    - ACC_FAST:     __nv_fpmp2_acc_fast (no normalization)
    - ACC_DEF:      __nv_fpmp2_acc (Dekker-style with normalization)
    - ACC_ACCURATE: __nv_fpmp2_acc_accurate (FPAN-style, maximum precision)
    - ADD_FAST:     Using add_fast with (c, 0.0) as second operand
    - ADD_DEF:      Using add with (c, 0.0) as second operand
    - ADD_ACCURATE: Using add_accurate with (c, 0.0) as second operand

    Configuration:
    -------------------------------------------------------------------------
    - REPS: Number of accumulations per thread (default: 1024)
    - UNROLL: Loop unroll factor (default: 64)
    - THREADS_PER_BLOCK: CUDA block size (default: 256)
    - NUM_BLOCKS: Number of CUDA blocks (default: 1024)
    - NUM_ITERATIONS: Timing iterations (default: 10)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <string.h>

// FPMP library
#include <cuda/fpmp>
#include <cuda/fpmp_math>

using namespace cuda::experimental;
using namespace cuda::experimental::fpmp;

// Configuration
#ifndef NUM_ITERATIONS
  #define NUM_ITERATIONS (10)
#endif

#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK (256)
#endif

#ifndef NUM_BLOCKS
  #define NUM_BLOCKS (1024)
#endif

#ifndef REPS
  #define REPS (1024)
#endif

#ifndef UNROLL
  #define UNROLL 64
#endif

#ifndef ACCURACY_LEN
  #define ACCURACY_LEN (64 * 1024)  // 64K elements for accuracy check
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

#if USE_CUDA
template <typename T>
HOST_DEVICE void update_bits_ff(T a, T& r) {
    constexpr int sz = sizeof(T);
    if constexpr (sz > 8) 
    {
        uint64_t a_bits = 0, r_bits = 0;
        memcpy(&a_bits, &a, 8);
        memcpy(&r_bits, &r, 8);
        r_bits = r_bits ^ (a_bits & 0x0000000100000001ul);
        memcpy(&r, &r_bits, 8);
    }
    else
    {
        uint64_t a_bits = 0, r_bits = 0;
        memcpy(&a_bits, &a, sz);
        memcpy(&r_bits, &r, sz);
        r_bits = r_bits ^ (a_bits & 0x0000000100000001ul);
        memcpy(&r, &r_bits, sz);
    }
}

HOST_DEVICE void update_bits_float(float a, float& r) {
    uint32_t a_bits = 0, r_bits = 0;
    memcpy(&a_bits, &a, sizeof(float));
    memcpy(&r_bits, &r, sizeof(float));
    r_bits = r_bits ^ (a_bits & 0x1);
    memcpy(&r, &r_bits, sizeof(float));
}
#endif

// ====================================================================================================
// PERFORMANCE KERNELS: ACC operations (optimized single-component accumulate)
// ====================================================================================================

#if USE_CUDA

// ACC Fast: Uses __nv_fpmp2_acc_fast via operator+= on fast type
KERNEL void acc_fast_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fp32mp2_low acc = fp32mp2_low(start_val + (double)tid * 0.001);
    float c = static_cast<float>(start_val + 0.5 + (double)tid * 0.0007);
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc += c;  // Uses __nv_fpmp2_acc_fast
        update_bits_float(static_cast<float>(acc), c);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

// ACC Default: Uses __nv_fpmp2_acc via operator+= on default type
KERNEL void acc_def_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fp32mp2 acc = fp32mp2(start_val + (double)tid * 0.001);
    float c = static_cast<float>(start_val + 0.5 + (double)tid * 0.0007);
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc += c;  // Uses __nv_fpmp2_acc
        update_bits_float(static_cast<float>(acc), c);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

// ACC Accurate: Uses __nv_fpmp2_acc_accurate via operator+= on accurate type
KERNEL void acc_accurate_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fp32mp2_high acc = fp32mp2_high(start_val + (double)tid * 0.001);
    float c = static_cast<float>(start_val + 0.5 + (double)tid * 0.0007);
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc += c;  // Uses __nv_fpmp2_acc_accurate
        update_bits_float(static_cast<float>(acc), c);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

// ====================================================================================================
// PERFORMANCE KERNELS: ADD operations (full mp2+mp2 addition with zero low part)
// ====================================================================================================

// ADD Fast: Uses full addition with (c, 0.0f) as second operand
KERNEL void add_fast_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fp32mp2_low acc = fp32mp2_low(start_val + (double)tid * 0.001);
    float c = static_cast<float>(start_val + 0.5 + (double)tid * 0.0007);
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc = acc + fp32mp2_low(c, 0.0f);  // Full mp2+mp2 addition
        update_bits_float(static_cast<float>(acc), c);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

// ADD Default: Uses full addition with (c, 0.0f) as second operand
KERNEL void add_def_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fp32mp2 acc = fp32mp2(start_val + (double)tid * 0.001);
    float c = static_cast<float>(start_val + 0.5 + (double)tid * 0.0007);
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc = acc + fp32mp2(c, 0.0f);  // Full mp2+mp2 addition
        update_bits_float(static_cast<float>(acc), c);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

// ADD Accurate: Uses full addition with (c, 0.0f) as second operand
KERNEL void add_accurate_perf(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    fp32mp2_high acc = fp32mp2_high(start_val + (double)tid * 0.001);
    float c = static_cast<float>(start_val + 0.5 + (double)tid * 0.0007);
    
    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc = acc + fp32mp2_high(c, 0.0f);  // Full mp2+mp2 addition
        update_bits_float(static_cast<float>(acc), c);
    }
    
    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

// ====================================================================================================
// ACCURACY KERNELS: Validate correctness on array inputs
// ====================================================================================================

// ACC accuracy kernels - accumulate single float into mp2
template<typename FpMp2Type>
KERNEL void acc_accuracy_kernel(const FpMp2Type* a, const float* b, FpMp2Type* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        FpMp2Type acc = a[idx];
        acc += b[idx];  // Uses optimized ACC
        result[idx] = acc;
    }
}

// ADD accuracy kernels - full mp2+mp2 addition with (b, 0.0)
template<typename FpMp2Type>
KERNEL void add_accuracy_kernel(const FpMp2Type* a, const float* b, FpMp2Type* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + FpMp2Type(b[idx], 0.0f);  // Full mp2+mp2 addition
    }
}

// Double reference kernel
KERNEL void double_acc_accuracy(const double* a, const double* b, double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] + b[idx];
}

// ====================================================================================================
// Timing and measurement utilities
// ====================================================================================================

template<typename KernelFunc>
double measure_kernel_time(KernelFunc kernel, double start_val, unsigned char* d_result, int iterations) {
    // Warmup
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(start_val, d_result, 0);
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(start_val, d_result, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;
}

struct BenchmarkResult {
    const char* name;
    double acc_time_ms;
    double add_time_ms;
    double acc_gops;
    double add_gops;
    double speedup;  // ACC speedup over ADD (>1.0 means ACC is faster)
};

void print_results(const BenchmarkResult& r) {
    printf("%-12s | %10.4f | %11.4f | %9.2f | %9.2f | %7.2fx\n",
           r.name, r.acc_time_ms, r.add_time_ms, r.acc_gops, r.add_gops, r.speedup);
}

// ====================================================================================================
// Accuracy validation
// ====================================================================================================

void generate_gaussian_data(double* data, int n, double mean, double stddev, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(mean, stddev);
    for (int i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
}

struct AccuracyResult {
    const char* method;
    double acc_max_err;
    double acc_avg_err;
    double add_max_err;
    double add_avg_err;
    bool acc_ok;
    bool add_ok;
};

void print_accuracy_result(const AccuracyResult& r) {
    printf("%-12s | %14.4e | %14.4e | %-4s | %14.4e | %14.4e | %-4s\n",
           r.method, 
           r.acc_max_err, r.acc_avg_err, r.acc_ok ? "OK" : "FAIL",
           r.add_max_err, r.add_avg_err, r.add_ok ? "OK" : "FAIL");
}

template<typename FpMp2Type>
AccuracyResult validate_method_accuracy(const char* method_name,
                                         FpMp2Type* d_a, float* d_b_flt, double* d_b_dbl,
                                         FpMp2Type* d_res, double* d_res_dbl,
                                         FpMp2Type* h_res, double* h_res_dbl,
                                         int n, int blocks) {
    AccuracyResult result;
    result.method = method_name;
    
    // Test ACC
    acc_accuracy_kernel<FpMp2Type><<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b_flt, d_res, n);
    double_acc_accuracy<<<blocks, THREADS_PER_BLOCK>>>(
        reinterpret_cast<double*>(d_a),  // Will be separate array
        d_b_dbl, d_res_dbl, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res, d_res, n * sizeof(FpMp2Type), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_dbl, d_res_dbl, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    result.acc_max_err = 0.0;
    result.acc_avg_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i]) / h_res_dbl[i]);
            result.acc_max_err = fmax(result.acc_max_err, err);
            result.acc_avg_err += err;
        }
    }
    result.acc_avg_err /= n;
    result.acc_ok = (result.acc_max_err <= ERR_THRESHOLD);
    
    // Test ADD (full mp2+mp2 with (b, 0.0))
    add_accuracy_kernel<FpMp2Type><<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b_flt, d_res, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_res, d_res, n * sizeof(FpMp2Type), cudaMemcpyDeviceToHost);
    
    result.add_max_err = 0.0;
    result.add_avg_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i]) / h_res_dbl[i]);
            result.add_max_err = fmax(result.add_max_err, err);
            result.add_avg_err += err;
        }
    }
    result.add_avg_err /= n;
    result.add_ok = (result.add_max_err <= ERR_THRESHOLD);
    
    return result;
}

int validate_accuracy() {
    const int n = ACCURACY_LEN;
    int errors = 0;
    
    printf("================================================================================\n");
    printf("  ACCURACY VALIDATION\n");
    printf("================================================================================\n");
    printf("Generating %d Gaussian random numbers for accuracy testing...\n\n", n);
    
    // Allocate host memory
    double* h_a_dbl = new double[n];
    double* h_b_dbl = new double[n];
    float* h_b_flt = new float[n];
    double* h_res_dbl = new double[n];
    
    fp32mp2_low* h_a_fast = new fp32mp2_low[n];
    fp32mp2* h_a_def = new fp32mp2[n];
    fp32mp2_high* h_a_accurate = new fp32mp2_high[n];
    
    fp32mp2_low* h_res_fast = new fp32mp2_low[n];
    fp32mp2* h_res_def = new fp32mp2[n];
    fp32mp2_high* h_res_accurate = new fp32mp2_high[n];
    
    // Generate test data
    generate_gaussian_data(h_a_dbl, n, 0.0, 1.0, 42);
    generate_gaussian_data(h_b_dbl, n, 0.0, 1.0, 999);
    
    for (int i = 0; i < n; i++) {
        h_a_fast[i] = fp32mp2_low(h_a_dbl[i]);
        h_a_def[i] = fp32mp2(h_a_dbl[i]);
        h_a_accurate[i] = fp32mp2_high(h_a_dbl[i]);
        h_b_flt[i] = static_cast<float>(h_b_dbl[i]);
        // Use the same float value for reference (fair comparison)
        h_b_dbl[i] = static_cast<double>(h_b_flt[i]);
    }
    
    printf("\n");
    
    // Allocate device memory
    fp32mp2_low *d_a_fast, *d_res_fast;
    fp32mp2 *d_a_def, *d_res_def;
    fp32mp2_high *d_a_accurate, *d_res_accurate;
    float* d_b_flt;
    double *d_a_dbl, *d_b_dbl, *d_res_dbl;
    
    cudaMalloc(&d_a_fast, n * sizeof(fp32mp2_low));
    cudaMalloc(&d_a_def, n * sizeof(fp32mp2));
    cudaMalloc(&d_a_accurate, n * sizeof(fp32mp2_high));
    cudaMalloc(&d_res_fast, n * sizeof(fp32mp2_low));
    cudaMalloc(&d_res_def, n * sizeof(fp32mp2));
    cudaMalloc(&d_res_accurate, n * sizeof(fp32mp2_high));
    cudaMalloc(&d_b_flt, n * sizeof(float));
    cudaMalloc(&d_a_dbl, n * sizeof(double));
    cudaMalloc(&d_b_dbl, n * sizeof(double));
    cudaMalloc(&d_res_dbl, n * sizeof(double));
    
    // Copy data to device
    cudaMemcpy(d_a_fast, h_a_fast, n * sizeof(fp32mp2_low), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_def, h_a_def, n * sizeof(fp32mp2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_accurate, h_a_accurate, n * sizeof(fp32mp2_high), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_flt, h_b_flt, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_dbl, h_a_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_dbl, h_b_dbl, n * sizeof(double), cudaMemcpyHostToDevice);
    
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    AccuracyResult acc_results[3];
    int acc_idx = 0;
    
    // Test FAST method
    // Run double reference once
    double_acc_accuracy<<<blocks, THREADS_PER_BLOCK>>>(d_a_dbl, d_b_dbl, d_res_dbl, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res_dbl, d_res_dbl, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    // FAST
    acc_accuracy_kernel<fp32mp2_low><<<blocks, THREADS_PER_BLOCK>>>(d_a_fast, d_b_flt, d_res_fast, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res_fast, d_res_fast, n * sizeof(fp32mp2_low), cudaMemcpyDeviceToHost);
    
    AccuracyResult fast_result;
    fast_result.method = "FAST";
    fast_result.acc_max_err = 0.0;
    fast_result.acc_avg_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_fast[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i]) / h_res_dbl[i]);
            fast_result.acc_max_err = fmax(fast_result.acc_max_err, err);
            fast_result.acc_avg_err += err;
        }
    }
    fast_result.acc_avg_err /= n;
    fast_result.acc_ok = (fast_result.acc_max_err <= ERR_THRESHOLD);
    
    // ADD for FAST
    add_accuracy_kernel<fp32mp2_low><<<blocks, THREADS_PER_BLOCK>>>(d_a_fast, d_b_flt, d_res_fast, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res_fast, d_res_fast, n * sizeof(fp32mp2_low), cudaMemcpyDeviceToHost);
    
    fast_result.add_max_err = 0.0;
    fast_result.add_avg_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_fast[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i]) / h_res_dbl[i]);
            fast_result.add_max_err = fmax(fast_result.add_max_err, err);
            fast_result.add_avg_err += err;
        }
    }
    fast_result.add_avg_err /= n;
    fast_result.add_ok = (fast_result.add_max_err <= ERR_THRESHOLD);
    acc_results[acc_idx++] = fast_result;
    
    // DEFAULT
    acc_accuracy_kernel<fp32mp2><<<blocks, THREADS_PER_BLOCK>>>(d_a_def, d_b_flt, d_res_def, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res_def, d_res_def, n * sizeof(fp32mp2), cudaMemcpyDeviceToHost);
    
    AccuracyResult def_result;
    def_result.method = "DEFAULT";
    def_result.acc_max_err = 0.0;
    def_result.acc_avg_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_def[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i]) / h_res_dbl[i]);
            def_result.acc_max_err = fmax(def_result.acc_max_err, err);
            def_result.acc_avg_err += err;
        }
    }
    def_result.acc_avg_err /= n;
    def_result.acc_ok = (def_result.acc_max_err <= ERR_THRESHOLD);
    
    add_accuracy_kernel<fp32mp2><<<blocks, THREADS_PER_BLOCK>>>(d_a_def, d_b_flt, d_res_def, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res_def, d_res_def, n * sizeof(fp32mp2), cudaMemcpyDeviceToHost);
    
    def_result.add_max_err = 0.0;
    def_result.add_avg_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_def[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i]) / h_res_dbl[i]);
            def_result.add_max_err = fmax(def_result.add_max_err, err);
            def_result.add_avg_err += err;
        }
    }
    def_result.add_avg_err /= n;
    def_result.add_ok = (def_result.add_max_err <= ERR_THRESHOLD);
    acc_results[acc_idx++] = def_result;
    
    // ACCURATE
    acc_accuracy_kernel<fp32mp2_high><<<blocks, THREADS_PER_BLOCK>>>(d_a_accurate, d_b_flt, d_res_accurate, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res_accurate, d_res_accurate, n * sizeof(fp32mp2_high), cudaMemcpyDeviceToHost);
    
    AccuracyResult accurate_result;
    accurate_result.method = "ACCURATE";
    accurate_result.acc_max_err = 0.0;
    accurate_result.acc_avg_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_accurate[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i]) / h_res_dbl[i]);
            accurate_result.acc_max_err = fmax(accurate_result.acc_max_err, err);
            accurate_result.acc_avg_err += err;
        }
    }
    accurate_result.acc_avg_err /= n;
    accurate_result.acc_ok = (accurate_result.acc_max_err <= ERR_THRESHOLD);
    
    add_accuracy_kernel<fp32mp2_high><<<blocks, THREADS_PER_BLOCK>>>(d_a_accurate, d_b_flt, d_res_accurate, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res_accurate, d_res_accurate, n * sizeof(fp32mp2_high), cudaMemcpyDeviceToHost);
    
    accurate_result.add_max_err = 0.0;
    accurate_result.add_avg_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ff_val = static_cast<double>(h_res_accurate[i]);
        if (h_res_dbl[i] != 0.0) {
            double err = fabs((ff_val - h_res_dbl[i]) / h_res_dbl[i]);
            accurate_result.add_max_err = fmax(accurate_result.add_max_err, err);
            accurate_result.add_avg_err += err;
        }
    }
    accurate_result.add_avg_err /= n;
    accurate_result.add_ok = (accurate_result.add_max_err <= ERR_THRESHOLD);
    acc_results[acc_idx++] = accurate_result;
    
    // Print accuracy results
    printf("Accuracy Results (vs Double Precision Reference):\n");
    printf("---------------------------------------------------------------------------------------------\n");
    printf("Method       |    ACC Max     |    ACC Avg     | ACC  |    ADD Max     |    ADD Avg     | ADD \n");
    printf("-------------+----------------+----------------+------+----------------+----------------+----\n");
    
    for (int i = 0; i < acc_idx; i++) {
        print_accuracy_result(acc_results[i]);
        if (!acc_results[i].acc_ok) errors++;
        if (!acc_results[i].add_ok) errors++;
    }
    
    printf("---------------------------------------------------------------------------------------------\n\n");
    
    printf("Note: ACC and ADD should produce identical results for the same method.\n");
    printf("      Both operations perform the same mathematical computation.\n\n");
    
    // Cleanup
    delete[] h_a_dbl;
    delete[] h_b_dbl;
    delete[] h_b_flt;
    delete[] h_res_dbl;
    delete[] h_a_fast;
    delete[] h_a_def;
    delete[] h_a_accurate;
    delete[] h_res_fast;
    delete[] h_res_def;
    delete[] h_res_accurate;
    
    cudaFree(d_a_fast);
    cudaFree(d_a_def);
    cudaFree(d_a_accurate);
    cudaFree(d_res_fast);
    cudaFree(d_res_def);
    cudaFree(d_res_accurate);
    cudaFree(d_b_flt);
    cudaFree(d_a_dbl);
    cudaFree(d_b_dbl);
    cudaFree(d_res_dbl);
    
    return errors;
}

#endif // USE_CUDA

// ====================================================================================================
// Main
// ====================================================================================================

int main() 
{

#if USE_CUDA
    int cuda_dev = 0;
    int clockRate;
    struct cudaDeviceProp props;
    cudaSetDevice(cuda_dev);
    cudaGetDeviceProperties(&props, cuda_dev);
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    int sm = props.multiProcessorCount;
#endif

    printf("================================================================================\n");
    printf("  ACCUMULATE (ACC) vs ADDITION (ADD) BENCHMARK\n");
    printf("  Comparing Optimized Single-Component Accumulate vs Full mp2+mp2 Addition\n");
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
    
    printf("Description:\n");
    printf("  ACC: Optimized accumulate - adds single float to mp2 (saves ~6 ops)\n");
    printf("  ADD: Full mp2+mp2 addition - treats single float as (c, 0.0)\n");
    printf("\n");

#if USE_CUDA
    // Allocate minimal memory for result (never used)
    unsigned char* d_result;
    cudaMalloc(&d_result, NUM_BLOCKS * THREADS_PER_BLOCK);
    
    double start_val = 1.0;
    
    BenchmarkResult results[3];
    int idx = 0;
    
    // Calculate total operations
    long long total_ops = (long long)NUM_BLOCKS * THREADS_PER_BLOCK * REPS;
    
    // Fast method
    double acc_fast_time = measure_kernel_time(acc_fast_perf, start_val, d_result, NUM_ITERATIONS);
    double add_fast_time = measure_kernel_time(add_fast_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"FAST", acc_fast_time, add_fast_time,
                      total_ops / (acc_fast_time * 1e6),
                      total_ops / (add_fast_time * 1e6),
                      add_fast_time / acc_fast_time};
    
    // Default method
    double acc_def_time = measure_kernel_time(acc_def_perf, start_val, d_result, NUM_ITERATIONS);
    double add_def_time = measure_kernel_time(add_def_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"DEFAULT", acc_def_time, add_def_time,
                      total_ops / (acc_def_time * 1e6),
                      total_ops / (add_def_time * 1e6),
                      add_def_time / acc_def_time};
    
    // Accurate method
    double acc_accurate_time = measure_kernel_time(acc_accurate_perf, start_val, d_result, NUM_ITERATIONS);
    double add_accurate_time = measure_kernel_time(add_accurate_perf, start_val, d_result, NUM_ITERATIONS);
    results[idx++] = {"ACCURATE", acc_accurate_time, add_accurate_time,
                      total_ops / (acc_accurate_time * 1e6),
                      total_ops / (add_accurate_time * 1e6),
                      add_accurate_time / acc_accurate_time};
    
    // Print results table
    printf("================================================================================\n");
    printf("  BENCHMARK RESULTS\n");
    printf("================================================================================\n");
    printf("Method       | ACC (ms)   | ADD (ms)    | ACC GOPS  | ADD GOPS  | Speedup\n");
    printf("-------------+------------+-------------+-----------+-----------+----------\n");
    
    for (int i = 0; i < idx; i++) {
        print_results(results[i]);
    }
    
    printf("================================================================================\n\n");
    
    printf("Legend:\n");
    printf("  ACC      - Optimized single-component accumulate (operator+=)\n");
    printf("  ADD      - Full mp2+mp2 addition with (c, 0.0) second operand\n");
    printf("  GOPS     - Giga Operations Per Second\n");
    printf("  Speedup  - ACC speedup over ADD (>1.0 means ACC is faster)\n");
    printf("\n");
    
    printf("Analysis:\n");
    printf("  The ACC operation saves ~6 FP operations per accumulation by:\n");
    printf("    1. Avoiding 2Sum on low parts (contribution has lo=0)\n");
    printf("    2. Direct error accumulation into existing low part\n");
    printf("  Expected speedup: ~1.3-1.5x for normalized methods, less for fast method\n");
    printf("\n");
    
    cudaFree(d_result);
    
    // Run accuracy validation
    int accuracy_errors = validate_accuracy();
    
    printf("Benchmark completed %s.\n\n", accuracy_errors == 0 ? "SUCCESSFULLY" : "with ERRORS");
#endif
    
    return 0;
}
