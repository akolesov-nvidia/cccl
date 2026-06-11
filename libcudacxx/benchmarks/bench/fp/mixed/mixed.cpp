/*
    mixed.cpp - Mixed Precision Arithmetic Throughput Benchmark
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This benchmark compares the throughput performance (operations per SM per GPU cycle) of various 
    floating-point data types for three fundamental arithmetic operations: ADD, MUL, DIV.

    Data Types Compared:
    -------------------------------------------------------------------------
    a) float         - Native IEEE 754 single precision (32-bit)
    b) float-float   - Double-float emulation (~48-bit mantissa) using fp32mp2
    c) double        - Native IEEE 754 double precision (64-bit)
    d) fpemu_accurate - FP64 emulation, accurate method (full mantissa, full IEEE-754 range)
    e) fpemu_def      - FP64 emulation, def method (up to 1-2 LSB error, relaxed denormals)
    f) fpemu_fast     - FP64 emulation, fast method (fp32-comparable precision)
    f) double-double - Double-double emulation (~106-bit mantissa) using fp64mp2
    g) fp128         - IEEE 754 quad precision (128-bit)

    Performance Metric:
    -------------------------------------------------------------------------
    Operations per SM per GPU cycle = (total_ops) / (time_ms * clock_rate_kHz * num_SMs)

    This metric normalizes performance across different GPUs and provides a hardware-independent
    measure of arithmetic throughput.

    Configuration:
    -------------------------------------------------------------------------
    - REPS: Number of operations per thread (default: 65536)
    - UNROLL: Loop unroll factor (default: 64)
    - THREADS_PER_BLOCK: CUDA block size (default: 2048)
    - NUM_BLOCKS: Number of CUDA blocks (default: 8192)
    - NUM_ITERATIONS: Timing iterations (default: 10)

    Data Type Enable/Disable (set to 0 to disable, 1 to enable):
    -------------------------------------------------------------------------
    - __FPMP_FP32_ENABLED__          - Native float (default: 1)
    - __FPMP_FP32MP2_ENABLED__       - Float-float (default: 1)
    - __FPMP_FP64_NATIVE_ENABLED__   - Native double (default: 1)
    - __FPEMU_ACCURATE_ENABLED__     - FPEMU accurate method (default: 0)
    - __FPEMU_DEF_ENABLED__          - FPEMU def method (default: 0)
    - __FPEMU_FAST_ENABLED__         - FPEMU fast method (default: 0)
    - __FPMP_FP64MP2_ENABLED__       - Double-double (default: 1)
    - __FPMP_FP128_ENABLED__ - Quad precision fp128 (default: 0, requires HW support)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <string.h>
#include <type_traits>

// Configuration
#ifndef NUM_ITERATIONS
  #define NUM_ITERATIONS (10)
#endif

#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK (2048)
#endif

#ifndef NUM_BLOCKS
  #define NUM_BLOCKS (8192)
#endif

#ifndef REPS
  #define REPS (65536)
#endif

#ifndef UNROLL
  #define UNROLL 64
#endif

// ====================================================================================================
// Data Type Enable/Disable Configuration
// ====================================================================================================
// Set to 1 to enable, 0 to disable each data type benchmark
// Can be overridden via compiler flags: -D__FPMP_FP32_ENABLED__=0

// Data type defaults: all default to 0 (disabled)
// Enable via Makefile TYPES parameter or -D compiler flags
#ifndef __FPMP_FP32_ENABLED__
  #define __FPMP_FP32_ENABLED__ 1
#endif

#ifndef __FPMP_FP32MP2_ENABLED__
  #define __FPMP_FP32MP2_ENABLED__ 1
#endif

#ifndef __FPMP_FP64_NATIVE_ENABLED__
  #define __FPMP_FP64_NATIVE_ENABLED__ 1
#endif

#ifndef __FPEMU_ACCURATE_ENABLED__
  #define __FPEMU_ACCURATE_ENABLED__ 0
#endif

#ifndef __FPEMU_DEF_ENABLED__
  #define __FPEMU_DEF_ENABLED__ 0
#endif

#ifndef __FPEMU_FAST_ENABLED__
  #define __FPEMU_FAST_ENABLED__ 0
#endif

#ifndef __FPMP_FP64_ENABLED__ 
  #define __FPMP_FP64_ENABLED__ 1
#endif

#ifndef __FPMP_FP64MP2_ENABLED__
    #define __FPMP_FP64MP2_ENABLED__ 1
#endif
#if (FPMP_FP64MP2_ENABLE == 0)
    #undef  __FPMP_FP64MP2_ENABLED__
    #define __FPMP_FP64MP2_ENABLED__ 0
#endif

// Note: FPMP_FP128_ENABLE defaults to 0 if not defined by fpmp library


// Multi-precision + emulated floating-point libraries
#include <cuda/fpmp>
#include <cuda/fpmp_math>
#include <cuda/fpemu>

using namespace cuda::experimental;
using namespace cuda::experimental::fpmp;

// Note: FPMP_FP128_ENABLE is defined by fpmp library or compiler flags

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
// CUDA Error Handling
// ====================================================================================================

#if USE_CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Error: %s\n", cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

// Check if a kernel executed successfully, return -1.0 on error
inline double check_kernel_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "  WARNING: Kernel failed - %s\n", cudaGetErrorString(err));
        return -1.0;
    }
    return 0.0;
}
#endif

// ====================================================================================================
// Type aliases for benchmarked types
// ====================================================================================================

// a) Native float
#if __FPMP_FP32_ENABLED__ == 1
using fp32_t = float;
#endif

// b) Float-float (double-float emulation)
#if __FPMP_FP32MP2_ENABLED__ == 1
using fp32mp2_bench_t = fp32mp2;
#endif

// c) Native double
#if __FPMP_FP64_NATIVE_ENABLED__ == 1
using fp64_t = double;
#endif

// d) FPEMU accurate
#if __FPEMU_ACCURATE_ENABLED__ == 1
using fpemu_accurate_t = fp64emu_t<fp64emu_accuracy::high>;
#endif

// e) FPEMU def
#if __FPEMU_DEF_ENABLED__ == 1
using fpemu_def_t = fp64emu_t<fp64emu_accuracy::mid>;
#endif

// f) FPEMU fast
#if __FPEMU_FAST_ENABLED__ == 1
using fpemu_fast_t = fp64emu_t<fp64emu_accuracy::low>;
#endif

// f) Double-double 
#if __FPMP_FP64MP2_ENABLED__ == 1
using fp64mp2_bench_t = fp64mp2;
#endif

// g) Quad precision - available with FPMP_FP128_ENABLE
//    Uses __float128 on x86 (libquadmath) or long double on ARM64
#if __FPMP_FP128_ENABLED__ == 1
    #ifndef MIXED_HAS_LIBQUADMATH
        #if (defined(__x86_64__) || defined(_M_X64) || \
             defined(__i386__)   || defined(_M_IX86)) \
            && !defined(_MSC_VER) && !defined(_WIN32)
            #define MIXED_HAS_LIBQUADMATH 1
        #else
            #define MIXED_HAS_LIBQUADMATH 0
        #endif
    #endif
    #ifndef MIXED_HAS_LDOUBLE128
        #if defined(__aarch64__) || defined(_M_ARM64) || \
            defined(__s390x__) || defined(__LONG_DOUBLE_IEEE128__)
            #define MIXED_HAS_LDOUBLE128 1
        #else
            #define MIXED_HAS_LDOUBLE128 0
        #endif
    #endif
    #if (MIXED_HAS_LIBQUADMATH == 1)
        typedef __float128 __mixed_fp128;
    #elif (MIXED_HAS_LDOUBLE128 == 1)
        typedef long double __mixed_fp128;
    #endif
    using fp128_t = __mixed_fp128;
#endif

// ====================================================================================================
// PERFORMANCE KERNELS (template on Type — one implementation per op)
// ====================================================================================================

#if USE_CUDA
template <typename Type>
KERNEL void add_perf_kernel(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = UNROLL;
    Type a = static_cast<Type>(start_val + (double)tid * 0.001);
    Type b = static_cast<Type>(start_val + 1.5 + (double)tid * 0.0007);

    for (int i = 0; i < REPS; i += u) {
        #pragma unroll u
        for (int j = 0; j < u; j++) {
            a = a + b;
        }
    }

    if (never) {
        result[tid] = a;
    }
}

template <typename Type>
KERNEL void mul_perf_kernel(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = UNROLL;
    Type a = static_cast<Type>(start_val + (double)tid * 0.001);
    Type b = static_cast<Type>(start_val + 1.0001 + (double)tid * 0.00001);

    for (int i = 0; i < REPS; i += u) {
        #pragma unroll u
        for (int j = 0; j < u; j++) {
            a = a * b;
        }
    }

    if (never) {
        result[tid] = a;
    }
}

template <typename Type>
KERNEL void div_perf_kernel(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int u = UNROLL;
    Type a = static_cast<Type>(start_val + (double)tid * 0.001 + 1.0);
    Type b = static_cast<Type>(start_val + 1.5 + (double)tid * 0.0007);

    for (int i = 0; i < REPS; i += u) {
        #pragma unroll u
        for (int j = 0; j < u; j++) {
            a = b = a / b;
        }
    }

    if (never) {
        result[tid] = b;
    }
}
#endif // USE_CUDA

// ====================================================================================================
// Timing Infrastructure
// ====================================================================================================

struct BenchmarkResult {
    const char* data_type;
    const char* operation;
    double time_ms;
    double ops_per_sm_per_cycle;
    double gflops;
};

#if USE_CUDA
// Returns negative value on error, positive time on success
double measure_kernel_time(void (*kernel)(double, unsigned char*, uint32_t), 
                           double start_val, unsigned char* result, int iterations) {
    cudaEvent_t start, stop;
    
    if (cudaEventCreate(&start) != cudaSuccess) return -1.0;
    if (cudaEventCreate(&stop) != cudaSuccess) {
        cudaEventDestroy(start);
        return -1.0;
    }
    
    // Warmup and check for kernel errors
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(start_val, result, 0);
    cudaError_t warmup_err = cudaDeviceSynchronize();
    if (warmup_err != cudaSuccess) {
        fprintf(stderr, "  WARNING: Kernel not supported on this device - %s\n", 
                cudaGetErrorString(warmup_err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        // Clear the error state
        cudaGetLastError();
        return -1.0;
    }
    
    // Check for launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "  WARNING: Kernel launch failed - %s\n", 
                cudaGetErrorString(launch_err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return -1.0;
    }
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(start_val, result, 0);
    }
    cudaEventRecord(stop);
    
    cudaError_t sync_err = cudaEventSynchronize(stop);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "  WARNING: Kernel execution failed - %s\n", 
                cudaGetErrorString(sync_err));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaGetLastError();
        return -1.0;
    }
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds / iterations;
}

// Calculate operations per SM per GPU cycle
double calc_ops_per_sm_cycle(double time_ms, long long total_ops, int num_sms, int clock_rate_khz) {
    // time in seconds
    double time_sec = time_ms / 1000.0;
    // number of cycles during the benchmark
    double cycles = time_sec * clock_rate_khz * 1000.0;
    // operations per SM per cycle
    return (double)total_ops / (cycles * num_sms);
}
#endif

void print_header() {
    printf("================================================================================\n");
    printf("  MIXED PRECISION THROUGHPUT BENCHMARK\n");
    printf("  Comparing operations per SM per GPU cycle across data types\n");
    printf("================================================================================\n");
}

void print_table_header() {
    printf("\n");
    printf("%-14s | %-6s | %12s | %18s | %12s\n", 
           "Data Type", "Op", "Time (ms)", "Ops/SM/Cycle", "GFLOPS");
    printf("---------------+--------+--------------+--------------------+--------------\n");
}

void print_result(const BenchmarkResult& r) {
    printf("%-14s | %-6s | %12.4f | %18.6f | %12.2f\n",
           r.data_type, r.operation, r.time_ms, r.ops_per_sm_per_cycle, r.gflops);
}

void print_separator() {
    printf("---------------+--------+--------------+--------------------+--------------\n");
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) 
{
    print_header();

#if USE_CUDA
    // ========================================
    // CUDA Initialization and Error Checking
    // ========================================
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "\nERROR: No CUDA-capable GPU detected.\n");
        fprintf(stderr, "       %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    int cuda_dev = 0;
    err = cudaSetDevice(cuda_dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "\nERROR: Failed to set CUDA device %d.\n", cuda_dev);
        fprintf(stderr, "       %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    struct cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, cuda_dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "\nERROR: Failed to get device properties.\n");
        fprintf(stderr, "       %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    int clockRate;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, cuda_dev);
    int num_sms = props.multiProcessorCount;
    
    printf("\nConfiguration:\n");
    printf("  GPU:                  %s\n", props.name);
    printf("  Compute Capability:   %d.%d\n", props.major, props.minor);
    printf("  SMs:                  %d\n", num_sms);
    printf("  Clock Rate:           %.2f MHz\n", clockRate / 1000.0);
    printf("  Blocks:               %d\n", NUM_BLOCKS);
    printf("  Threads/block:        %d\n", THREADS_PER_BLOCK);
    printf("  Repetitions/thread:   %d\n", REPS);
    printf("  Unroll factor:        %d\n", UNROLL);
    printf("  Timing iterations:    %d\n", NUM_ITERATIONS);
    
    // Check for fp128 support
#if __FPMP_FP128_ENABLED__ == 1
    {
        int compute_cap = props.major * 10 + props.minor;
        if (compute_cap < 100) {
            printf("\n  NOTE: FP128 enabled but compute capability < 10.0\n");
            printf("        FP128 benchmarks may not work correctly.\n");
        }
    }
#endif
    
    // Validate configuration
    if (clockRate <= 0) {
        fprintf(stderr, "\nERROR: Invalid clock rate detected (%d).\n", clockRate);
        fprintf(stderr, "       This may indicate a driver or hardware issue.\n");
        return 1;
    }
    
    if (num_sms <= 0) {
        fprintf(stderr, "\nERROR: Invalid SM count detected (%d).\n", num_sms);
        return 1;
    }
        
    // Allocate minimal memory for result (never used)
    unsigned char* d_result = nullptr;
    err = cudaMalloc(&d_result, NUM_BLOCKS * THREADS_PER_BLOCK);
    if (err != cudaSuccess) {
        fprintf(stderr, "\nERROR: Failed to allocate device memory.\n");
        fprintf(stderr, "       %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    [[maybe_unused]] double start_val = argc;
    [[maybe_unused]] long long total_ops = (long long)NUM_BLOCKS * THREADS_PER_BLOCK * REPS;
    [[maybe_unused]] int skipped_benchmarks = 0;
    
    // Results storage
    BenchmarkResult results[64];
    int idx = 0;
    
    // Helper macro to add result only if benchmark succeeded
    #define ADD_RESULT_IF_VALID(type_name, op_name, time_val) \
        do { \
            if (time_val > 0) { \
                results[idx++] = {type_name, op_name, time_val, \
                                  calc_ops_per_sm_cycle(time_val, total_ops, num_sms, clockRate), \
                                  total_ops / (time_val * 1e6)}; \
            } else { \
                skipped_benchmarks++; \
            } \
        } while(0)
    
    printf("\nRunning benchmarks...\n");
    
    // ========================================
    // Benchmark: Native Float (fp32)
    // ========================================
#if __FPMP_FP32_ENABLED__ == 1
    printf("  Testing float...\n");
    {
        double t_add = measure_kernel_time(&add_perf_kernel<fp32_t>, start_val, d_result, NUM_ITERATIONS);
        double t_mul = measure_kernel_time(&mul_perf_kernel<fp32_t>, start_val, d_result, NUM_ITERATIONS);
        double t_div = measure_kernel_time(&div_perf_kernel<fp32_t>, start_val, d_result, NUM_ITERATIONS);
        
        ADD_RESULT_IF_VALID("float", "ADD", t_add);
        ADD_RESULT_IF_VALID("float", "MUL", t_mul);
        ADD_RESULT_IF_VALID("float", "DIV", t_div);
    }
#endif
    
    // ========================================
    // Benchmark: Float-Float (fp32mp2)
    // ========================================
#if __FPMP_FP32MP2_ENABLED__ == 1
    printf("  Testing float-float...\n");
    {
        double t_add = measure_kernel_time(&add_perf_kernel<fp32mp2_bench_t>, start_val, d_result, NUM_ITERATIONS);
        double t_mul = measure_kernel_time(&mul_perf_kernel<fp32mp2_bench_t>, start_val, d_result, NUM_ITERATIONS);
        double t_div = measure_kernel_time(&div_perf_kernel<fp32mp2_bench_t>, start_val, d_result, NUM_ITERATIONS);
        
        ADD_RESULT_IF_VALID("float-float", "ADD", t_add);
        ADD_RESULT_IF_VALID("float-float", "MUL", t_mul);
        ADD_RESULT_IF_VALID("float-float", "DIV", t_div);
    }
#endif
    
    // ========================================
    // Benchmark: Native Double (fp64)
    // ========================================
#if __FPMP_FP64_NATIVE_ENABLED__ == 1
    printf("  Testing double...\n");
    {
        double t_add = measure_kernel_time(&add_perf_kernel<fp64_t>, start_val, d_result, NUM_ITERATIONS);
        double t_mul = measure_kernel_time(&mul_perf_kernel<fp64_t>, start_val, d_result, NUM_ITERATIONS);
        double t_div = measure_kernel_time(&div_perf_kernel<fp64_t>, start_val, d_result, NUM_ITERATIONS);
        
        ADD_RESULT_IF_VALID("double", "ADD", t_add);
        ADD_RESULT_IF_VALID("double", "MUL", t_mul);
        ADD_RESULT_IF_VALID("double", "DIV", t_div);
    }
#endif
    
    // ========================================
    // Benchmark: FPEMU accurate
    // ========================================
#if __FPEMU_ACCURATE_ENABLED__ == 1
    printf("  Testing FPEMU accurate...\n");
    {
        double t_add = measure_kernel_time(&add_perf_kernel<fpemu_accurate_t>, start_val, d_result, NUM_ITERATIONS);
        double t_mul = measure_kernel_time(&mul_perf_kernel<fpemu_accurate_t>, start_val, d_result, NUM_ITERATIONS);
        double t_div = measure_kernel_time(&div_perf_kernel<fpemu_accurate_t>, start_val, d_result, NUM_ITERATIONS);
        
        ADD_RESULT_IF_VALID("fpemu_acc", "ADD", t_add);
        ADD_RESULT_IF_VALID("fpemu_acc", "MUL", t_mul);
        ADD_RESULT_IF_VALID("fpemu_acc", "DIV", t_div);
    }
#endif
    
    // ========================================
    // Benchmark: FPEMU def
    // ========================================
#if __FPEMU_DEF_ENABLED__ == 1
    printf("  Testing FPEMU def...\n");
    {
        double t_add = measure_kernel_time(&add_perf_kernel<fpemu_def_t>, start_val, d_result, NUM_ITERATIONS);
        double t_mul = measure_kernel_time(&mul_perf_kernel<fpemu_def_t>, start_val, d_result, NUM_ITERATIONS);
        double t_div = measure_kernel_time(&div_perf_kernel<fpemu_def_t>, start_val, d_result, NUM_ITERATIONS);
        
        ADD_RESULT_IF_VALID("fpemu_def", "ADD", t_add);
        ADD_RESULT_IF_VALID("fpemu_def", "MUL", t_mul);
        ADD_RESULT_IF_VALID("fpemu_def", "DIV", t_div);
    }
#endif
    
    // ========================================
    // Benchmark: FPEMU fast
    // ========================================
#if __FPEMU_FAST_ENABLED__ == 1
    printf("  Testing FPEMU fast...\n");
    {
        double t_add = measure_kernel_time(&add_perf_kernel<fpemu_fast_t>, start_val, d_result, NUM_ITERATIONS);
        double t_mul = measure_kernel_time(&mul_perf_kernel<fpemu_fast_t>, start_val, d_result, NUM_ITERATIONS);
        double t_div = measure_kernel_time(&div_perf_kernel<fpemu_fast_t>, start_val, d_result, NUM_ITERATIONS);
        
        ADD_RESULT_IF_VALID("fpemu_fast", "ADD", t_add);
        ADD_RESULT_IF_VALID("fpemu_fast", "MUL", t_mul);
        ADD_RESULT_IF_VALID("fpemu_fast", "DIV", t_div);
    }
#endif
    
    // ========================================
    // Benchmark: Double-Double (fp64mp2)
    // ========================================
#if __FPMP_FP64MP2_ENABLED__ == 1
    printf("  Testing double-double...\n");
    {
        double t_add = measure_kernel_time(&add_perf_kernel<fp64mp2_bench_t>, start_val, d_result, NUM_ITERATIONS);
        double t_mul = measure_kernel_time(&mul_perf_kernel<fp64mp2_bench_t>, start_val, d_result, NUM_ITERATIONS);
        double t_div = measure_kernel_time(&div_perf_kernel<fp64mp2_bench_t>, start_val, d_result, NUM_ITERATIONS);
        
        ADD_RESULT_IF_VALID("double-double", "ADD", t_add);
        ADD_RESULT_IF_VALID("double-double", "MUL", t_mul);
        ADD_RESULT_IF_VALID("double-double", "DIV", t_div);
    }
#endif
    
    // ========================================
    // Benchmark: Quad Precision (fp128)
    // ========================================
#if __FPMP_FP128_ENABLED__ == 1
    printf("  Testing fp128...\n");
    {
        double t_add = measure_kernel_time(&add_perf_kernel<fp128_t>, start_val, d_result, NUM_ITERATIONS);
        double t_mul = measure_kernel_time(&mul_perf_kernel<fp128_t>, start_val, d_result, NUM_ITERATIONS);
        double t_div = measure_kernel_time(&div_perf_kernel<fp128_t>, start_val, d_result, NUM_ITERATIONS);
        
        ADD_RESULT_IF_VALID("fp128", "ADD", t_add);
        ADD_RESULT_IF_VALID("fp128", "MUL", t_mul);
        ADD_RESULT_IF_VALID("fp128", "DIV", t_div);
    }
#endif
    
    // Undefine the helper macro
    #undef ADD_RESULT_IF_VALID
    
    // ========================================
    // Print Results Table
    // ========================================
    print_table_header();
    
    const char* current_type = "";
    for (int i = 0; i < idx; i++) {
        if (strcmp(current_type, results[i].data_type) != 0) {
            if (i > 0) print_separator();
            current_type = results[i].data_type;
        }
        print_result(results[i]);
    }
    
    printf("================================================================================\n");
    
    // ========================================
    // Print Summary by Operation
    // ========================================
    printf("\nSUMMARY: Relative Performance (normalized to native float)\n");
    printf("================================================================================\n");
    
    // Find baseline (float) times for each operation
    double base_add = 0, base_mul = 0, base_div = 0;
    for (int i = 0; i < idx; i++) {
        if (strcmp(results[i].data_type, "float") == 0) {
            if (strcmp(results[i].operation, "ADD") == 0) base_add = results[i].time_ms;
            if (strcmp(results[i].operation, "MUL") == 0) base_mul = results[i].time_ms;
            if (strcmp(results[i].operation, "DIV") == 0) base_div = results[i].time_ms;
        }
    }
    
    printf("\n%-14s | %12s | %12s | %12s\n", "Data Type", "ADD", "MUL", "DIV");
    printf("---------------+--------------+--------------+--------------\n");
    
    current_type = "";
    double rel_add = 0, rel_mul = 0, rel_div = 0;
    for (int i = 0; i < idx; i++) {
        if (strcmp(current_type, results[i].data_type) != 0) {
            if (strlen(current_type) > 0) {
                printf("%-14s | %11.2fx | %11.2fx | %11.2fx\n", 
                       current_type, rel_add, rel_mul, rel_div);
            }
            current_type = results[i].data_type;
            rel_add = rel_mul = rel_div = 0;
        }
        if (strcmp(results[i].operation, "ADD") == 0) rel_add = results[i].time_ms / base_add;
        if (strcmp(results[i].operation, "MUL") == 0) rel_mul = results[i].time_ms / base_mul;
        if (strcmp(results[i].operation, "DIV") == 0) rel_div = results[i].time_ms / base_div;
    }
    // Print last type
    if (strlen(current_type) > 0) {
        printf("%-14s | %11.2fx | %11.2fx | %11.2fx\n", 
               current_type, rel_add, rel_mul, rel_div);
    }
    
    printf("================================================================================\n");
    printf("\nNote: Lower relative values indicate better performance.\n");
    printf("      1.00x = same speed as native float\n");
    
    printf("\nData Type Legend:\n");
    printf("  float        - Native IEEE 754 single precision (32-bit, 24-bit mantissa)\n");
    printf("  float-float  - Double float emulation (up to 48-bit mantissa)\n");
    printf("  double       - Native IEEE 754 double precision (64-bit, 53-bit mantissa)\n");
    printf("  fpemu_acc    - FPEMU accurate (full mantissa, full IEEE-754 range)\n");
    printf("  fpemu_def    - FPEMU def (up to 1-2 LSB error, relaxed denormals)\n");
    printf("  fpemu_fast   - FPEMU fast (fp32-comparable precision)\n");
    printf("  double-double- Double double emulation (up to 106-bit mantissa)\n");
    printf("  fp128        - IEEE 754 quad precision (128-bit, 113-bit mantissa)\n");
    printf("\n");
    
    // Cleanup
    cudaFree(d_result);
    
    // Final status
    if (skipped_benchmarks > 0) {
        printf("WARNING: %d benchmark(s) failed or skipped due to unsupported operations.\n", 
               skipped_benchmarks);
        printf("         This may indicate hardware limitations or driver issues.\n\n");
    }
    
    if (idx == 0) {
        printf("ERROR: No benchmarks completed successfully.\n");
        printf("       Please check your GPU and driver configuration.\n\n");
        return 1;
    }
    
    printf("Benchmark completed. %d tests passed, %d skipped.\n\n", idx, skipped_benchmarks);
    return (skipped_benchmarks > 0) ? 2 : 0;  // Return 2 for partial success
    
#else // !USE_CUDA
    printf("\nConfiguration:\n");
    printf("  Execution:            CPU\n");
    printf("\n  NOTE: This benchmark requires CUDA GPU for performance testing.\n");
    printf("        Skipping benchmark (build successful, no GPU available).\n\n");
    return 0;
#endif
}
