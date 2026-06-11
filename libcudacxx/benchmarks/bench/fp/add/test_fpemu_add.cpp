/*
    test_fpemu_add.cpp - Accuracy & Performance test for standalone FP64 emulation
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This test validates the correctness and measures the performance of two
    standalone integer-only double-precision addition emulation functions:

      __nv_internal_fp64_add_int32(double x, int32_t i) — double + int32
      __nv_internal_fp64_add_fp64 (double x, double  y) — double + double

    Results are compared against native double-precision addition as the reference.

    Test Categories (Host + Device):
    -------------------------------------------------------------------------
    - BASIC:       Simple cases (zero, one, negative one, identity)
    - BOUNDARY:    INT32_MIN, INT32_MAX, large/small doubles
    - RANDOM:      Gaussian-distributed random doubles with random int32 values
    - ACCUMULATE:  Repeated accumulation to test error propagation

    CUDA Performance Comparison:
    -------------------------------------------------------------------------
    - EMU_INT32: __nv_internal_fp64_add_int32  (integer-only, double + int32)
    - EMU_FP64:  __nv_internal_fp64_add_fp64   (integer-only, double + double)
    - NATIVE:    double + (double)int32         (native FP64 DADD)

    Configuration:
    -------------------------------------------------------------------------
    - RANDOM_LEN:      Number of random test pairs (default: 64K)
    - ACCUMULATE_LEN:  Number of accumulation steps (default: 16K)
    - SEED:            RNG seed (default: 42)
    - REPS:            Operations per thread for perf (default: 1024)
    - UNROLL:          Loop unroll factor (default: 64)
    - THREADS_PER_BLOCK: CUDA block size (default: 256)
    - NUM_BLOCKS:      Number of CUDA blocks (default: 1024)
    - NUM_ITERATIONS:  Timing iterations (default: 10)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include <string.h>
#include <climits>
#include <cinttypes>

// Functions under test
#include "fpemu_add.hpp"

// Configuration
#ifndef RANDOM_LEN
  #define RANDOM_LEN (64 * 1024)
#endif

#ifndef ACCUMULATE_LEN
  #define ACCUMULATE_LEN (16 * 1024)
#endif

#ifndef SEED
  #define SEED 42
#endif

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

// Host/device macros
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
// Bit-level comparison utilities
// ====================================================================================================

static inline uint64_t double_to_bits(double d) {
    uint64_t bits;
    memcpy(&bits, &d, sizeof(uint64_t));
    return bits;
}

static inline int64_t ulp_distance(double a, double b) {
    uint64_t a_bits = double_to_bits(a);
    uint64_t b_bits = double_to_bits(b);

    int64_t ai = (int64_t)a_bits;
    int64_t bi = (int64_t)b_bits;
    if (ai < 0) ai = (int64_t)0x8000000000000000LL - ai;
    if (bi < 0) bi = (int64_t)0x8000000000000000LL - bi;

    int64_t diff = ai - bi;
    return (diff < 0) ? -diff : diff;
}

// ====================================================================================================
// Test result tracking
// ====================================================================================================

struct TestStats {
    const char* category;
    int total;
    int exact;
    int within_1ulp;
    int within_2ulp;
    int failed;
    int64_t max_ulp;
    double max_rel_err;
};

static void print_stats(const TestStats& s) {
    printf("  %-14s | %7d | %7d (%5.1f%%) | %6d (%5.1f%%) | %6d (%5.1f%%) | %4d | %4" PRId64 " | %10.2e\n",
           s.category,
           s.total,
           s.exact,  100.0 * s.exact / s.total,
           s.within_1ulp, 100.0 * s.within_1ulp / s.total,
           s.within_2ulp, 100.0 * s.within_2ulp / s.total,
           s.failed,
           s.max_ulp,
           s.max_rel_err);
}

static void update_stats(TestStats& s, double got, double expected) {
    s.total++;

    if (double_to_bits(got) == double_to_bits(expected)) {
        s.exact++;
        s.within_1ulp++;
        s.within_2ulp++;
        return;
    }

    int64_t ulps = ulp_distance(got, expected);
    double rel = (expected != 0.0) ? fabs((got - expected) / expected) : fabs(got);

    if (ulps <= 1) s.within_1ulp++;
    if (ulps <= 2) s.within_2ulp++;
    if (ulps > 2) s.failed++;

    if (ulps > s.max_ulp) s.max_ulp = ulps;
    if (rel > s.max_rel_err) s.max_rel_err = rel;
}

static void print_stats_header() {
    printf("  Category       |   Total |   Exact          |  <=1 ULP        |  <=2 ULP        | Fail | Max ULP | Max RelErr\n");
    printf("  ---------------+---------+------------------+-----------------+-----------------+------+---------+-----------\n");
}

static void print_stats_separator() {
    printf("  ---------------+---------+------------------+-----------------+-----------------+------+---------+-----------\n");
}

// Accumulate per-op totals
struct OpTotals {
    int total, exact, ulp1, ulp2, fail;
    int64_t max_ulp;
    double max_rel;

    void add(const TestStats& s) {
        total   += s.total;
        exact   += s.exact;
        ulp1    += s.within_1ulp;
        ulp2    += s.within_2ulp;
        fail    += s.failed;
        if (s.max_ulp > max_ulp) max_ulp = s.max_ulp;
        if (s.max_rel_err > max_rel) max_rel = s.max_rel_err;
    }

    void print_total() const {
        printf("  %-14s | %7d | %7d (%5.1f%%) | %6d (%5.1f%%) | %6d (%5.1f%%) | %4d | %4" PRId64 " | %10.2e\n",
               "TOTAL", total,
               exact, 100.0 * exact / total,
               ulp1,  100.0 * ulp1  / total,
               ulp2,  100.0 * ulp2  / total,
               fail,  max_ulp, max_rel);
    }

    bool pass() const { return ulp1 == total; }
};

// ====================================================================================================
// Host accuracy tests: __nv_internal_fp64_add_int32
// ====================================================================================================

static TestStats test_basic_int32() {
    TestStats stats = {"BASIC", 0, 0, 0, 0, 0, 0, 0.0};

    struct { double x; int32_t i; } cases[] = {
        {  0.0,          0 },
        {  0.0,          1 },
        {  0.0,         -1 },
        {  1.0,          0 },
        {  1.0,          1 },
        {  1.0,         -1 },
        { -1.0,          1 },
        { -1.0,         -1 },
        {  1.5,          1 },
        {  1.5,         -1 },
        { -3.0,          2 },
        {  0.5,          0 },
        { -0.5,          1 },
        {  100.75,    -100 },
        {  0.1,          3 },
        {  1234.5678,  -1234 },
        { -9999.9,     10000 },
        {  0.0,        100 },
        {  0.0,       -100 },
        { -0.0,          0 },
        { -0.0,          1 },
        { -0.0,         -1 },
    };

    int n = sizeof(cases) / sizeof(cases[0]);
    for (int t = 0; t < n; t++) {
        double expected = cases[t].x + (double)cases[t].i;
        double got = __nv_internal_fp64_add_int32(cases[t].x, cases[t].i);
        update_stats(stats, got, expected);
    }

    return stats;
}

static TestStats test_boundary_int32() {
    TestStats stats = {"BOUNDARY", 0, 0, 0, 0, 0, 0, 0.0};

    struct { double x; int32_t i; } cases[] = {
        {  0.0,         INT32_MAX },
        {  0.0,         INT32_MIN },
        {  1.0,         INT32_MAX },
        { -1.0,         INT32_MIN },
        {  1.0,         INT32_MIN },
        { -1.0,         INT32_MAX },
        {  1e10,        1000000 },
        {  1e15,              1 },
        {  1e18,             -1 },
        { -1e15,              1 },
        { -1e18,             -1 },
        {  1e-10,             1 },
        { -1e-10,            -1 },
        {  1e-300,            1 },
        { -1e-300,           -1 },
        {  (double)INT32_MAX,            -INT32_MAX },
        {  (double)INT32_MIN,  0 },
        { -(double)INT32_MAX,             INT32_MAX },
        {  1.0,               2 },
        {  2.0,              -2 },
        {  4.0,              -4 },
        {  0.25,              0 },
        {  0.125,             1 },
        {  5e-324,            1 },
        { -5e-324,           -1 },
        {  (double)(1LL << 52),  1 },
        { -(double)(1LL << 52), -1 },
    };

    int n = sizeof(cases) / sizeof(cases[0]);
    for (int t = 0; t < n; t++) {
        double expected = cases[t].x + (double)cases[t].i;
        double got = __nv_internal_fp64_add_int32(cases[t].x, cases[t].i);
        update_stats(stats, got, expected);
    }

    return stats;
}

static TestStats test_random_int32() {
    TestStats stats = {"RANDOM", 0, 0, 0, 0, 0, 0, 0.0};
    const int n = RANDOM_LEN;

    std::mt19937 gen(SEED);
    std::normal_distribution<double> dist_x(0.0, 1e6);
    std::uniform_int_distribution<int32_t> dist_i(INT32_MIN, INT32_MAX);

    for (int t = 0; t < n; t++) {
        double x = dist_x(gen);
        int32_t i = dist_i(gen);
        double expected = x + (double)i;
        double got = __nv_internal_fp64_add_int32(x, i);
        update_stats(stats, got, expected);
    }

    return stats;
}

static TestStats test_random_small_int32() {
    TestStats stats = {"RANDOM_SMALL", 0, 0, 0, 0, 0, 0, 0.0};
    const int n = RANDOM_LEN;

    std::mt19937 gen(SEED + 7);
    std::normal_distribution<double> dist_x(0.0, 10.0);
    std::uniform_int_distribution<int32_t> dist_i(-100, 100);

    for (int t = 0; t < n; t++) {
        double x = dist_x(gen);
        int32_t i = dist_i(gen);
        double expected = x + (double)i;
        double got = __nv_internal_fp64_add_int32(x, i);
        update_stats(stats, got, expected);
    }

    return stats;
}

static TestStats test_accumulate_int32() {
    TestStats stats = {"ACCUMULATE", 0, 0, 0, 0, 0, 0, 0.0};
    const int n = ACCUMULATE_LEN;

    std::mt19937 gen(SEED + 99);
    std::uniform_int_distribution<int32_t> dist_i(-1000, 1000);

    double acc_emu = 0.0;
    double acc_ref = 0.0;

    for (int t = 0; t < n; t++) {
        int32_t i = dist_i(gen);
        acc_ref = acc_ref + (double)i;
        acc_emu = __nv_internal_fp64_add_int32(acc_emu, i);
        update_stats(stats, acc_emu, acc_ref);
    }

    return stats;
}

static TestStats test_accumulate_frac_int32() {
    TestStats stats = {"ACCUM_FRAC", 0, 0, 0, 0, 0, 0, 0.0};
    const int n = ACCUMULATE_LEN;

    std::mt19937 gen(SEED + 123);
    std::uniform_int_distribution<int32_t> dist_i(-10000, 10000);

    double acc_emu = 0.123456789012345;
    double acc_ref = 0.123456789012345;

    for (int t = 0; t < n; t++) {
        int32_t i = dist_i(gen);
        acc_ref = acc_ref + (double)i;
        acc_emu = __nv_internal_fp64_add_int32(acc_emu, i);
        update_stats(stats, acc_emu, acc_ref);
    }

    return stats;
}

// ====================================================================================================
// Host accuracy tests: __nv_internal_fp64_add_fp64
// ====================================================================================================

static TestStats test_basic_fp64() {
    TestStats stats = {"BASIC", 0, 0, 0, 0, 0, 0, 0.0};

    struct { double x; double y; } cases[] = {
        {  0.0,           0.0 },
        {  0.0,           1.0 },
        {  0.0,          -1.0 },
        {  1.0,           0.0 },
        {  1.0,           1.0 },
        {  1.0,          -1.0 },
        { -1.0,           1.0 },
        { -1.0,          -1.0 },
        {  1.5,           1.0 },
        {  1.5,          -1.0 },
        { -3.0,           2.0 },
        {  0.5,           0.0 },
        { -0.5,           1.0 },
        {  100.75,     -100.0 },
        {  0.1,           3.0 },
        {  1234.5678,  -1234.0 },
        { -9999.9,     10000.0 },
        { -0.0,           0.0 },
        { -0.0,           1.0 },
        { -0.0,          -1.0 },
        {  0.1,           0.2 },
        { -0.1,          -0.2 },
        {  1e-15,         1e-15 },
        {  1.0,           1e-16 },
        {  3.14159265358979, 2.71828182845905 },
    };

    int n = sizeof(cases) / sizeof(cases[0]);
    for (int t = 0; t < n; t++) {
        double expected = cases[t].x + cases[t].y;
        double got = __nv_internal_fp64_add_fp64(cases[t].x, cases[t].y);
        update_stats(stats, got, expected);
    }

    return stats;
}

static TestStats test_boundary_fp64() {
    TestStats stats = {"BOUNDARY", 0, 0, 0, 0, 0, 0, 0.0};

    // Normal range only — no Inf/NaN/subnormal inputs or overflow results
    struct { double x; double y; } cases[] = {
        {  1e307,         1e307 },
        { -1e307,        -1e307 },
        {  1e308,        -1e308 },
        {  1e-300,        1e-300 },
        { -1e-300,       -1e-300 },
        {  1e15,          1.0 },
        {  1e18,         -1.0 },
        { -1e15,          1.0 },
        { -1e18,         -1.0 },
        {  1e-10,         1.0 },
        { -1e-10,        -1.0 },
        {  1e-300,        1.0 },
        { -1e-300,       -1.0 },
        {  (double)(1LL << 52),   1.0 },
        { -(double)(1LL << 52),  -1.0 },
        {  (double)(1LL << 52),  -(double)(1LL << 52) },
        {  0.25,          0.75 },
        {  0.125,         0.875 },
        {  1.0,          -1.0 + 1e-16 },
        {  (double)INT32_MAX,  -(double)INT32_MAX },
    };

    int n = sizeof(cases) / sizeof(cases[0]);
    for (int t = 0; t < n; t++) {
        double expected = cases[t].x + cases[t].y;
        double got = __nv_internal_fp64_add_fp64(cases[t].x, cases[t].y);
        update_stats(stats, got, expected);
    }

    return stats;
}

static TestStats test_random_fp64() {
    TestStats stats = {"RANDOM", 0, 0, 0, 0, 0, 0, 0.0};
    const int n = RANDOM_LEN;

    std::mt19937 gen(SEED + 200);
    std::normal_distribution<double> dist(0.0, 1e6);

    for (int t = 0; t < n; t++) {
        double x = dist(gen);
        double y = dist(gen);
        double expected = x + y;
        double got = __nv_internal_fp64_add_fp64(x, y);
        update_stats(stats, got, expected);
    }

    return stats;
}

static TestStats test_random_small_fp64() {
    TestStats stats = {"RANDOM_SMALL", 0, 0, 0, 0, 0, 0, 0.0};
    const int n = RANDOM_LEN;

    std::mt19937 gen(SEED + 207);
    std::normal_distribution<double> dist(0.0, 10.0);

    for (int t = 0; t < n; t++) {
        double x = dist(gen);
        double y = dist(gen);
        double expected = x + y;
        double got = __nv_internal_fp64_add_fp64(x, y);
        update_stats(stats, got, expected);
    }

    return stats;
}

// ====================================================================================================
// CUDA: Bit manipulation to prevent compiler optimization
// ====================================================================================================

#if USE_CUDA

HOST_DEVICE void update_bits_double(double a, double& r) {
    uint64_t a_bits = 0, r_bits = 0;
    memcpy(&a_bits, &a, sizeof(double));
    memcpy(&r_bits, &r, sizeof(double));
    r_bits = r_bits ^ (a_bits & 0x0000000100000001ULL);
    memcpy(&r, &r_bits, sizeof(double));
}

HOST_DEVICE void update_bits_int(int32_t a, int32_t& r) {
    r = r ^ (a & 0x1);
}

// ====================================================================================================
// CUDA: Accuracy kernels
// ====================================================================================================

KERNEL void accuracy_kernel_emu_int32(const double* x, const int32_t* ival,
                                      double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = __nv_internal_fp64_add_int32(x[idx], ival[idx]);
    }
}

KERNEL void accuracy_kernel_emu_fp64(const double* x, const double* y,
                                     double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = __nv_internal_fp64_add_fp64(x[idx], y[idx]);
    }
}

KERNEL void accuracy_kernel_native_int32(const double* x, const int32_t* ival,
                                         double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = x[idx] + (double)ival[idx];
    }
}

KERNEL void accuracy_kernel_native_fp64(const double* x, const double* y,
                                        double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = x[idx] + y[idx];
    }
}

// ====================================================================================================
// CUDA: Performance kernels — int32 group (addend is int32_t, anti-opt via update_bits_int)
// ====================================================================================================

KERNEL void perf_kernel_emu_int32(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double acc = start_val + (double)tid * 0.001;
    int32_t iv = (int32_t)(tid & 0xFFFF) - 32768;

    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc = __nv_internal_fp64_add_int32(acc, iv);
        update_bits_int(i, iv);
    }

    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void perf_kernel_native_int32(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double acc = start_val + (double)tid * 0.001;
    int32_t iv = (int32_t)(tid & 0xFFFF) - 32768;

    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc = acc + (double)iv;
        update_bits_int(i, iv);
    }

    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

// ====================================================================================================
// CUDA: Performance kernels — fp64 group (addend is double, anti-opt via update_bits_double)
// ====================================================================================================

KERNEL void perf_kernel_emu_fp64(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double acc = start_val + (double)tid * 0.001;
    double addend = (double)((int32_t)(tid & 0xFFFF) - 32768);

    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc = __nv_internal_fp64_add_fp64(acc, addend);
        update_bits_double((double)i, addend);
    }

    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

KERNEL void perf_kernel_native_fp64(double start_val, unsigned char* result, uint32_t never = 0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double acc = start_val + (double)tid * 0.001;
    double addend = (double)((int32_t)(tid & 0xFFFF) - 32768);

    const int u = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; i++) {
        acc = acc + addend;
        update_bits_double((double)i, addend);
    }

    if (never) {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&acc;
        for (int b = 0; b < (int)sizeof(acc); b++) sum += ptr[b];
        result[tid] = sum;
    }
}

// ====================================================================================================
// CUDA: Timing utility
// ====================================================================================================

template<typename KernelFunc>
double measure_kernel_time(KernelFunc kernel, double start_val,
                           unsigned char* d_result, int iterations) {
    kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(start_val, d_result, 0);
    cudaDeviceSynchronize();

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

// ====================================================================================================
// CUDA: Device accuracy validation
// ====================================================================================================

int validate_device_accuracy() {
    const int n = RANDOM_LEN;
    int errors = 0;

    // --- INT32 accuracy on device ---
    {
        printf("================================================================================\n");
        printf("  DEVICE ACCURACY: add_int32 (GPU emu vs GPU native)\n");
        printf("================================================================================\n");

        double* h_x      = new double[n];
        int32_t* h_i     = new int32_t[n];
        double* h_emu    = new double[n];
        double* h_native = new double[n];

        std::mt19937 gen(SEED + 777);
        std::normal_distribution<double> dist_x(0.0, 1e6);
        std::uniform_int_distribution<int32_t> dist_i(INT32_MIN, INT32_MAX);

        for (int t = 0; t < n; t++) {
            h_x[t] = dist_x(gen);
            h_i[t] = dist_i(gen);
        }

        double *d_x, *d_emu, *d_native;
        int32_t *d_i;
        cudaMalloc(&d_x,      n * sizeof(double));
        cudaMalloc(&d_i,      n * sizeof(int32_t));
        cudaMalloc(&d_emu,    n * sizeof(double));
        cudaMalloc(&d_native, n * sizeof(double));

        cudaMemcpy(d_x, h_x, n * sizeof(double),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_i, h_i, n * sizeof(int32_t), cudaMemcpyHostToDevice);

        int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        accuracy_kernel_emu_int32<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_i, d_emu, n);
        accuracy_kernel_native_int32<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_i, d_native, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_emu,    d_emu,    n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_native, d_native, n * sizeof(double), cudaMemcpyDeviceToHost);

        TestStats stats = {"DEV_INT32", 0, 0, 0, 0, 0, 0, 0.0};
        for (int t = 0; t < n; t++) {
            update_stats(stats, h_emu[t], h_native[t]);
        }

        print_stats_header();
        print_stats(stats);
        printf("================================================================================\n");
        if (stats.within_1ulp == stats.total)
            printf("  add_int32 device: PASS (all %d within 1 ULP)\n\n", stats.total);
        else {
            printf("  add_int32 device: FAIL (%d/%d exceeded 1 ULP)\n\n",
                   stats.total - stats.within_1ulp, stats.total);
            errors++;
        }

        delete[] h_x; delete[] h_i; delete[] h_emu; delete[] h_native;
        cudaFree(d_x); cudaFree(d_i); cudaFree(d_emu); cudaFree(d_native);
    }

    // --- FP64 accuracy on device ---
    {
        printf("================================================================================\n");
        printf("  DEVICE ACCURACY: add_fp64 (GPU emu vs GPU native)\n");
        printf("================================================================================\n");

        double* h_x      = new double[n];
        double* h_y      = new double[n];
        double* h_emu    = new double[n];
        double* h_native = new double[n];

        std::mt19937 gen(SEED + 888);
        std::normal_distribution<double> dist(0.0, 1e6);

        for (int t = 0; t < n; t++) {
            h_x[t] = dist(gen);
            h_y[t] = dist(gen);
        }

        double *d_x, *d_y, *d_emu, *d_native;
        cudaMalloc(&d_x,      n * sizeof(double));
        cudaMalloc(&d_y,      n * sizeof(double));
        cudaMalloc(&d_emu,    n * sizeof(double));
        cudaMalloc(&d_native, n * sizeof(double));

        cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, n * sizeof(double), cudaMemcpyHostToDevice);

        int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        accuracy_kernel_emu_fp64<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_y, d_emu, n);
        accuracy_kernel_native_fp64<<<blocks, THREADS_PER_BLOCK>>>(d_x, d_y, d_native, n);
        cudaDeviceSynchronize();

        cudaMemcpy(h_emu,    d_emu,    n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_native, d_native, n * sizeof(double), cudaMemcpyDeviceToHost);

        TestStats stats = {"DEV_FP64", 0, 0, 0, 0, 0, 0, 0.0};
        for (int t = 0; t < n; t++) {
            update_stats(stats, h_emu[t], h_native[t]);
        }

        print_stats_header();
        print_stats(stats);
        printf("================================================================================\n");
        if (stats.within_1ulp == stats.total)
            printf("  add_fp64  device: PASS (all %d within 1 ULP)\n\n", stats.total);
        else {
            printf("  add_fp64  device: FAIL (%d/%d exceeded 1 ULP)\n\n",
                   stats.total - stats.within_1ulp, stats.total);
            errors++;
        }

        delete[] h_x; delete[] h_y; delete[] h_emu; delete[] h_native;
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_emu); cudaFree(d_native);
    }

    return errors;
}

// ====================================================================================================
// CUDA: Performance benchmark
// ====================================================================================================

void run_performance_benchmark() {
    unsigned char* d_result;
    cudaMalloc(&d_result, NUM_BLOCKS * THREADS_PER_BLOCK);

    double start_val = 1.0;
    long long total_ops = (long long)NUM_BLOCKS * THREADS_PER_BLOCK * REPS;

    // --- Group 1: double + int32 (same addend type, same anti-opt) ---
    double native_i32_time = measure_kernel_time(perf_kernel_native_int32, start_val, d_result, NUM_ITERATIONS);
    double emu_i32_time    = measure_kernel_time(perf_kernel_emu_int32,    start_val, d_result, NUM_ITERATIONS);

    // --- Group 2: double + double (same addend type, same anti-opt) ---
    double native_f64_time = measure_kernel_time(perf_kernel_native_fp64, start_val, d_result, NUM_ITERATIONS);
    double emu_f64_time    = measure_kernel_time(perf_kernel_emu_fp64,    start_val, d_result, NUM_ITERATIONS);

    double native_i32_gops = total_ops / (native_i32_time * 1e6);
    double emu_i32_gops    = total_ops / (emu_i32_time    * 1e6);
    double native_f64_gops = total_ops / (native_f64_time * 1e6);
    double emu_f64_gops    = total_ops / (emu_f64_time    * 1e6);

    double ratio_i32 = native_i32_time / emu_i32_time;
    double ratio_f64 = native_f64_time / emu_f64_time;

    printf("================================================================================\n");
    printf("  PERFORMANCE: double + int32  (EMU vs NATIVE)\n");
    printf("================================================================================\n");
    printf("  Method       |  Time (ms) |   GOPS    | Ratio\n");
    printf("  -------------+------------+-----------+--------\n");
    printf("  NATIVE       | %10.4f | %9.2f |  1.00x\n", native_i32_time, native_i32_gops);
    printf("  EMU_INT32    | %10.4f | %9.2f | %5.2fx\n", emu_i32_time, emu_i32_gops, ratio_i32);
    printf("================================================================================\n\n");

    printf("================================================================================\n");
    printf("  PERFORMANCE: double + double (EMU vs NATIVE)\n");
    printf("================================================================================\n");
    printf("  Method       |  Time (ms) |   GOPS    | Ratio\n");
    printf("  -------------+------------+-----------+--------\n");
    printf("  NATIVE       | %10.4f | %9.2f |  1.00x\n", native_f64_time, native_f64_gops);
    printf("  EMU_FP64     | %10.4f | %9.2f | %5.2fx\n", emu_f64_time, emu_f64_gops, ratio_f64);
    printf("================================================================================\n\n");

    printf("  Total ops:        %lld\n", total_ops);
    printf("  Blocks:           %d\n", NUM_BLOCKS);
    printf("  Threads/block:    %d\n", THREADS_PER_BLOCK);
    printf("  Reps/thread:      %d\n", REPS);
    printf("  Timing iters:     %d\n\n", NUM_ITERATIONS);

    cudaFree(d_result);
}

#endif // USE_CUDA

// ====================================================================================================
// Helper: run a set of per-op tests and print results
// ====================================================================================================

static bool run_host_accuracy(const char* func_name, TestStats* per_op, int n_per_op,
                              TestStats* accum, int n_accum) {
    printf("================================================================================\n");
    printf("  HOST: %s — PER-OPERATION ACCURACY\n", func_name);
    printf("  Pass criteria: every single operation within 1 ULP\n");
    printf("================================================================================\n");
    print_stats_header();

    OpTotals tot = {};
    for (int i = 0; i < n_per_op; i++) {
        print_stats(per_op[i]);
        tot.add(per_op[i]);
    }

    print_stats_separator();
    tot.print_total();
    printf("================================================================================\n\n");

    if (n_accum > 0) {
        printf("================================================================================\n");
        printf("  HOST: %s — ACCUMULATED ERROR (informational)\n", func_name);
        printf("  HA-accuracy mode: ~1 ULP truncation per op accumulates over iterations\n");
        printf("================================================================================\n");
        print_stats_header();

        for (int i = 0; i < n_accum; i++) {
            print_stats(accum[i]);
        }

        printf("================================================================================\n\n");
    }

    bool pass = tot.pass();
    if (pass) {
        printf("  %s HOST: PASS - all %d per-op tests within 1 ULP\n", func_name, tot.total);
        printf("    Exact matches: %d/%d (%.1f%%)\n", tot.exact, tot.total,
               100.0 * tot.exact / tot.total);
        printf("    Max ULP error: %" PRId64 "\n\n", tot.max_ulp);
    } else {
        printf("  %s HOST: FAIL - %d/%d exceeded 1 ULP\n", func_name,
               tot.total - tot.ulp1, tot.total);
        printf("    Max ULP error: %" PRId64 "\n", tot.max_ulp);
        printf("    Max relative error: %.4e\n\n", tot.max_rel);
    }

    return pass;
}

// ====================================================================================================
// Main
// ====================================================================================================

int main()
{
    printf("================================================================================\n");
    printf("  STANDALONE FP64 EMULATION — ACCURACY & PERFORMANCE TEST\n");
    printf("  Integer-Only Double-Precision Addition Emulation\n");
    printf("================================================================================\n\n");

    printf("Functions under test:\n");
    printf("  __nv_internal_fp64_add_int32(double x, int32_t i)  — double + int32\n");
    printf("  __nv_internal_fp64_add_fp64 (double x, double  y)  — double + double\n\n");

    printf("Configuration:\n");
    printf("  Random test length:       %d\n", RANDOM_LEN);
    printf("  Accumulation length:      %d\n", ACCUMULATE_LEN);
    printf("  RNG seed:                 %d\n", SEED);
    printf("  Reference:                native double-precision addition\n");
#if USE_CUDA
    int cuda_dev = 0;
    int clockRate;
    struct cudaDeviceProp props;
    cudaSetDevice(cuda_dev);
    cudaGetDeviceProperties(&props, cuda_dev);
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    printf("  Execution:            %s (host + device)\n", props.name);
    printf("  Clock Rate:           %.2f MHz\n", clockRate / 1000.0);
    printf("  SMs:                  %d\n", props.multiProcessorCount);
    printf("  Reps/thread:          %d\n", REPS);
    printf("  Blocks:               %d\n", NUM_BLOCKS);
    printf("  Threads/block:        %d\n", THREADS_PER_BLOCK);
#else
    printf("  Execution:                CPU (host only)\n");
#endif
    printf("\n");

    // ================================================================
    // HOST: add_int32 accuracy
    // ================================================================

    TestStats int32_per_op[] = {
        test_basic_int32(),
        test_boundary_int32(),
        test_random_int32(),
        test_random_small_int32(),
    };
    TestStats int32_accum[] = {
        test_accumulate_int32(),
        test_accumulate_frac_int32(),
    };
    bool int32_host_pass = run_host_accuracy("add_int32",
        int32_per_op, sizeof(int32_per_op)/sizeof(int32_per_op[0]),
        int32_accum,  sizeof(int32_accum)/sizeof(int32_accum[0]));

    // ================================================================
    // HOST: add_fp64 accuracy
    // ================================================================

    TestStats fp64_per_op[] = {
        test_basic_fp64(),
        test_boundary_fp64(),
        test_random_fp64(),
        test_random_small_fp64(),
    };
    bool fp64_host_pass = run_host_accuracy("add_fp64",
        fp64_per_op, sizeof(fp64_per_op)/sizeof(fp64_per_op[0]),
        nullptr, 0);

    bool host_pass = int32_host_pass && fp64_host_pass;

    // ================================================================
    // DEVICE: accuracy + performance
    // ================================================================

#if USE_CUDA
    int device_errors = validate_device_accuracy();
    run_performance_benchmark();

    bool device_pass = (device_errors == 0);
    printf("DEVICE RESULT: %s\n\n", device_pass ? "PASS" : "FAIL");

    bool all_pass = host_pass && device_pass;
#else
    printf("  NOTE: Build with nvcc (TARGET=device) to enable GPU accuracy and performance tests.\n\n");
    bool all_pass = host_pass;
#endif

    printf("OVERALL: %s\n\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
