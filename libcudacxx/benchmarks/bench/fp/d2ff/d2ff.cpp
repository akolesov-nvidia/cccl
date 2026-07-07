/*
    d2ff.cpp - Double/fp32mp2 Conversion Throughput Benchmark
    =========================================================
    Author:  Andrei Kolesov
    Date:    2025

    Compares throughput of the standard (FP64-based) and optimized
    (integer-only) conversion paths in both directions:

      1. double -> fp32mp2  (CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP)
      2. fp32mp2 -> double  (CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE)

    Standard paths use FP64 arithmetic (F2D, D2F, DADD, DSUB),
    bottlenecked on GPUs with limited FP64 throughput (1:64 ratio).
    Optimized paths use integer bit manipulation only (no FP64).

    All four kernel variants are compiled into the same binary via
    separate compilation and compared in a single run.

    Usage:
      make                   # build and run comparison
      make rerun             # clean, build, and run
      make T=host rerun      # compare on CPU
*/

#include <stdio.h>
#include <math.h>

#if defined(__CUDACC__)
    #include <cuda_runtime.h>
#endif

#include "d2ff_common.hpp"

static void print_comparison(const char* title,
                             d2ff_result std_res, d2ff_result opt_res,
                             long long total_conv)
{
    double std_gconv = total_conv / (std_res.time_ms * 1e6);
    double opt_gconv = total_conv / (opt_res.time_ms * 1e6);
    double speedup   = std_res.time_ms / opt_res.time_ms;

    printf("%s\n", title);
    printf("  %-38s %10s %10s %8s\n",
           "Conversion", "Time (ms)", "Gconv/s", "Speedup");
    printf("  %-38s %10s %10s %8s\n",
           "--------------------------------------",
           "----------", "----------", "--------");
    printf("  %-38s %10.3f %10.1f %8s\n",
           "Standard (FP64 ops)",
           std_res.time_ms, std_gconv, "ref");
    printf("  %-38s %10.3f %10.1f %7.2fx\n",
           "Optimized (INT32 bit ops, no FP64)",
           opt_res.time_ms, opt_gconv, speedup);

    const char* match;
    if (std_res.sample == opt_res.sample)
        match = "OK (exact)";
    else if (fabs(std_res.sample - opt_res.sample) < 1e-10 * fabs(std_res.sample))
        match = "OK (approx)";
    else
        match = "MISMATCH";

    printf("  Sanity: std=%.10f  opt=%.10f  %s\n\n",
           std_res.sample, opt_res.sample, match);
}

int main()
{
    printf("Double/fp32mp2 Conversion Benchmark\n");
    printf("===================================\n");

#if defined(__CUDACC__)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Target: GPU (%s)\n", prop.name);
#else
    printf("Target: CPU (host)\n");
#endif

    printf("Threads: %d  REPS: %d  DEPTH: %d  UNROLL: %d  Timing iterations: %d\n",
           TOTAL_THREADS, REPS, DEPTH, UNROLL, NUM_ITERATIONS);

    const long long total_conv = (long long)TOTAL_THREADS * REPS * DEPTH;
    printf("Conversions per iteration: %lld\n\n", total_conv);

    // Direction 1: double -> fp32mp2
    d2ff_result d2ff_std = d2ff_run_std();
    d2ff_result d2ff_opt = d2ff_run_opt();
    print_comparison("double -> fp32mp2:", d2ff_std, d2ff_opt, total_conv);

    // Direction 2: fp32mp2 -> double
    d2ff_result ff2d_std = ff2d_run_std();
    d2ff_result ff2d_opt = ff2d_run_opt();
    print_comparison("fp32mp2 -> double:", ff2d_std, ff2d_opt, total_conv);

    return 0;
}
