/*
    atomic.cpp - Float-Float AtomicAdd Benchmark
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Measure the performance and basic numeric behavior of atomicAdd on native double (baseline)
    and FPMP float-float fp32mp2 under high contention.

    What This Benchmark Does:
    -------------------------------------------------------------------------
    - Launches a kernel where all threads repeatedly call atomicAdd on the SAME memory location.
      This is a deliberate worst-case contention scenario (serialization dominates).
    - Times one kernel run (after a short warmup) and prints:
      - elapsed time (ms)
      - slowdown vs. native double atomics
      - final accumulated result and relative error vs. the expected sum

    Configuration (compile-time):
    -------------------------------------------------------------------------
    The Makefile passes these as -D... defines:
      - NUM_ITERATIONS    (iterations per thread)
      - THREADS_PER_BLOCK (threads per block)
      - NUM_BLOCKS        (number of blocks)

    Building and Running:
    -------------------------------------------------------------------------
    Preferred (via this benchmark's Makefile):
        make                                          # Build and run
        make NUM_ITER=100 THREADS=128 NUM_BLOCKS=32 run
        make ARCH=89 rebuild                          # Ada / RTX 40xx

    Notes / Limitations:
    -------------------------------------------------------------------------
    - This file benchmarks atomicAdd only (not atomicSub).
    - Results reflect extreme contention; they are not representative of low-contention atomics.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <cuda/fpmp>
#include <cuda/fpmp_math>

using namespace cuda::experimental;
using namespace cuda::experimental::fpmp;

// Type alias for the multi-precision floating-point type
using fptype_t = fp32mp2;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Benchmark parameters - can be overridden via Makefile
#ifndef NUM_ITERATIONS
  #define NUM_ITERATIONS (50)
#endif
#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK (64)
#endif
#ifndef NUM_BLOCKS
  #define NUM_BLOCKS (16)
#endif

// Total operations
constexpr int TOTAL_OPS = NUM_ITERATIONS * THREADS_PER_BLOCK * NUM_BLOCKS;

// Kernel for fptype_t atomicAdd
__global__ void atomicAdd_ffloat_kernel(fptype_t* result, int iterations)
{
    fptype_t increment(0.001f);
    
    for (int i = 0; i < iterations; i++) {
        atomicAdd(result, increment);
    }
}

// Kernel for native double atomicAdd
__global__ void atomicAdd_double_kernel(double* result, int iterations)
{
    double increment = 0.001;
    
    for (int i = 0; i < iterations; i++) {
        atomicAdd(result, increment);
    }
}

// Timing helper with progress indication
// WARNING: Does NOT reset memory between runs - memory will accumulate!
template<typename Func, typename ResetFunc>
double benchmark_kernel(Func kernel_func, ResetFunc reset_func, const char* name, int warmup_runs = 2)
{
    // Warmup (reset before to avoid accumulation)
    reset_func();
    for (int i = 0; i < warmup_runs; i++) {
        kernel_func();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // Reset before actual benchmark
    reset_func();
    // Benchmark - measure single run after warmup
    auto start = std::chrono::high_resolution_clock::now();
    kernel_func();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time_ms = elapsed.count();    
    return avg_time_ms;
}

int main()
{
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Atomic Operations Performance Benchmark" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Get device properties
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "\nDevice: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "\nBenchmark Configuration:" << std::endl;
    std::cout << "  Threads per block: " << THREADS_PER_BLOCK << std::endl;
    std::cout << "  Number of blocks:  " << NUM_BLOCKS << std::endl;
    std::cout << "  Iterations/thread: " << NUM_ITERATIONS << std::endl;
    std::cout << "  Total operations:  " << TOTAL_OPS << " (" 
              << (TOTAL_OPS / 1e6) << "M ops)" << std::endl;
    std::cout << "\nNote: All threads atomically update the SAME memory location" << std::endl;
    std::cout << "      This creates extreme contention (realistic worst-case scenario)" << std::endl;
    std::cout << std::endl;
    
    // Allocate device memory
    fptype_t* d_efloat_result;
    double* d_double_result;
    
    CUDA_CHECK(cudaMalloc(&d_efloat_result, sizeof(fptype_t)));
    CUDA_CHECK(cudaMalloc(&d_double_result, sizeof(double)));
    
    // Initialize to zero
    fptype_t h_efloat_zero(0.0f);
    double h_double_zero = 0.0;
    
    CUDA_CHECK(cudaMemcpy(d_efloat_result, &h_efloat_zero, sizeof(fptype_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_double_result, &h_double_zero, sizeof(double), cudaMemcpyHostToDevice));
    
    std::cout << std::string(78, '-') << std::endl;
    std::cout << std::left << std::setw(35) << "Operation Type" 
              << std::right << std::setw(15) << "Time (ms)" 
              << std::setw(12) << "Slowdown" 
              << std::setw(8) << "Result" << std::endl;
    std::cout << std::string(78, '-') << std::endl;
    
    // Benchmark 1: Native double atomicAdd (baseline)
    double double_time = benchmark_kernel
    (
        [&]() {
            atomicAdd_double_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_double_result, NUM_ITERATIONS);
        },
        [&]() {
            double zero = 0.0;
            CUDA_CHECK(cudaMemcpy(d_double_result, &zero, sizeof(double), cudaMemcpyHostToDevice));
        },
        "Native double atomicAdd"
    );
    
    double h_double_result;
    CUDA_CHECK(cudaMemcpy(&h_double_result, d_double_result, sizeof(double), cudaMemcpyDeviceToHost));
    
    std::cout << std::left << std::setw(35) << "Native double atomicAdd" 
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << double_time
              << std::setw(12) << std::setprecision(2) << std::fixed << "1.00x"
              << std::right << std::setw(12) << std::setprecision(6) << h_double_result << std::endl;
    
    // Benchmark 2: fptype_t atomicAdd (fp32mp2)
    double efloat_time = benchmark_kernel
    (
        [&]() {
            atomicAdd_ffloat_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_efloat_result, NUM_ITERATIONS);
        },
        [&]() {
            fptype_t zero(0.0f);
            CUDA_CHECK(cudaMemcpy(d_efloat_result, &zero, sizeof(fptype_t), cudaMemcpyHostToDevice));
        },
        "fptype_t atomicAdd"
    );
    
    fptype_t h_efloat_result;
    CUDA_CHECK(cudaMemcpy(&h_efloat_result, d_efloat_result, sizeof(fptype_t), cudaMemcpyDeviceToHost));
    
    double slowdown = efloat_time / double_time;
    std::cout << std::left << std::setw(35) << "fptype_t atomicAdd" 
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << efloat_time
              << std::setw(11) << std::setprecision(2) << std::fixed << slowdown << "x"
              << std::right << std::setw(12) << std::setprecision(6) << static_cast<double>(h_efloat_result) << std::endl;
    
    std::cout << std::string(78, '-') << std::endl;
    
    // Performance analysis
    std::cout << "\nPerformance Analysis:" << std::endl;
    std::cout << "  fptype_t atomicAdd is " << std::setprecision(2) << slowdown 
              << "x slower than native double" << std::endl;
    
    // Verify correctness
    double expected = TOTAL_OPS * 0.001;
    double error_double = std::abs(h_double_result - expected) / expected;
    double error_efloat = std::abs(static_cast<double>(h_efloat_result) - expected) / expected;
    
    std::cout << "\nAccuracy (vs expected " << std::fixed << std::setprecision(2) << expected << "):" << std::endl;
    std::cout << "  Native double:      " << std::setprecision(6) << std::scientific << error_double << " rel error" << std::endl;
    std::cout << "  fptype_t:     " << std::setprecision(6) << std::scientific << error_efloat << " rel error" << std::endl;
    
    // Throughput analysis
    double ops_per_sec_double = TOTAL_OPS / (double_time / 1000.0);
    double ops_per_sec_efloat = TOTAL_OPS / (efloat_time / 1000.0);
    
    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  Native double:   " << std::setprecision(2) << (ops_per_sec_double / 1e6) << " MOp/s" << std::endl;
    std::cout << "  fptype_t:  " << std::setprecision(2) << (ops_per_sec_efloat / 1e6) << " MOp/s" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_efloat_result));
    CUDA_CHECK(cudaFree(d_double_result));
    
    std::cout << "\n" << std::string(78, '=') << std::endl;
    
    return 0;
}

