/*
    d2ff_kernel.hpp - Kernel + wrapper for the d2ff benchmark
    =========================================================
    Included by d2ff_std.cpp and d2ff_opt.cpp with different settings.

    Required macros (must be defined before including):
      D2FF_KERNEL_NAME  — name of the __global__ kernel function
      D2FF_WRAPPER_NAME — name of the host wrapper that launches + times it

    CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP must be set (or left at default 0) before
    this include to select the conversion path under test.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <cstring>
#include <chrono>

#include <cuda/fpmp>
#include <cuda/fpmp_math>

using namespace cuda::experimental;

#include "d2ff_common.hpp"

#if defined(__CUDACC__)
    #define D2FF_USE_CUDA 1
    #include <cuda_runtime.h>
#else
    #define D2FF_USE_CUDA 0
#endif

/* ------------------------------------------------------------------ */
/* CUDA helpers                                                        */
/* ------------------------------------------------------------------ */
#if D2FF_USE_CUDA
#define D2FF_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)
#endif

/* ------------------------------------------------------------------ */
/* Bit feedback: XOR LSB of result into double input.                  */
/* Creates a true data dependency: next iteration's conversion input   */
/* depends on previous iteration's output.                             */
/* ------------------------------------------------------------------ */
#if D2FF_USE_CUDA
static __device__ __forceinline__
#else
static inline
#endif
void d2ff_feedback_bits(fp32mp2 acc, double& val)
{
    uint32_t acc_bits;
    float hi = acc.hi();
    memcpy(&acc_bits, &hi, sizeof(uint32_t));
    uint64_t val_bits;
    memcpy(&val_bits, &val, sizeof(uint64_t));
    val_bits ^= (uint64_t)(acc_bits & 1u);
    memcpy(&val, &val_bits, sizeof(double));
}

/* ------------------------------------------------------------------ */
/* Kernel: DEPTH parallel conversion chains per thread.                */
/* Each chain has its own double val, fp32mp2 acc, and scale.        */
/* Higher DEPTH = more live registers = more pressure on the register  */
/* file, revealing whether the optimized conversion's extra integer    */
/* temporaries cause spills.                                           */
/* ------------------------------------------------------------------ */
#if D2FF_USE_CUDA
__global__
#endif
void D2FF_KERNEL_NAME(fp32mp2* __restrict__ output, int total_threads)
{
#if D2FF_USE_CUDA
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
    (void)total_threads;
    for (int tid = 0; tid < TOTAL_THREADS; ++tid)
    {
#endif

    double     val[DEPTH];
    fp32mp2  acc[DEPTH];
    fp32mp2  scale[DEPTH];

    for (int d = 0; d < DEPTH; ++d) {
        val[d]   = 1.0 + (double)tid * 1.23456789e-7 + (double)d * 0.1;
        acc[d]   = fp32mp2(0.0f, 0.0f);
        scale[d] = fp32mp2(1.000001f + (float)d * 0.0000001f, 0.0f);
    }

    const double evolve_mul = 1.0 + 1e-9;
    const double evolve_add = 1e-11;

    const int u __attribute__((unused)) = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; ++i) {
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            fp32mp2 converted(val[d]);
            acc[d] = acc[d] + converted * scale[d];
            d2ff_feedback_bits(acc[d], val[d]);
            val[d] = val[d] * evolve_mul + evolve_add;
        }
    }

    fp32mp2 total(0.0f, 0.0f);
    #pragma unroll
    for (int d = 0; d < DEPTH; ++d)
        total = total + acc[d];

    output[tid] = total;

#if !D2FF_USE_CUDA
    }
#endif
}

/* ------------------------------------------------------------------ */
/* Wrapper: allocate, warmup, time, return result                      */
/* ------------------------------------------------------------------ */
d2ff_result D2FF_WRAPPER_NAME()
{
    const int total_threads = TOTAL_THREADS;

    fp32mp2* output;
#if D2FF_USE_CUDA
    D2FF_CHECK(cudaMalloc(&output, total_threads * sizeof(fp32mp2)));

    D2FF_KERNEL_NAME<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(output, total_threads);
    D2FF_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    D2FF_CHECK(cudaEventCreate(&start));
    D2FF_CHECK(cudaEventCreate(&stop));

    D2FF_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        D2FF_KERNEL_NAME<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(output, total_threads);
    D2FF_CHECK(cudaEventRecord(stop));
    D2FF_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    D2FF_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    double avg_ms = (double)total_ms / NUM_ITERATIONS;

    D2FF_CHECK(cudaEventDestroy(start));
    D2FF_CHECK(cudaEventDestroy(stop));
#else
    output = new fp32mp2[total_threads];

    D2FF_KERNEL_NAME(output, total_threads);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        D2FF_KERNEL_NAME(output, total_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                    / NUM_ITERATIONS;
#endif

    fp32mp2 sample_val;
#if D2FF_USE_CUDA
    D2FF_CHECK(cudaMemcpy(&sample_val, output, sizeof(fp32mp2), cudaMemcpyDeviceToHost));
    D2FF_CHECK(cudaFree(output));
#else
    sample_val = output[0];
    delete[] output;
#endif

    d2ff_result res;
    res.time_ms = avg_ms;
    res.sample = (double)sample_val;
    return res;
}
