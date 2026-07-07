/*
    ff2d_kernel.hpp - Kernel + wrapper for fp32mp2 -> double benchmark
    ==================================================================
    Included by ff2d_std.cpp and ff2d_opt.cpp with different settings.

    Required macros (must be defined before including):
      FF2D_KERNEL_NAME  — name of the __global__ kernel function
      FF2D_WRAPPER_NAME — name of the host wrapper that launches + times it

    CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE must be set (or left at default 0) before
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
using namespace cuda::experimental::fpmp;

#include "d2ff_common.hpp"

#if defined(__CUDACC__)
    #define FF2D_USE_CUDA 1
    #include <cuda_runtime.h>
#else
    #define FF2D_USE_CUDA 0
#endif

/* ------------------------------------------------------------------ */
/* CUDA helpers                                                        */
/* ------------------------------------------------------------------ */
#if FF2D_USE_CUDA
#define FF2D_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)
#endif

/* ------------------------------------------------------------------ */
/* Bit feedback: XOR LSB of double result into fp32mp2 input.          */
/* Creates a true data dependency: next iteration's conversion input   */
/* depends on previous iteration's output.                             */
/* ------------------------------------------------------------------ */
#if FF2D_USE_CUDA
static __device__ __forceinline__
#else
static inline
#endif
void ff2d_feedback_bits(double result, fp32mp2& val)
{
    uint64_t res_bits;
    memcpy(&res_bits, &result, sizeof(uint64_t));
    uint32_t hi_bits;
    float hi = val.hi();
    memcpy(&hi_bits, &hi, sizeof(uint32_t));
    hi_bits ^= (uint32_t)(res_bits & 1u);
    memcpy(&hi, &hi_bits, sizeof(float));
    float lo = val.lo();
    val = fp32mp2(hi, lo);
}

/* ------------------------------------------------------------------ */
/* Kernel: DEPTH parallel conversion chains per thread.                */
/* Each chain converts fp32mp2 -> double (THE operation under test), */
/* accumulates, feeds back, and evolves the fp32mp2 input.             */
/* ------------------------------------------------------------------ */
#if FF2D_USE_CUDA
__global__
#endif
void FF2D_KERNEL_NAME(double* __restrict__ output, int total_threads)
{
#if FF2D_USE_CUDA
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
    (void)total_threads;
    for (int tid = 0; tid < TOTAL_THREADS; ++tid)
    {
#endif

    fp32mp2 val[DEPTH];
    double    acc[DEPTH];
    double    scale[DEPTH];

    for (int d = 0; d < DEPTH; ++d) {
        float hi = 1.0f + (float)tid * 1.23456789e-4f + (float)d * 0.1f;
        float lo = hi * 1.0e-8f;
        val[d]   = fp32mp2(hi, lo);
        acc[d]   = 0.0;
        scale[d] = 1.000001 + (double)d * 0.0000001;
    }

    const int u __attribute__((unused)) = UNROLL;
    #pragma unroll u
    for (int i = 0; i < REPS; ++i) {
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            double converted = (double)val[d];   // THE operation under test
            acc[d] = acc[d] + converted * scale[d];
            ff2d_feedback_bits(acc[d], val[d]);
        }
    }

    double total = 0.0;
    #pragma unroll
    for (int d = 0; d < DEPTH; ++d)
        total = total + acc[d];

    output[tid] = total;

#if !FF2D_USE_CUDA
    }
#endif
}

/* ------------------------------------------------------------------ */
/* Wrapper: allocate, warmup, time, return result                      */
/* ------------------------------------------------------------------ */
d2ff_result FF2D_WRAPPER_NAME()
{
    const int total_threads = TOTAL_THREADS;

    double* output;
#if FF2D_USE_CUDA
    FF2D_CHECK(cudaMalloc(&output, total_threads * sizeof(double)));

    FF2D_KERNEL_NAME<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(output, total_threads);
    FF2D_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    FF2D_CHECK(cudaEventCreate(&start));
    FF2D_CHECK(cudaEventCreate(&stop));

    FF2D_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        FF2D_KERNEL_NAME<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(output, total_threads);
    FF2D_CHECK(cudaEventRecord(stop));
    FF2D_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0;
    FF2D_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    double avg_ms = (double)total_ms / NUM_ITERATIONS;

    FF2D_CHECK(cudaEventDestroy(start));
    FF2D_CHECK(cudaEventDestroy(stop));
#else
    output = new double[total_threads];

    FF2D_KERNEL_NAME(output, total_threads);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
        FF2D_KERNEL_NAME(output, total_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                    / NUM_ITERATIONS;
#endif

    double sample_val;
#if FF2D_USE_CUDA
    FF2D_CHECK(cudaMemcpy(&sample_val, output, sizeof(double), cudaMemcpyDeviceToHost));
    FF2D_CHECK(cudaFree(output));
#else
    sample_val = output[0];
    delete[] output;
#endif

    d2ff_result res;
    res.time_ms = avg_ms;
    res.sample = sample_val;
    return res;
}
