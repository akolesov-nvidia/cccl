//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/*
    fp64mp2_thread.cpp - Thread Cooperation Primitives Demo for fp64mp2
    ======================================================================================================

    Minimal example demonstrating CUDA thread-cooperation intrinsics overloaded for fp64mp2.
    All these primitives are device-side only - the host build prints a skip message.

    Primitives exercised:
    -------------------------------------------------------------------------
      Atomics        : atomicAdd, atomicSub  (REQUIRES sm_90+ / Hopper for 128-bit atomicCAS)
      Warp shuffles  : __shfl_sync, __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync
                       (sm_70+ / Volta - works on any modern GPU)

    Three use-case kernels:
      1) Grid-wide atomic reduction via atomicAdd                    [sm_90+ only]
      2) Warp-level sum via __shfl_down_sync butterfly               [any modern GPU]
      3) Warp-level broadcast via __shfl_sync                        [any modern GPU]

    If compiled for an architecture below sm_90, the atomic kernel is omitted (compile-time
    gate via __CUDA_ARCH__) and a runtime message explains the skip.

    For exhaustive correctness tests, see:
        units/fpmp_atomic_dd.cpp
        units/fpmp_shfl.cpp

    Build (using the provided Makefile):
        make EXAMPLE=fp64mp2_thread               # GPU (atomics need ARCH=90)
        make EXAMPLE=fp64mp2_thread ARCH=90       # GPU on Hopper / Blackwell
        make EXAMPLE=fp64mp2_thread TARGET=host   # CPU stub
        make run EXAMPLE=fp64mp2_thread
*/
#include <cstdio>

#include <cuda/std/cstdlib>

#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if !_CCCL_CUDA_COMPILATION()

int main()
{
    fprintf(stderr,
            "Example skipped: thread-cooperation primitives (atomics, warp shuffles)\n"
            "are CUDA device-side and require the nvcc toolchain.\n");
    return 0;
}

#else // CUDA + fp64mp2 enabled

#include <cuda_runtime.h>

using fptype_t = fp64mp2;

#define CUDA_OK(call)                                                                                \
    do {                                                                                             \
        cudaError_t _e = (call);                                                                     \
        if (_e != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
            exit(1);                                                                                 \
        }                                                                                            \
    } while (0)

/* ---- (1) Grid-wide atomic reduction (sm_90+ only) ---------------------------
 * 128-bit atomicCAS is required for fp64mp2 (16 bytes); only available
 * starting Hopper (sm_90).  Guarded at device-compile time. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__global__ void atomic_reduce_kernel(fptype_t* accumulator, double step)
{
    fptype_t v(step);
    atomicAdd(accumulator, v);
}
#else
/* No-op kernel for sm < 900; main() reports the skip and does not launch it. */
__global__ void atomic_reduce_kernel(fptype_t* /*accumulator*/, double /*step*/) { }
#endif

/* ---- (2) Warp-level reduction via shfl_down --------------------------------- */
__global__ void warp_reduce_kernel(fptype_t* warp_sum)
{
    const unsigned int FULL_MASK = 0xffffffffu;
    int lane = threadIdx.x & 31;
    fptype_t v(static_cast<double>(lane + 1));
    for (int offset = 16; offset > 0; offset >>= 1)
        v = v + __shfl_down_sync(FULL_MASK, v, offset, 32);
    if (lane == 0)
        *warp_sum = v;
}

/* ---- (3) Warp-level broadcast via shfl_sync --------------------------------- */
__global__ void warp_broadcast_kernel(fptype_t* per_lane)
{
    const unsigned int FULL_MASK = 0xffffffffu;
    int  lane    = threadIdx.x & 31;
    fptype_t v   ( (lane == 0) ? 12345.0 : 0.0 );
    fptype_t got = __shfl_sync(FULL_MASK, v, /*src_lane=*/0, /*width=*/32);
    per_lane[threadIdx.x] = got;
}

int main()
{
    printf("\n================================================================================\n");
    printf("  FP64MP2 THREAD COOPERATION PRIMITIVES DEMO\n");
    printf("================================================================================\n\n");

    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    const int sm = prop.major * 100 + prop.minor * 10;
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    /* ---- (1) atomicAdd reduction (sm_90+) ----------------------------- */
    {
        printf("(1) atomicAdd grid reduction:\n");
        if (sm < 900) {
            printf("    SKIPPED - 128-bit atomicCAS requires compute capability >= 9.0 (Hopper).\n"
                   "    Current device sm_%d%d.\n", prop.major, prop.minor);
        } else {
            const int    blocks  = 4;
            const int    threads = 128;
            const int    total   = blocks * threads;
            const double step    = 0.1;

            fptype_t* d_acc = nullptr;
            CUDA_OK(cudaMallocManaged(&d_acc, sizeof(fptype_t)));
            *d_acc = fptype_t(0.0);

            atomic_reduce_kernel<<<blocks, threads>>>(d_acc, step);
            CUDA_OK(cudaDeviceSynchronize());

            printf("    %d threads x %.2f -> expected %.4f, computed %.16f\n",
                   total, step, total * step, static_cast<double>(*d_acc));
            CUDA_OK(cudaFree(d_acc));
        }
    }

    /* ---- (2) warp shuffle: down-butterfly sum ------------------------- */
    {
        fptype_t* d_sum = nullptr;
        CUDA_OK(cudaMallocManaged(&d_sum, sizeof(fptype_t)));

        warp_reduce_kernel<<<1, 32>>>(d_sum);
        CUDA_OK(cudaDeviceSynchronize());

        printf("\n(2) __shfl_down_sync warp sum (lanes hold 1..32):\n");
        printf("    expected 528, computed %.1f\n", static_cast<double>(*d_sum));
        CUDA_OK(cudaFree(d_sum));
    }

    /* ---- (3) warp shuffle: broadcast from lane 0 ---------------------- */
    {
        const int LANES = 32;
        fptype_t* d_lanes = nullptr;
        CUDA_OK(cudaMallocManaged(&d_lanes, LANES * sizeof(fptype_t)));

        warp_broadcast_kernel<<<1, LANES>>>(d_lanes);
        CUDA_OK(cudaDeviceSynchronize());

        bool   all_equal = true;
        double bcast     = static_cast<double>(d_lanes[0]);
        for (int i = 1; i < LANES; i++)
            if (static_cast<double>(d_lanes[i]) != bcast) { all_equal = false; break; }

        printf("\n(3) __shfl_sync broadcast from lane 0:\n");
        printf("    value = %.1f, all 32 lanes match: %s\n", bcast, all_equal ? "yes" : "NO");
        CUDA_OK(cudaFree(d_lanes));
    }

    printf("\n================================================================================\n");
    printf("  DEMO COMPLETED\n");
    printf("================================================================================\n\n");
    return 0;
}

#endif // _CCCL_CUDA_COMPILATION()
