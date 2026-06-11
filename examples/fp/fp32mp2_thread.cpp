/*
    fp32mp2_thread.cpp - Thread Cooperation Primitives Demo for fp32mp2
    ======================================================================================================

    Minimal example demonstrating CUDA thread-cooperation intrinsics overloaded for fp32mp2.
    All these primitives are device-side only - the host build prints a skip message.

    Primitives exercised:
    -------------------------------------------------------------------------
      Atomics        : atomicAdd, atomicSub (CAS-based; 64-bit atomicCAS, any modern GPU)
      Warp shuffles  : __shfl_sync, __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync
                       (require sm_70+ / Volta)

    Three use-case kernels:
      1) Grid-wide atomic reduction (atomicAdd) into a single accumulator cell.
      2) Warp-level sum via __shfl_down_sync butterfly (lane 0 ends up with the warp total).
      3) Warp-level broadcast via __shfl_sync (lane 0's value is replicated to all 32 lanes).

    For exhaustive correctness tests, see:
        units/fpmp_atomic_ff.cpp
        units/fpmp_shfl.cpp

    Build (using the provided Makefile):
        make EXAMPLE=fp32mp2_thread               # GPU
        make EXAMPLE=fp32mp2_thread TARGET=host   # CPU stub (prints skip message)
        make run EXAMPLE=fp32mp2_thread
*/
#include <stdio.h>
#include <stdlib.h>

#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if !defined(__CUDACC__)

int main()
{
    fprintf(stderr,
            "Example skipped: thread-cooperation primitives (atomics, warp shuffles)\n"
            "are CUDA device-side and require the nvcc toolchain.\n");
    return 0;
}

#else // __CUDACC__

#include <cuda_runtime.h>

using fptype_t = fp32mp2;

#define CUDA_OK(call)                                                                                \
    do {                                                                                             \
        cudaError_t _e = (call);                                                                     \
        if (_e != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
            exit(1);                                                                                 \
        }                                                                                            \
    } while (0)

/* ---- (1) Grid-wide atomic reduction ----------------------------------------
 * Every thread adds `step` to a single accumulator cell.
 * Final value should equal `step * num_threads`. */
__global__ void atomic_reduce_kernel(fptype_t* accumulator, float step)
{
    fptype_t v(step);
    atomicAdd(accumulator, v);
}

/* ---- (2) Warp-level reduction via shfl_down ---------------------------------
 * Each lane starts with `lane + 1`.  After the butterfly, lane 0 holds
 * the warp sum (= 1+2+...+32 = 528). */
__global__ void warp_reduce_kernel(fptype_t* warp_sum)
{
    const unsigned int FULL_MASK = 0xffffffffu;
    int lane = threadIdx.x & 31;
    fptype_t v(static_cast<float>(lane + 1));
    for (int offset = 16; offset > 0; offset >>= 1)
        v = v + __shfl_down_sync(FULL_MASK, v, offset, 32);
    if (lane == 0)
        *warp_sum = v;
}

/* ---- (3) Warp-level broadcast via shfl_sync ---------------------------------
 * Lane 0 holds a known value; every other lane fetches it from lane 0
 * with __shfl_sync.  All 32 results should be identical. */
__global__ void warp_broadcast_kernel(fptype_t* per_lane)
{
    const unsigned int FULL_MASK = 0xffffffffu;
    int  lane    = threadIdx.x & 31;
    fptype_t v   ( (lane == 0) ? 12345.0f : 0.0f );          // only lane 0's value matters
    fptype_t got = __shfl_sync(FULL_MASK, v, /*src_lane=*/0, /*width=*/32);
    per_lane[threadIdx.x] = got;
}

int main()
{
    printf("\n================================================================================\n");
    printf("  FP32MP2 THREAD COOPERATION PRIMITIVES DEMO\n");
    printf("================================================================================\n\n");

    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    /* ---- (1) atomicAdd reduction -------------------------------------- */
    {
        const int   blocks  = 4;
        const int   threads = 128;
        const int   total   = blocks * threads;
        const float step    = 0.1f;

        fptype_t* d_acc = nullptr;
        CUDA_OK(cudaMallocManaged(&d_acc, sizeof(fptype_t)));
        *d_acc = fptype_t(0.0f);

        atomic_reduce_kernel<<<blocks, threads>>>(d_acc, step);
        CUDA_OK(cudaDeviceSynchronize());

        printf("(1) atomicAdd grid reduction:\n");
        printf("    %d threads x %.2f -> expected %.4f, computed %.6f\n",
               total, step, total * step, static_cast<double>(*d_acc));
        CUDA_OK(cudaFree(d_acc));
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

#endif // __CUDACC__
