/*
    fpmp_shfl.cpp - Unit test for fpmp warp shuffle overloads
    ==========================================================
    Author:  generated
    Date:    2026

    Tests CUDA warp shuffle overloads for fpmp2:
      - __shfl_sync
      - __shfl_xor_sync
      - __shfl_down_sync
      - __shfl_up_sync

    Validation strategy:
      For each operation, compare fpmp overload output against scalar CUDA
      intrinsics applied independently to hi/lo lanes:
        ref_hi = ::__shfl_*_sync(..., x.hi(), ...)
        ref_lo = ::__shfl_*_sync(..., x.lo(), ...)

      Mismatch count must be zero.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if !defined(__CUDACC__)

int main() {
    printf("fpmp_shfl: CUDA compiler required, skipping.\n");
    return 0;
}

#else

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static int g_pass = 0;
static int g_fail = 0;

template <typename MP2>
__device__ MP2 make_lane_value(int lane) {
    using FpType = decltype(MP2().hi());
    FpType hi = static_cast<FpType>(lane) + static_cast<FpType>(0.25);
    FpType lo = static_cast<FpType>(lane - 8) * static_cast<FpType>(0.03125);
    return MP2(hi, lo);
}

template <typename MP2>
__global__ void kernel_shfl_sync(int* mismatches) {
    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xFFFFFFFFu;
    const int src_lane = 3;
    const int width = 16;

    MP2 x = make_lane_value<MP2>(lane);
    MP2 y = __shfl_sync(mask, x, src_lane, width);

    auto ref_hi = ::__shfl_sync(mask, x.hi(), src_lane, width);
    auto ref_lo = ::__shfl_sync(mask, x.lo(), src_lane, width);

    if (y.hi() != ref_hi || y.lo() != ref_lo) {
        atomicAdd(mismatches, 1);
    }
}

template <typename MP2>
__global__ void kernel_shfl_xor_sync(int* mismatches) {
    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xFFFFFFFFu;
    const int lane_mask = 5;
    const int width = 16;

    MP2 x = make_lane_value<MP2>(lane);
    MP2 y = __shfl_xor_sync(mask, x, lane_mask, width);

    auto ref_hi = ::__shfl_xor_sync(mask, x.hi(), lane_mask, width);
    auto ref_lo = ::__shfl_xor_sync(mask, x.lo(), lane_mask, width);

    if (y.hi() != ref_hi || y.lo() != ref_lo) {
        atomicAdd(mismatches, 1);
    }
}

template <typename MP2>
__global__ void kernel_shfl_down_sync(int* mismatches) {
    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xFFFFFFFFu;
    const unsigned delta = 2;
    const int width = 16;

    MP2 x = make_lane_value<MP2>(lane);
    MP2 y = __shfl_down_sync(mask, x, delta, width);

    auto ref_hi = ::__shfl_down_sync(mask, x.hi(), delta, width);
    auto ref_lo = ::__shfl_down_sync(mask, x.lo(), delta, width);

    if (y.hi() != ref_hi || y.lo() != ref_lo) {
        atomicAdd(mismatches, 1);
    }
}

template <typename MP2>
__global__ void kernel_shfl_up_sync(int* mismatches) {
    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xFFFFFFFFu;
    const unsigned delta = 2;
    const int width = 16;

    MP2 x = make_lane_value<MP2>(lane);
    MP2 y = __shfl_up_sync(mask, x, delta, width);

    auto ref_hi = ::__shfl_up_sync(mask, x.hi(), delta, width);
    auto ref_lo = ::__shfl_up_sync(mask, x.lo(), delta, width);

    if (y.hi() != ref_hi || y.lo() != ref_lo) {
        atomicAdd(mismatches, 1);
    }
}

static void report_op(const char* type_name, const char* op_name, int mismatches) {
    if (mismatches == 0) {
        printf("  PASS  %-9s %-18s mismatches=%d\n", type_name, op_name, mismatches);
        g_pass++;
    } else {
        printf("  FAIL  %-9s %-18s mismatches=%d\n", type_name, op_name, mismatches);
        g_fail++;
    }
}

template <typename MP2>
void run_type_suite(const char* type_name) {
    int* d_mismatches = nullptr;
    CUDA_CHECK(cudaMallocManaged(&d_mismatches, sizeof(int)));

    *d_mismatches = 0;
    kernel_shfl_sync<MP2><<<1, 32>>>(d_mismatches);
    CUDA_CHECK(cudaDeviceSynchronize());
    report_op(type_name, "__shfl_sync", *d_mismatches);

    *d_mismatches = 0;
    kernel_shfl_xor_sync<MP2><<<1, 32>>>(d_mismatches);
    CUDA_CHECK(cudaDeviceSynchronize());
    report_op(type_name, "__shfl_xor_sync", *d_mismatches);

    *d_mismatches = 0;
    kernel_shfl_down_sync<MP2><<<1, 32>>>(d_mismatches);
    CUDA_CHECK(cudaDeviceSynchronize());
    report_op(type_name, "__shfl_down_sync", *d_mismatches);

    *d_mismatches = 0;
    kernel_shfl_up_sync<MP2><<<1, 32>>>(d_mismatches);
    CUDA_CHECK(cudaDeviceSynchronize());
    report_op(type_name, "__shfl_up_sync", *d_mismatches);

    CUDA_CHECK(cudaFree(d_mismatches));
}

int main() {
    printf("\n  fpmp_shfl: unit test for fpmp warp shuffle overloads\n");
    printf("  =====================================================\n");

    run_type_suite<fp32mp2>("fp32mp2");

    run_type_suite<fp64mp2>("fp64mp2");

    printf("\n  =====================================================\n");
    printf("  Total: %d passed, %d failed\n\n", g_pass, g_fail);
    return g_fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}

#endif // __CUDACC__
