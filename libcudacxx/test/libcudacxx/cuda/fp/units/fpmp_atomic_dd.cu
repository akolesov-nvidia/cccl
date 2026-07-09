// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: atomicAdd / atomicSub on fp64mp2 (double-double).
//
//  Device-only test (multi-block 128-bit atomics; requires compute capability
//  >= 9.0 / Hopper for 128-bit atomicCAS). Two checks:
//    - Atomicity: every thread does atomicAdd(1.0) then atomicSub(1.0); with
//      correct atomics the shared accumulator cancels back to ~0.
//    - Accuracy: many threads accumulate/subtract a small value and the result is
//      compared against the analytic sum within a relative tolerance.
//
//  On a host-only build (no CUDA) the test is SKIP()ed; under CUDA it runs on the
//  device, and is SKIP()ed at runtime on devices with compute capability < 9.0.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cmath>

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the double-double multi-precision floating-point type.
using dfloat = fp64mp2;

#if _CCCL_CUDA_COMPILATION()
// Each thread adds then subtracts 1.0; the accumulator must cancel to ~0.
__global__ void test_atomicity_kernel_dd(unsigned int* idx, dfloat* res)
{
  dfloat val(1.0);
  atomicAdd(idx, 1u);
  atomicAdd(res, val);
  atomicSub(res, val);
}

__global__ void test_atomicAdd_accuracy_kernel_dd(dfloat* res, double value_to_add)
{
  dfloat val(value_to_add);
  atomicAdd(res, val);
}

__global__ void test_atomicSub_accuracy_kernel_dd(dfloat* res, double value_to_sub)
{
  dfloat val(value_to_sub);
  atomicSub(res, val);
}

// 128-bit atomicCAS (needed by the double-double atomics) is only available on
// compute capability >= 9.0. Skip on older devices.
static bool device_supports_dd_atomics()
{
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
  {
    return false;
  }
  return prop.major >= 9;
}
#endif // _CCCL_CUDA_COMPILATION()

// Always-run host smoke check. The double-double atomic tests below are
// Hopper-only and SKIP() on host-only builds or on devices with compute
// capability < 9.0. If every test case in this binary skips, Catch2 returns a
// non-zero "all tests skipped" exit code and CTest reports the whole binary as
// failed. This trivial case guarantees at least one non-skipped assertion runs.
C2H_TEST("fpmp fp64mp2 host smoke", "[fpmp]")
{
  fp_ran_on_host();
  const dfloat a(1.5);
  const dfloat b(-0.25);
  REQUIRE(static_cast<double>(a) == 1.5);
  REQUIRE(static_cast<double>(b) == -0.25);
}

C2H_TEST("fpmp double-double atomicAdd/atomicSub atomicity", "[fpmp]")
{
#if !_CCCL_CUDA_COMPILATION()
  SKIP("atomicAdd/atomicSub on fp64mp2 are device-only");
#else
  if (!device_supports_dd_atomics())
  {
    SKIP("double-double atomics require compute capability >= 9.0 (Hopper)");
  }
  fp_ran_on_device();
  const int num_threads = 512;
  const int num_blocks  = 4;

  unsigned int* d_idx = nullptr;
  dfloat* d_res       = nullptr;
  REQUIRE_CUDART(cudaMalloc(&d_idx, sizeof(unsigned int)));
  REQUIRE_CUDART(cudaMalloc(&d_res, sizeof(dfloat)));

  unsigned int h_idx = 0;
  dfloat h_res(0.0);
  REQUIRE_CUDART(cudaMemcpy(d_idx, &h_idx, sizeof(unsigned int), cudaMemcpyHostToDevice));
  REQUIRE_CUDART(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice));

  test_atomicity_kernel_dd<<<num_blocks, num_threads>>>(d_idx, d_res);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());

  REQUIRE_CUDART(cudaMemcpy(&h_idx, d_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  REQUIRE_CUDART(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost));

  const double result = static_cast<double>(h_res);
  ::printf("  atomicity: threads=%u  result=%.16e\n", h_idx, result);

  REQUIRE(h_idx == static_cast<unsigned int>(num_threads * num_blocks));
  REQUIRE(::cuda::std::fabs(result) < 1e-14);

  REQUIRE_CUDART(cudaFree(d_idx));
  REQUIRE_CUDART(cudaFree(d_res));
#endif // _CCCL_CUDA_COMPILATION()
}

C2H_TEST("fpmp double-double atomicAdd/atomicSub accuracy", "[fpmp]")
{
#if !_CCCL_CUDA_COMPILATION()
  SKIP("atomicAdd/atomicSub on fp64mp2 are device-only");
#else
  if (!device_supports_dd_atomics())
  {
    SKIP("double-double atomics require compute capability >= 9.0 (Hopper)");
  }
  fp_ran_on_device();
  const int num_threads   = 512;
  const int num_blocks    = 4;
  const int total_threads = num_threads * num_blocks;

  // Test 1: add a small value from all threads.
  {
    dfloat* d_res = nullptr;
    REQUIRE_CUDART(cudaMalloc(&d_res, sizeof(dfloat)));
    dfloat h_res(0.0);
    REQUIRE_CUDART(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice));

    const double value_to_add = 0.1;
    test_atomicAdd_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value_to_add);
    REQUIRE_CUDART(cudaGetLastError());
    REQUIRE_CUDART(cudaDeviceSynchronize());
    REQUIRE_CUDART(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost));

    const double result    = static_cast<double>(h_res);
    const double expected  = value_to_add * total_threads;
    const double rel_error = ::cuda::std::fabs(result - expected) / expected;
    ::printf("  add:      expected=%.16e computed=%.16e rel=%.4e\n", expected, result, rel_error);
    REQUIRE(rel_error <= 1e-14);

    REQUIRE_CUDART(cudaFree(d_res));
  }

  // Test 2: add then subtract the same value (start at 100.0).
  {
    dfloat* d_res = nullptr;
    REQUIRE_CUDART(cudaMalloc(&d_res, sizeof(dfloat)));
    dfloat h_res(100.0);
    REQUIRE_CUDART(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice));

    const double value = 0.5;
    test_atomicAdd_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value);
    REQUIRE_CUDART(cudaDeviceSynchronize());
    test_atomicSub_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value);
    REQUIRE_CUDART(cudaDeviceSynchronize());
    REQUIRE_CUDART(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost));

    const double result    = static_cast<double>(h_res);
    const double expected  = 100.0;
    const double rel_error = ::cuda::std::fabs(result - expected) / expected;
    ::printf("  add/sub:  expected=%.16e computed=%.16e rel=%.4e\n", expected, result, rel_error);
    REQUIRE(rel_error <= 1e-14);

    REQUIRE_CUDART(cudaFree(d_res));
  }

  // Test 3: subtract a value from all threads (start at 1000.0).
  {
    dfloat* d_res = nullptr;
    REQUIRE_CUDART(cudaMalloc(&d_res, sizeof(dfloat)));
    dfloat h_res(1000.0);
    REQUIRE_CUDART(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice));

    const double value_to_sub = 0.25;
    test_atomicSub_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value_to_sub);
    REQUIRE_CUDART(cudaGetLastError());
    REQUIRE_CUDART(cudaDeviceSynchronize());
    REQUIRE_CUDART(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost));

    const double result    = static_cast<double>(h_res);
    const double expected  = 1000.0 - (value_to_sub * total_threads);
    const double rel_error = ::cuda::std::fabs(result - expected) / ::cuda::std::fabs(expected);
    ::printf("  sub:      expected=%.16e computed=%.16e rel=%.4e\n", expected, result, rel_error);
    REQUIRE(rel_error <= 1e-14);

    REQUIRE_CUDART(cudaFree(d_res));
  }
#endif // _CCCL_CUDA_COMPILATION()
}
