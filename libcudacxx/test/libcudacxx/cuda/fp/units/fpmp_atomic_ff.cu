// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: atomicAdd / atomicSub on fp32mp2 (float-float).
//
//  Device-only test (multi-block atomics). Two checks:
//    - Atomicity: every thread does atomicAdd(1.0) then atomicSub(1.0); with
//      correct atomics the shared accumulator cancels back to ~0.
//    - Accuracy: many threads accumulate a small value and the result is compared
//      against the analytic sum within a relative tolerance.
//
//  On a host-only build (no CUDA) the test is SKIP()ed; under CUDA it runs on the
//  device.
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

// Type alias for the multi-precision floating-point type.
using ffloat = fp32mp2;

#if _CCCL_CUDA_COMPILATION()
// Each thread adds then subtracts 1.0; the accumulator must cancel to ~0.
__global__ void test_atomicity_kernel(unsigned int* idx, ffloat* res)
{
  ffloat val(1.0f);
  atomicAdd(idx, 1u);
  atomicAdd(res, val);
  atomicSub(res, val);
}

__global__ void test_atomicAdd_accuracy_kernel(ffloat* res, float value_to_add)
{
  ffloat val(value_to_add);
  atomicAdd(res, val);
}

__global__ void test_atomicSub_accuracy_kernel(ffloat* res, float value_to_sub)
{
  ffloat val(value_to_sub);
  atomicSub(res, val);
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp atomicAdd/atomicSub atomicity", "[fpmp]")
{
#if !_CCCL_CUDA_COMPILATION()
  SKIP("atomicAdd/atomicSub on fp32mp2 are device-only");
#else
  fp_ran_on_device();
  const int num_threads = 512;
  const int num_blocks  = 4;

  unsigned int* d_idx = nullptr;
  ffloat* d_res       = nullptr;
  REQUIRE_CUDART(cudaMalloc(&d_idx, sizeof(unsigned int)));
  REQUIRE_CUDART(cudaMalloc(&d_res, sizeof(ffloat)));

  unsigned int h_idx = 0;
  ffloat h_res(0.0f);
  REQUIRE_CUDART(cudaMemcpy(d_idx, &h_idx, sizeof(unsigned int), cudaMemcpyHostToDevice));
  REQUIRE_CUDART(cudaMemcpy(d_res, &h_res, sizeof(ffloat), cudaMemcpyHostToDevice));

  test_atomicity_kernel<<<num_blocks, num_threads>>>(d_idx, d_res);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());

  REQUIRE_CUDART(cudaMemcpy(&h_idx, d_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  REQUIRE_CUDART(cudaMemcpy(&h_res, d_res, sizeof(ffloat), cudaMemcpyDeviceToHost));

  const double result = static_cast<double>(h_res);
  ::printf("  atomicity: threads=%u  result=%.10e\n", h_idx, result);

  // All threads participated and add/sub cancelled to ~0.
  REQUIRE(h_idx == static_cast<unsigned int>(num_threads * num_blocks));
  REQUIRE(::cuda::std::fabs(result) < 1e-6);

  REQUIRE_CUDART(cudaFree(d_idx));
  REQUIRE_CUDART(cudaFree(d_res));
#endif // _CCCL_CUDA_COMPILATION()
}

C2H_TEST("fpmp atomicAdd/atomicSub accuracy", "[fpmp]")
{
#if !_CCCL_CUDA_COMPILATION()
  SKIP("atomicAdd/atomicSub on fp32mp2 are device-only");
#else
  fp_ran_on_device();
  const int num_threads   = 512;
  const int num_blocks    = 4;
  const int total_threads = num_threads * num_blocks;

  // Test 1: add a small value from all threads.
  {
    ffloat* d_res = nullptr;
    REQUIRE_CUDART(cudaMalloc(&d_res, sizeof(ffloat)));
    ffloat h_res(0.0f);
    REQUIRE_CUDART(cudaMemcpy(d_res, &h_res, sizeof(ffloat), cudaMemcpyHostToDevice));

    const float value_to_add = 0.1f;
    test_atomicAdd_accuracy_kernel<<<num_blocks, num_threads>>>(d_res, value_to_add);
    REQUIRE_CUDART(cudaGetLastError());
    REQUIRE_CUDART(cudaDeviceSynchronize());
    REQUIRE_CUDART(cudaMemcpy(&h_res, d_res, sizeof(ffloat), cudaMemcpyDeviceToHost));

    const double result    = static_cast<double>(h_res);
    const double expected  = static_cast<double>(value_to_add) * total_threads;
    const double rel_error = ::cuda::std::fabs(result - expected) / expected;
    ::printf("  add:      expected=%.10e computed=%.10e rel=%.4e\n", expected, result, rel_error);
    REQUIRE(rel_error <= 1e-5);

    REQUIRE_CUDART(cudaFree(d_res));
  }

  // Test 2: add then subtract the same value (start at 100.0).
  {
    ffloat* d_res = nullptr;
    REQUIRE_CUDART(cudaMalloc(&d_res, sizeof(ffloat)));
    ffloat h_res(100.0f);
    REQUIRE_CUDART(cudaMemcpy(d_res, &h_res, sizeof(ffloat), cudaMemcpyHostToDevice));

    const float value = 0.5f;
    test_atomicAdd_accuracy_kernel<<<num_blocks, num_threads>>>(d_res, value);
    REQUIRE_CUDART(cudaDeviceSynchronize());
    test_atomicSub_accuracy_kernel<<<num_blocks, num_threads>>>(d_res, value);
    REQUIRE_CUDART(cudaDeviceSynchronize());
    REQUIRE_CUDART(cudaMemcpy(&h_res, d_res, sizeof(ffloat), cudaMemcpyDeviceToHost));

    const double result    = static_cast<double>(h_res);
    const double expected  = 100.0;
    const double rel_error = ::cuda::std::fabs(result - expected) / expected;
    ::printf("  add/sub:  expected=%.10e computed=%.10e rel=%.4e\n", expected, result, rel_error);
    REQUIRE(rel_error <= 1e-5);

    REQUIRE_CUDART(cudaFree(d_res));
  }

  // Test 3: subtract a value from all threads (start at 1000.0).
  {
    ffloat* d_res = nullptr;
    REQUIRE_CUDART(cudaMalloc(&d_res, sizeof(ffloat)));
    ffloat h_res(1000.0f);
    REQUIRE_CUDART(cudaMemcpy(d_res, &h_res, sizeof(ffloat), cudaMemcpyHostToDevice));

    const float value_to_sub = 0.25f;
    test_atomicSub_accuracy_kernel<<<num_blocks, num_threads>>>(d_res, value_to_sub);
    REQUIRE_CUDART(cudaGetLastError());
    REQUIRE_CUDART(cudaDeviceSynchronize());
    REQUIRE_CUDART(cudaMemcpy(&h_res, d_res, sizeof(ffloat), cudaMemcpyDeviceToHost));

    const double result    = static_cast<double>(h_res);
    const double expected  = 1000.0 - (static_cast<double>(value_to_sub) * total_threads);
    const double rel_error = ::cuda::std::fabs(result - expected) / expected;
    ::printf("  sub:      expected=%.10e computed=%.10e rel=%.4e\n", expected, result, rel_error);
    REQUIRE(rel_error <= 1e-5);

    REQUIRE_CUDART(cudaFree(d_res));
  }
#endif // _CCCL_CUDA_COMPILATION()
}
