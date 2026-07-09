// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: compile-time double -> fp32mp2 lookup table.
//
//  A constexpr fp32mp2 lookup table is initialized directly from double literals
//  (compile-time conversion, zero runtime overhead). The test loads pairs of
//  entries, multiplies them, and verifies the fp32mp2 products track the double
//  reference within tolerance, plus that each stored entry round-trips to its
//  literal within tolerance. The same _CCCL_HOST_DEVICE run_test() runs on the
//  host and, under CUDA, on the device.
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

using ffloat = fp32mp2;

// Same list builds both an fp32mp2 array (F = LUT_FF) and a double reference
// array (F = LUT_ID).
#define LUT_ID(x)   x
#define LUT_FF(x)   ffloat(x)
#define LUT_LIST(F)                                                                             \
  F(3.14159265358979323846), F(2.71828182845904523536), F(1.41421356237309504880),             \
    F(1.73205080756887729352), F(0.69314718055994530942), F(0.43429448190325182765),           \
    F(1.61803398874989484820), F(0.57721566490153286060), F(299792458.0), F(6.62607015e-34),   \
    F(1.602176634e-19), F(9.10938356e-31), F(1.234567890123456789), F(9.876543210987654321),   \
    F(0.123456789012345678), F(123456.789012345678)

constexpr int LUT_SIZE = 16;

// Multiply pairs of LUT entries and verify against the double reference; also
// check the round-trip precision of every stored entry.
_CCCL_HOST_DEVICE bool run_test()
{
  constexpr ffloat lut[] = {LUT_LIST(LUT_FF)};
  constexpr double ref[] = {LUT_LIST(LUT_ID)};

  const int idx[][2] = {{0, 0}, {0, 1}, {2, 3}, {4, 5}, {6, 7}, {12, 13}, {14, 15}, {0, 2}};
  const int num_ops  = (int) (sizeof(idx) / sizeof(idx[0]));

  const double tol = 1e-12;
  bool ok          = true;

  for (int i = 0; i < num_ops; i++)
  {
    const int a       = idx[i][0];
    const int b       = idx[i][1];
    const double prod = (double) (lut[a] * lut[b]);
    const double r    = ref[a] * ref[b];
    const double rel  = (r != 0.0) ? ::cuda::std::fabs(prod - r) / ::cuda::std::fabs(r) : 0.0;
    ok                = ok && (rel < tol);
  }

  for (int i = 0; i < LUT_SIZE; i++)
  {
    const double rel = (ref[i] != 0.0) ? ::cuda::std::fabs((double) lut[i] - ref[i]) / ::cuda::std::fabs(ref[i]) : 0.0;
    ok               = ok && (rel < tol);
  }

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp compile-time lookup table", "[fpmp]")
{
  fp_ran_on_host();
  REQUIRE(run_test());

#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
