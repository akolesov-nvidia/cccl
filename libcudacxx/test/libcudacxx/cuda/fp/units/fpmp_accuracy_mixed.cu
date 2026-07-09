// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: mixed-type accuracy-explicit overloads.
//
//  Companion to fpmp_accuracy. Tests the mixed-type overloads of add<m>, sub<m>,
//  mul<m>, div<m>, fma<m>, mad<m> that accept any mix of fpmp2 and built-in
//  arithmetic operands (at least one fpmp2). Each mixed-type call must be
//  bit-identical to the strict all-fpmp2 call where the scalar(s) are wrapped in
//  the participating type; result-type preservation is pinned by static_asserts.
//  The same _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on the
//  device.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <type_traits>
#include <utility>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Result type must match the participating fpmp2 type, regardless of operand
// order or which argument is the scalar.
static_assert(std::is_same<decltype(add<fpmp2_accuracy::high>(std::declval<fp32mp2>(), 1.0f)), fp32mp2>::value,
              "add(mp2_def, float) must return mp2_def");
static_assert(std::is_same<decltype(sub<fpmp2_accuracy::low>(1.0f, std::declval<fp32mp2_low>())), fp32mp2_low>::value,
              "sub(float, mp2_low) must return mp2_low");
static_assert(std::is_same<decltype(mul<fpmp2_accuracy::high>(std::declval<fp32mp2_high>(), 2)), fp32mp2_high>::value,
              "mul(mp2_high, int) must return mp2_high");
static_assert(std::is_same<decltype(fma<fpmp2_accuracy::low>(1.0f, std::declval<fp32mp2_low>(), 2.0f)), fp32mp2_low>::value,
              "fma(float, mp2_low, float) must return mp2_low");
static_assert(std::is_same<decltype(mad<fpmp2_accuracy::def>(1.0f, 2.0f, std::declval<fp32mp2>())), fp32mp2>::value,
              "mad(float, float, mp2_def) must return mp2_def");

// Each mixed-type call must equal the strict form (scalars wrapped in the fpmp2
// type) bit-for-bit.
_CCCL_HOST_DEVICE bool run_test()
{
  using ff = fp32mp2_low;
  ff a(1.234567890f), b(2.345678901f);
  const float s = 0.5f;
  const float t = 3.0f;

  bool ok = true;

  // Binary, both argument orders.
  ok = ok && ((double) add<fpmp2_accuracy::high>(a, s) == (double) add<fpmp2_accuracy::high>(a, ff(s)));
  ok = ok && ((double) add<fpmp2_accuracy::high>(s, a) == (double) add<fpmp2_accuracy::high>(ff(s), a));
  ok = ok && ((double) sub<fpmp2_accuracy::high>(a, s) == (double) sub<fpmp2_accuracy::high>(a, ff(s)));
  ok = ok && ((double) sub<fpmp2_accuracy::high>(s, a) == (double) sub<fpmp2_accuracy::high>(ff(s), a));
  ok = ok && ((double) mul<fpmp2_accuracy::low>(a, s) == (double) mul<fpmp2_accuracy::low>(a, ff(s)));
  ok = ok && ((double) mul<fpmp2_accuracy::low>(s, a) == (double) mul<fpmp2_accuracy::low>(ff(s), a));
  ok = ok && ((double) div<fpmp2_accuracy::def>(a, s) == (double) div<fpmp2_accuracy::def>(a, ff(s)));
  ok = ok && ((double) div<fpmp2_accuracy::def>(s, a) == (double) div<fpmp2_accuracy::def>(ff(s), a));

  // Ternary fma: scalar in every position.
  ok = ok && ((double) fma<fpmp2_accuracy::high>(a, s, t) == (double) fma<fpmp2_accuracy::high>(a, ff(s), ff(t)));
  ok = ok && ((double) fma<fpmp2_accuracy::high>(s, a, t) == (double) fma<fpmp2_accuracy::high>(ff(s), a, ff(t)));
  ok = ok && ((double) fma<fpmp2_accuracy::high>(s, t, a) == (double) fma<fpmp2_accuracy::high>(ff(s), ff(t), a));

  // Ternary mad: one scalar, two fpmp2 operands.
  ok = ok && ((double) mad<fpmp2_accuracy::low>(a, b, s) == (double) mad<fpmp2_accuracy::low>(a, b, ff(s)));
  ok = ok && ((double) mad<fpmp2_accuracy::low>(a, s, b) == (double) mad<fpmp2_accuracy::low>(a, ff(s), b));
  ok = ok && ((double) mad<fpmp2_accuracy::low>(s, a, b) == (double) mad<fpmp2_accuracy::low>(ff(s), a, b));

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp mixed-type accuracy-explicit overloads", "[fpmp]")
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
