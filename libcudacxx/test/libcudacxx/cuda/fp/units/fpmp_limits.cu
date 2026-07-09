// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: cuda::std::numeric_limits<fpmp2> specialization.
//
//  Validates the numeric_limits<> specialization for the double-word types
//  fp32mp2 (double-float) and fp64mp2 (double-double). Compile-time static_asserts
//  cover every reported characteristic (integer traits and the exact power-of-two
//  hi/lo components of min()/max()/lowest()/epsilon(), plus cv-qualified and
//  accuracy-variant forwarding). The _CCCL_HOST_DEVICE run_test() then exercises
//  the value members at runtime on the host and, under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

namespace cs = cuda::std;

template <class T>
using nl = cs::numeric_limits<T>;

//==========================================================================================
// Compile-time checks (evaluated for both host and device builds)
//==========================================================================================

// ----- fp32mp2 (double-float) -----
static_assert(nl<fp32mp2>::is_specialized, "fp32mp2 must be specialized");
static_assert(nl<fp32mp2>::is_signed, "fp32mp2 is signed");
static_assert(!nl<fp32mp2>::is_integer, "fp32mp2 is not integer");
static_assert(!nl<fp32mp2>::is_exact, "fp32mp2 is not exact");
static_assert(nl<fp32mp2>::radix == 2, "fp32mp2 radix is 2");
static_assert(nl<fp32mp2>::digits == 46, "fp32mp2 has 2*24-2 = 46 mantissa bits");
static_assert(nl<fp32mp2>::digits10 == 13, "fp32mp2 digits10");
static_assert(nl<fp32mp2>::max_digits10 == 15, "fp32mp2 max_digits10");
static_assert(nl<fp32mp2>::max_exponent == nl<float>::max_exponent, "fp32mp2 shares float's max exponent");
static_assert(nl<fp32mp2>::min_exponent == nl<float>::min_exponent + nl<float>::digits, "fp32mp2 min exponent");
static_assert(!nl<fp32mp2>::is_iec559, "double-word is not an IEEE-754 format");
static_assert(nl<fp32mp2>::is_bounded, "fp32mp2 is bounded");
static_assert(nl<fp32mp2>::has_infinity, "fp32mp2 has infinity");
static_assert(nl<fp32mp2>::has_quiet_NaN, "fp32mp2 has quiet NaN");
static_assert(nl<fp32mp2>::round_style == cs::round_to_nearest, "fp32mp2 rounds to nearest");

// exact (hi, lo) constants -- all powers of two, so equality is exact.
static_assert(nl<fp32mp2>::epsilon().hi() == 0x1p-45f, "fp32mp2 epsilon = 2^(1-46) = 2^-45");
static_assert(nl<fp32mp2>::epsilon().lo() == 0.0f, "fp32mp2 epsilon lo is zero");
static_assert(nl<fp32mp2>::max().hi() == nl<float>::max(), "fp32mp2 max hi = FLT_MAX");
static_assert(nl<fp32mp2>::max().lo() == nl<float>::max() * 0x1p-25f, "fp32mp2 max lo");
static_assert(nl<fp32mp2>::min().hi() == 0x1p-102f, "fp32mp2 min hi = 2^-102 (smallest all-normal)");
static_assert(nl<fp32mp2>::lowest().hi() == -nl<float>::max(), "fp32mp2 lowest = -max");
static_assert(nl<fp32mp2>::round_error().hi() == 0.5f, "fp32mp2 round_error = 0.5");

// ----- fp64mp2 (double-double) -----
static_assert(nl<fp64mp2>::is_specialized, "fp64mp2 must be specialized");
static_assert(nl<fp64mp2>::is_signed, "fp64mp2 is signed");
static_assert(nl<fp64mp2>::radix == 2, "fp64mp2 radix is 2");
static_assert(nl<fp64mp2>::digits == 104, "fp64mp2 has 2*53-2 = 104 mantissa bits");
static_assert(nl<fp64mp2>::digits10 == 31, "fp64mp2 digits10");
static_assert(nl<fp64mp2>::max_digits10 == 33, "fp64mp2 max_digits10");
static_assert(nl<fp64mp2>::max_exponent == nl<double>::max_exponent, "fp64mp2 shares double's max exponent");
static_assert(nl<fp64mp2>::min_exponent == nl<double>::min_exponent + nl<double>::digits, "fp64mp2 min exponent");
static_assert(nl<fp64mp2>::min_exponent == -968, "fp64mp2 min exponent matches __ibm128 (-968)");
static_assert(!nl<fp64mp2>::is_iec559, "double-word is not an IEEE-754 format");
static_assert(nl<fp64mp2>::has_infinity, "fp64mp2 has infinity");
static_assert(nl<fp64mp2>::has_quiet_NaN, "fp64mp2 has quiet NaN");

// exact (hi, lo) constants.
static_assert(nl<fp64mp2>::epsilon().hi() == 0x1p-103, "fp64mp2 epsilon = 2^(1-104) = 2^-103");
static_assert(nl<fp64mp2>::max().hi() == nl<double>::max(), "fp64mp2 max hi = DBL_MAX");
static_assert(nl<fp64mp2>::max().lo() == nl<double>::max() * 0x1p-54, "fp64mp2 max lo");
static_assert(nl<fp64mp2>::min().hi() == 0x1p-969, "fp64mp2 min hi = 2^-969 (matches __ibm128)");
static_assert(nl<fp64mp2>::lowest().hi() == -nl<double>::max(), "fp64mp2 lowest = -max");

// ----- accuracy variants and cv-qualified forwarding -----
static_assert(nl<fp32mp2_low>::digits == 46, "low variant is specialized");
static_assert(nl<fp32mp2_high>::is_specialized, "high variant is specialized");
static_assert(nl<fp64mp2_low>::digits == 104, "low variant is specialized");
static_assert(nl<fp64mp2_high>::max_exponent == nl<double>::max_exponent, "high variant is specialized");
static_assert(nl<const fp32mp2>::digits == 46, "const-qualified forwards to the specialization");
static_assert(nl<volatile fp64mp2>::digits == 104, "volatile-qualified forwards to the specialization");

// the value members are usable in a constexpr context (they use the constexpr (hi, lo) ctor).
static constexpr fp32mp2 kEps32 = nl<fp32mp2>::epsilon();
static constexpr fp64mp2 kMax64 = nl<fp64mp2>::max();
static_assert(kEps32.hi() > 0.0f && kMax64.hi() > 0.0, "constexpr value members");

//==========================================================================================
// Runtime checks (host + device)
//==========================================================================================

// (1 + epsilon) != 1, infinity() > max(), quiet_NaN() != quiet_NaN(), lowest() < 0.
_CCCL_HOST_DEVICE bool run_test()
{
  bool ok = true;

  const fp32mp2 one32(1.0f);
  ok = ok && ((one32 + nl<fp32mp2>::epsilon()) != one32);

  const fp64mp2 one64(1.0);
  ok = ok && ((one64 + nl<fp64mp2>::epsilon()) != one64);

  ok = ok && ((double) nl<fp32mp2>::infinity() > (double) nl<fp32mp2>::max());

  const double nan64 = (double) nl<fp64mp2>::quiet_NaN();
  ok                 = ok && (nan64 != nan64); // NaN compares unequal to itself

  ok = ok && ((double) nl<fp64mp2>::lowest() < 0.0);

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp numeric_limits specialization", "[fpmp]")
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
