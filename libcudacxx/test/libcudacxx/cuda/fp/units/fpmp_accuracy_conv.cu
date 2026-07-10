// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: cross-accuracy conversion semantics.
//
//  Different fpmp2<FpType, met> specializations share the same (hi, lo)
//  representation; only the accuracy tag differs. fpmp.h provides an explicit
//  cross-accuracy converting constructor that bit-copies (hi, lo). Compile-time
//  static_asserts pin the contract (explicit-only across accuracy, implicit
//  same-type, cross-FpType widening rules, and the assignment side). The
//  _CCCL_HOST_DEVICE run_test() then confirms the conversions are bit-exact at
//  runtime, on the host and, under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <type_traits>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// ---------------------------------------------------------------------------
// Compile-time contract for fp32mp2 (FpType == float).
// ---------------------------------------------------------------------------
// explicit construction across accuracy levels is allowed ...
static_assert(std::is_constructible<fp32mp2_low, fp32mp2_high>::value, "");
static_assert(std::is_constructible<fp32mp2_low, fp32mp2>::value, "");
static_assert(std::is_constructible<fp32mp2, fp32mp2_low>::value, "");
static_assert(std::is_constructible<fp32mp2, fp32mp2_high>::value, "");
static_assert(std::is_constructible<fp32mp2_high, fp32mp2>::value, "");
static_assert(std::is_constructible<fp32mp2_high, fp32mp2_low>::value, "");

// ... but implicit conversion across accuracy levels is NOT.
static_assert(!std::is_convertible<fp32mp2_high, fp32mp2_low>::value, "");
static_assert(!std::is_convertible<fp32mp2, fp32mp2_low>::value, "");
static_assert(!std::is_convertible<fp32mp2_low, fp32mp2>::value, "");
static_assert(!std::is_convertible<fp32mp2_high, fp32mp2>::value, "");
static_assert(!std::is_convertible<fp32mp2, fp32mp2_high>::value, "");
static_assert(!std::is_convertible<fp32mp2_low, fp32mp2_high>::value, "");

// Same-type implicit conversion (copy) is unaffected.
static_assert(std::is_convertible<fp32mp2_low, fp32mp2_low>::value, "");
static_assert(std::is_convertible<fp32mp2, fp32mp2>::value, "");
static_assert(std::is_convertible<fp32mp2_high, fp32mp2_high>::value, "");

// Cross-FpType conversion contract: upconvert implicit, downconvert honors the
// CCCL_FPMP_EXPLICIT_CASTS knob.
static_assert(std::is_convertible<fp32mp2, fp64mp2>::value, "fp32mp2 -> fp64mp2 must be implicit (lossless upconvert)");
#if CCCL_FPMP_EXPLICIT_CASTS == 1
static_assert(!std::is_convertible<fp64mp2, fp32mp2>::value,
              "fp64mp2 -> fp32mp2 must be explicit under EXPLICIT_CASTS=1");
#else
static_assert(std::is_convertible<fp64mp2, fp32mp2>::value,
              "fp64mp2 -> fp32mp2 implicit by default (matches double -> fp32mp2)");
#endif

// Assignment side of the contract: cross-accuracy assignment must fail (the
// explicit ctor is not visible to copy-assignment); same-type assignment works.
static_assert(!std::is_assignable<fp32mp2_low&, fp32mp2_high>::value, "");
static_assert(!std::is_assignable<fp32mp2_low&, fp32mp2>::value, "");
static_assert(!std::is_assignable<fp32mp2&, fp32mp2_low>::value, "");
static_assert(!std::is_assignable<fp32mp2&, fp32mp2_high>::value, "");
static_assert(!std::is_assignable<fp32mp2_high&, fp32mp2>::value, "");
static_assert(!std::is_assignable<fp32mp2_high&, fp32mp2_low>::value, "");
static_assert(std::is_assignable<fp32mp2_low&, fp32mp2_low>::value, "");
static_assert(std::is_assignable<fp32mp2&, fp32mp2>::value, "");
static_assert(std::is_assignable<fp32mp2_high&, fp32mp2_high>::value, "");

// The new ctor preserves result type (no type inference surprises).
static_assert(std::is_same<decltype(fp32mp2_low(std::declval<fp32mp2_high>())), fp32mp2_low>::value, "");

// ---------------------------------------------------------------------------
// Compile-time contract for fp64mp2 (FpType == double).
// ---------------------------------------------------------------------------
static_assert(std::is_constructible<fp64mp2_low, fp64mp2_high>::value, "");
static_assert(std::is_constructible<fp64mp2_high, fp64mp2>::value, "");
static_assert(!std::is_convertible<fp64mp2_high, fp64mp2_low>::value, "");
static_assert(!std::is_convertible<fp64mp2, fp64mp2_low>::value, "");
static_assert(!std::is_assignable<fp64mp2_low&, fp64mp2_high>::value, "");
static_assert(!std::is_assignable<fp64mp2_low&, fp64mp2>::value, "");

// ---------------------------------------------------------------------------
// Runtime bit-equality check.
// ---------------------------------------------------------------------------
template <typename Dst, typename Src>
_CCCL_HOST_DEVICE static bool bit_exact(Src src)
{
  Dst dst(src); // explicit cross-accuracy ctor
  return (dst.hi() == src.hi()) && (dst.lo() == src.lo());
}

_CCCL_HOST_DEVICE bool run_test()
{
  bool ok = true;

  // Representative (hi, lo) pairs: regular, near-max, tiny, negative.
  const float f32[4][2] = {
    {1.2345678f, 1.0e-9f},
    {0x1.fffffep+126f, -0x1.0p+102f},
    {1.0e-30f, 1.0e-38f},
    {-3.1415927f, 1.5e-8f},
  };
  for (int i = 0; i < 4; ++i)
  {
    fp32mp2 sd(f32[i][0], f32[i][1]);
    ok = ok && bit_exact<fp32mp2_low>(sd) && bit_exact<fp32mp2_high>(sd);
    fp32mp2_low sl(f32[i][0], f32[i][1]);
    ok = ok && bit_exact<fp32mp2>(sl) && bit_exact<fp32mp2_high>(sl);
    fp32mp2_high sh(f32[i][0], f32[i][1]);
    ok = ok && bit_exact<fp32mp2>(sh) && bit_exact<fp32mp2_low>(sh);
  }

  const double f64[4][2] = {
    {1.234567890123456, 1.0e-18},
    {1.0e+300, 1.0e+283},
    {-2.7182818284590452, 1.5e-17},
    {1.0e-200, 1.0e-217},
  };
  for (int i = 0; i < 4; ++i)
  {
    fp64mp2_high sh(f64[i][0], f64[i][1]);
    ok = ok && bit_exact<fp64mp2_low>(sh);
    fp64mp2 sd(f64[i][0], f64[i][1]);
    ok = ok && bit_exact<fp64mp2_low>(sd);
  }

  // Every explicit-conversion shape routes through the bit-exact ctor.
  {
    fp32mp2_high src(0x1.23p+4f, 0x1.0p-20f);
    fp32mp2_low a(src); // direct-init
    fp32mp2_low b = fp32mp2_low(src); // functional cast in copy-init
    fp32mp2_low c = static_cast<fp32mp2_low>(src); // static_cast in copy-init
    fp32mp2_low d;
    d = fp32mp2_low(src); // explicit assign (functional)
    fp32mp2_low e;
    e  = static_cast<fp32mp2_low>(src); // explicit assign (static_cast)
    ok = ok && (a.hi() == src.hi() && a.lo() == src.lo()) && (b.hi() == src.hi() && b.lo() == src.lo())
      && (c.hi() == src.hi() && c.lo() == src.lo()) && (d.hi() == src.hi() && d.lo() == src.lo())
      && (e.hi() == src.hi() && e.lo() == src.lo());
  }

  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp cross-accuracy conversion", "[fpmp]")
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
