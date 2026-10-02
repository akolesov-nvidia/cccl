// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp64emu operand alignment for large exponent gaps.
//
//  add / sub / fma align the smaller operand by the exponent difference, which
//  can be far wider than the 64-bit (add) or 128-bit (fma) significand. Every
//  gap from 1 to 1022 is swept with both operand orders, so an alignment shift
//  that wraps modulo the register width (instead of saturating) shows up as a
//  grossly wrong result. high must match native double exactly; mid is allowed
//  a few ulp.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

TEST_HOST_DEVICE_FUNC void check(double got, double ref, bool exact)
{
  if (exact)
  {
    assert(got == ref);
  }
  else
  {
    assert(cuda::std::fabs(got - ref) <= cuda::std::ldexp(cuda::std::fabs(ref), -50));
  }
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_gap(double x, double y, bool exact)
{
  T ex(x);
  T ey(y);
  T one(1.0);
  check(static_cast<double>(ex + ey), x + y, exact);
  check(static_cast<double>(ey + ex), y + x, exact);
  check(static_cast<double>(ex - ey), x - y, exact);
  check(static_cast<double>(ey - ex), y - x, exact);
  // Addend much smaller than the product, then product much smaller than the addend.
  check(static_cast<double>(cudax::fma(ex, one, ey)), cuda::std::fma(x, 1.0, y), exact);
  check(static_cast<double>(cudax::fma(ey, one, ex)), cuda::std::fma(y, 1.0, x), exact);
}

template <class T>
TEST_HOST_DEVICE_FUNC void test(bool exact)
{
  for (int gap = 1; gap <= 1022; ++gap)
  {
    test_gap<T>(1.0, cuda::std::ldexp(1.5, -gap), exact);
    test_gap<T>(-1.75, cuda::std::ldexp(1.25, -gap), exact);
  }
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_subnormal()
{
  // Gaps reaching past the bottom of the normal range (operand exponent field 0).
  for (int gap = 1022; gap <= 1074; ++gap)
  {
    test_gap<T>(1.0, cuda::std::ldexp(1.0, -gap), true);
    test_gap<T>(cuda::std::ldexp(1.0, 1000), cuda::std::ldexp(1.0, -gap), true);
  }
}

int main(int, char**)
{
  test<cudax::fp64emu_high>(true);
  test<cudax::fp64emu_unpacked_high>(true);
  test<cudax::fp64emu_mid>(false);
  test<cudax::fp64emu_unpacked_mid>(false);

  test_subnormal<cudax::fp64emu_high>();
  test_subnormal<cudax::fp64emu_unpacked_high>();

  return 0;
}
