// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: float <-> double emulated conversions (fp64emu <-> float).
//
//  Validates that fpemu float->double (widening, exact) and double->float
//  (narrowing, round-to-nearest-even) conversions produce bit-identical results
//  to native casts across all value classes: normals, subnormals, +/-0, Inf, NaN
//  and rounding-boundary cases.
//
//  NaN payloads: when the packed API is routed through the unpacked cores
//  (_CCCL_FPEMU_PACKED_VIA_UNPACKED == 1) the unpack/pack round-trip
//  canonicalizes NaNs, so a NaN-vs-NaN bit difference is tolerated in that mode
//  only; otherwise the conversion must be strictly bit-exact.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpemu>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_HAS_FLOAT128()
// __float128 -> double is a lossy narrowing (and otherwise makes construction
// ambiguous with the float/double ctors), so quad construction is deliberately
// deleted for the single-double emulated types, mirroring the deleted 128-bit
// integer ctors. (fp64mp2's double-double CAN hold a quad and keeps its ctor.)
static_assert(!cuda::std::is_constructible_v<fpemu<double>, __float128>);
static_assert(!cuda::std::is_constructible_v<fpemu_unpacked<double>, __float128>);
#endif // _CCCL_HAS_FLOAT128()

// Bit-reinterpret helpers (host + device via cuda::std::bit_cast).
_CCCL_HOST_DEVICE uint64_t d_bits(double d)
{
  return cuda::std::bit_cast<uint64_t>(d);
}
_CCCL_HOST_DEVICE uint32_t f_bits(float f)
{
  return cuda::std::bit_cast<uint32_t>(f);
}
_CCCL_HOST_DEVICE double from_d_bits(uint64_t b)
{
  return cuda::std::bit_cast<double>(b);
}
_CCCL_HOST_DEVICE float from_f_bits(uint32_t b)
{
  return cuda::std::bit_cast<float>(b);
}

_CCCL_HOST_DEVICE bool is_nan_d(uint64_t b)
{
  return ((b >> 52) & 0x7FF) == 0x7FF && (b & 0x000FFFFFFFFFFFFFull) != 0;
}
_CCCL_HOST_DEVICE bool is_nan_f(uint32_t b)
{
  return ((b >> 23) & 0xFF) == 0xFF && (b & 0x007FFFFFu) != 0;
}

_CCCL_HOST_DEVICE constexpr bool relax_nan_payload()
{
#if (_CCCL_FPEMU_PACKED_VIA_UNPACKED == 1)
  return true;
#else
  return false;
#endif
}

// float -> fp64emu -> double (widening, exact).
_CCCL_HOST_DEVICE bool f2d_ok(float v)
{
  fp64emu e(v);
  const uint64_t be = d_bits((double) e);
  const uint64_t br = d_bits((double) v);
  if (be == br)
  {
    return true;
  }
  return relax_nan_payload() && is_nan_d(be) && is_nan_d(br);
}

// double -> fp64emu -> float (narrowing, round-to-nearest-even).
_CCCL_HOST_DEVICE bool d2f_ok(double v)
{
  fp64emu e(v);
  const uint32_t be = f_bits((float) e);
  const uint32_t br = f_bits((float) v);
  if (be == br)
  {
    return true;
  }
  return relax_nan_payload() && is_nan_f(be) && is_nan_f(br);
}

// float -> double widening over every value class (incl. subnormals / NaN / Inf).
TEST_FUNC void test_f2d()
{
  const float vals[] = {
    0.0f,
    -0.0f,
    1.0f,
    -1.0f,
    0.5f,
    -0.5f,
    2.0f,
    -2.0f,
    from_f_bits(0x7F800000u), // +Inf
    from_f_bits(0xFF800000u), // -Inf
    from_f_bits(0x7FC00000u), // +qNaN
    from_f_bits(0xFFC00000u), // -qNaN
    from_f_bits(0x7F800001u), // +sNaN
    from_f_bits(0x7FC0DEADu), // qNaN with payload
    from_f_bits(0x00800000u), // min normal float
    from_f_bits(0x7F7FFFFFu), // max finite float
    from_f_bits(0xFF7FFFFFu), // -max finite float
    from_f_bits(0x00000001u), // min positive subnormal
    from_f_bits(0x80000001u), // -min positive subnormal
    from_f_bits(0x007FFFFFu), // max subnormal
    from_f_bits(0x00400000u), // subnormal with 1 bit
    from_f_bits(0x00000100u), // small subnormal
    3.14159265f,
    1.0e38f,
    -1.0e38f,
    1.0e-38f,
    -1.0e-38f,
    1.0e-45f, // near min subnormal
  };
  for (const float v : vals)
  {
    assert(f2d_ok(v));
  }
}

// double -> float narrowing (round-to-nearest-even) over every value class.
TEST_FUNC void test_d2f()
{
  const double vals[] = {
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    from_d_bits(0x7FF0000000000000ULL), // +Inf
    from_d_bits(0xFFF0000000000000ULL), // -Inf
    from_d_bits(0x7FF8000000000000ULL), // +qNaN
    from_d_bits(0xFFF8000000000000ULL), // -qNaN
    from_d_bits(0x7FF0000000000001ULL), // +sNaN
    from_d_bits(0x7FF80000DEADBEEFull), // qNaN with payload
    (double) from_f_bits(0x7F7FFFFFu), // max finite float as double
    -(double) from_f_bits(0x7F7FFFFFu), // -max finite float as double
    (double) from_f_bits(0x00800000u), // min normal float as double
    from_d_bits(0x3800000000000000ULL), // 2^(-127) -> subnormal float
    from_d_bits(0x3690000000000000ULL), // 2^(-150)
    from_d_bits(0x36A0000000000000ULL), // 2^(-149) = min subnormal float
    from_d_bits(0x380FFFFFFFFFE000ULL), // max subnormal float
    1.0e39,
    -1.0e39, // overflow to float Inf
    3.5e38,
    -3.5e38,
    1.0e-300,
    -1.0e-300, // near-zero underflow
    5.0e-324,
    0.25,
    0.125,
    0.0625, // exact conversions
    100.0,
    -100.0,
    3.14159265358979,
  };
  for (const double v : vals)
  {
    assert(d2f_ok(v));
  }
}

int main(int, char**)
{
  test_f2d();
  test_d2f();

  return 0;
}
