// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: rounding corners of the double <-> fp32mp2 conversions.
//
//  fp32mp2 -> double is the correctly rounded value of hi + lo, what the FP64
//  sum (double) hi + (double) lo gives: ties included, and the binade below a
//  power-of-two head included ({1, -2^-53} is 1 - 2^-53).
//
//  double -> fp32mp2 must stay finite for every finite double that rounds to
//  a finite float head: just below the float overflow threshold the head is
//  FLT_MAX and the tail +ulp/2, and the pair must not be renormalized into
//  {inf, -inf}. A double at or beyond the threshold converts to {+-inf, +-0},
//  the infinities to {+-inf, +-0} (not inf - inf = NaN) and -0 to {-0, -0}.
//  Below the float range the head is the correctly rounded float denormal
//  (what the cast gives) with a zero tail, and the conversion back is exact.
//  None of this uses an FP64 instruction on the integer-only paths.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "test_macros.h"

using ffloat = cuda::fp32mp2;

TEST_HOST_DEVICE_FUNC double ref_to_double(float hi, float lo)
{
  return static_cast<double>(hi) + static_cast<double>(lo);
}

TEST_HOST_DEVICE_FUNC bool same_bits(double a, double b)
{
  return ::cuda::std::bit_cast<::cuda::std::uint64_t>(a) == ::cuda::std::bit_cast<::cuda::std::uint64_t>(b);
}

// fp32mp2 -> double: the reported pair, the binade-below cases, the ties.
TEST_HOST_DEVICE_FUNC void test_to_double_corners()
{
  const float pairs[][2] = {
    {1.0f, -0x1p-53f}, // reported: 1 - 2^-53 exactly
    {1.0f, 0x1p-53f}, // tie: 1
    {1.0f, -0x1p-54f}, // below the lower binade's half ulp: 1
    {1.0f, -0x1p-30f},
    {2.0f, -0x1p-52f},
    {0.5f, -0x1p-54f},
    {1.5f, 0x1p-53f}, // tie in the upper binade: 1.5
    {-1.0f, 0x1p-53f},
    {0x1p+84f, -0x1.f9d176p+42f},
    {0x1p-26f, -0x1.41c644p-80f},
    {3.0f, 0x1p-23f},
    {0x1.fffffep+0f, 0x1p-24f},
  };
  for (const auto& p : pairs)
  {
    assert(same_bits(static_cast<double>(ffloat(p[0], p[1])), ref_to_double(p[0], p[1])));
  }
}

// fp32mp2 -> double: a deterministic sweep of canonical pairs, including
// heads that are powers of two with tails of either sign and tails at exactly
// half an ulp.
TEST_HOST_DEVICE_FUNC void test_to_double_sweep()
{
  ::cuda::std::uint32_t s = 0x9E3779B9u;
  auto next               = [&]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };
  for (int i = 0; i < 20000; ++i)
  {
    const ::cuda::std::uint32_t r = next();
    const int e                   = static_cast<int>(r % 200u) - 100;
    float hi                      = ::cuda::std::ldexp(1.0f + static_cast<float>(next() & 0xFFFFFFu) * 0x1p-24f, e);
    if ((r & 0x300u) == 0u)
    {
      hi = ::cuda::std::ldexp(1.0f, e); // a power of two one time in four
    }
    if (r & 0x400u)
    {
      hi = -hi;
    }
    const float ulp = ::cuda::std::ldexp(1.0f, ::cuda::std::ilogb(hi) - 23);
    float lo;
    switch ((r >> 11) & 3u)
    {
      case 0:
        lo = (static_cast<float>(next() & 0xFFFFFFu) * 0x1p-24f - 0.5f) * ulp;
        break;
      case 1:
        lo = ::cuda::std::ldexp(static_cast<float>(next() & 0xFFFFFFu) * 0x1p-24f - 0.5f,
                                e - 30 - static_cast<int>(next() % 50u));
        break;
      default:
        lo = ((r & 0x2000u) ? -0.5f : 0.5f) * ulp;
        break;
    }
    assert(same_bits(static_cast<double>(ffloat(hi, lo)), ref_to_double(hi, lo)));
  }
}

// double -> fp32mp2 around the float overflow threshold.
TEST_HOST_DEVICE_FUNC void test_from_double_overflow()
{
  constexpr float flt_max  = ::cuda::std::numeric_limits<float>::max();
  constexpr float inf      = ::cuda::std::numeric_limits<float>::infinity();
  const double just_below  = 3.40282356779733624e+38; // reported: below the threshold 2^128 - 2^103
  const double threshold   = 0x1.fffffffp+127; // 2^128 - 2^103: rounds to even, i.e. to 2^128
  const double way_above   = 3.5e38;
  const double dbl_flt_max = static_cast<double>(flt_max);

  const ffloat a(just_below);
  assert(a.hi() == flt_max && a.lo() == 0x1p103f);
  assert(static_cast<double>(a) == just_below || static_cast<double>(a) == dbl_flt_max + 0x1p103);

  const ffloat b(-just_below);
  assert(b.hi() == -flt_max && b.lo() == -0x1p103f);

  const ffloat c(threshold);
  assert(c.hi() == inf && c.lo() == 0.0f);

  const ffloat d(way_above);
  assert(d.hi() == inf && d.lo() == 0.0f);

  const ffloat e(-way_above);
  assert(e.hi() == -inf && e.lo() == 0.0f);

  const ffloat f(dbl_flt_max);
  assert(f.hi() == flt_max && f.lo() == 0.0f);

  const ffloat g(static_cast<double>(inf));
  assert(g.hi() == inf && g.lo() == 0.0f);

  const ffloat h(static_cast<double>(-inf));
  assert(h.hi() == -inf && h.lo() == 0.0f);

  const ffloat z(-0.0);
  assert(z.hi() == 0.0f && ::cuda::std::signbit(z.hi()) && z.lo() == 0.0f && ::cuda::std::signbit(z.lo()));

  // Below the float range: the head is the cast (a float denormal, rounded to
  // nearest even), the tail 0; converting back gives the head exactly.
  const double tiny[] = {1.0e-40, -1.0e-40, 1.4e-45, 7.0e-46, 2.1e-45, 1.1754942e-38, -3.0e-39, 5.0e-324, 1.0e-310};
  for (double d : tiny)
  {
    const ffloat q(d);
    const float c = static_cast<float>(d);
    assert(q.hi() == c && ::cuda::std::signbit(q.hi()) == ::cuda::std::signbit(c));
    assert(q.lo() == 0.0f && ::cuda::std::signbit(q.lo()) == ::cuda::std::signbit(c));
    assert(same_bits(static_cast<double>(q), static_cast<double>(c)));
  }

  // Pair -> double with a denormal or tiny head (the integer-only cold path).
  const float cold[][2] = {
    {0x1p-149f, 0.0f},
    {-0x1p-149f, 0.0f},
    {1.0e-40f, 0.0f},
    {0.0f, 3.0e-39f},
    {0x1p-126f, 0.0f},
    {0x1.fffffcp-127f, 0.0f},
    {1.0e-30f, 1.0e-45f},
    {0x1p-100f, 0x1p-130f},
    {0x1p-75f, 0x1p-100f},
    {0x1.8p-76f, -0x1p-101f}};
  for (const auto& p : cold)
  {
    assert(same_bits(static_cast<double>(ffloat(p[0], p[1])), ref_to_double(p[0], p[1])));
  }
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_to_double_corners();
  test_to_double_sweep();
  test_from_double_overflow();
}

int main(int, char**)
{
  test();

  return 0;
}
