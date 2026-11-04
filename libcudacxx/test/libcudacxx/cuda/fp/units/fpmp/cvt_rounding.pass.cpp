// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: rounding corners of the double <-> fp32mp2 conversions, per
//  accuracy level.
//
//  fp32mp2_high (exact conversions):
//    fp32mp2 -> double is the correctly rounded value of hi + lo, what the FP64
//    sum (double) hi + (double) lo gives: ties included, and the binade below
//    a power-of-two head included ({1, -2^-53} is 1 - 2^-53).
//    double -> fp32mp2 stays finite for every finite double that rounds to a
//    finite float head: just below the float overflow threshold the head is
//    FLT_MAX and the tail +ulp/2. A double at or beyond the threshold converts
//    to {+-inf, +-0}, the infinities to {+-inf, +-0} and -0 to {-0, -0}. Below
//    the float range the head is the correctly rounded float denormal (what
//    the cast gives) with a zero tail, and the conversion back is exact.
//    None of this uses an FP64 instruction on the integer-only paths.
//
//  fp32mp2 (mid, the default) and fp32mp2_low (fast conversions):
//    fp32mp2 -> double can be one double ulp above the correctly rounded value
//    below a power-of-two head ({1, -2^-53} gives 1), and is exact otherwise.
//    double -> fp32mp2 gives {+-inf, 0} for the infinities and for every
//    double at or beyond the float overflow threshold, never a NaN or an
//    infinite tail; within 8 float ulps of FLT_MAX the tail is flushed
//    ({FLT_MAX, 0} for the doubles just below the threshold), -0 gives
//    {-0, +0}.
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

TEST_HOST_DEVICE_FUNC double ref_to_double(float hi, float lo)
{
  return static_cast<double>(hi) + static_cast<double>(lo);
}

TEST_HOST_DEVICE_FUNC bool same_bits(double a, double b)
{
  return ::cuda::std::bit_cast<::cuda::std::uint64_t>(a) == ::cuda::std::bit_cast<::cuda::std::uint64_t>(b);
}

TEST_HOST_DEVICE_FUNC bool is_pow2(float x)
{
  const auto b = ::cuda::std::bit_cast<::cuda::std::uint32_t>(x);
  return (b & 0x7FFFFFu) == 0u && (b & 0x7F800000u) != 0u;
}

// fp32mp2 -> double: exact for the exact level; for the fast one, exact unless
// the head is a power of two with a tail of the other sign, where one double
// ulp of the lower binade is allowed (the result is on the upper binade's
// grid).
template <typename MP2, bool Exact>
TEST_HOST_DEVICE_FUNC void check_to_double(float hi, float lo)
{
  const double got = static_cast<double>(MP2(hi, lo));
  const double ref = ref_to_double(hi, lo);
  if (Exact || !(is_pow2(hi) && lo != 0.0f && (lo < 0.0f) != (hi < 0.0f)))
  {
    assert(same_bits(got, ref));
  }
  else
  {
    const double diff = got > ref ? got - ref : ref - got;
    assert(diff <= ::cuda::std::ldexp(1.0, ::cuda::std::ilogb(ref) - 52));
  }
}

// fp32mp2 -> double: the reported pair, the binade-below cases, the ties.
template <typename MP2, bool Exact>
TEST_HOST_DEVICE_FUNC void test_to_double_corners()
{
  const float pairs[][2] = {
    {1.0f, -0x1p-53f}, // reported: 1 - 2^-53 exactly (exact level)
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
    check_to_double<MP2, Exact>(p[0], p[1]);
  }
  if constexpr (Exact)
  {
    assert(same_bits(static_cast<double>(MP2(1.0f, -0x1p-53f)), 1.0 - 0x1p-53));
  }
}

// fp32mp2 -> double: a deterministic sweep of canonical pairs, including
// heads that are powers of two with tails of either sign and tails at exactly
// half an ulp.
template <typename MP2, bool Exact>
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
    check_to_double<MP2, Exact>(hi, lo);
  }
}

// double -> fp32mp2 around the float overflow threshold, the infinities, the
// zeros and below the float range.
template <typename MP2, bool Exact>
TEST_HOST_DEVICE_FUNC void test_from_double_corners()
{
  constexpr float flt_max  = ::cuda::std::numeric_limits<float>::max();
  constexpr float inf      = ::cuda::std::numeric_limits<float>::infinity();
  const double just_below  = 3.40282356779733624e+38; // reported: below the threshold 2^128 - 2^103
  const double threshold   = 0x1.fffffffp+127; // 2^128 - 2^103: rounds to even, i.e. to 2^128
  const double way_above   = 3.5e38;
  const double dbl_flt_max = static_cast<double>(flt_max);

  const MP2 a(just_below);
  assert(a.hi() == flt_max);
  assert(a.lo() == (Exact ? 0x1p103f : 0.0f));
  if constexpr (Exact)
  {
    assert(static_cast<double>(a) == just_below || static_cast<double>(a) == dbl_flt_max + 0x1p103);
  }

  const MP2 b(-just_below);
  assert(b.hi() == -flt_max);
  assert(b.lo() == (Exact ? -0x1p103f : 0.0f));

  const MP2 c(threshold);
  assert(c.hi() == inf && c.lo() == 0.0f);

  const MP2 d(way_above);
  assert(d.hi() == inf && d.lo() == 0.0f);

  const MP2 e(-way_above);
  assert(e.hi() == -inf && e.lo() == 0.0f);

  const MP2 f(dbl_flt_max);
  assert(f.hi() == flt_max && f.lo() == 0.0f);

  const MP2 g(static_cast<double>(inf));
  assert(g.hi() == inf && g.lo() == 0.0f);

  const MP2 h(static_cast<double>(-inf));
  assert(h.hi() == -inf && h.lo() == 0.0f);

  const MP2 n(::cuda::std::numeric_limits<double>::quiet_NaN());
  assert(n.hi() != n.hi() && n.lo() == 0.0f);

  const MP2 z(-0.0);
  assert(z.hi() == 0.0f && ::cuda::std::signbit(z.hi()) && z.lo() == 0.0f);
  if constexpr (Exact)
  {
    assert(::cuda::std::signbit(z.lo()));
  }

  // The top binade keeps its tail up to 8 float ulps below the threshold
  // (both levels), the exact level up to the threshold. The inputs are pair
  // representable (a float head plus a float tail), so the round trip is exact.
  const double top[] = {
    0x1.8p+127,
    0x1.fffffcp+127, // a float one ulp below FLT_MAX: a zero tail in both levels
    0x1.ffffp+127 + 0x1p+80,
    -(0x1.8p+127 + 0x1p+95),
    0x1.fffffp+127 - 0x1p+75, // 2^128 - 2^107 - 2^75: the last double before the fast level's flush
  };
  for (double x : top)
  {
    const MP2 q(x);
    assert(q.hi() == static_cast<float>(x));
    assert(static_cast<double>(q.hi()) + static_cast<double>(q.lo()) == x);
  }
  const MP2 cut(0x1.fffffp+127); // 2^128 - 2^107 itself, a float: {x, 0} in both levels
  assert(cut.hi() == 0x1.fffffp+127f && cut.lo() == 0.0f);
  const MP2 fl(0x1.fffffep+127 + 0x1p+80); // within 8 ulps of FLT_MAX: the fast level flushes the tail
  assert(fl.hi() == flt_max && fl.lo() == (Exact ? 0x1p+80f : 0.0f));

  // Below the float range: the head is the cast (a float denormal, rounded to
  // nearest even), the tail 0; converting back gives the head exactly.
  const double tiny[] = {1.0e-40, -1.0e-40, 1.4e-45, 7.0e-46, 2.1e-45, 1.1754942e-38, -3.0e-39, 5.0e-324, 1.0e-310};
  for (double x : tiny)
  {
    const MP2 q(x);
    const float c = static_cast<float>(x);
    assert(q.hi() == c && ::cuda::std::signbit(q.hi()) == ::cuda::std::signbit(c));
    assert(q.lo() == 0.0f);
    if constexpr (Exact)
    {
      assert(::cuda::std::signbit(q.lo()) == ::cuda::std::signbit(c));
    }
    assert(same_bits(static_cast<double>(q), static_cast<double>(c)));
  }

  // Doubles whose tail is a float denormal (2^-126 <= |x| < 2^-98): the exact
  // level gives the value of the cast's pair (head and tail each rounded
  // once) in canonical form; the fast levels may be one 2^-149 off.
  const double small[] = {
    0x1.40a6aaa000000p-121, // the harness's pattern input: tail -10 * 2^-149 exactly
    -0x1.40a6aaa000000p-121,
    0x1.ba636333e7ea7p-103, // tails the hot path rounds twice
    0x1.34e6e94142f7bp-105,
    -0x1.4bf0c4be52502p-106,
    0x1.5263017bdeb56p-102,
    0x1.ffffffp-101, // head rounds up into the next binade
    0x1.000001p-126,
    -0x1.0000018p-126,
    0x1.fffffep-99 + 0x1p-123, // tie at half an ulp
    0x1.0000008p-126,
    0x1.0000008000001p-120,
  };
  for (double x : small)
  {
    const MP2 q(x);
    const float ch = static_cast<float>(x);
    const float cl = static_cast<float>(x - static_cast<double>(ch));
    const double v = static_cast<double>(q.hi()) + static_cast<double>(q.lo());
    if constexpr (Exact)
    {
      // the value of the cast's pair, in canonical form (fl(hi + lo) == hi)
      assert(v == static_cast<double>(ch) + static_cast<double>(cl));
      assert(q.hi() + q.lo() == q.hi());
    }
    else
    {
      // the fast levels round the tail twice, and a tail that underflows to
      // zero no longer renormalizes a head rounded half away: the pair is
      // within one 2^-149 of x
      const double d = v > x ? v - x : x - v;
      assert(d <= 0x1p-149);
    }
  }

  // Pair -> double with a denormal or tiny head (the cold path of both levels).
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
    check_to_double<MP2, Exact>(p[0], p[1]);
  }
}

template <typename MP2, bool Exact>
TEST_HOST_DEVICE_FUNC void test_level()
{
  test_to_double_corners<MP2, Exact>();
  test_to_double_sweep<MP2, Exact>();
  test_from_double_corners<MP2, Exact>();
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_level<cuda::fp32mp2_high, true>();
  test_level<cuda::fp32mp2, false>();
  test_level<cuda::fp32mp2_low, false>();
}

int main(int, char**)
{
  test();

  return 0;
}
