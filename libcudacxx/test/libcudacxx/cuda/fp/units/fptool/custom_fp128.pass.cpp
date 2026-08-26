// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fptool's fp_custom over a binary128 base, spelled fp128_custom.
//
//  custom.pass.cpp and custom_fp32.pass.cpp cover the same type over `double`
//  and over `float`. This file covers what the widest base type adds: 15
//  exponent and 112 mantissa bits leave headroom above every format the other
//  two can express, which makes binary64 and binary32 themselves emulable
//  rather than merely native.
//
//  The central section checks that exactly. Rounding a result twice - once to
//  the base type and once to the requested format - lands where rounding it
//  directly would for +, -, *, / and sqrt, provided the intermediate carries
//  2p + 2 bits of the target's p; binary128's 113 clear that for both
//  binary64's 53 and binary32's 24. So fp128_custom<11, 52> reproduces `double`
//  bit for bit and fp128_custom<8, 23> reproduces `float`, throughout the
//  normal range. Subnormals are the exception, the exponent reduction having no
//  encoding for them, so the values below stay clear of that range.
//
//  Everything here is host-side: sqrt and fma over this base type reach a host
//  backend, and its arithmetic is device-callable only where
//  _CCCL_FP_CUSTOM_FP128_DEVICE_OPS says so.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fptool>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Where the platform offers no binary128 base type this file is an empty program. NVRTC is
// excluded as well: it compiles device code only, and there is no host side here to run.
#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1) && !_CCCL_COMPILER(NVRTC)

// The base type itself, whichever 128-bit type the platform provided.
using quad = cudax::__fp_custom_fp128;

// The native binary128 format, where both reductions compile out entirely.
using fp128_native = cudax::fp128_custom<>;
// binary64 and binary32 emulated over it, which is what this base type is for.
using fp128_fp64 = cudax::fp128_custom<11, 52>;
using fp128_fp32 = cudax::fp128_custom<8, 23>;
// A mantissa between the two, wider than any narrower base could hold.
using fp128_mant64 = cudax::fp128_custom<15, 64>;
// binary64's exponent range over the full mantissa, so only the exponent is reduced.
using fp128_exp11 = cudax::fp128_custom<11, 112>;

// A double lifted into the base type, which is exact. Spelled out as a function because a
// binary128 literal needs a q suffix that not every supported configuration accepts.
quad to_quad(double d)
{
  return static_cast<quad>(d);
}

// === the native format is a drop-in for binary128 ===
// Every operation is the base type's own, so the results are exact rather than close.
bool test_native_matches_quad()
{
  bool ok = true;

  const quad a = to_quad(1.0) / to_quad(3.0);
  const quad b = to_quad(1.0) / to_quad(7.0);
  const quad c = to_quad(0.5);

  const fp128_native ra = a, rb = b, rc = c; // implicit: the native format holds every binary128

  ok = ok && static_cast<quad>(ra + rb) == a + b;
  ok = ok && static_cast<quad>(ra - rb) == a - b;
  ok = ok && static_cast<quad>(ra * rb) == a * b;
  ok = ok && static_cast<quad>(ra / rb) == a / b;
  ok = ok && static_cast<quad>(-ra) == -a;

  // Square roots and products that every binary format represents exactly, which pins the two
  // math functions down without a binary128 reference to compare against.
  ok = ok && static_cast<quad>(sqrt(fp128_native{to_quad(2.25)})) == to_quad(1.5);
  ok = ok && static_cast<quad>(fma(fp128_native{to_quad(2.0)}, fp128_native{to_quad(3.0)}, rc)) == to_quad(6.5);

  // The mantissa is the point of this base type: a term 100 binades down survives here, and
  // is gone in binary64, whose 52 bits reach 52.
  const quad tiny = to_quad(0x1p-100);
  ok              = ok && static_cast<quad>(fp128_native{to_quad(1.0)} + fp128_native{tiny}) == to_quad(1.0) + tiny;
  ok              = ok && 1.0 + 0x1p-100 == 1.0;

  // Compound assignment and inc/dec reach the same operators.
  {
    fp128_native x = to_quad(1.0);
    x += ra;
    x -= rc;
    x *= rb;
    x /= fp128_native{to_quad(2.0)};
    ++x;
    --x;

    quad ref = to_quad(1.0);
    ref += a;
    ref -= c;
    ref *= b;
    ref /= to_quad(2.0);
    ref += to_quad(1.0);
    ref -= to_quad(1.0);
    ok = ok && static_cast<quad>(x) == ref;
  }

  return ok;
}

// === binary64 reproduced over binary128 ===
// The claim from the header comment, checked value by value: the two roundings compose into
// the one binary64 would have performed alone.
bool test_reproduces_double()
{
  // Nothing here reaches the subnormal range, which the exponent reduction flushes to zero
  // while binary64 encodes it: the smallest product is 1e-200, and binary64 stays normal down
  // to 1e-308.
  const double values[] = {
    1.0, 0.5, 1.0 / 3.0, 0.1, 1.234567890123456, 12345.678, -7.0 / 9.0, 0x1p-30, 1e100, 1e-100, 3.0};

  bool ok = true;

  for (const double x : values)
  {
    for (const double y : values)
    {
      const fp128_fp64 rx = x, ry = y; // implicit: 11 and 52 bits hold every double

      ok = ok && static_cast<double>(rx + ry) == x + y;
      ok = ok && static_cast<double>(rx - ry) == x - y;
      ok = ok && static_cast<double>(rx * ry) == x * y;
      ok = ok && static_cast<double>(rx / ry) == x / y;
    }

    if (x > 0.0)
    {
      ok = ok && static_cast<double>(sqrt(fp128_fp64{x})) == ::cuda::std::sqrt(x);
    }
  }

  // Outside the reduced exponent range the clamping and binary64's own overflow agree.
  ok = ok && ::cuda::std::isinf(static_cast<double>(fp128_fp64{1e300} * fp128_fp64{1e300}));
  ok = ok && ::cuda::std::isinf(1e300 * 1e300);

  return ok;
}

// === binary32 reproduced over binary128 ===
bool test_reproduces_float()
{
  // The smallest product here is 1e-20, and binary32 stays normal down to 1e-38.
  const float values[] = {1.0f, 0.5f, 1.0f / 3.0f, 0.1f, 1.1234567f, 12345.678f, -7.0f / 9.0f, 0x1p-30f, 1e10f, 1e-10f};

  bool ok = true;

  for (const float x : values)
  {
    for (const float y : values)
    {
      const fp128_fp32 rx = x, ry = y; // implicit: 8 and 23 bits hold every float

      ok = ok && static_cast<float>(rx + ry) == x + y;
      ok = ok && static_cast<float>(rx - ry) == x - y;
      ok = ok && static_cast<float>(rx * ry) == x * y;
      ok = ok && static_cast<float>(rx / ry) == x / y;
    }

    if (x > 0.0f)
    {
      ok = ok && static_cast<float>(sqrt(fp128_fp32{x})) == ::cuda::std::sqrt(x);
    }
  }

  return ok;
}

// === mantissa reduction between the two ===
// A 64-bit mantissa, which no narrower base type could request. The reduction applies to the
// operands as well as the result, so adding zero is enough to observe it.
bool test_mantissa_reduction()
{
  const fp128_mant64 zero{to_quad(0.0)};

  bool ok = true;

  // The 64th mantissa bit is the last one kept, so a term exactly there survives.
  const quad kept = to_quad(1.0) + to_quad(0x1p-64);
  ok              = ok && static_cast<quad>(fp128_mant64{kept} + zero) == kept;

  // One bit below it the discarded part is exactly half an ulp and the kept mantissa is even,
  // so round-to-nearest-even rounds down.
  ok = ok && static_cast<quad>(fp128_mant64{to_quad(1.0) + to_quad(0x1p-65)} + zero) == to_quad(1.0);

  // Half an ulp again, but now over an odd mantissa, so the same rule rounds up.
  const quad odd_tie = to_quad(1.0) + to_quad(0x1p-64) + to_quad(0x1p-65);
  ok                 = ok && static_cast<quad>(fp128_mant64{odd_tie} + zero) == to_quad(1.0) + to_quad(0x1p-63);

  // The native format keeps every one of them, the base type reaching 112 bits.
  ok = ok && static_cast<quad>(fp128_native{odd_tie} + fp128_native{to_quad(0.0)}) == odd_tie;

  return ok;
}

// === exponent reduction ===
// binary64's 11 bits over the full binary128 mantissa, so the range clamps while the precision
// stays where it is.
bool test_exponent_reduction()
{
  bool ok = true;

  // Above the reduced range, and the sign survives the clamp.
  const quad huge   = to_quad(1e300);
  const double over = static_cast<double>(fp128_exp11{huge} * fp128_exp11{huge});
  ok                = ok && ::cuda::std::isinf(over) && over > 0.0;
  const double under_sign = static_cast<double>(fp128_exp11{huge} * fp128_exp11{-huge});
  ok                      = ok && ::cuda::std::isinf(under_sign) && under_sign < 0.0;

  // Below the reduced range, where the base type itself is nowhere near its own limit.
  const quad tiny = to_quad(1e-300);
  ok              = ok && static_cast<double>(fp128_exp11{tiny} * fp128_exp11{tiny}) == 0.0;
  ok              = ok && static_cast<quad>(fp128_native{tiny} * fp128_native{tiny}) != to_quad(0.0);

  // The mantissa is untouched, so a term 100 binades down still survives.
  const quad near_one = to_quad(1.0) + to_quad(0x1p-100);
  ok = ok && static_cast<quad>(fp128_exp11{near_one} + fp128_exp11{to_quad(0.0)}) == near_one;

  return ok;
}

// === the type surface ===
static_assert(sizeof(fp128_native) == 16, "");
static_assert(sizeof(fp128_fp64) == sizeof(fp128_native), "the requested sizes cost no storage");
static_assert(::cuda::std::is_trivially_copyable_v<fp128_native>, "");
static_assert(::cuda::std::is_trivially_copyable_v<fp128_fp64>, "");

// The base type is part of the type, so binary64's format over binary128 and the same format
// over `double` are two distinct types, which no conversion between them is offered for.
static_assert(!::cuda::std::is_same_v<fp128_fp64, cudax::fp64_custom<>>, "");
static_assert(!::cuda::std::is_same_v<fp128_fp32, cudax::fp32_custom<>>, "");

// === conversions out ===
// operator double() is neither a template nor explicit, so it is what reaches a sink of any
// arithmetic type - and over this base type it rounds, binary64 not holding the format.
TEST_HOST_DEVICE_FUNC int pick(float)
{
  return 1;
}
TEST_HOST_DEVICE_FUNC int pick(double)
{
  return 2;
}

template <class _Tp, class = void>
struct picks_ambiguously : ::cuda::std::true_type
{};
template <class _Tp>
struct picks_ambiguously<_Tp, decltype(void(pick(::cuda::std::declval<_Tp>())))> : ::cuda::std::false_type
{};

// Wider than binary32 in the exponent, so float stays explicit and double is the only way in.
static_assert(!picks_ambiguously<fp128_native>::value, "");
static_assert(!picks_ambiguously<fp128_fp64>::value, "");
// Within binary32 on both axes, so float and double are offered on equal terms.
static_assert(picks_ambiguously<fp128_fp32>::value, "");

static_assert(::cuda::std::is_convertible_v<fp128_native, double>, "");
static_assert(::cuda::std::is_convertible_v<fp128_native, float>, "");

// The explicit binary128 conversion decides which function a binary128 sink picks, not whether
// it can be reached: operator double() plus the standard conversion is a path out of any
// format, as it is to a float. So the cast is what makes the way out exact, not what makes it
// possible; test_conversions checks that difference on a value.
static_assert(::cuda::std::is_convertible_v<fp128_native, quad>, "");
static_assert(::cuda::std::is_constructible_v<quad, fp128_native>, "");
// Every base type offers the binary128 conversion, not just this one.
static_assert(::cuda::std::is_constructible_v<quad, cudax::fp64_custom<>>, "");
static_assert(::cuda::std::is_constructible_v<quad, cudax::fp32_custom<>>, "");

// === conversions in ===
// The same rank rule as over the narrower bases: a source is implicit where the requested
// format holds it in both fields. Only the native format holds a binary128, while a double and
// a float are implicit into more formats here than over their own bases.
static_assert(::cuda::std::is_convertible_v<quad, fp128_native>, "");
static_assert(::cuda::std::is_convertible_v<double, fp128_native>, "");
static_assert(::cuda::std::is_convertible_v<float, fp128_native>, "");
static_assert(::cuda::std::is_convertible_v<int, fp128_native>, "");
static_assert(::cuda::std::is_convertible_v<double, fp128_fp64>, "");
static_assert(::cuda::std::is_convertible_v<float, fp128_fp32>, "");

#  if CCCL_FP_CUSTOM_EXPLICIT_CASTS == 1

// Narrower than binary128, so a binary128 is a cast even into a format that holds every
// double.
static_assert(!::cuda::std::is_convertible_v<quad, fp128_fp64>, "");
static_assert(!::cuda::std::is_convertible_v<quad, fp128_exp11>, "");
static_assert(!::cuda::std::is_convertible_v<quad, fp128_mant64>, "");
// And into the narrower base types, where it used to be deleted outright.
static_assert(!::cuda::std::is_convertible_v<quad, cudax::fp64_custom<>>, "");
static_assert(!::cuda::std::is_convertible_v<quad, cudax::fp32_custom<>>, "");
// What a double cannot reach is still a format narrower than itself, over this base type as
// over any other.
static_assert(!::cuda::std::is_convertible_v<double, fp128_fp32>, "");
static_assert(!::cuda::std::is_convertible_v<quad, cudax::fp128_custom<15, cudax::fp_custom_dynamic_size>>, "");

#  else // ^^^ CCCL_FP_CUSTOM_EXPLICIT_CASTS == 1 ^^^ / vvv == 0 vvv

static_assert(::cuda::std::is_convertible_v<quad, fp128_fp64>, "");
static_assert(::cuda::std::is_convertible_v<quad, fp128_exp11>, "");
static_assert(::cuda::std::is_convertible_v<quad, fp128_mant64>, "");
static_assert(::cuda::std::is_convertible_v<quad, cudax::fp64_custom<>>, "");
static_assert(::cuda::std::is_convertible_v<quad, cudax::fp32_custom<>>, "");
static_assert(::cuda::std::is_convertible_v<double, fp128_fp32>, "");
static_assert(::cuda::std::is_convertible_v<quad, cudax::fp128_custom<15, cudax::fp_custom_dynamic_size>>, "");

#  endif // CCCL_FP_CUSTOM_EXPLICIT_CASTS == 0

// Explicit, not absent: every source is still constructible into every format.
static_assert(::cuda::std::is_constructible_v<fp128_fp64, quad>, "");
static_assert(::cuda::std::is_constructible_v<cudax::fp64_custom<>, quad>, "");
static_assert(::cuda::std::is_constructible_v<cudax::fp32_custom<>, quad>, "");

// A 128-bit integer stays deleted: 112 mantissa bits are more than any other base type has and
// still 16 short of holding one.
static_assert(!::cuda::std::is_constructible_v<fp128_native, __uint128_t>, "");
static_assert(!::cuda::std::is_constructible_v<fp128_native, __int128_t>, "");

bool test_conversions()
{
  bool ok = true;

  // The base type is the one boundary that binds at construction, and over the widest one
  // nothing binds: a double arrives exactly, whatever the requested format.
  const quad third = to_quad(1.0) / to_quad(3.0);
  ok               = ok && static_cast<quad>(fp128_fp64{1.0 / 3.0}) == to_quad(1.0 / 3.0);
  ok               = ok && to_quad(1.0 / 3.0) != third;

  // The requested format is applied by the first operation, so a value it cannot hold reads
  // back whole and rounds on use.
  ok = ok && static_cast<quad>(fp128_fp64{third}) == third;
  ok = ok && static_cast<quad>(fp128_fp64{third} + fp128_fp64{to_quad(0.0)}) == to_quad(1.0 / 3.0);

  // Out through double the value rounds to binary64, which is what keeps the type a drop-in;
  // out through the base type it does not.
  ok = ok && static_cast<double>(fp128_native{third}) == 1.0 / 3.0;
  ok = ok && static_cast<quad>(fp128_native{third}) == third;

  // Which is the whole difference between the two ways to a binary128 sink. Implicitly the
  // value goes through operator double() and arrives rounded; the cast above is exact.
  const quad implicit_out = fp128_native{third};
  ok                      = ok && implicit_out == to_quad(1.0 / 3.0) && implicit_out != third;

  // Mixed arithmetic and comparison take a scalar operand and stay in fp_custom.
  const fp128_fp64 three{3};
  ok = ok && static_cast<double>(three + 1.0) == 4.0;
  ok = ok && static_cast<double>(1.0f + three) == 4.0;
  ok = ok && static_cast<double>(three * 2) == 6.0;
  ok = ok && three > 2.0 && three < 4;
  ok = ok && ::cuda::std::is_same_v<decltype(three + 1.0), fp128_fp64>;
  ok = ok && ::cuda::std::is_same_v<decltype(2 * three), fp128_fp64>;

  return ok;
}

// A qualified cuda::std::sqrt or cuda::std::fma call suppresses ADL, and a plain double operand
// in an fma makes ::fma(double, double, double) viable. Both must still reduce rather than
// quietly compute at full binary128 precision.
bool test_standard_spellings()
{
  bool ok = true;

  const fp128_fp64 two{2.0};
  const double reduced_root = static_cast<double>(sqrt(two));
  ok                        = ok && static_cast<double>(::cuda::std::sqrt(two)) == reduced_root;
  ok                        = ok && reduced_root == ::cuda::std::sqrt(2.0);

  // A first operand that only a mantissa wider than binary64's could keep.
  const quad wide = to_quad(1.0) + to_quad(0x1p-60);
  const fp128_fp64 a{wide}, b{1.0}, zero{0.0};
  const double reduced_fma = static_cast<double>(fma(a, b, zero));
  ok                       = ok && static_cast<double>(::cuda::std::fma(a, b, zero)) == reduced_fma;
  ok                       = ok && static_cast<double>(fma(a, b, 0.0)) == reduced_fma;
  ok                       = ok && static_cast<double>(::cuda::std::fma(a, b, 0.0)) == reduced_fma;
  ok                       = ok && reduced_fma == 1.0;

  // The same call over the native format keeps it, which is what the reduction above dropped.
  const fp128_native wa{wide}, wb = to_quad(1.0), wzero = to_quad(0.0);
  ok = ok && static_cast<quad>(fma(wa, wb, wzero)) == wide;

  return ok;
}

// === runtime sizes ===
// Each base type holds its own pair, so sizing binary128 must leave the double and float ones
// where they were.
void test_host_runtime_sizes()
{
  using fp128_dynamic = cudax::fp128_custom<cudax::fp_custom_dynamic_size, cudax::fp_custom_dynamic_size>;

  const fp128_dynamic one{to_quad(1.0)}, tiny{to_quad(0x1p-60)};
  const quad sum_full = to_quad(1.0) + to_quad(0x1p-60);

  // Untouched, the sizes are binary128's, so a term 60 binades down survives.
  assert(cudax::fp_custom_get_host_mantissa_size<quad>() == 112);
  assert(cudax::fp_custom_get_host_exponent_size<quad>() == 15);
  assert(static_cast<quad>(one + tiny) == sum_full);

  // binary64's 52 bits cannot hold it.
  cudax::fp_custom_set_host_mantissa_size<quad>(52);
  assert(cudax::fp_custom_get_host_mantissa_size<quad>() == 52);
  assert(static_cast<quad>(one + tiny) == to_quad(1.0));

  // The double and float sizes are separate state, which that write must not have touched.
  assert(cudax::fp_custom_get_host_mantissa_size() == 52);
  assert(cudax::fp_custom_get_host_exponent_size() == 11);
  assert(cudax::fp_custom_get_host_mantissa_size<float>() == 23);
  assert(cudax::fp_custom_get_host_exponent_size<float>() == 8);

  // The exponent is the other axis: 11 bits keep 1.0 but flush 1e-300 squared.
  cudax::fp_custom_set_host_exponent_size<quad>(11);
  assert(cudax::fp_custom_get_host_exponent_size<quad>() == 11);
  assert(static_cast<double>(fp128_dynamic{to_quad(1e-300)} * fp128_dynamic{to_quad(1e-300)}) == 0.0);

  // Leave the native format behind for anything that runs later.
  cudax::fp_custom_set_host_mantissa_size<quad>(112);
  cudax::fp_custom_set_host_exponent_size<quad>(15);
  assert(static_cast<quad>(one + tiny) == sum_full);
}

void test()
{
  assert(test_native_matches_quad());
  assert(test_reproduces_double());
  assert(test_reproduces_float());
  assert(test_mantissa_reduction());
  assert(test_exponent_reduction());
  assert(test_conversions());
  assert(test_standard_spellings());
  test_host_runtime_sizes();
}

#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1 && !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1) && !_CCCL_COMPILER(NVRTC)
  // force_include.h runs this main on the host and then inside a kernel; the binary128
  // functions above are host-only, so NV_IS_HOST selects the driver, not the code tested.
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1 && !_CCCL_COMPILER(NVRTC)

  return 0;
}
