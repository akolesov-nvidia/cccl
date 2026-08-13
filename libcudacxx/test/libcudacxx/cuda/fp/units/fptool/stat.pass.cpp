// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// note: the host half of this test resets and reads the device record through the CUDA
// runtime API, which is not available in NVRTC's device-only translation unit
// UNSUPPORTED: enable-tile
// error: the instrumentation updates its counters with atomics, which are unsupported in
// tile code

//===----------------------------------------------------------------------===//
//
//  Unit test: statistics-collecting fpmp2 wrapper (fpmp2_stat).
//
//  The wrapper must observe arithmetic without changing it, so the test computes
//  everything twice - once on fp32mp2 and once on fp32mp2_stat - and requires the
//  two to agree bit for bit: operators, compound assignments, increments, mixed
//  scalar and mixed wrapped-type forms, the untraced helpers (sqrt, rsqrt, fma,
//  mad, renormalize) and the math wrappers. It also covers the layout promise
//  (same size, alignment and trivial copyability as the wrapped type) and the
//  conversion surface.
//
//  Under CUDA a second part checks the record itself: that a reset arms the range
//  sentinels and zeroes the counters, and that a kernel with a known operation mix
//  produces exactly the expected counts. The parity part runs on the host and, under
//  CUDA, on the device.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fptool_math>
#include <cuda/std/cassert>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

using base_t     = cudax::fp32mp2;
using stat_t     = cudax::fp32mp2_stat;
using base_low_t = cudax::fp32mp2_low;
using stat_low_t = cudax::fp32mp2_stat_low;

// The drop-in promise in memory.
static_assert(sizeof(stat_t) == sizeof(base_t));
static_assert(alignof(stat_t) == alignof(base_t));
static_assert(cuda::std::is_trivially_copyable_v<stat_t>);
static_assert(sizeof(cudax::fp64mp2_stat) == sizeof(cudax::fp64mp2));
static_assert(cuda::std::is_trivially_copyable_v<cudax::fp64mp2_stat>);

// The wrapper reports the same characteristics as the wrapped type.
static_assert(cuda::std::numeric_limits<stat_t>::digits == cuda::std::numeric_limits<base_t>::digits);
static_assert(cuda::std::numeric_limits<stat_t>::is_specialized);

// A pair of limbs, compared bitwise so that a sign of zero or a NaN payload cannot hide
// a difference.
struct pair_t
{
  float hi;
  float lo;
};

TEST_HOST_DEVICE_FUNC bool same(const pair_t& lhs, const pair_t& rhs)
{
  return cuda::std::memcmp(&lhs, &rhs, sizeof(pair_t)) == 0;
}

template <class _Tp>
TEST_HOST_DEVICE_FUNC pair_t limbs(const _Tp& value)
{
  return pair_t{value.hi(), value.lo()};
}

// Every operation below is computed on both types and the two results compared, so the
// harness needs no expected values: the wrapped type is the reference.
TEST_HOST_DEVICE_FUNC void test_parity()
{
  const float fa = 1.123456789f;
  const float fb = 2.987654321f;
  const float fc = 0.577215664f;

  const base_t ba(fa), bb(fb), bc(fc);
  const stat_t sa(fa), sb(fb), sc(fc);

  assert(same(limbs(ba + bb), limbs(sa + sb)));
  assert(same(limbs(ba - bb), limbs(sa - sb)));
  assert(same(limbs(ba * bb), limbs(sa * sb)));
  assert(same(limbs(ba / bb), limbs(sa / sb)));
  assert(same(limbs(-ba), limbs(-sa)));
  assert(same(limbs(renormalize(ba)), limbs(renormalize(sa))));

  { // compound assignment chain
    base_t b = ba;
    stat_t s = sa;
    b += bb;
    s += sb;
    b -= bc;
    s -= sc;
    b *= bb;
    s *= sb;
    b /= bc;
    s /= sc;
    b += fc;
    s += fc;
    b -= fc;
    s -= fc;
    assert(same(limbs(b), limbs(s)));
  }

  { // increment and decrement, prefix and postfix
    base_t b = ba;
    stat_t s = sa;
    ++b;
    ++s;
    --b;
    --s;
    b++;
    s++;
    b--;
    s--;
    assert(same(limbs(b), limbs(s)));
  }

  // mixed with a built-in scalar, both operand orders
  assert(same(limbs(fc * ba), limbs(fc * sa)));
  assert(same(limbs(ba * fc), limbs(sa * fc)));
  assert(same(limbs(ba + 2), limbs(sa + 2)));
  assert(same(limbs(3 - ba), limbs(3 - sa)));
  assert(same(limbs(ba / 2.0f), limbs(sa / 2.0f)));

  // mixed with the wrapped type, both operand orders: these must stay instrumented and
  // must not be ambiguous
  assert(same(limbs(ba + bb), limbs(sa + bb)));
  assert(same(limbs(ba + bb), limbs(ba + sb)));
  assert(same(limbs(ba - bb), limbs(sa - bb)));
  assert(same(limbs(ba * bb), limbs(ba * sb)));
  assert(same(limbs(ba / bb), limbs(sa / bb)));
  static_assert(cuda::std::is_same_v<decltype(sa + bb), stat_t>);
  static_assert(cuda::std::is_same_v<decltype(ba + sb), stat_t>);

  // untraced arithmetic helpers
  assert(same(limbs(sqrt(bb)), limbs(sqrt(sb))));
  assert(same(limbs(rsqrt(bb)), limbs(rsqrt(sb))));
  assert(same(limbs(fma(ba, bb, bc)), limbs(fma(sa, sb, sc))));
  assert(same(limbs(mad(ba, bb, bc)), limbs(mad(sa, sb, sc))));
  assert(same(limbs(fma(ba, bb, base_t(2.0f))), limbs(fma(sa, sb, 2.0f))));

  // math wrappers against the same functions on the wrapped type
  assert(same(limbs(exp(bc)), limbs(exp(sc))));
  assert(same(limbs(log(bb)), limbs(log(sb))));
  assert(same(limbs(pow(bb, bc)), limbs(pow(sb, sc))));
  assert(same(limbs(hypot(ba, bb)), limbs(hypot(sa, sb))));
  assert(same(limbs(fabs(-ba)), limbs(fabs(-sa))));
  assert(same(limbs(fmax(ba, bb)), limbs(fmax(sa, sb))));
  assert(same(limbs(ldexp(ba, 3)), limbs(ldexp(sa, 3))));
  assert(same(limbs(norm3d(ba, bb, bc)), limbs(norm3d(sa, sb, sc))));
  assert(ilogb(bb) == ilogb(sb));
  assert(lround(bb) == lround(sb));

  // the standard spellings must reach the emulated implementation, not narrow to double
  assert(same(limbs(cuda::std::exp(bc)), limbs(cuda::std::exp(sc))));
  assert(same(limbs(cuda::std::hypot(ba, bb)), limbs(cuda::std::hypot(sa, sb))));

  { // out-pointer functions
    base_t bs, bcos;
    stat_t ss, scos;
    sincos(bc, &bs, &bcos);
    sincos(sc, &ss, &scos);
    assert(same(limbs(bs), limbs(ss)));
    assert(same(limbs(bcos), limbs(scos)));

    base_t bi;
    stat_t si;
    const base_t bf = modf(bb, &bi);
    const stat_t sf = modf(sb, &si);
    assert(same(limbs(bi), limbs(si)));
    assert(same(limbs(bf), limbs(sf)));

    int bq = 0, sq = 0;
    assert(same(limbs(remquo(bb, bc, &bq)), limbs(remquo(sb, sc, &sq))));
    assert(bq == sq);
  }

  // comparisons, against the wrapper, the wrapped type and a scalar
  assert(sa == sa);
  assert(sa != sb);
  assert(sa < sb);
  assert(!(sa > sb));
  assert(sa <= sb);
  assert(sb >= sa);
  assert(sa == ba);
  assert(ba == sa);
  assert(sa < bb);
  assert(bb > sa);
  assert(sa != 0.0f);
  assert(0.0f < sa);

  // classification
  assert(fpmp_isfinite(sa) == fpmp_isfinite(ba));
  assert(fpmp_isnan(sa) == fpmp_isnan(ba));
  assert(isfinite(sa) != 0);
  assert(isnan(sa) == 0);

  // conversions: to and from the wrapped type, to built-in types, and across accuracy
  const base_t to_base   = sa; // implicit
  const stat_t from_base = ba; // implicit
  assert(same(limbs(to_base), limbs(sa)));
  assert(same(limbs(from_base), limbs(ba)));
  assert(static_cast<double>(sa) == static_cast<double>(ba));
  assert(static_cast<float>(sa) == static_cast<float>(ba));
  assert(static_cast<int>(sb) == static_cast<int>(bb));

  const base_low_t base_low       = static_cast<base_low_t>(ba);
  const stat_low_t stat_low       = static_cast<stat_low_t>(sa);
  const stat_low_t stat_from_base = static_cast<stat_low_t>(ba);
  assert(same(limbs(base_low), limbs(stat_low)));
  assert(same(limbs(base_low), limbs(stat_from_base)));

  { // volatile storage round-trip, the pattern used for shared-memory scalars
    volatile stat_t vs = sa;
    const stat_t loaded(vs);
    assert(same(limbs(sa), limbs(loaded)));
    assert(vs.hi() == sa.hi() && vs.lo() == sa.lo());
    vs = sb;
    const stat_t reloaded(vs);
    assert(same(limbs(sb), limbs(reloaded)));
  }

  { // trivially copyable in practice: a byte copy carries the value
    stat_t dst(0.0f);
    cuda::std::memcpy(&dst, &sa, sizeof(stat_t));
    assert(same(limbs(dst), limbs(sa)));
  }

  { // the double-double instantiation, so both limb types are exercised
    using dbase_t = cudax::fp64mp2;
    using dstat_t = cudax::fp64mp2_stat;
    const dbase_t db(1.25);
    const dstat_t ds(1.25);
    const dbase_t br = (db + db) * db / db - db;
    const dstat_t sr = (ds + ds) * ds / ds - ds;
    assert(br.hi() == sr.hi() && br.lo() == sr.lo());
    // sqrt is native to fpmp for both limb types; the transcendentals of a double-double
    // fall back to binary128 on the host, which would pull libquadmath into this test.
    assert(sqrt(db).hi() == sqrt(ds).hi());
    assert(fabs(-db).hi() == fabs(-ds).hi());
  }
}

#if _CCCL_CUDA_COMPILATION()

__global__ void parity_kernel()
{
  test_parity();
}

// A kernel by one thread with a hand-counted operation mix.
constexpr unsigned long long int expected_add = 3ull;
constexpr unsigned long long int expected_sub = 3ull;
constexpr unsigned long long int expected_mul = 2ull;
constexpr unsigned long long int expected_div = 2ull;
constexpr unsigned long long int expected_ops = expected_add + expected_sub + expected_mul + expected_div;

__global__ void counting_kernel(float* sink)
{
  stat_t a(1.5f), b(0.25f);

  stat_t s = a + b; // add
  s        = s - b; // sub
  s        = s * b; // mul
  s        = s / b; // div
  s += a; // add
  s -= b; // sub
  s *= a; // mul
  s /= a; // div
  ++s; // add
  --s; // sub

  *sink = static_cast<float>(s);
}

// The exact operands of the counting kernel produce a zero lo limb, so a gap is only
// sampled where the arithmetic actually needed the second limb: the range stays empty
// otherwise, which is the sentinel pair rather than an ordered range.
void check_slot_sampled(const cudax::fpmp2_stat_value& slot)
{
  assert(slot.min_exp <= slot.max_exp);
  assert(slot.min_hi_lo_mantissa_gap <= slot.max_hi_lo_mantissa_gap
         || (slot.min_hi_lo_mantissa_gap == cuda::std::numeric_limits<int>::max()
             && slot.max_hi_lo_mantissa_gap == cuda::std::numeric_limits<int>::min()));
  // The operands are small exact values, so nothing degenerate should appear.
  assert(slot.nan_count == 0ull);
  assert(slot.inf_count == 0ull);
  assert(slot.infnan_count == 0ull);
  assert(slot.zero_count == 0ull);
}

// An inexact quotient needs both limbs, so its summary must carry a gap sample. In a
// normalized double-float |lo| <= ulp(hi)/2, which puts the exponents at least
// digits(float) = 24 places apart.
__global__ void gap_kernel(float* sink)
{
  const stat_t third = stat_t(1.0f) / stat_t(3.0f);
  *sink              = third.lo();
}

void test_device_record()
{
  const int sentinel_max = cuda::std::numeric_limits<int>::max();
  const int sentinel_min = cuda::std::numeric_limits<int>::min();

  // The parity kernel runs first and also counts, so the record is reset afterwards.
  parity_kernel<<<1, 1>>>();
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  float* sink = nullptr;
  assert(cudaMalloc(&sink, sizeof(float)) == cudaSuccess);

  assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);

  cudax::fpmp2_stat_data after_reset{};
  assert(cudax::fpmp2_stat_read_device_data(&after_reset) == cudaSuccess);

  assert(after_reset.ops_count == 0ull);
  assert(after_reset.add_count == 0ull);
  assert(after_reset.sub_count == 0ull);
  assert(after_reset.mul_count == 0ull);
  assert(after_reset.div_count == 0ull);
  // Armed ranges: empty, so the first sample replaces both ends.
  assert(after_reset.result.min_exp == sentinel_max);
  assert(after_reset.result.max_exp == sentinel_min);
  assert(after_reset.result.min_hi_lo_mantissa_gap == sentinel_max);
  assert(after_reset.result.max_hi_lo_mantissa_gap == sentinel_min);

  counting_kernel<<<1, 1>>>(sink);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudax::fpmp2_stat_data after_run{};
  assert(cudax::fpmp2_stat_read_device_data(&after_run) == cudaSuccess);

  assert(after_run.add_count == expected_add);
  assert(after_run.sub_count == expected_sub);
  assert(after_run.mul_count == expected_mul);
  assert(after_run.div_count == expected_div);
  assert(after_run.ops_count == expected_ops);
  assert(after_run.ops_count == after_run.add_count + after_run.sub_count + after_run.mul_count + after_run.div_count);

  check_slot_sampled(after_run.arg[0]);
  check_slot_sampled(after_run.arg[1]);
  check_slot_sampled(after_run.result);

  // arg[2] is reserved for a future ternary operation and must stay untouched.
  assert(after_run.arg[2].min_exp == sentinel_max);
  assert(after_run.arg[2].max_exp == sentinel_min);

  // A second reset clears what the run recorded.
  assert(cudax::fpmp2_stat_reset_device_data() == cudaSuccess);
  cudax::fpmp2_stat_data after_second_reset{};
  assert(cudax::fpmp2_stat_read_device_data(&after_second_reset) == cudaSuccess);
  assert(after_second_reset.ops_count == 0ull);

  // An inexact result must be summarized with a gap that reflects a normalized pair.
  gap_kernel<<<1, 1>>>(sink);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  cudax::fpmp2_stat_data after_gap{};
  assert(cudax::fpmp2_stat_read_device_data(&after_gap) == cudaSuccess);

  assert(after_gap.div_count == 1ull);
  assert(after_gap.result.min_hi_lo_mantissa_gap <= after_gap.result.max_hi_lo_mantissa_gap);
  assert(after_gap.result.min_hi_lo_mantissa_gap >= cuda::std::numeric_limits<float>::digits);
  // The operands 1 and 3 are exact, so their own lo limbs are zero.
  assert(after_gap.arg[0].zero_lo_count == 1ull);
  assert(after_gap.arg[1].zero_lo_count == 1ull);
  assert(after_gap.result.zero_lo_count == 0ull);

  assert(cudaFree(sink) == cudaSuccess);
}

#endif // _CCCL_CUDA_COMPILATION()

int main(int, char**)
{
  test_parity();
#if _CCCL_CUDA_COMPILATION()
  test_device_record();
#endif // _CCCL_CUDA_COMPILATION()
  return 0;
}
