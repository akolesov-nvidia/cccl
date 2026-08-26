// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fptool's fp_custom over a `float` base, spelled fp32_custom.
//
//  custom.pass.cpp covers the same type over `double`; this file covers what
//  changes when the base type is binary32: the storage narrows to 4 bytes, the
//  arithmetic runs in float, the field sizes top out at 8 and 23, and the
//  conversion rules shift one rank down, a `double` now being the wider format
//  that has to be cast in.
//
//  Sections: the native format as a drop-in for float, mantissa reduction
//  (TF32-like), exponent reduction (FP16-like range), the zero-mantissa
//  power-of-two case, the type surface, the conversions in and out, agreement
//  with the same format emulated over `double`, the standard math spellings,
//  and the runtime sizes, which each base type holds separately.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fptool>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

// <cuda/stream> is only usable where the CUDA runtime is: under NVRTC cuda::stream_ref is left
// undefined while get_stream.h still returns it by value. The stream-based runtime-size tests
// below are host-side and carry the same guard.
#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
#  include <cuda/stream>
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// The native binary32 format, where both reductions compile out entirely.
using fp32_native = cudax::fp32_custom<>;
// TF32: binary32's exponent over a 10-bit mantissa, so only the mantissa is reduced.
using fp32_tf32 = cudax::fp32_custom<8, 10>;
// FP16's two field sizes, which reduces both.
using fp32_half = cudax::fp32_custom<5, 10>;
// The smallest mantissa the type accepts, which quantizes to powers of two.
using fp32_po2 = cudax::fp32_custom<8, 0>;

// Bits dropped by a 10-bit mantissa over binary32's 23.
constexpr ::cuda::std::uint32_t tf32_dropped_mask = (1u << 13) - 1u;

// === the native format is a drop-in for float ===
// Every operation is the float one, so the results are exact rather than close.
TEST_HOST_DEVICE_FUNC bool test_native_matches_float()
{
  bool ok = true;

  float a = 1.1234567f;
  float b = 2.1234567f;
  float c = 0.5f;

  fp32_native ra = a, rb = b, rc = c; // implicit: the native format holds every float

  ok = ok && static_cast<float>(ra + rb) == a + b;
  ok = ok && static_cast<float>(ra - rb) == a - b;
  ok = ok && static_cast<float>(ra * rb) == a * b;
  ok = ok && static_cast<float>(ra / rb) == a / b;
  ok = ok && static_cast<float>(-ra) == -a;

  ok = ok && static_cast<float>(sqrt(ra)) == ::cuda::std::sqrt(a);
  ok = ok && static_cast<float>(fma(ra, rb, rc)) == ::cuda::std::fma(a, b, c);

  // Compound assignment and inc/dec reach the same operators. The reference repeats the
  // increment as an addition, ±1 not being exactly invertible in binary32.
  {
    fp32_native x = 1.0f;
    x += ra;
    x -= rc;
    x *= rb;
    x /= fp32_native(2.0f);
    ++x;
    --x;

    float ref = 1.0f;
    ref += a;
    ref -= c;
    ref *= b;
    ref /= 2.0f;
    ref += 1.0f;
    ref -= 1.0f;
    ok = ok && static_cast<float>(x) == ref;
  }

  // Accumulating a harmonic sum step for step, which any deviation from float would show.
  {
    fp32_native sum = 0.0f;
    float ref       = 0.0f;
    for (int n = 1; n <= 1000; ++n)
    {
      const float term = 1.0f / static_cast<float>(n);
      sum += fp32_native(term);
      ref += term;
    }
    ok = ok && static_cast<float>(sum) == ref;
  }

  return ok;
}

// === mantissa reduction ===
// The reduction applies to the operands as well as the result, so adding zero is enough to
// observe it.
TEST_HOST_DEVICE_FUNC bool test_mantissa_reduction()
{
  bool ok = true;

  const fp32_tf32 zero{0.0f};

  // A third has every low mantissa bit occupied, so 13 of them are lost and the value moves.
  const float third      = 1.0f / 3.0f;
  const float third_tf32 = static_cast<float>(fp32_tf32{third} + zero);
  ok                     = ok && (::cuda::std::bit_cast<::cuda::std::uint32_t>(third) & tf32_dropped_mask) != 0u;
  ok                     = ok && (::cuda::std::bit_cast<::cuda::std::uint32_t>(third_tf32) & tf32_dropped_mask) == 0u;
  ok                     = ok && third_tf32 != third;
  ok                     = ok && ::cuda::std::fabs(third_tf32 - third) < 0x1p-11f;

  // A term 13 binades down is exactly what a 10-bit mantissa cannot hold beside 1.
  ok = ok && static_cast<float>(fp32_tf32{1.0f} + fp32_tf32{0x1p-13f}) == 1.0f;
  ok = ok && (1.0f + 0x1p-13f) != 1.0f; // native float does hold it

  // Round-to-nearest-even at the halfway point, whose two instances here go opposite ways:
  // 1 + 2^-11 sits between 1 and 1 + 2^-10 and lands on 1, whose mantissa is even, while
  // 1 + 3*2^-11 sits between 1 + 2^-10 and 1 + 2^-9 and lands on the latter, the former's
  // mantissa being odd.
  ok = ok && static_cast<float>(fp32_tf32{1.0f + 0x1p-11f} + zero) == 1.0f;
  ok = ok && static_cast<float>(fp32_tf32{1.0f + 0x1p-10f + 0x1p-11f} + zero) == 1.0f + 0x1p-9f;

  // Specials are left alone.
  ok = ok && ::cuda::std::isinf(static_cast<float>(fp32_tf32{::cuda::std::numeric_limits<float>::infinity()} + zero));
  ok = ok && ::cuda::std::isnan(static_cast<float>(fp32_tf32{::cuda::std::numeric_limits<float>::quiet_NaN()} + zero));

  return ok;
}

// === exponent reduction ===
// A 5-bit exponent covers 2^-14 to just under 2^16, and anything outside is clamped rather
// than represented, infinity above and a signed zero below.
TEST_HOST_DEVICE_FUNC bool test_exponent_reduction()
{
  bool ok = true;

  const fp32_half one{1.0f};

  // Inside the range, so only the mantissa is touched.
  ok = ok && static_cast<float>(fp32_half{0x1p15f} * one) == 0x1p15f;
  ok = ok && static_cast<float>(fp32_half{0x1p-14f} * one) == 0x1p-14f;
  ok = ok && static_cast<float>(fp32_half{65504.0f} * one) == 65504.0f; // FP16's largest finite

  // Outside it, with the sign preserved.
  ok = ok && ::cuda::std::isinf(static_cast<float>(fp32_half{0x1p16f} * one));
  ok = ok && ::cuda::std::isinf(static_cast<float>(fp32_half{-0x1p16f} * one));
  ok = ok && ::cuda::std::signbit(static_cast<float>(fp32_half{-0x1p16f} * one));
  ok = ok && static_cast<float>(fp32_half{0x1p-15f} * one) == 0.0f;
  ok = ok && ::cuda::std::signbit(static_cast<float>(fp32_half{-0x1p-15f} * one));

  // Overflow of the reduced range is reached by arithmetic too, not only by construction.
  ok = ok && ::cuda::std::isinf(static_cast<float>(fp32_half{0x1p10f} * fp32_half{0x1p10f}));

  // NaN must not be mistaken for a large finite value and clamped to infinity.
  ok = ok && ::cuda::std::isnan(static_cast<float>(fp32_half{::cuda::std::numeric_limits<float>::quiet_NaN()} * one));

  return ok;
}

// The smallest mantissa the type accepts keeps only the implicit leading 1, so every value
// collapses to the nearest power of two. Ties land on the even exponent, which is why 3
// rounds down to 2 while 6 rounds up to 8 -- the same values custom.pass.cpp checks over
// `double`, the rounding position being the same distance from the mantissa's top.
TEST_HOST_DEVICE_FUNC bool test_power_of_two()
{
  bool ok = true;

  const fp32_po2 one{1.0f};
  ok = ok && static_cast<float>(fp32_po2{1.4f} * one) == 1.0f;
  ok = ok && static_cast<float>(fp32_po2{1.5f} * one) == 2.0f;
  ok = ok && static_cast<float>(fp32_po2{3.0f} * one) == 2.0f;
  ok = ok && static_cast<float>(fp32_po2{6.0f} * one) == 8.0f;
  ok = ok && static_cast<float>(fp32_po2{0.3f} * one) == 0.25f;

  ok = ok && static_cast<float>(fp32_po2{-3.0f} * one) == -2.0f;
  ok = ok && static_cast<float>(fp32_po2{-0.0f} + fp32_po2{-0.0f}) == 0.0f;
  ok = ok && ::cuda::std::signbit(static_cast<float>(fp32_po2{-0.0f} + fp32_po2{-0.0f}));

  return ok;
}

// === type surface ===
// The type stays a transparent stand-in for float: same layout, trivially copyable so it
// can be bit_cast and copied by memcpy, and usable as volatile storage.
static_assert(::cuda::std::is_trivially_copyable_v<fp32_tf32>, "");
static_assert(::cuda::std::is_trivially_copy_constructible_v<fp32_tf32>, "");
static_assert(::cuda::std::is_trivially_copy_assignable_v<fp32_tf32>, "");
static_assert(sizeof(fp32_tf32) == sizeof(float), "");
static_assert(alignof(fp32_tf32) == alignof(float), "");
// Which is half of what the same format costs over `double`, the point of the float base.
static_assert(sizeof(fp32_tf32) * 2 == sizeof(cudax::fp64_custom<8, 10>), "");

TEST_HOST_DEVICE_FUNC bool test_type_surface()
{
  bool ok = true;

  // bit_cast in both directions, which trivial copyability is what enables.
  const auto bits = ::cuda::std::bit_cast<::cuda::std::uint32_t>(fp32_tf32{1.0f});
  ok              = ok && bits == 0x3f800000u;
  ok              = ok && static_cast<float>(::cuda::std::bit_cast<fp32_tf32>(bits)) == 1.0f;

  // bool and character types convert, as they do for float.
  ok = ok && static_cast<float>(fp32_tf32{true}) == 1.0f;
  ok = ok && static_cast<float>(fp32_tf32{'A'}) == 65.0f;

  // Volatile storage: load, store and a volatile-to-volatile copy all preserve bits.
  volatile fp32_tf32 vsrc = fp32_tf32{0.5f};
  const fp32_tf32 loaded  = vsrc;
  volatile fp32_tf32 vdst = fp32_tf32{0.0f};
  vdst                    = loaded;
  volatile fp32_tf32 vcpy = fp32_tf32{0.0f};
  vcpy                    = vdst;
  ok                      = ok && static_cast<float>(loaded) == 0.5f;
  ok                      = ok && static_cast<float>(fp32_tf32{vcpy}) == 0.5f;

  return ok;
}

// === conversions out ===
// float is implicit for every fp32_custom, the value being held in one, and double is
// implicit as well, being wider still. Offering both on equal terms is what makes an
// overload set holding float and double ambiguous for these types, which is the cost of
// never narrowing silently.
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

static_assert(picks_ambiguously<fp32_native>::value, "");
static_assert(picks_ambiguously<fp32_tf32>::value, "");
static_assert(picks_ambiguously<fp32_half>::value, "");
static_assert(picks_ambiguously<fp32_po2>::value, "");
// A runtime size is unknown at compile time, but it cannot exceed binary32's own, so a float
// base fits whatever it turns out to be. This is where the float base differs from the
// double one, for which the same spelling takes the explicit conversion.
static_assert(picks_ambiguously<cudax::fp32_custom<8, cudax::fp_custom_dynamic_size>>::value, "");
static_assert(picks_ambiguously<cudax::fp32_custom<cudax::fp_custom_dynamic_size, 23>>::value, "");
static_assert(!picks_ambiguously<cudax::fp64_custom<8, cudax::fp_custom_dynamic_size>>::value, "");

static_assert(::cuda::std::is_convertible_v<fp32_native, float>, "");
static_assert(::cuda::std::is_convertible_v<fp32_half, float>, "");
static_assert(::cuda::std::is_convertible_v<fp32_native, double>, "");
static_assert(::cuda::std::is_convertible_v<fp32_half, double>, "");

// === conversions in ===
// The same rank rule as over `double`, read against binary32: a float is implicit where the
// requested format holds it, which is only the native one, and a double or an integer is a
// cast everywhere, binary64 being wider than any format a float base can offer.
static_assert(::cuda::std::is_convertible_v<float, fp32_native>, "");

#if CCCL_FP_CUSTOM_EXPLICIT_CASTS == 1

static_assert(!::cuda::std::is_convertible_v<double, fp32_native>, "");
static_assert(!::cuda::std::is_convertible_v<int, fp32_native>, "");
static_assert(!::cuda::std::is_convertible_v<bool, fp32_native>, "");
static_assert(!::cuda::std::is_convertible_v<char, fp32_native>, "");

// Narrower than binary32 in either field, so a float is a cast as well.
static_assert(!::cuda::std::is_convertible_v<float, fp32_tf32>, "");
static_assert(!::cuda::std::is_convertible_v<float, fp32_half>, "");
static_assert(!::cuda::std::is_convertible_v<float, fp32_po2>, "");
static_assert(!::cuda::std::is_convertible_v<double, fp32_tf32>, "");

// A runtime size is unknown here, so every source takes the explicit constructor: unlike the
// conversion out, this direction has to know the format exactly.
static_assert(!::cuda::std::is_convertible_v<float, cudax::fp32_custom<8, cudax::fp_custom_dynamic_size>>, "");

#else // ^^^ CCCL_FP_CUSTOM_EXPLICIT_CASTS == 1 ^^^ / vvv == 0 vvv

static_assert(::cuda::std::is_convertible_v<double, fp32_native>, "");
static_assert(::cuda::std::is_convertible_v<int, fp32_native>, "");
static_assert(::cuda::std::is_convertible_v<bool, fp32_native>, "");
static_assert(::cuda::std::is_convertible_v<char, fp32_native>, "");
static_assert(::cuda::std::is_convertible_v<float, fp32_tf32>, "");
static_assert(::cuda::std::is_convertible_v<float, fp32_half>, "");
static_assert(::cuda::std::is_convertible_v<float, fp32_po2>, "");
static_assert(::cuda::std::is_convertible_v<double, fp32_tf32>, "");
static_assert(::cuda::std::is_convertible_v<float, cudax::fp32_custom<8, cudax::fp_custom_dynamic_size>>, "");

#endif // CCCL_FP_CUSTOM_EXPLICIT_CASTS == 0

// Explicit, not absent: every source is still constructible into every format.
static_assert(::cuda::std::is_constructible_v<fp32_native, double>, "");
static_assert(::cuda::std::is_constructible_v<fp32_native, int>, "");
static_assert(::cuda::std::is_constructible_v<fp32_half, float>, "");
static_assert(::cuda::std::is_constructible_v<fp32_half, bool>, "");

// The base type is part of the type, so the same format over the two bases gives two
// distinct types, which no conversion between them is offered for.
static_assert(!::cuda::std::is_same_v<cudax::fp32_custom<8, 10>, cudax::fp64_custom<8, 10>>, "");

TEST_HOST_DEVICE_FUNC bool test_conversions()
{
  bool ok = true;

  // The base type is the one boundary that binds at construction: a double is rounded to
  // binary32 on the way in, so what binary32 cannot hold is gone immediately, where over a
  // double base the same value survives until the first operation.
  const fp32_native narrowed{1.0 + 0x1p-30};
  ok = ok && static_cast<double>(narrowed) == 1.0;
  ok = ok && static_cast<double>(cudax::fp64_custom<8, 23>{1.0 + 0x1p-30}) == 1.0 + 0x1p-30;

  // The requested format, in contrast, is applied by the first operation, so a value the
  // reduced exponent cannot hold reads back whole and clamps on use.
  const fp32_half big{0x1p20f};
  ok = ok && static_cast<float>(big) == 0x1p20f;
  ok = ok && ::cuda::std::isinf(static_cast<float>(big + fp32_half{0.0f}));

  // A double or an integer arrives rounded to the base type, and nothing further.
  ok = ok && static_cast<float>(fp32_native{1.0 / 3.0}) == 1.0f / 3.0f;
  ok = ok && static_cast<float>(fp32_native{3}) == 3.0f;

  // Mixed arithmetic and comparison take a scalar operand of either width, and stay in
  // fp_custom rather than promoting out of it.
  const fp32_native three{3};
  ok = ok && static_cast<float>(three + 1.0f) == 4.0f;
  ok = ok && static_cast<float>(1.0 + three) == 4.0f;
  ok = ok && static_cast<float>(three * 2) == 6.0f;
  ok = ok && three > 2.0f && three < 4;
  ok = ok && ::cuda::std::is_same_v<decltype(three + 1.0f), fp32_native>;
  ok = ok && ::cuda::std::is_same_v<decltype(1.0 * three), fp32_native>;

  return ok;
}

// === agreement with the same format over `double` ===
// A format that fits in binary32 can be emulated over either base type, and for a value the
// base type holds exactly the reduction lands in the same place: the rounding position sits
// the same distance from the top of the mantissa, and the bits below it are the same.
TEST_HOST_DEVICE_FUNC bool test_cross_base_agreement()
{
  using fp64_tf32 = cudax::fp64_custom<8, 10>;

  bool ok = true;

  const float values[] = {1.0f / 3.0f, 1.1234567f, 0.1f, 12345.678f, -7.0f / 9.0f, 0x1p-30f};
  for (const float v : values)
  {
    const float over_float   = static_cast<float>(fp32_tf32{v} + fp32_tf32{0.0f});
    const double over_double = static_cast<double>(fp64_tf32{static_cast<double>(v)} + fp64_tf32{0.0});
    ok                       = ok && static_cast<double>(over_float) == over_double;
  }

  return ok;
}

// A qualified cuda::std::sqrt or cuda::std::fma call suppresses ADL, and a plain float
// operand in an fma makes ::fma(float, float, float) viable. Both must still reduce rather
// than quietly compute at full binary32 precision.
TEST_HOST_DEVICE_FUNC bool test_standard_spellings()
{
  bool ok = true;

  const fp32_tf32 two{2.0f};
  const float reduced_root = static_cast<float>(sqrt(two));
  ok                       = ok && static_cast<float>(::cuda::std::sqrt(two)) == reduced_root;
  ok                       = ok && reduced_root != ::cuda::std::sqrt(2.0f);

  // A tiny third operand that only the unreduced mantissa could keep.
  const fp32_tf32 a{1.0f + 0x1p-13f}, b{1.0f}, zero{0.0f};
  const float reduced_fma = static_cast<float>(fma(a, b, zero));
  ok                      = ok && static_cast<float>(::cuda::std::fma(a, b, zero)) == reduced_fma;
  ok                      = ok && static_cast<float>(fma(a, b, 0.0f)) == reduced_fma;
  ok                      = ok && static_cast<float>(::cuda::std::fma(a, b, 0.0f)) == reduced_fma;
  ok                      = ok && reduced_fma == 1.0f && ::cuda::std::fma(1.0f + 0x1p-13f, 1.0f, 0.0f) != 1.0f;

  return ok;
}

TEST_HOST_DEVICE_FUNC void test()
{
  assert(test_native_matches_float());
  assert(test_mantissa_reduction());
  assert(test_exponent_reduction());
  assert(test_power_of_two());
  assert(test_type_surface());
  assert(test_conversions());
  assert(test_cross_base_agreement());
  assert(test_standard_spellings());
}

#if _CCCL_CUDA_COMPILATION()
// Runtime sizes, which fp_custom_dynamic_size selects, live in a device variable rather than
// in the type -- one variable per base type, which is what this section is about.
using fp32_dynamic = cudax::fp32_custom<cudax::fp_custom_dynamic_size, cudax::fp_custom_dynamic_size>;

// Reports both what the arithmetic did with the current sizes and what the sizes are, so a
// stale device copy would be visible either way.
__global__ void dynamic_size_kernel(float* sum, int* mant_size)
{
  const fp32_dynamic one{1.0f}, tiny{0x1p-13f};
  *sum       = static_cast<float>(one + tiny);
  *mant_size = cudax::fp_custom_get_device_mantissa_size<float>();
}

// The device-side setter, which is what a JIT-compiled program has instead of the host one.
// One block, so nothing else is reading the size while thread 0 writes it.
__global__ void device_set_size_kernel(int new_size, int* observed)
{
  cudax::fp_custom_set_device_mantissa_size<float>(new_size);
  __syncthreads();
  *observed = cudax::fp_custom_get_device_mantissa_size<float>();
}
#endif // _CCCL_CUDA_COMPILATION()

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
// The float sizes are separate state from the double ones, so setting either must leave the
// other at its own base type's native values.
void test_runtime_sizes(cuda::stream_ref stream)
{
  float* sum            = nullptr;
  int* mant_size        = nullptr;
  const float sum_full  = 1.0f + 0x1p-13f;
  assert(cudaMallocManaged(&sum, sizeof(float)) == cudaSuccess);
  assert(cudaMallocManaged(&mant_size, sizeof(int)) == cudaSuccess);

  // Untouched, the sizes are binary32's, so the small term survives.
  assert(cudax::fp_custom_get_device_mantissa_size<float>(stream) == 23);
  assert(cudax::fp_custom_get_device_exponent_size<float>(stream) == 8);

  dynamic_size_kernel<<<1, 1, 0, stream.get()>>>(sum, mant_size);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  assert(*sum == sum_full);
  assert(*mant_size == 23);

  // 10 bits cannot hold a term 13 binades down, and the kernel needs no synchronization to
  // see the new size: the copy is ahead of it on the stream.
  cudax::fp_custom_set_device_mantissa_size<float>(10, stream);
  dynamic_size_kernel<<<1, 1, 0, stream.get()>>>(sum, mant_size);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  assert(*sum == 1.0f);
  assert(*mant_size == 10);
  assert(cudax::fp_custom_get_device_mantissa_size<float>(stream) == 10);

  // The exponent is the other axis, and independent.
  cudax::fp_custom_set_device_exponent_size<float>(5, stream);
  assert(cudax::fp_custom_get_device_exponent_size<float>(stream) == 5);
  assert(cudax::fp_custom_get_device_mantissa_size<float>(stream) == 10);

  // The double sizes are separate state, which the float writes must not have touched, and
  // so is the host copy of the float ones.
  assert(cudax::fp_custom_get_device_mantissa_size(stream) == 52);
  assert(cudax::fp_custom_get_device_exponent_size(stream) == 11);
  assert(cudax::fp_custom_get_host_mantissa_size<float>() == 23);
  assert(cudax::fp_custom_get_host_exponent_size<float>() == 8);

  // A write from device code reaches the same variable the host accessors see.
  int* observed = nullptr;
  assert(cudaMallocManaged(&observed, sizeof(int)) == cudaSuccess);
  device_set_size_kernel<<<1, 32, 0, stream.get()>>>(16, observed);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  assert(*observed == 16);
  assert(cudax::fp_custom_get_device_mantissa_size<float>(stream) == 16);

  // Leave the native format behind for anything that runs later.
  cudax::fp_custom_set_device_mantissa_size<float>(23, stream);
  cudax::fp_custom_set_device_exponent_size<float>(8, stream);
  assert(cudax::fp_custom_get_device_mantissa_size<float>(stream) == 23);
  assert(cudax::fp_custom_get_device_exponent_size<float>(stream) == 8);

  assert(cudaFree(sum) == cudaSuccess);
  assert(cudaFree(mant_size) == cudaSuccess);
  assert(cudaFree(observed) == cudaSuccess);
}
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

#if !_CCCL_COMPILER(NVRTC)
// The host sizes, which are the only ones a host-only compilation has, and which each base
// type again holds separately.
void test_host_runtime_sizes()
{
  using fp32_dynamic_host = cudax::fp32_custom<cudax::fp_custom_dynamic_size, cudax::fp_custom_dynamic_size>;
  using fp64_dynamic_host = cudax::fp64_custom<cudax::fp_custom_dynamic_size, cudax::fp_custom_dynamic_size>;

  const fp32_dynamic_host one32{1.0f}, tiny32{0x1p-13f};
  const fp64_dynamic_host one64{1.0}, tiny64{0x1p-13};

  assert(cudax::fp_custom_get_host_mantissa_size<float>() == 23);
  assert(static_cast<float>(one32 + tiny32) == 1.0f + 0x1p-13f);

  cudax::fp_custom_set_host_mantissa_size<float>(10);
  assert(cudax::fp_custom_get_host_mantissa_size<float>() == 10);
  assert(static_cast<float>(one32 + tiny32) == 1.0f);

  // The double sizes are untouched, so the same term still survives there.
  assert(cudax::fp_custom_get_host_mantissa_size() == 52);
  assert(static_cast<double>(one64 + tiny64) == 1.0 + 0x1p-13);

  // The exponent is the other axis: 5 bits keep 1.0 but flush 2^-13 * 2^-13.
  cudax::fp_custom_set_host_exponent_size<float>(5);
  assert(cudax::fp_custom_get_host_exponent_size<float>() == 5);
  assert(cudax::fp_custom_get_host_exponent_size() == 11);
  assert(static_cast<float>(fp32_dynamic_host{0x1p-20f} * one32) == 0.0f);

  // Leave the native format behind for anything that runs later.
  cudax::fp_custom_set_host_mantissa_size<float>(23);
  cudax::fp_custom_set_host_exponent_size<float>(8);
  assert(static_cast<float>(one32 + tiny32) == 1.0f + 0x1p-13f);
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  test();

#if !_CCCL_COMPILER(NVRTC)
  // force_include.h runs this main on the host and then inside a kernel; the host sizes only
  // exist in the host pass, so NV_IS_HOST selects the driver, not the code tested.
  NV_IF_TARGET(NV_IS_HOST, (test_host_runtime_sizes();))
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST,
               (cudaStream_t raw_stream = nullptr; //
                assert(cudaStreamCreate(&raw_stream) == cudaSuccess);
                test_runtime_sizes(cuda::stream_ref{raw_stream});
                assert(cudaStreamDestroy(raw_stream) == cudaSuccess);))
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

  return 0;
}
