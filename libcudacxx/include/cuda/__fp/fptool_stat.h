//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPTOOL_STAT_H
#define _CUDA___FP_FPTOOL_STAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! @file fptool_stat.h
//! @brief fpmp2_stat - a statistics-collecting drop-in replacement for fpmp2
//!
//! `fpmp2_stat<_FpType, _TypeAcc>` holds an `fpmp2<_FpType, _TypeAcc>` and mirrors its
//! arithmetic API. Results are bit-identical to the wrapped type: the wrapper only
//! observes them, so swapping `fp32mp2` for `fp32mp2_stat` never changes what a program
//! computes.
//!
//! ## Quick Start
//!
//! ```cpp
//! #include <cuda/fptool>
//!
//! using namespace cuda::experimental;
//!
//! using Real = fp32mp2_stat; // instead of fp32mp2
//!
//! fpmp2_stat_reset_device_data();  // clear the counters
//! my_kernel<<<blocks, threads>>>(); // run the region of interest
//! cudaDeviceSynchronize();
//!
//! fpmp2_stat_data stats{};
//! fpmp2_stat_read_device_data(&stats);
//! printf("%llu adds, %llu muls\n", stats.add_count, stats.mul_count);
//! ```
//!
//! ## What Is Collected
//!
//! Each binary `+`, `-`, `*`, `/` - including the compound assignments, `++`, `--`, the
//! mixed value/scalar overloads and `atomicAdd`/`atomicSub` - increments its counter and
//! summarizes its two operands and its result into `fpmp2_stat_data::arg[0]`, `arg[1]`
//! and `result`. A summary (`fpmp2_stat_value`) records the exponent range of the `hi`
//! limb, how often the value or its `lo` limb was zero, how often infinities, NaNs and
//! subnormals appeared, and the range of the gap between the `hi` and `lo` limbs, which tells
//! how much of the double-word precision the computation actually uses: the gap is 0 or 1 for
//! a normalized pair, negative where the limbs overlap and larger where precision is held in
//! reserve.
//!
//! Each of the three slots receives exactly one value per counted operation, so
//! `ops_count` is the total to divide by when turning a count into a share, as in
//! `zero_lo_count / ops_count`.
//!
//! `sqrt`, `rsqrt`, `fma`, `mad`, `renormalize` and the math functions from
//! `<cuda/fptool_math>` are not counted: they are composites whose internal operations
//! would swamp the counters. `arg[2]` is reserved for a future ternary operation.
//!
//! ## Collection Is Device-Only
//!
//! The counters live in device memory and are updated with atomics, so only work that
//! runs on the GPU is observed. The very same code compiles and runs on the host, where
//! the wrapper is a transparent pass-through and no counters are gathered.
//!
//! | Function                                        | Description                             |
//! |-------------------------------------------------|-----------------------------------------|
//! | `fpmp2_stat_reset_device_data()`                | Clear counters, arm the range sentinels |
//! | `fpmp2_stat_read_device_data(fpmp2_stat_data*)` | Copy the record back to the host        |
//!
//! Both return the `cudaError_t` of the underlying symbol copy, and both are host-only,
//! so they do not exist under NVRTC, whose translation unit has no host side.
//!
//! The record lives in one program-wide copy shared by all translation units. On the
//! device that sharing needs relocatable device code (`-rdc=true`); in whole-program
//! mode each translation unit gets its own device copy, and a reset or read then only
//! sees the copy belonging to its own translation unit.
//!
//! @note The limb gap sample (`min_hi_lo_gap_sample_hi` / `_lo`) is best-effort
//! under concurrency: a thread with a larger gap never overwrites it, but two threads
//! lowering the minimum at once may leave the sample of either one.
//! @note Instrumentation costs a handful of atomics per operation, so a `_stat` type is
//! meant for analysis runs rather than production ones.

#include <cuda/__fp/fpmp.h>
#include <cuda/__fp/fpmp_limits.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_CUDA_COMPILATION() && _CCCL_HOST_COMPILATION()
// Include the CUDA runtime for host-side functions like cudaMemcpyToSymbol
#  include <cuda_runtime.h>
#endif // _CCCL_CUDA_COMPILATION() && _CCCL_HOST_COMPILATION()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
// === collected record ===

//! @brief Summary of the fpmp2 values that passed through one operand or result slot
//!
//! Exponents are unbiased, as `ilogb` reports them, and are sampled from the `hi` limb
//! of finite non-zero values only. The two sentinel-initialized ranges say "no sample
//! yet" through an empty range, i.e. `min > max`; `fpmp2_stat_reset_device_data()` arms
//! them.
//!
//! The field types are the ones the device-side atomics take, which is also what makes
//! them the portable choice here: `unsigned long long int` is what `atomicAdd` accepts
//! and what `%llu` prints, and it is at least 64 bits by the standard and exactly 64 on
//! every platform CUDA supports. (`uint64_t` would not do: it is `unsigned long` on LP64
//! platforms, a distinct type for which no `atomicAdd` overload exists.) The exponents
//! are `int`, the type of `ilogb`, of `numeric_limits<>::min_exponent` and of
//! `atomicMin`/`atomicMax`, and identical to `int32_t` wherever CCCL builds.
struct fpmp2_stat_value
{
  //! @brief Largest exponent of `hi` seen, `numeric_limits<int>::min()` until a value is
  //! sampled
  int max_exp;
  //! @brief Smallest exponent of `hi` seen, `numeric_limits<int>::max()` until a value is
  //! sampled
  int min_exp;
  //! @brief Values whose `hi` and `lo` limbs were both zero
  unsigned long long int zero_count;
  //! @brief Finite values with a non-zero `hi` and a zero `lo`, i.e. no extra precision
  unsigned long long int zero_lo_count;
  //! @brief Values with an infinite `hi` or `lo`, counted once per value
  unsigned long long int inf_count;
  //! @brief Values with a NaN `hi` or `lo`, counted once per value
  unsigned long long int nan_count;
  //! @brief Values whose limbs were infinities of opposite signs, whose sum is a NaN
  unsigned long long int infnan_count;
  //! @brief Values with a subnormal `hi` or `lo`, counted once per value
  //!
  //! A non-zero count says the computation reached the bottom of the exponent range, where
  //! precision degrades gradually. It matters for two reasons beyond the usual performance
  //! concern: the `low` and `mid` accuracy levels support the normal range only, so any
  //! subnormal means such a configuration is being used outside its domain; and a subnormal
  //! `lo` costs the pair its tail, since `lo` normally sits `digits` binades below `hi` and
  //! cannot go there once it is subnormal.
  //!
  //! `lo` reaches that limit long before `hi` does, so this counter usually reports the
  //! precision loss rather than a subnormal result.
  unsigned long long int denorm_count;
  //! @brief Largest gap between the limbs seen, `numeric_limits<int>::min()` until a value
  //! is sampled
  //!
  //! The gap is `exp(hi) - exp(lo) - digits`, i.e. the raw exponent difference with the
  //! mantissa width of the base type taken out, so it says how far the pair is from being
  //! tightly normalized rather than how wide the format is:
  //!
  //! - `0` or `1`: normalized. A normalized `lo` is at most half an ulp of `hi`, which puts
  //!   its exponent exactly `digits` places below, so no bits are wasted and none overlap.
  //! - negative: the limbs overlap, so the pair carries fewer significant bits than its two
  //!   limbs suggest. Only the `low` accuracy level, which skips renormalization, produces
  //!   this.
  //! - much greater than `1`: `lo` sits well below half an ulp of `hi`, so the second limb
  //!   is barely carrying anything and the pair has that many bits of accuracy in reserve.
  //!
  //! Subnormal limbs are measured by their leading significant bit, as `ilogb` would report
  //! them, so a subnormal `lo` does not fake an overlap. It does mean the pair lost part of
  //! its tail, which `denorm_count` reports.
  int max_hi_lo_gap;
  //! @brief Smallest gap between the limbs seen, `numeric_limits<int>::max()` until a value
  //! is sampled
  //!
  //! Defined as for `max_hi_lo_gap`. This is the interesting end of the range: a negative
  //! minimum reports that unnormalized pairs occurred, and how badly the limbs overlapped.
  int min_hi_lo_gap;
  //! @brief `hi` limb of a value that lowered `min_hi_lo_gap`, for inspection
  double min_hi_lo_gap_sample_hi;
  //! @brief `lo` limb of the same value
  double min_hi_lo_gap_sample_lo;
};

//! @brief The record a `_stat` type fills in on the device
//!
//! Counters are shared by every `fpmp2_stat` instantiation, so a program that uses more
//! than one of them sees their operations summed.
struct fpmp2_stat_data
{
  //! @brief Instrumented binary operations, the sum of the four counters below
  unsigned long long int ops_count;
  //! @brief Instrumented additions, including `+=` and `++`
  unsigned long long int add_count;
  //! @brief Instrumented subtractions, including `-=` and `--`
  unsigned long long int sub_count;
  //! @brief Instrumented multiplications, including `*=`
  unsigned long long int mul_count;
  //! @brief Instrumented divisions, including `/=`
  unsigned long long int div_count;
  //! @brief Operand summaries: `arg[0]` and `arg[1]` are the binary operands, `arg[2]`
  //! is reserved for a future ternary operation
  fpmp2_stat_value arg[3];
  //! @brief Result summary
  fpmp2_stat_value result;
};

// Which counter an instrumented operation bumps. A template argument rather than a
// runtime one, so the increment folds down to a single atomic.
enum class __fpmp2_stat_binop
{
  __add,
  __sub,
  __mul,
  __div,
};

// A slot with both ranges armed: empty ranges that the first sample replaces.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr fpmp2_stat_value __fpmp2_stat_cleared_value() noexcept
{
  fpmp2_stat_value __slot{};
  __slot.max_exp       = ::cuda::std::numeric_limits<int>::min();
  __slot.min_exp       = ::cuda::std::numeric_limits<int>::max();
  __slot.max_hi_lo_gap = ::cuda::std::numeric_limits<int>::min();
  __slot.min_hi_lo_gap = ::cuda::std::numeric_limits<int>::max();
  return __slot;
}

// A zeroed record with every slot armed, used both as the initial value of the device
// record and as the source of a reset.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr fpmp2_stat_data __fpmp2_stat_cleared_data() noexcept
{
  fpmp2_stat_data __data{};
  for (int __i = 0; __i < 3; ++__i)
  {
    __data.arg[__i] = __fpmp2_stat_cleared_value();
  }
  __data.result = __fpmp2_stat_cleared_value();
  return __data;
}

#if _CCCL_CUDA_COMPILATION()
// The record every instrumented operation updates.
//
// A variable template rather than a plain variable on purpose: a variable template has
// vague linkage, so all translation units share one copy, while a plain
// `inline _CCCL_DEVICE` variable is rejected by nvcc outside relocatable device code.
// The dummy parameter exists only to make it a template; the counters are deliberately
// shared by all instantiations of fpmp2_stat.
template <class _Void = void>
_CCCL_DEVICE fpmp2_stat_data __fpmp2_stat_device_data = __fpmp2_stat_cleared_data();
#endif // _CCCL_CUDA_COMPILATION()

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

//! @brief Clear the device record and arm its range sentinels
//!
//! Call it before the region of interest. Counting starts as soon as the next kernel
//! runs, so no synchronization beyond the usual stream ordering is needed.
//!
//! @return The `cudaError_t` of the symbol copy
_CCCL_HOST_API inline cudaError_t fpmp2_stat_reset_device_data() noexcept
{
  const fpmp2_stat_data __cleared = __fpmp2_stat_cleared_data();
  return cudaMemcpyToSymbol(__fpmp2_stat_device_data<>, &__cleared, sizeof(fpmp2_stat_data));
}

//! @brief Copy the device record to the host
//!
//! The kernels that produced the numbers must have finished, e.g. through
//! `cudaDeviceSynchronize()`.
//!
//! @param __dst Destination record
//! @return The `cudaError_t` of the symbol copy
_CCCL_HOST_API inline cudaError_t fpmp2_stat_read_device_data(fpmp2_stat_data* __dst) noexcept
{
  return cudaMemcpyFromSymbol(__dst, __fpmp2_stat_device_data<>, sizeof(fpmp2_stat_data));
}

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

// === value inspection ===

// The IEEE-754 fields of one limb, as the accumulation below needs them.
struct __fpmp2_stat_parts
{
  // Unbiased exponent, i.e. the encoded field minus the bias. Zero and subnormals share
  // the lowest value, -bias, and infinity and NaN the highest one. Use it to classify a
  // limb, not to measure it.
  int __exp;
  // What ilogb would report: the position of the leading significant bit. Identical to
  // __exp for normal values, but a subnormal's encoded field is pinned at its minimum and
  // says nothing about the magnitude, so the leading mantissa bit has to be located.
  // Meaningless for zero, infinity and NaN, which the callers exclude.
  int __exp_ilogb;
  bool __mant_is_zero;
  bool __exp_is_max;
  bool __sign;
};

// Splits a limb without touching the FPU, so the same code serves float and double.
// The two constants come from numeric_limits: digits is the mantissa size including the
// implicit bit, and max_exponent is the bias plus one.
template <class _FpType>
[[nodiscard]] _CCCL_HOST_DEVICE_API __fpmp2_stat_parts __fpmp2_stat_split(_FpType __x) noexcept
{
  using _UInt = ::cuda::std::
    conditional_t<sizeof(_FpType) == sizeof(::cuda::std::uint32_t), ::cuda::std::uint32_t, ::cuda::std::uint64_t>;

  constexpr int __mant_size = ::cuda::std::numeric_limits<_FpType>::digits - 1;
  constexpr int __bias      = ::cuda::std::numeric_limits<_FpType>::max_exponent - 1;
  constexpr int __exp_max   = 2 * __bias + 1;

  const _UInt __bits = ::cuda::std::bit_cast<_UInt>(__x);
  const int __exp    = static_cast<int>((__bits >> __mant_size) & static_cast<_UInt>(__exp_max));
  const _UInt __mant = __bits & ((_UInt{1} << __mant_size) - _UInt{1});

  // A subnormal is `mant * 2^(1 - bias - mant_size)`, so its leading bit sits at
  // `msb(mant) + 1 - bias - mant_size`. The count is computed unconditionally, which is one
  // instruction, rather than behind a branch that would practically never be taken.
  const int __msb        = static_cast<int>(8 * sizeof(_UInt)) - 1 - ::cuda::std::countl_zero(__mant);
  const bool __is_denorm = __exp == 0 && __mant != _UInt{0};
  const int __exp_ilogb  = __is_denorm ? (__msb + 1 - __bias - __mant_size) : (__exp - __bias);

  return {__exp - __bias,
          __exp_ilogb,
          __mant == _UInt{0},
          __exp == __exp_max,
          (__bits >> (8 * sizeof(_UInt) - 1)) != _UInt{0}};
}

#if _CCCL_CUDA_COMPILATION()
// Folds one fpmp2 value into a slot. Device-only: the updates are atomic because every
// thread of a grid writes the same slot.
template <class _FpType>
_CCCL_DEVICE_API inline void __fpmp2_stat_accumulate(fpmp2_stat_value* __slot, _FpType __hi, _FpType __lo) noexcept
{
  const __fpmp2_stat_parts __p_hi = __fpmp2_stat_split(__hi);
  const __fpmp2_stat_parts __p_lo = __fpmp2_stat_split(__lo);

  constexpr int __exp_min = 1 - ::cuda::std::numeric_limits<_FpType>::max_exponent;

  const bool __hi_is_zero = __p_hi.__exp == __exp_min && __p_hi.__mant_is_zero;
  const bool __lo_is_zero = __p_lo.__exp == __exp_min && __p_lo.__mant_is_zero;
  const bool __is_zero    = __hi_is_zero && __lo_is_zero;
  const bool __is_nan     = (__p_hi.__exp_is_max && !__p_hi.__mant_is_zero) //
                     || (__p_lo.__exp_is_max && !__p_lo.__mant_is_zero);
  const bool __is_inf = (__p_hi.__exp_is_max && __p_hi.__mant_is_zero) //
                     || (__p_lo.__exp_is_max && __p_lo.__mant_is_zero);
  const bool __is_finite = !__p_hi.__exp_is_max && !__p_lo.__exp_is_max;
  // A subnormal is the minimum exponent field with a non-zero mantissa, which is what the
  // zero tests above rule out. Either limb counts.
  const bool __is_denorm = (__p_hi.__exp == __exp_min && !__p_hi.__mant_is_zero) //
                        || (__p_lo.__exp == __exp_min && !__p_lo.__mant_is_zero);

  if (__is_nan)
  {
    ::atomicAdd(&__slot->nan_count, 1ull);
  }
  if (__is_inf)
  {
    ::atomicAdd(&__slot->inf_count, 1ull);
  }
  // Infinities of opposite signs: the value the pair stands for is a NaN, which usually
  // means an overflow the algorithm did not expect.
  if (__p_hi.__exp_is_max && __p_hi.__mant_is_zero && __p_lo.__exp_is_max && __p_lo.__mant_is_zero
      && __p_hi.__sign != __p_lo.__sign)
  {
    ::atomicAdd(&__slot->infnan_count, 1ull);
  }
  if (__is_denorm)
  {
    ::atomicAdd(&__slot->denorm_count, 1ull);
  }
  if (__is_zero)
  {
    ::atomicAdd(&__slot->zero_count, 1ull);
  }
  if (__is_finite && !__hi_is_zero && __lo_is_zero)
  {
    ::atomicAdd(&__slot->zero_lo_count, 1ull);
  }

  if (__is_finite && !__is_zero)
  {
    ::atomicMax(&__slot->max_exp, __p_hi.__exp_ilogb);
    ::atomicMin(&__slot->min_exp, __p_hi.__exp_ilogb);

    if (!__lo_is_zero)
    {
      // Measured against a tightly normalized pair rather than as a raw exponent
      // difference: a normalized `lo` is at most half an ulp of `hi`, which puts its
      // exponent `digits` places below, so subtracting `digits` makes 0 the tight case and
      // a negative value an overlap. See the field documentation.
      constexpr int __digits = ::cuda::std::numeric_limits<_FpType>::digits;

      const int __gap      = __p_hi.__exp_ilogb - __p_lo.__exp_ilogb - __digits;
      const int __prev_min = ::atomicMin(&__slot->min_hi_lo_gap, __gap);
      ::atomicMax(&__slot->max_hi_lo_gap, __gap);

      // Best-effort sample of the tightest pair seen, see the note in the file comment.
      if (__gap < __prev_min)
      {
        __slot->min_hi_lo_gap_sample_hi = static_cast<double>(__hi);
        __slot->min_hi_lo_gap_sample_lo = static_cast<double>(__lo);
      }
    }
  }
}

// Records one instrumented binary operation: the counters plus the three value slots.
template <__fpmp2_stat_binop _Kind, class _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_DEVICE_API inline void __fpmp2_stat_note_binop(
  const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y, const fpmp2<_FpType, _TypeAcc>& __r) noexcept
{
  fpmp2_stat_data& __data = __fpmp2_stat_device_data<>;

  ::atomicAdd(&__data.ops_count, 1ull);
  if constexpr (_Kind == __fpmp2_stat_binop::__add)
  {
    ::atomicAdd(&__data.add_count, 1ull);
  }
  else if constexpr (_Kind == __fpmp2_stat_binop::__sub)
  {
    ::atomicAdd(&__data.sub_count, 1ull);
  }
  else if constexpr (_Kind == __fpmp2_stat_binop::__mul)
  {
    ::atomicAdd(&__data.mul_count, 1ull);
  }
  else
  {
    ::atomicAdd(&__data.div_count, 1ull);
  }

  __fpmp2_stat_accumulate(&__data.arg[0], __x.hi(), __x.lo());
  __fpmp2_stat_accumulate(&__data.arg[1], __y.hi(), __y.lo());
  __fpmp2_stat_accumulate(&__data.result, __r.hi(), __r.lo());
}
#endif // _CCCL_CUDA_COMPILATION()

// === main class definition ===

//! @brief Statistics-collecting drop-in replacement for `fpmp2`
//!
//! Wraps an `fpmp2<_FpType, _TypeAcc>` and mirrors its arithmetic, comparison and
//! conversion API. Every instrumented operation is computed by the wrapped type first
//! and only then observed, so results are bit-identical to the plain type.
//!
//! ## Memory Layout
//! Same size and alignment as the wrapped `fpmp2`, and trivially copyable, so arrays
//! and kernel arguments can be reinterpreted between the two.
//!
//! ## Interoperability
//! Conversion to and from the wrapped `fpmp2` of the same accuracy is implicit, and
//! mixing the two in one expression yields an instrumented result. Accuracy levels do
//! not mix implicitly, exactly as they do not for `fpmp2` itself: convert explicitly,
//! e.g. `fp32mp2_stat_low(x)`.
//!
//! @tparam _FpType Limb type: `float` for double-float, `double` for double-double
//! @tparam _TypeAcc Arithmetic accuracy level, see `fpmp2_accuracy`
//!
//! @note Statistics are collected on the device only; on the host the wrapper is a
//! transparent pass-through.
template <class _FpType, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
class alignas(alignof(fpmp2<_FpType, _TypeAcc>)) fpmp2_stat
{
public:
  //! @brief The wrapped type, whose results this type reproduces exactly
  using base_type = fpmp2<_FpType, _TypeAcc>;

private:
  base_type __stat_v_;

  // Instrumentation hook. Compiled out on the host, where there is no record to update.
  template <__fpmp2_stat_binop _Kind>
  _CCCL_HOST_DEVICE_API static void __trace(const base_type& __x, const base_type& __y, const base_type& __r) noexcept
  {
    NV_IF_TARGET(NV_IS_DEVICE, (__fpmp2_stat_note_binop<_Kind>(__x, __y, __r);))
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr fpmp2_stat __from_base(const base_type& __v) noexcept
  {
    return fpmp2_stat(__v);
  }

public:
  //! @brief Read the high limb
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _FpType hi() const noexcept
  {
    return __stat_v_.hi();
  }
  //! @brief Read the low limb
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _FpType lo() const noexcept
  {
    return __stat_v_.lo();
  }
  //! @brief Read the high limb of a volatile object
  [[nodiscard]] _CCCL_HOST_DEVICE_API _FpType hi() const volatile noexcept
  {
    return __stat_v_.hi();
  }
  //! @brief Read the low limb of a volatile object
  [[nodiscard]] _CCCL_HOST_DEVICE_API _FpType lo() const volatile noexcept
  {
    return __stat_v_.lo();
  }

  //! @brief Access the wrapped value, e.g. to hand it to an API taking `fpmp2`
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const base_type& as_fpmp2() const noexcept
  {
    return __stat_v_;
  }
  //! @brief Access the wrapped value for modification, which is not instrumented
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr base_type& as_fpmp2() noexcept
  {
    return __stat_v_;
  }

  // === constructors ===

  //! @brief Default constructor, leaving the value uninitialized like `fpmp2`
  _CCCL_HIDE_FROM_ABI fpmp2_stat() = default;

  //! @brief Construct from the two limbs directly
  _CCCL_HOST_DEVICE_API constexpr fpmp2_stat(_FpType __hi, _FpType __lo) noexcept
      : __stat_v_{__hi, __lo}
  {}

  //! @brief Copy constructor, defaulted so the type stays trivially copyable
  //! @note NVCC implicitly makes defaulted special members __host__ __device__
  _CCCL_HIDE_FROM_ABI fpmp2_stat(const fpmp2_stat&) = default;
  //! @brief Copy assignment, defaulted so the type stays trivially copyable
  _CCCL_HIDE_FROM_ABI constexpr fpmp2_stat& operator=(const fpmp2_stat&) = default;

  // Volatile support, mirroring fpmp2: storage only, i.e. load, store and a
  // limb-preserving round-trip. Each overload is wrapped in a dummy template so that
  // the C++ standard does not consider it a copy constructor or copy assignment
  // operator, which preserves trivial copyability.

  //! @brief Copy constructor from a volatile object
  template <class _Dummy = void>
  _CCCL_HOST_DEVICE_API fpmp2_stat(const volatile fpmp2_stat& __other) noexcept
      : __stat_v_{__other.hi(), __other.lo()}
  {}

  //! @brief Assignment to a volatile object
  //! @note Returns void to avoid the C++20 deprecation of a volatile return type
  template <class _Dummy = void>
  _CCCL_HOST_DEVICE_API void operator=(const fpmp2_stat& __other) volatile noexcept
  {
    __stat_v_ = __other.__stat_v_;
  }

  //! @brief Assignment from a volatile object
  template <class _Dummy = void>
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator=(const volatile fpmp2_stat& __other) noexcept
  {
    __stat_v_ = base_type{__other.hi(), __other.lo()};
    return *this;
  }

  //! @brief Implicit conversion from the wrapped type, the symmetric counterpart of
  //! `operator base_type()`
  _CCCL_HOST_DEVICE_API constexpr fpmp2_stat(const base_type& __other) noexcept
      : __stat_v_{__other}
  {}

  //! @brief Explicit conversion from another accuracy level, which copies the limbs
  //! without renormalizing, as the `fpmp2` counterpart does
  _CCCL_TEMPLATE(fpmp2_accuracy _TypeAcc2)
  _CCCL_REQUIRES((_TypeAcc2 != _TypeAcc))
  _CCCL_HOST_DEVICE_API constexpr explicit fpmp2_stat(const fpmp2<_FpType, _TypeAcc2>& __other) noexcept
      : __stat_v_{base_type{__other}}
  {}

  //! @brief Explicit conversion from a `_stat` type of another accuracy level
  _CCCL_TEMPLATE(fpmp2_accuracy _TypeAcc2)
  _CCCL_REQUIRES((_TypeAcc2 != _TypeAcc))
  _CCCL_HOST_DEVICE_API constexpr explicit fpmp2_stat(const fpmp2_stat<_FpType, _TypeAcc2>& __other) noexcept
      : __stat_v_{base_type{__other.as_fpmp2()}}
  {}

  //! @brief Implicit conversion from a single limb, leaving `lo` zero
  _CCCL_HOST_DEVICE_API constexpr fpmp2_stat(_FpType __f) noexcept
      : __stat_v_{__f}
  {}

  //! @brief Construct a double-float from a double, splitting it into the two limbs
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_HOST_DEVICE_API constexpr _CCCL_FPMP_EXPLICIT fpmp2_stat(double __d) noexcept
      : __stat_v_{__d}
  {}

#if _CCCL_FPMP_FP128_ENABLE == 1
  //! @brief Construct a double-double from a binary128 value
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  _CCCL_FPMP_FP128_API constexpr _CCCL_FPMP_EXPLICIT fpmp2_stat(__fpmp_fp128 __d) noexcept
      : __stat_v_{__d}
  {}
  //! @brief Explicit conversion of a double-double to binary128
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  [[nodiscard]] _CCCL_FPMP_FP128_API explicit operator __fpmp_fp128() const noexcept
  {
    return static_cast<__fpmp_fp128>(__stat_v_);
  }
  // A double-float has no binary128 interchange in either direction, deleted rather
  // than absent for the same reason as in fpmp2: the diagnostic then names the rule.
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_FPMP_FP128_API _CCCL_FPMP_EXPLICIT fpmp2_stat(__fpmp_fp128) = delete;
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  _CCCL_FPMP_FP128_API explicit operator __fpmp_fp128() const = delete;
#endif // _CCCL_FPMP_FP128_ENABLE == 1

  //! @brief Construct from any standard integer type
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2_stat(_Tp __i) noexcept
      : __stat_v_{__i}
  {}

  //! @brief Construct from `bool` or a character type, which `__cccl_is_integer_v`
  //! excludes but `double` accepts
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND(!::cuda::std::__cccl_is_integer_v<_Tp>))
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2_stat(_Tp __i) noexcept
      : __stat_v_{__i}
  {}

#if _CCCL_HAS_INT128()
  // Deleted for the same reason as in fpmp2: a 128-bit integer would silently truncate.
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2_stat(__int128_t)  = delete;
  _CCCL_HOST_DEVICE_API _CCCL_FPMP_EXPLICIT fpmp2_stat(__uint128_t) = delete;
#endif // _CCCL_HAS_INT128()

  // === conversions out ===

  //! @brief Implicit conversion to the wrapped type
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr operator base_type() const noexcept
  {
    return __stat_v_;
  }

  //! @brief Conversion to the wrapped type from a volatile object
  [[nodiscard]] _CCCL_HOST_DEVICE_API operator base_type() const volatile noexcept
  {
    return base_type{hi(), lo()};
  }

  //! @brief Explicit conversion to the wrapped type of another accuracy level
  _CCCL_TEMPLATE(fpmp2_accuracy _TypeAcc2)
  _CCCL_REQUIRES((_TypeAcc2 != _TypeAcc))
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit constexpr operator fpmp2<_FpType, _TypeAcc2>() const noexcept
  {
    return fpmp2<_FpType, _TypeAcc2>{__stat_v_};
  }

  // Conversion to double follows the wrapped type: implicit out of a double-float,
  // where the pair sums exactly, explicit out of a double-double, where the low limb is
  // dropped. See the corresponding comment in fpmp.h.
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API operator double() const noexcept
  {
    return static_cast<double>(__stat_v_);
  }
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp32_v<_Up>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API operator double() const volatile noexcept
  {
    return static_cast<double>(base_type{hi(), lo()});
  }
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator double() const noexcept
  {
    return static_cast<double>(__stat_v_);
  }
  _CCCL_TEMPLATE(class _Up = _FpType)
  _CCCL_REQUIRES(__fpmp2_is_fp64_v<_Up>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator double() const volatile noexcept
  {
    return static_cast<double>(base_type{hi(), lo()});
  }

  //! @brief Explicit conversion to float
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator float() const noexcept
  {
    return static_cast<float>(__stat_v_);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator float() const volatile noexcept
  {
    return static_cast<float>(base_type{hi(), lo()});
  }

  //! @brief Explicit conversion to any standard integer type
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator _Tp() const noexcept
  {
    return static_cast<_Tp>(__stat_v_);
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API explicit operator _Tp() const volatile noexcept
  {
    return static_cast<_Tp>(base_type{hi(), lo()});
  }

#if _CCCL_HAS_INT128()
  _CCCL_HOST_DEVICE_API explicit operator __int128_t() const           = delete;
  _CCCL_HOST_DEVICE_API explicit operator __uint128_t() const          = delete;
  _CCCL_HOST_DEVICE_API explicit operator __int128_t() const volatile  = delete;
  _CCCL_HOST_DEVICE_API explicit operator __uint128_t() const volatile = delete;
#endif // _CCCL_HAS_INT128()

  // === arithmetic ===

  //! @brief Renormalize the pair, which is not an instrumented operation
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat renormalize(const fpmp2_stat& __x) noexcept
  {
    return __from_base(renormalize(__x.__stat_v_));
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator+(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    const base_type __r = __x.__stat_v_ + __y.__stat_v_;
    __trace<__fpmp2_stat_binop::__add>(__x.__stat_v_, __y.__stat_v_, __r);
    return __from_base(__r);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator-(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    const base_type __r = __x.__stat_v_ - __y.__stat_v_;
    __trace<__fpmp2_stat_binop::__sub>(__x.__stat_v_, __y.__stat_v_, __r);
    return __from_base(__r);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator*(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    const base_type __r = __x.__stat_v_ * __y.__stat_v_;
    __trace<__fpmp2_stat_binop::__mul>(__x.__stat_v_, __y.__stat_v_, __r);
    return __from_base(__r);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator/(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    const base_type __r = __x.__stat_v_ / __y.__stat_v_;
    __trace<__fpmp2_stat_binop::__div>(__x.__stat_v_, __y.__stat_v_, __r);
    return __from_base(__r);
  }

  //! @brief Negation, which is exact and therefore not instrumented
  [[nodiscard]] _CCCL_HOST_DEVICE_API fpmp2_stat operator-() const noexcept
  {
    return __from_base(-__stat_v_);
  }

  // Mixing the wrapped type into an expression. Without these the operand would have to
  // convert - either way round - and the two conversions would be equally good, so the
  // call would be ambiguous. Being exact matches, they also keep the result
  // instrumented.
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator+(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x + __from_base(__y);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator+(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __from_base(__x) + __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator-(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x - __from_base(__y);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator-(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __from_base(__x) - __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator*(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x * __from_base(__y);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator*(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __from_base(__x) * __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator/(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x / __from_base(__y);
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator/(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __from_base(__x) / __y;
  }

  // === compound assignment ===

  _CCCL_HOST_DEVICE_API fpmp2_stat& operator+=(const fpmp2_stat& __other) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ += __other.__stat_v_;
    __trace<__fpmp2_stat_binop::__add>(__x, __other.__stat_v_, __stat_v_);
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator-=(const fpmp2_stat& __other) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ -= __other.__stat_v_;
    __trace<__fpmp2_stat_binop::__sub>(__x, __other.__stat_v_, __stat_v_);
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator*=(const fpmp2_stat& __other) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ *= __other.__stat_v_;
    __trace<__fpmp2_stat_binop::__mul>(__x, __other.__stat_v_, __stat_v_);
    return *this;
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator/=(const fpmp2_stat& __other) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ /= __other.__stat_v_;
    __trace<__fpmp2_stat_binop::__div>(__x, __other.__stat_v_, __stat_v_);
    return *this;
  }

  //! @brief Add a single limb, mirroring the `fpmp2` overload that skips building a pair
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator+=(const _FpType __c) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ += __c;
    __trace<__fpmp2_stat_binop::__add>(__x, base_type{__c}, __stat_v_);
    return *this;
  }
  //! @brief Subtract a single limb
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator-=(const _FpType __c) noexcept
  {
    const base_type __x = __stat_v_;
    __stat_v_ -= __c;
    __trace<__fpmp2_stat_binop::__sub>(__x, base_type{__c}, __stat_v_);
    return *this;
  }

  _CCCL_HOST_DEVICE_API fpmp2_stat& operator++() noexcept
  {
    return *this += _FpType(1);
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat& operator--() noexcept
  {
    return *this -= _FpType(1);
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat operator++(int) noexcept
  {
    const fpmp2_stat __old = *this;
    ++(*this);
    return __old;
  }
  _CCCL_HOST_DEVICE_API fpmp2_stat operator--(int) noexcept
  {
    const fpmp2_stat __old = *this;
    --(*this);
    return __old;
  }

  // === comparisons ===

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ == __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ != __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ < __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ > __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ <= __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const fpmp2_stat& __x, const fpmp2_stat& __y) noexcept
  {
    return __x.__stat_v_ >= __y.__stat_v_;
  }

  // Comparing against the wrapped type, for the same ambiguity reason as the arithmetic
  // overloads above.
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ == __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x == __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ != __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x != __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ < __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x < __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ > __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x > __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ <= __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x <= __y.__stat_v_;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const fpmp2_stat& __x, const base_type& __y) noexcept
  {
    return __x.__stat_v_ >= __y;
  }
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const base_type& __x, const fpmp2_stat& __y) noexcept
  {
    return __x >= __y.__stat_v_;
  }

  // === mixed arithmetic with built-in scalars ===
  // Same shape as the fpmp2 overloads: the scalar is promoted to the pair type, so
  // `2.0f * x` and `x / 3` behave as they would for the wrapped type, instrumented.

  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator+(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x) + fpmp2_stat(__y);
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator-(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x) - fpmp2_stat(__y);
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator*(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x) * fpmp2_stat(__y);
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend fpmp2_stat operator/(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x) / fpmp2_stat(__y);
  }

  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator==(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ == fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator!=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ != fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ < fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ > fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator<=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ <= fpmp2_stat(__y).__stat_v_;
  }
  _CCCL_TEMPLATE(class _T1, class _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fpmp2_stat> || ::cuda::std::is_same_v<_T2, fpmp2_stat>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend bool operator>=(const _T1& __x, const _T2& __y) noexcept
  {
    return fpmp2_stat(__x).__stat_v_ >= fpmp2_stat(__y).__stat_v_;
  }
};

// Trait: detect any specialization of fpmp2_stat<_FpType, _TypeAcc>, the counterpart of
// __fpmp_is_fpmp2_v.
template <class _Tp>
inline constexpr bool __fpmp_is_fpmp2_stat_v = false;
template <class _FpType, fpmp2_accuracy _TypeAcc>
inline constexpr bool __fpmp_is_fpmp2_stat_v<fpmp2_stat<_FpType, _TypeAcc>> = true;

// === math free functions mirroring the fpmp2 ones ===
// None of these are instrumented: they are composites, and counting the operations
// inside them would drown the counters of the surrounding algorithm.

//! @brief Square root
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
sqrt(const fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(sqrt(__x.as_fpmp2()));
}

//! @brief Reciprocal square root
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
rsqrt(const fpmp2_stat<_FpType, _TypeAcc>& __x) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(rsqrt(__x.as_fpmp2()));
}

//! @brief Fused multiply-add
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
fma(const fpmp2_stat<_FpType, _TypeAcc>& __x,
    const fpmp2_stat<_FpType, _TypeAcc>& __y,
    const fpmp2_stat<_FpType, _TypeAcc>& __z) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(fma(__x.as_fpmp2(), __y.as_fpmp2(), __z.as_fpmp2()));
}

//! @brief Multiply-add
template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
mad(const fpmp2_stat<_FpType, _TypeAcc>& __x,
    const fpmp2_stat<_FpType, _TypeAcc>& __y,
    const fpmp2_stat<_FpType, _TypeAcc>& __z) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(mad(__x.as_fpmp2(), __y.as_fpmp2(), __z.as_fpmp2()));
}

//! @brief Fused multiply-add with built-in scalars mixed in
_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(
  ((__fpmp_is_fpmp2_stat_v<_T1> || __fpmp_is_fpmp2_stat_v<_T2> || __fpmp_is_fpmp2_stat_v<_T3>)
   && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using __stat = ::cuda::std::
    conditional_t<__fpmp_is_fpmp2_stat_v<_T1>, _T1, ::cuda::std::conditional_t<__fpmp_is_fpmp2_stat_v<_T2>, _T2, _T3>>;
  return fma(__stat(__x), __stat(__y), __stat(__z));
}

//! @brief Multiply-add with built-in scalars mixed in
_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(
  ((__fpmp_is_fpmp2_stat_v<_T1> || __fpmp_is_fpmp2_stat_v<_T2> || __fpmp_is_fpmp2_stat_v<_T3>)
   && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>) ))
[[nodiscard]] _CCCL_HOST_DEVICE_API inline auto mad(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using __stat = ::cuda::std::
    conditional_t<__fpmp_is_fpmp2_stat_v<_T1>, _T1, ::cuda::std::conditional_t<__fpmp_is_fpmp2_stat_v<_T2>, _T2, _T3>>;
  return mad(__stat(__x), __stat(__y), __stat(__z));
}

#if _CCCL_CUDA_COMPILATION()
// === atomics ===
// Atomic accumulation into a shared or global value, mirroring the fpmp2 overloads and
// returning the old value as they do. These are instrumented, since they perform the
// arithmetic the counters are about, with one caveat: the result summary is the sum
// recomputed from the returned old value rather than one observed inside the atomic, so
// a value that another thread's update changed in between is summarized as this thread
// computed it.

template <class _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
atomicAdd(fpmp2_stat<_FpType, _TypeAcc>* __address, const fpmp2_stat<_FpType, _TypeAcc>& __val) noexcept
{
  const fpmp2<_FpType, _TypeAcc> __old = atomicAdd(&__address->as_fpmp2(), __val.as_fpmp2());
  __fpmp2_stat_note_binop<__fpmp2_stat_binop::__add>(__old, __val.as_fpmp2(), __old + __val.as_fpmp2());
  return fpmp2_stat<_FpType, _TypeAcc>(__old);
}

template <class _FpType, fpmp2_accuracy _TypeAcc>
_CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
atomicSub(fpmp2_stat<_FpType, _TypeAcc>* __address, const fpmp2_stat<_FpType, _TypeAcc>& __val) noexcept
{
  const fpmp2<_FpType, _TypeAcc> __old = atomicSub(&__address->as_fpmp2(), __val.as_fpmp2());
  __fpmp2_stat_note_binop<__fpmp2_stat_binop::__sub>(__old, __val.as_fpmp2(), __old - __val.as_fpmp2());
  return fpmp2_stat<_FpType, _TypeAcc>(__old);
}

// === warp shuffles ===
// Overloads of CUDA's __shfl_sync family, mirroring the fpmp2 ones so that a kernel
// written against the wrapped type keeps compiling after the swap. Thread-cooperation
// primitives, not arithmetic, so they are not instrumented.

template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
__shfl_sync(unsigned mask, const fpmp2_stat<_FpType, _TypeAcc>& var, int srcLane, int width = warpSize) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(__shfl_sync(mask, var.as_fpmp2(), srcLane, width));
}

template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc>
__shfl_xor_sync(unsigned mask, const fpmp2_stat<_FpType, _TypeAcc>& var, int laneMask, int width = warpSize) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(__shfl_xor_sync(mask, var.as_fpmp2(), laneMask, width));
}

template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc> __shfl_down_sync(
  unsigned mask, const fpmp2_stat<_FpType, _TypeAcc>& var, unsigned int delta, int width = warpSize) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(__shfl_down_sync(mask, var.as_fpmp2(), delta, width));
}

template <class _FpType, fpmp2_accuracy _TypeAcc>
[[nodiscard]] _CCCL_DEVICE_API inline fpmp2_stat<_FpType, _TypeAcc> __shfl_up_sync(
  unsigned mask, const fpmp2_stat<_FpType, _TypeAcc>& var, unsigned int delta, int width = warpSize) noexcept
{
  return fpmp2_stat<_FpType, _TypeAcc>(__shfl_up_sync(mask, var.as_fpmp2(), delta, width));
}
#endif // _CCCL_CUDA_COMPILATION()

// === type aliases ===

//! @brief Instrumented double-float, default accuracy
using fp32mp2_stat = fpmp2_stat<float, fpmp2_accuracy::def>;
//! @brief Instrumented double-float, `low` accuracy
using fp32mp2_stat_low = fpmp2_stat<float, fpmp2_accuracy::low>;
//! @brief Instrumented double-float, `mid` accuracy
using fp32mp2_stat_mid = fpmp2_stat<float, fpmp2_accuracy::mid>;
//! @brief Instrumented double-float, `high` accuracy
using fp32mp2_stat_high = fpmp2_stat<float, fpmp2_accuracy::high>;

//! @brief Instrumented double-double, default accuracy
using fp64mp2_stat = fpmp2_stat<double, fpmp2_accuracy::def>;
//! @brief Instrumented double-double, `low` accuracy
using fp64mp2_stat_low = fpmp2_stat<double, fpmp2_accuracy::low>;
//! @brief Instrumented double-double, `mid` accuracy
using fp64mp2_stat_mid = fpmp2_stat<double, fpmp2_accuracy::mid>;
//! @brief Instrumented double-double, `high` accuracy
using fp64mp2_stat_high = fpmp2_stat<double, fpmp2_accuracy::high>;

// The drop-in promise in memory: same footprint as the wrapped type, and copyable with
// a plain memcpy, so buffers can be reinterpreted between the two.
static_assert(sizeof(fp32mp2_stat) == sizeof(fp32mp2) && alignof(fp32mp2_stat) == alignof(fp32mp2));
static_assert(sizeof(fp64mp2_stat) == sizeof(fp64mp2) && alignof(fp64mp2_stat) == alignof(fp64mp2));
static_assert(::cuda::std::is_trivially_copyable_v<fp32mp2_stat>);
static_assert(::cuda::std::is_trivially_copyable_v<fp64mp2_stat>);
} // namespace cuda::experimental

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! @brief numeric_limits for the instrumented types
//!
//! Inherits every characteristic from the wrapped type's specialization and only
//! rewraps the values it hands out, so `numeric_limits<fp32mp2_stat>::epsilon()` reports
//! the same number as `numeric_limits<fp32mp2>::epsilon()`.
template <class _FpType, ::cuda::experimental::fpmp2_accuracy _TypeAcc>
class numeric_limits<::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>>
    : public numeric_limits<::cuda::experimental::fpmp2<_FpType, _TypeAcc>>
{
private:
  using __base = numeric_limits<::cuda::experimental::fpmp2<_FpType, _TypeAcc>>;

public:
  using type = ::cuda::experimental::fpmp2_stat<_FpType, _TypeAcc>;

  _CCCL_HOST_DEVICE_API static constexpr type min() noexcept
  {
    return type(__base::min());
  }
  _CCCL_HOST_DEVICE_API static constexpr type max() noexcept
  {
    return type(__base::max());
  }
  _CCCL_HOST_DEVICE_API static constexpr type lowest() noexcept
  {
    return type(__base::lowest());
  }
  _CCCL_HOST_DEVICE_API static constexpr type epsilon() noexcept
  {
    return type(__base::epsilon());
  }
  _CCCL_HOST_DEVICE_API static constexpr type round_error() noexcept
  {
    return type(__base::round_error());
  }
  _CCCL_HOST_DEVICE_API static constexpr type infinity() noexcept
  {
    return type(__base::infinity());
  }
  _CCCL_HOST_DEVICE_API static constexpr type quiet_NaN() noexcept
  {
    return type(__base::quiet_NaN());
  }
  _CCCL_HOST_DEVICE_API static constexpr type signaling_NaN() noexcept
  {
    return type(__base::signaling_NaN());
  }
  _CCCL_HOST_DEVICE_API static constexpr type denorm_min() noexcept
  {
    return type(__base::denorm_min());
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPTOOL_STAT_H
