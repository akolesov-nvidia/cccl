//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_IMPL_CVT_H
#define _CUDA___FP_FPMP_IMPL_CVT_H

#include <cuda/std/detail/__config>

#include <cuda/std/__bit/countl.h> // countl_zero for the float-denormal head in __fpmp2_to_double

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_impl_cvt.h - fpmp2 conversions, casts and bit-cast
    ==================================================================================================
    Per-operation implementation core split out of <cuda/__fp/fpmp_impl.h>. It carries the
    conversions, casts and bit-cast
    for the fpmp2 double-word type, for both the header-only (inline) mode and the library
    (_CCCL_FPMP_USE_LIB) mode. All shared macros, the fp128 vocabulary type, and the __fpmp_*
    error-free-transform primitives live in <cuda/__fp/fpmp_impl.h>, which this header includes.
*/

#include <cuda/__fp/fpmp_impl.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
#if !(defined _CCCL_FPMP_USE_LIB)
/*
 * --------------------------------------------------------------------
 * Conversion operations
 * --------------------------------------------------------------------
 */
/*
// -----------------------------------------------------------------------
// __fpmp2_from_double: Convert double → fpmp2 (hi, lo) pair
// -----------------------------------------------------------------------
// Splits a 64-bit double into two FpType components such that:
//   x ≈ hi + lo    (with hi carrying the leading bits, lo the remainder)
//
// Two versions, selected by the accuracy level of the fpmp2 type:
//   __fpmp2_high_from_double (fpmp2_accuracy::high): the exact one, described
//     first, with an integer-only cold branch for zero, denormals, the band
//     where the tail is a float denormal (rounded once there, where the hot
//     path would round it twice), the top float binade, inf and NaN and the
//     finite / signed corners: the pair has the value of the FP64 cast path's
//     pair on every input, in canonical form;
//   __fpmp2_from_double (low, mid, the default): the original fast one, the
//     hot path covers 2^-126 <= |x| < 2^128 - 2^107, an FP64 cast fallback
//     with a zero tail takes the rest. So +-inf give {+-inf, 0}, -0 gives
//     {-0, +0}, and the tail is flushed within 8 float ulps of the overflow
//     threshold (where the Fast2Sum of the hot path could overflow).
// Both are __internal_fpmp2_from_double<_FpType, _Exact>.
//
// When CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP == 1 and FpType == float:
//   Uses integer bit manipulation to extract the hi and lo components
//   directly from the IEEE 754 double bit pattern, avoiding all FP64
//   arithmetic. This is beneficial on GPUs with limited FP64 throughput
//   (e.g., consumer GPUs with 1:64 FP64:FP32 ratio).
//
//   Notes:
//   * hi rounding is round-to-nearest, ties AWAY-FROM-ZERO (single
//     add+shift), instead of ties-to-even. As a result, hi may differ
//     from (float)x at exact tie midpoints; for non-tie inputs the
//     two rules agree.
//   * lo is computed by reinterpreting the bottom 29 mantissa bits as
//     a signed 32-bit integer (the round bit is placed at the sign
//     position so a rounded-up hi automatically yields a negative
//     residual via two's complement), converting it to float with
//     round-to-nearest-even, and rescaling by exact powers of two.
//   * A final Fast2Sum re-establishes the canonical fl(hi+lo) == hi
//     invariant. This is required because the round-to-nearest lo
//     can land at +/-ulp(hi)/2 and overflow the canonical range.
//
// When CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP == 0 or FpType != float:
//   Uses the standard cast-based approach:
//     hi = (FpType)x;  lo = (FpType)(x - (double)hi);
//   This relies on two FP64 operations (cast + subtract).
//
// IEEE 754 bit layout reference:
//   double (64-bit): [1 sign][11 exponent][52 mantissa]
//   float  (32-bit): [1 sign][ 8 exponent][23 mantissa]
//   Exponent bias: double = 1023, float = 127, difference = 896
//
// The 52-bit double mantissa is split into:
//   - hi: top 23 bits  → float mantissa (bits [29:51] of double mantissa)
//   - lo: bottom 29 bits → second float  (bits [0:28] of double mantissa)
// -----------------------------------------------------------------------
*/
template <typename _FpType, bool _Exact>
_CCCL_FPMP_CORE_API void __internal_fpmp2_from_double(const double __x, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
#  if _CCCL_FPMP_USE_OPT_FROM_DOUBLE == 1
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    if constexpr (_Exact)
    {
      uint64_t __dbits = ::cuda::std::bit_cast<uint64_t>(__x);
      uint32_t __sign  = (uint32_t) (__dbits >> 63);
      uint32_t __d_exp = (uint32_t) ((__dbits >> 52) & 0x7FFU);
      uint64_t __mant  = __dbits & 0x000FFFFFFFFFFFFFULL;

      // hi biased exponent in float space: f_exp = (d_exp - 1023) + 127.
      int32_t __f_exp = (int32_t) __d_exp - 896;

      // Hot path for f_exp in [29, 253], i.e. |x| in [2^-98, 2^127): the head
      // is a normal float and cannot round up to inf, and the tail is a
      // normal float whenever the int -> float conversion of the residual
      // below has to round (a residual of 2^24 units or more is 2^(e-28) or
      // more, above 2^-126 for e >= -98), so it is rounded once. One compare;
      // the cold branch below takes zero and the double denormals (d_exp ==
      // 0), the float underflow range (f_exp <= 0), the band where the tail
      // can be a float denormal (f_exp in [1, 28]), the top float binade
      // (f_exp == 254, where the head may carry into inf and the Fast2Sum
      // could overflow) and inf / NaN (f_exp >= 255) -- all without FP64.
      if (_CCCL_BUILTIN_EXPECT((uint32_t) (__f_exp - 29) < 225U, 1))
      {
        // hi mantissa: top 23 explicit bits with round-to-nearest,
        // ties away from zero. (((mant >> 28) + 1) >> 1) takes the
        // top 24 bits of mant, adds 1 at the round position, then
        // drops it. The carry can ripple into the exponent (when
        // hi_round == 0x800000), so we use '+' (not '|') to merge
        // hi_round into the shifted exponent field.
        uint32_t __hi_round = (((uint32_t) (__mant >> 28)) + 1U) >> 1;
        uint32_t __hi_bits  = (__sign << 31) | (((uint32_t) __f_exp << 23) + __hi_round);
        *__res_hi           = ::cuda::std::bit_cast<float>(__hi_bits);

        // Encode the residual as a signed 32-bit integer with the
        // round bit placed at the sign position. Bottom 32 bits of
        // mant are bits [31:0]; shifting left by 3 in 32-bit
        // arithmetic discards bits [31:29] (already absorbed into
        // hi) and places bit 28 (the round bit) at bit 31. When the
        // round bit is 1 (hi was rounded up), rsd is negative in
        // two's complement, exactly representing the signed residual
        // x - hi at mantissa scale * 2^3.
        int32_t __rsd = (int32_t) ((uint32_t) __mant << 3);

        // Convert rsd to float with round-to-nearest-even (default for
        // host int->float and CUDA cvt.rn.f32.s32). Then scale:
        //   * 2^-55  : undoes the << 3 (-3) and the mantissa-position
        //              offset (-52) to recover residual at unit scale.
        //   * scale  : 2^(f_exp - 127) with the sign of x. Both
        //              multiplications are exact (powers of two).
        float __scale = ::cuda::std::bit_cast<float>((__sign << 31) | ((uint32_t) __f_exp << 23));
        *__res_lo     = (static_cast<float>(__rsd) * 0x1p-55f) * __scale;

        // Fast2Sum to enforce canonical form fl(hi+lo) == hi.
        // Required because round-to-nearest on r can leave |lo|
        // exactly at ulp(hi)/2; if hi has an odd low mantissa bit,
        // fl(hi+lo) would otherwise round away from hi.
        *__res_hi = ::cuda::__fpmp_fast_two_sum(*__res_hi, *__res_lo, __res_lo);
      }
      else
      {
        // Cold branch, integer only. The tail is a zero with the head's
        // sign in every case but the top binade: -0 -> {-0, -0}, +-inf ->
        // {+-inf, +-0}.
        const float __szero = ::cuda::std::bit_cast<float>(__sign << 31);
        if (__d_exp == 0U)
        {
          // zero and the double denormals (< 2^-1022, far below float)
          *__res_hi = __szero;
          *__res_lo = __szero;
        }
        else if (__d_exp == 0x7FFU)
        {
          // +-inf, or NaN (quiet, keeping the top 23 payload bits)
          const uint32_t __nan = (__mant != 0ULL) ? (0x400000U | (uint32_t) (__mant >> 29)) : 0U;
          *__res_hi            = ::cuda::std::bit_cast<float>((__sign << 31) | 0x7F800000U | __nan);
          *__res_lo            = __szero;
        }
        else if (__f_exp <= 0)
        {
          // float underflow: |x| < 2^-126, the head is the float denormal
          // round-to-nearest-even of x in units of 2^-149, i.e. the 53-bit
          // significand shifted right by 30 - f_exp (>= 30; the shift is
          // clamped at 63, where the result is 0 anyway); a field of 2^23
          // is the smallest normal, which the encoding gives for free.
          const uint64_t __m53 = __mant | (1ULL << 52);
          const uint32_t __sh  = (uint32_t) (30 - __f_exp) < 63U ? (uint32_t) (30 - __f_exp) : 63U;
          uint64_t __q         = __m53 >> __sh;
          const uint64_t __rem = __m53 - (__q << __sh);
          const uint64_t __hlf = 1ULL << (__sh - 1);
          __q += (uint64_t) ((__rem > __hlf) || (__rem == __hlf && (__q & 1ULL)));
          *__res_hi = ::cuda::std::bit_cast<float>((__sign << 31) | (uint32_t) __q);
          *__res_lo = __szero;
        }
        else if (__f_exp <= 28)
        {
          // 2^-126 <= |x| < 2^-98: the tail can be a float denormal, which the
          // hot path's two-step scaling would round twice. Head and tail come
          // from the integer significand, each rounded once to nearest even:
          // the head at 24 bits (a carry to 2^24 moves it to the next binade
          // through the '+' into the exponent field); the tail -- the signed
          // residual r in units of 2^(e - 52), |r| <= 2^28 -- as a normal
          // float through the int -> float conversion and two exact scalings
          // when it is 2^-126 or more (|r| >= 2^(53 - f_exp)), else on the
          // 2^-149 grid, r / 2^(30 - f_exp): t <= 2^23 is then the denormal
          // encoding of t * 2^-149 (t == 2^23 is the smallest normal, which
          // the encoding gives for free). A zero tail takes the head's sign.
          const uint64_t __m53 = __mant | (1ULL << 52);
          uint64_t __q         = __m53 >> 29;
          const uint64_t __rem = __m53 & ((1ULL << 29) - 1);
          __q += (uint64_t) ((__rem > (1ULL << 28)) || (__rem == (1ULL << 28) && (__q & 1ULL)));
          const int64_t __r     = (int64_t) __m53 - (int64_t) (__q << 29);
          const uint64_t __ar   = (uint64_t) (__r < 0 ? -__r : __r);
          const uint32_t __tneg = (uint32_t) ((__r < 0) != (__sign != 0U)) << 31;
          uint32_t __lo_bits;
          if (__ar >= (1ULL << (53 - __f_exp)))
          {
            const float __tn = (static_cast<float>((int32_t) __ar) * 0x1p-53f)
                             * ::cuda::std::bit_cast<float>((uint32_t) (__f_exp + 1) << 23);
            __lo_bits = __tneg | ::cuda::std::bit_cast<uint32_t>(__tn);
          }
          else
          {
            const uint32_t __s    = 30U - (uint32_t) __f_exp;
            uint64_t __t          = __ar >> __s;
            const uint64_t __trem = __ar & ((1ULL << __s) - 1);
            const uint64_t __th   = 1ULL << (__s - 1);
            __t += (uint64_t) ((__trem > __th) || (__trem == __th && (__t & 1ULL)));
            __lo_bits = ((__t == 0ULL) ? (__sign << 31) : __tneg) | (uint32_t) __t;
          }
          // Fast2Sum for the canonical form fl(hi + lo) == hi, as on the hot
          // path: a tail rounded up to exactly half an ulp beside an odd head
          // moves to {hi + ulp, -ulp/2}, the same value (the cast path keeps
          // the first form).
          const float __hi_r =
            ::cuda::std::bit_cast<float>((__sign << 31) | ((((uint32_t) __f_exp - 1U) << 23) + (uint32_t) __q));
          const float __lo_r = ::cuda::std::bit_cast<float>(__lo_bits);
          *__res_hi          = ::cuda::__fpmp_fast_two_sum(__hi_r, __lo_r, __res_lo);
        }
        else if (__f_exp >= 255)
        {
          // |x| >= 2^128: past the float range
          *__res_hi = ::cuda::std::bit_cast<float>((__sign << 31) | 0x7F800000U);
          *__res_lo = __szero;
        }
        else
        {
          // the top float binade, f_exp == 254: |x| in [2^127, 2^128), the
          // hot path's arithmetic, except that the head may carry into inf
          // (x >= 2^128 - 2^103: {+-inf, +-0}) and that for the 16 doubles
          // right below that threshold the head is +-FLT_MAX with a tail of
          // +-ulp/2, whose Fast2Sum would overflow (FLT_MAX is odd, the tie
          // rounds to 2^128): the finite pair is kept as it is there.
          const uint32_t __hi_round = (((uint32_t) (__mant >> 28)) + 1U) >> 1;
          const uint32_t __hi_mag   = (254U << 23) + __hi_round;
          if (__hi_mag >= 0x7F800000U)
          {
            *__res_hi = ::cuda::std::bit_cast<float>((__sign << 31) | 0x7F800000U);
            *__res_lo = __szero;
          }
          else
          {
            const int32_t __rsd = (int32_t) ((uint32_t) __mant << 3);
            const float __scale = ::cuda::std::bit_cast<float>((__sign << 31) | (254U << 23));
            const float __hi_f  = ::cuda::std::bit_cast<float>((__sign << 31) | __hi_mag);
            const float __lo_f  = (static_cast<float>(__rsd) * 0x1p-55f) * __scale;
            if (__hi_mag == 0x7F7FFFFFU && __rsd > 0)
            {
              *__res_hi = __hi_f;
              *__res_lo = __lo_f;
            }
            else
            {
              *__res_hi = ::cuda::__fpmp_fast_two_sum(__hi_f, __lo_f, __res_lo);
            }
          }
        }
      }
    }
    else
    {
      uint64_t __dbits = ::cuda::std::bit_cast<uint64_t>(__x);
      uint32_t __sign  = (uint32_t) (__dbits >> 63);
      uint64_t __mant  = __dbits & 0x000FFFFFFFFFFFFFULL;
      // top 32 bits of |x|: (d_exp << 20) | the top 20 mantissa bits
      const uint32_t __top = (uint32_t) (__dbits >> 32) & 0x7FFFFFFFU;

      // hi biased exponent in float space: f_exp = (d_exp - 1023) + 127.
      int32_t __f_exp = (int32_t) (__top >> 20) - 896;

      // The hot path takes 2^-126 <= |x| < 2^128 - 2^107, one range test on
      // __top: d_exp in [897, 1150] without the top 8 float ulps of the top
      // binade, where the Fast2Sum below could overflow (the head carries into
      // inf, or is FLT_MAX with a tail of +ulp/2, giving {inf, NaN} or
      // {inf, -inf}). Zero, the double denormals, float underflow, those top
      // ulps, inf and NaN take the cast with a zero tail: +-inf give
      // {+-inf, 0}, and the tail is flushed only where (float) x is within 8
      // ulps of FLT_MAX or infinite. (One instruction over the plain exponent
      // test; a test on the rounded head instead would delay the branch.)
      if (__top - 0x38100000U >= 0x0FDFFFFFU)
      {
        *__res_hi = (float) __x;
        *__res_lo = 0.0f;
      }
      else
      {
        // hi mantissa: top 23 explicit bits with round-to-nearest,
        // ties away from zero. (((mant >> 28) + 1) >> 1) takes the
        // top 24 bits of mant, adds 1 at the round position, then
        // drops it. The carry can ripple into the exponent (when
        // hi_round == 0x800000), so we use '+' (not '|') to merge
        // hi_round into the shifted exponent field.
        uint32_t __hi_round = (((uint32_t) (__mant >> 28)) + 1U) >> 1;
        uint32_t __hi_bits  = (__sign << 31) | (((uint32_t) __f_exp << 23) + __hi_round);
        *__res_hi           = ::cuda::std::bit_cast<float>(__hi_bits);

        // Encode the residual as a signed 32-bit integer with the
        // round bit placed at the sign position. Bottom 32 bits of
        // mant are bits [31:0]; shifting left by 3 in 32-bit
        // arithmetic discards bits [31:29] (already absorbed into
        // hi) and places bit 28 (the round bit) at bit 31. When the
        // round bit is 1 (hi was rounded up), rsd is negative in
        // two's complement, exactly representing the signed residual
        // x - hi at mantissa scale * 2^3.
        int32_t __rsd = (int32_t) ((uint32_t) __mant << 3);

        // Convert rsd to float with round-to-nearest-even (default for
        // host int->float and CUDA cvt.rn.f32.s32). Then scale:
        //   * 2^-55  : undoes the << 3 (-3) and the mantissa-position
        //              offset (-52) to recover residual at unit scale.
        //   * scale  : 2^(f_exp - 127) with the sign of x. Both
        //              multiplications are exact (powers of two).
        float __scale = ::cuda::std::bit_cast<float>((__sign << 31) | ((uint32_t) __f_exp << 23));
        *__res_lo     = (static_cast<float>(__rsd) * 0x1p-55f) * __scale;

        // Fast2Sum to enforce canonical form fl(hi+lo) == hi.
        // Required because round-to-nearest on r can leave |lo|
        // exactly at ulp(hi)/2; if hi has an odd low mantissa bit,
        // fl(hi+lo) would otherwise round away from hi.
        *__res_hi = ::cuda::__fpmp_fast_two_sum(*__res_hi, *__res_lo, __res_lo);
      }
    }
  }
  else if constexpr (__fpmp2_is_fp64_v<_FpType>)
  {
    // FpType == double (fp64mp2): the cast-based split below would
    // compute (double)(x - (double)x) == 0.0 and the compiler folds
    // it; spell that out at the source level so the intent is
    // explicit and the lo store is guaranteed not to depend on any
    // FP64 instruction.
    *__res_hi = __x;
    *__res_lo = 0.0;
  }
  else
  {
    // Generic fallback for any future non-float, non-double FpType:
    // cast-based split. Where x is already representable (+-0, +-inf
    // included) or hi has overflowed, the tail is a zero (with hi's sign
    // for +-0) rather than x - hi, which was inf - inf = NaN for the
    // infinities, -inf past the float range and +0 for -0.
    *__res_hi = static_cast<_FpType>(__x);
    {
      const double __hi_d = static_cast<double>(*__res_hi);
      const bool __exact  = (__x == __hi_d) || (__hi_d - __hi_d != 0.0); /* representable, or overflowed to +-inf */
      *__res_lo           = __exact ? ((__x == 0.0) ? *__res_hi : _FpType(0)) /* +-0 keeps its sign */
                                    : static_cast<_FpType>(__x - __hi_d);
    }
  }
#  else // !_CCCL_FPMP_USE_OPT_FROM_DOUBLE == 1
  if constexpr (__fpmp2_is_fp64_v<_FpType>)
  {
    // FpType == double (fp64mp2): trivial split, see comment above.
    *__res_hi = __x;
    *__res_lo = 0.0;
  }
  else
  {
    // Non-optimized path: two FP64 operations (cast + subtract). Where x is
    // already representable (+-0, +-inf included) or hi has overflowed, the
    // tail is a zero (with hi's sign for +-0) rather than x - hi, which was
    // inf - inf = NaN for the infinities, -inf past the float range and +0
    // for -0.
    *__res_hi = static_cast<_FpType>(__x);
    {
      const double __hi_d = static_cast<double>(*__res_hi);
      const bool __exact  = (__x == __hi_d) || (__hi_d - __hi_d != 0.0); /* representable, or overflowed to +-inf */
      *__res_lo           = __exact ? ((__x == 0.0) ? *__res_hi : _FpType(0)) /* +-0 keeps its sign */
                                    : static_cast<_FpType>(__x - __hi_d);
    }
  }
#  endif // !_CCCL_FPMP_USE_OPT_FROM_DOUBLE == 1
} // __internal_fpmp2_from_double

// double -> (hi, lo): the entry points of the accuracy levels (see above)
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_from_double(const double __x, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_from_double<_FpType, false>(__x, __res_hi, __res_lo);
}
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_high_from_double(const double __x, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  ::cuda::__internal_fpmp2_from_double<_FpType, true>(__x, __res_hi, __res_lo);
}

// int -> (hi, lo) conversions
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_from_int(const int32_t __i, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = ::cuda::__fpmp_int2fp_rz<_FpType>(__i);
  *__res_lo = ::cuda::__fpmp_int2fp_rz<_FpType>(__i - ::cuda::__fpmp_fp2int_rz(*__res_hi));
}

// uint -> (hi, lo) conversions
// Note: Use signed arithmetic to compute residual, since __fpmp_fp2uint_rz(*res_hi)
// might be larger than i when rounding direction differs
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_from_uint(const uint32_t __i, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = ::cuda::__fpmp_uint2fp_rz<_FpType>(__i);
  // Compute residual using signed arithmetic to handle case where hi rounds up
  int32_t __residual = static_cast<int32_t>(__i) - static_cast<int32_t>(::cuda::__fpmp_fp2uint_rz(*__res_hi));
  *__res_lo          = ::cuda::__fpmp_int2fp_rz<_FpType>(__residual);
}

// ll -> (hi, lo) conversions
// With __fpmp_ll2fp_rz properly rounding toward zero, hi is always <= i for positive i
// and >= i for negative i, so __fpmp_fp2ll_rz(hi) is always representable as int64_t.
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_from_ll(const int64_t __i, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = ::cuda::__fpmp_ll2fp_rz<_FpType>(__i);
  *__res_lo = ::cuda::__fpmp_ll2fp_rz<_FpType>(__i - ::cuda::__fpmp_fp2ll_rz(*__res_hi));
}

// ull -> (hi, lo) conversions
// With ull2fp_rz properly rounding toward zero, hi <= i always,
// so the residual i - hi is always non-negative.
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_from_ull(const uint64_t __i, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = ::cuda::__fpmp_ull2fp_rz<_FpType>(__i);
  // Residual is always non-negative and fits in int64_t (< 2^53 for double)
  uint64_t __residual = __i - ::cuda::__fpmp_fp2ull_rz(*__res_hi);
  *__res_lo           = ::cuda::__fpmp_ull2fp_rz<_FpType>(__residual);
}

// (hi, lo) -> double conversions
//
// Two versions, selected by the accuracy level of the fpmp2 type:
//   __fpmp2_high_to_double (fpmp2_accuracy::high): the one described here,
//     correctly rounded for every pair (the binade below a power-of-two head
//     included) with an integer-only cold path;
//   __fpmp2_to_double (low, mid, the default): the original fast one, the
//     same hot path without the lower-binade handling, so that below a
//     power-of-two head the result can be one double ulp above the correctly
//     rounded value ({1, -2^-53} gives 1), and an FP64 cold path (two
//     conversions and a sum).
// Both are __internal_fpmp2_to_double<_FpType, _Exact>.
//
// Optimized path (_CCCL_FPMP_USE_OPT_TO_DOUBLE == 1):
//   Reconstructs the IEEE 754 double bit pattern from two float
//   components using FP32 + integer arithmetic. No FP64 instructions on
//   the hot path — avoids the slow FP64 pipeline on GPUs with limited
//   double-precision throughput (1:64 ratio on consumer GPUs).
//
//   Assumes both inputs are normal floats (no denormals, inf, or NaN).
//   This is always true for well-formed fp32mp2 values produced by the
//   library's double→fp32mp2 conversion.
//
//   Step 0 — input renormalization (2Sum):
//     The hot path below relies on the canonical-form invariant
//     |lo| <= ulp(hi)/2 in two places:
//       (a) x_lo * scale must be exact (24-bit mantissa headroom);
//       (b) the resulting r must fit in int32 (|r| < 2^31, i.e.
//           |lo|/|hi| < 2^-21).
//     Pairs produced by add_fast or long FAST accumulator chains may
//     have |lo|/|hi| as large as ~2^-8, which silently overflows r.
//     A 2Sum at the top makes (hi, lo) canonical (|err| <= ulp(s)/2)
//     for any input magnitudes — 6 FP32 ops on the hot path, runs
//     largely in parallel with the bit extraction below on the FP/INT
//     pipes.
//
//   Hot-path algorithm:
//     1. Build a signed power-of-two scale = (sign_a ? -1 : +1) * 2^(179 - fexp_a)
//        by direct bit construction. The sign of hi is baked into the
//        scale so the next step yields lo's contribution *relative to hi*
//        regardless of lo's own sign — no same-sign / diff-sign branch.
//     2. r = (int32_t)(x_lo * scale)  — one FMUL + one F2I. For canonical
//        fp32mp2 (|lo| <= ulp(hi)/2) the multiplication is exact and
//        |r| <= 2^28. The signed r is exactly lo's contribution to the
//        double mantissa at hi's scale.
//     3. M = (mantissa-of-hi-with-implicit-1 at bit 52) + r — single
//        64-bit add. Range [2^52 - 2^28, 2^53 - 2^29 + 2^28], always > 0.
//     4. Renormalize by at most 1 bit (subtraction can borrow the
//        implicit-1 down to bit 51). Single conditional shift; no CLZ.
//     5. Splice sign / biased-exponent / mantissa into the double.
//
//   Cold path (integer only as well, no FP64 anywhere) covers the cases
//   where the FP32 scale would overflow / be ill-defined:
//     - fexp_a == 0      : hi is +/-0 or subnormal (the tail is then 0)
//     - fexp_a in [1,51] : 2^(179 - fexp_a) overflows float (need biased
//                          exponent <= 254 ⇒ fexp_a >= 52); the scale is
//                          applied in two exact products instead
//     - fexp_a == 0xFF   : hi is +/-Inf or NaN
//   Both ends of this range are extremely rare for typical fp32mp2 data.
//
//   The result is the correctly rounded value of hi + lo: step 2 rounds lo
//   to nearest even at the resolution of hi's binade, which is the
//   double's own rounding there since hi's mantissa is even at that scale;
//   when hi is a power of two and lo pulls below it (the binade below,
//   one more bit), the product is doubled before the rounding and the
//   borrow leaves the field in place, so {1, -2^-53} is 1 - 2^-53 exactly.
//   Cost of that case: five instructions, 141 -> 147 clk/ev on the L4.
//
// Non-optimized path:
//   static_cast<double>(x_hi) + static_cast<double>(x_lo)
//   (2x F2D + 1x DADD = 3 FP64 operations)
//
template <typename _FpType, bool _Exact>
_CCCL_FPMP_CORE_API double __internal_fpmp2_to_double(const _FpType __x_hi_in, const _FpType __x_lo_in) noexcept
{
#  if _CCCL_FPMP_USE_OPT_TO_DOUBLE == 1
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    if constexpr (_Exact)
    {
      // Renormalize input to canonical form. See "Step 0" in the
      // function's docstring for why this is required for the integer
      // path to be safe on non-canonical pairs (e.g. produced by
      // add_fast / long FAST accumulator chains).
      float __x_lo;
      float __x_hi = ::cuda::__fpmp_two_sum(__x_hi_in, __x_lo_in, &__x_lo);

      uint32_t __hi_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
      uint32_t __sign_a  = __hi_bits >> 31;
      uint32_t __fexp_a  = (__hi_bits >> 23) & 0xFFU;
      uint32_t __fmant_a = __hi_bits & 0x7FFFFFU;

      // Cold fallback for hi outside the FP32-scale-representable range:
      // zero/subnormal, very tiny normal (fexp_a in [1, 51]), Inf, NaN.
      // NB: sum the ORIGINAL inputs, not the 2Sum-renormalized (__x_hi, __x_lo).
      // The 2Sum prologue exists only to canonicalize finite operands for the
      // integer hot path; on special values it poisons the low word (e.g.
      // two_sum(inf, 0) yields lo = inf - inf = NaN), which would turn a clean
      // (inf, 0) / (nan, 0) into NaN here. The plain double sum of the originals
      // handles inf/nan/zero/subnormal correctly.
      if (__fexp_a < 52U || __fexp_a == 0xFFU)
      {
        // Cold path, integer only (no FP64 even here).
        if (__fexp_a == 0xFFU)
        {
          const uint32_t __hb = ::cuda::std::bit_cast<uint32_t>(__x_hi_in);
          const uint32_t __lb = ::cuda::std::bit_cast<uint32_t>(__x_lo_in);
          if ((__hb & 0x7F800000U) != 0x7F800000U)
          {
            // The 2Sum overflowed on finite limbs: |hi| ~ FLT_MAX with a tail of
            // +-ulp/2 or more (the pair of the 16 doubles right below the float
            // overflow threshold). The value is finite and the hot path's
            // arithmetic holds on the original limbs.
            const uint32_t __sg = __hb >> 31, __fe = (__hb >> 23) & 0xFFU, __fm = __hb & 0x7FFFFFU;
            const float __sc    = ::cuda::std::bit_cast<float>((__sg << 31) | ((306U - __fe) << 23));
            const int32_t __r0  = ::cuda::__fpmp_fp2int_rn(__x_lo_in * __sc);
            uint64_t __m0       = (uint64_t) ((int64_t) (((uint64_t) (0x800000U | __fm)) << 29) + (int64_t) __r0);
            const uint64_t __n0 = ((__m0 >> 52) & 1ULL) ^ 1ULL;
            __m0 <<= __n0;
            return ::cuda::std::bit_cast<double>(
              ((uint64_t) __sg << 63) | ((uint64_t) (__fe + 896U - (uint32_t) __n0) << 52)
              | (__m0 & 0x000FFFFFFFFFFFFFULL));
          }
          // +-inf or NaN head, from the original limbs (the 2Sum above poisons
          // the tail on these): hi + lo in IEEE terms, i.e. NaN for a NaN head,
          // a NaN tail or the opposite infinity, +-inf otherwise (quiet NaN,
          // the head's payload kept).
          const uint64_t __hm  = __hb & 0x7FFFFFU;
          const bool __lo_nan  = (__lb & 0x7FFFFFFFU) > 0x7F800000U;
          const bool __opp_inf = ((__lb & 0x7FFFFFFFU) == 0x7F800000U) && (((__lb ^ __hb) >> 31) != 0U);
          const uint64_t __pl =
            (__hm != 0ULL) ? ((1ULL << 51) | (__hm << 29)) : ((__lo_nan || __opp_inf) ? (1ULL << 51) : 0ULL);
          return ::cuda::std::bit_cast<double>(((uint64_t) (__hb >> 31) << 63) | (0x7FFULL << 52) | __pl);
        }
        if (__fexp_a == 0U)
        {
          // +-0 or a float denormal head; the tail of a canonical pair is 0
          // there (|lo| <= 2^-150 is no float), so the double is the head:
          // its field normalized with the leading one at bit 52.
          if (__fmant_a == 0U)
          {
            return ::cuda::std::bit_cast<double>((uint64_t) __sign_a << 63);
          }
          const uint32_t __lz  = (uint32_t) ::cuda::std::countl_zero(__fmant_a) - 9U; /* within the 23-bit field */
          const uint64_t __m52 = ((uint64_t) __fmant_a << (30U + __lz)) & 0x000FFFFFFFFFFFFFULL;
          return ::cuda::std::bit_cast<double>(((uint64_t) __sign_a << 63) | ((uint64_t) (896U - __lz) << 52) | __m52);
        }
        // fexp_a in [1, 51]: the hot path's arithmetic with the scale
        // 2^(179 - fexp_a), which is no float, split into +-2^127 and
        // 2^(52 - fexp_a); both products are exact.
        const float __scale1 = ::cuda::std::bit_cast<float>((__sign_a << 31) | (254U << 23));
        const float __scale2 = ::cuda::std::bit_cast<float>((179U - __fexp_a) << 23);
        const bool __below_c =
          (__fmant_a == 0U) && (static_cast<int32_t>(::cuda::std::bit_cast<uint32_t>(__x_lo) ^ __hi_bits) < 0);
        const float __tc    = (__x_lo * __scale1) * __scale2;
        const int32_t __r   = ::cuda::__fpmp_fp2int_rn(__below_c ? __tc + __tc : __tc);
        const int64_t __m   = (int64_t) (((uint64_t) (0x800000U | __fmant_a)) << 29) + (int64_t) __r;
        uint64_t __mu       = (uint64_t) __m;
        const uint64_t __ns = (((__mu >> 52) & 1ULL) ^ 1ULL);
        __mu <<= (__ns & (uint64_t) !__below_c);
        return ::cuda::std::bit_cast<double>(
          ((uint64_t) __sign_a << 63) | ((uint64_t) (__fexp_a + 896U - (uint32_t) __ns) << 52)
          | (__mu & 0x000FFFFFFFFFFFFFULL));
      }

      // scale = (sign_a ? -1 : +1) * 2^(179 - fexp_a). Always a normal
      // float for fexp_a in [52, 254] (biased scale_exp = 306 - fexp_a
      // in [52, 254]). Sign of hi baked in so the signed r below is
      // already lo's contribution relative to hi.
      float __scale = ::cuda::std::bit_cast<float>((__sign_a << 31) | ((306U - __fexp_a) << 23));

      // t exactly represents lo at hi's mantissa scale (signed): for
      // canonical fp32mp2 (|lo| <= ulp(hi)/2) the multiplication is exact
      // (power-of-two scaling) and |t| <= 2^28. When hi is a power of two
      // and lo pulls below it, the value lies in the binade below, where the
      // double resolves one more bit: t is doubled (exactly) so that r comes
      // out at that binade's scale, and the borrow below then leaves the
      // field in place. r = round-to-nearest-even of t, |r| <= 2^29.
      const bool __below =
        (__fmant_a == 0U) && (static_cast<int32_t>(::cuda::std::bit_cast<uint32_t>(__x_lo) ^ __hi_bits) < 0);
      const float __t = __x_lo * __scale;
      int32_t __r     = ::cuda::__fpmp_fp2int_rn(__below ? __t + __t : __t);

      // M = (hi's 53-bit mantissa with implicit 1, at bit 52)
      //     + (signed lo contribution at the same scale).
      // Range: [2^52 - 2^28, 2^53 - 2^29 + 2^28]. Always positive.
      int64_t __m = (int64_t) (((uint64_t) (0x800000U | __fmant_a)) << 29) + (int64_t) __r;

      // Subtraction can borrow at most one bit (|r| << 2^52), so the
      // implicit-1 lands at bit 52 (no shift) or bit 51 (shift up by 1,
      // unless the pair was handled at the lower binade's scale, where the
      // field is already right and only the exponent moves). Single
      // conditional shift, no __clzll on the critical path.
      uint64_t __mu         = (uint64_t) __m;
      uint64_t __need_shift = ((__mu >> 52) & 1ULL) ^ 1ULL;
      __mu <<= (__need_shift & (uint64_t) !__below);

      return ::cuda::std::bit_cast<double>(
        ((uint64_t) __sign_a << 63) | ((uint64_t) (__fexp_a + 896U - (uint32_t) __need_shift) << 52)
        | (__mu & 0x000FFFFFFFFFFFFFULL));
    }
    else
    {
      // Renormalize input to canonical form. See "Step 0" in the
      // function's docstring for why this is required for the integer
      // path to be safe on non-canonical pairs (e.g. produced by
      // add_fast / long FAST accumulator chains).
      float __x_lo;
      float __x_hi = ::cuda::__fpmp_two_sum(__x_hi_in, __x_lo_in, &__x_lo);

      uint32_t __hi_bits = ::cuda::std::bit_cast<uint32_t>(__x_hi);
      uint32_t __sign_a  = __hi_bits >> 31;
      uint32_t __fexp_a  = (__hi_bits >> 23) & 0xFFU;
      uint32_t __fmant_a = __hi_bits & 0x7FFFFFU;

      // Cold fallback for hi outside the FP32-scale-representable range:
      // zero/subnormal, very tiny normal (fexp_a in [1, 51]), Inf, NaN.
      // NB: sum the ORIGINAL inputs, not the 2Sum-renormalized (__x_hi, __x_lo).
      // The 2Sum prologue exists only to canonicalize finite operands for the
      // integer hot path; on special values it poisons the low word (e.g.
      // two_sum(inf, 0) yields lo = inf - inf = NaN), which would turn a clean
      // (inf, 0) / (nan, 0) into NaN here. The plain double sum of the originals
      // handles inf/nan/zero/subnormal correctly.
      if (__fexp_a < 52U || __fexp_a == 0xFFU)
      {
        return static_cast<double>(__x_hi_in) + static_cast<double>(__x_lo_in);
      }

      // scale = (sign_a ? -1 : +1) * 2^(179 - fexp_a). Always a normal
      // float for fexp_a in [52, 254] (biased scale_exp = 306 - fexp_a
      // in [52, 254]). Sign of hi baked in so the signed r below is
      // already lo's contribution relative to hi.
      float __scale = ::cuda::std::bit_cast<float>((__sign_a << 31) | ((306U - __fexp_a) << 23));

      // r exactly represents lo at hi's mantissa scale (signed). For
      // canonical fp32mp2 (|lo| <= ulp(hi)/2) the multiplication is
      // exact (power-of-two scaling) and |r| <= 2^28.
      int32_t __r = ::cuda::__fpmp_fp2int_rn(__x_lo * __scale);

      // M = (hi's 53-bit mantissa with implicit 1, at bit 52)
      //     + (signed lo contribution at the same scale).
      // Range: [2^52 - 2^28, 2^53 - 2^29 + 2^28]. Always positive.
      int64_t __m = (int64_t) (((uint64_t) (0x800000U | __fmant_a)) << 29) + (int64_t) __r;

      // Subtraction can borrow at most one bit (|r| << 2^52), so the
      // implicit-1 lands at bit 52 (no shift) or bit 51 (shift up by 1).
      // Single conditional shift, no __clzll on the critical path.
      uint64_t __mu         = (uint64_t) __m;
      uint64_t __need_shift = ((__mu >> 52) & 1ULL) ^ 1ULL;
      __mu <<= __need_shift;

      return ::cuda::std::bit_cast<double>(
        ((uint64_t) __sign_a << 63) | ((uint64_t) (__fexp_a + 896U - (uint32_t) __need_shift) << 52)
        | (__mu & 0x000FFFFFFFFFFFFFULL));
    }
  }
  else
  {
    return static_cast<double>(__x_hi_in) + static_cast<double>(__x_lo_in);
  }
#  else
  return static_cast<double>(__x_hi_in) + static_cast<double>(__x_lo_in);
#  endif
} // __internal_fpmp2_to_double

// (hi, lo) -> double: the entry points of the accuracy levels (see above)
template <typename _FpType>
_CCCL_FPMP_CORE_API double __fpmp2_to_double(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  return ::cuda::__internal_fpmp2_to_double<_FpType, false>(__x_hi, __x_lo);
}
template <typename _FpType>
_CCCL_FPMP_CORE_API double __fpmp2_high_to_double(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  return ::cuda::__internal_fpmp2_to_double<_FpType, true>(__x_hi, __x_lo);
}

// (hi, lo) -> float conversions (returns the sum as single FpType)
template <typename _FpType>
_CCCL_FPMP_CORE_API _FpType __fpmp2_to_float(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  return __x_hi + __x_lo;
}

// (hi, lo) -> int conversions
template <typename _FpType>
_CCCL_FPMP_CORE_API int32_t __fpmp2_to_int(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  _FpType __abs_hi = ::cuda::__fpmp_internal_fabs(__x_hi);
  // Check threshold BEFORE computing sum - for large values, addition loses precision
  // 2^24 for float, 2^53 for double
  _FpType __threshold = __fpmp2_is_fp32_v<_FpType> ? 0x1.0p24f : 0x1.0p53;
  if (__abs_hi < __threshold)
  {
    // Small value: use round-toward-zero addition
    _FpType __res = ::cuda::__fpmp_add_rz(__x_hi, __x_lo);
    return ::cuda::__fpmp_fp2int_rz(__res);
  }
  else
  {
    // Large value: integer addition. hi is an integer here (|hi| >= 2^24 for
    // float, 2^53 for double), so the truncation of hi + lo is hi + trunc(lo),
    // one step toward zero less when lo has the opposite sign and a
    // fractional part (2^24 - 0.75 truncates to 2^24 - 1)
    int32_t __hi_int = ::cuda::__fpmp_fp2int_rz(__x_hi);
    int32_t __lo_int = ::cuda::__fpmp_fp2int_rz(__x_lo);
    if ((static_cast<_FpType>(__lo_int) != __x_lo) && ((__x_lo < _FpType(0)) != (__x_hi < _FpType(0))))
    {
      __hi_int += (__x_hi < _FpType(0)) ? 1 : -1;
    }
    return __hi_int + __lo_int;
  }
} // __fpmp2_to_int

// (hi, lo) -> uint conversions
template <typename _FpType>
_CCCL_FPMP_CORE_API uint32_t __fpmp2_to_uint(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  // Check threshold BEFORE computing sum
  // 2^24 for float, 2^53 for double
  _FpType __threshold = __fpmp2_is_fp32_v<_FpType> ? 0x1.0p24f : 0x1.0p53;
  if (__x_hi < __threshold)
  {
    // Small value: use round-toward-zero addition
    _FpType __res = ::cuda::__fpmp_add_rz(__x_hi, __x_lo);
    return ::cuda::__fpmp_fp2uint_rz(__res);
  }
  else
  {
    // Large value: integer addition; a negative tail with a fractional part
    // takes the truncation one step down (see __fpmp2_to_int)
    uint32_t __hi_uint = ::cuda::__fpmp_fp2uint_rz(__x_hi);
    int32_t __lo_int   = ::cuda::__fpmp_fp2int_rz(__x_lo);
    if ((__x_lo < _FpType(0)) && (static_cast<_FpType>(__lo_int) != __x_lo))
    {
      __hi_uint -= 1u;
    }
    return __hi_uint + __lo_int;
  }
} // __fpmp2_to_uint

// (hi, lo) -> ll conversions
template <typename _FpType>
_CCCL_FPMP_CORE_API int64_t __fpmp2_to_ll(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  _FpType __abs_hi = ::cuda::__fpmp_internal_fabs(__x_hi);
  // Check threshold BEFORE computing sum
  // 2^24 for float, 2^53 for double
  _FpType __threshold = __fpmp2_is_fp32_v<_FpType> ? 0x1.0p24f : 0x1.0p53;
  if (__abs_hi < __threshold)
  {
    // Small value: use round-toward-zero addition
    _FpType __res = ::cuda::__fpmp_add_rz(__x_hi, __x_lo);
    return ::cuda::__fpmp_fp2ll_rz(__res);
  }
  else
  {
    // Large value: integer addition; a tail of the opposite sign with a
    // fractional part takes the truncation one step toward zero (see
    // __fpmp2_to_int)
    int64_t __hi_ll = ::cuda::__fpmp_fp2ll_rz(__x_hi);
    int64_t __lo_ll = ::cuda::__fpmp_fp2ll_rz(__x_lo);
    if ((static_cast<_FpType>(__lo_ll) != __x_lo) && ((__x_lo < _FpType(0)) != (__x_hi < _FpType(0))))
    {
      __hi_ll += (__x_hi < _FpType(0)) ? 1 : -1;
    }
    return __hi_ll + __lo_ll;
  }
} // __fpmp2_to_ll

// (hi, lo) -> ull conversions
template <typename _FpType>
_CCCL_FPMP_CORE_API uint64_t __fpmp2_to_ull(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  // Check threshold BEFORE computing sum
  // 2^24 for float, 2^53 for double
  _FpType __threshold = __fpmp2_is_fp32_v<_FpType> ? 0x1.0p24f : 0x1.0p53;
  if (__x_hi < __threshold)
  {
    // Small value: use round-toward-zero addition
    _FpType __res = ::cuda::__fpmp_add_rz(__x_hi, __x_lo);
    return ::cuda::__fpmp_fp2ull_rz(__res);
  }
  else
  {
    // Large value: integer addition; a negative tail with a fractional part
    // takes the truncation one step down (see __fpmp2_to_int)
    uint64_t __hi_ull = ::cuda::__fpmp_fp2ull_rz(__x_hi);
    int64_t __lo_ll   = ::cuda::__fpmp_fp2ll_rz(__x_lo);
    if ((__x_lo < _FpType(0)) && (static_cast<_FpType>(__lo_ll) != __x_lo))
    {
      __hi_ull -= 1u;
    }
    return __hi_ull + __lo_ll;
  }
} // __fpmp2_to_ull

// __fpmp_fp128 operations (only for FpType == double)
// declared wherever the 128-bit type can be named (_CCCL_FPMP_FP128_ENABLE), device-callable
// only where fp128 arithmetic is (_CCCL_FPMP_FP128_DEVICE_OPS)
#  if _CCCL_FPMP_FP128_ENABLE == 1
template <typename _FpType>
constexpr _CCCL_FPMP_FP128_CORE_API void
__fpmp2_from_quad(const __fpmp_fp128 __x, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = static_cast<_FpType>(__x);
  *__res_lo = static_cast<_FpType>(__x - static_cast<__fpmp_fp128>(*__res_hi));
}

template <typename _FpType>
_CCCL_FPMP_FP128_CORE_API __fpmp_fp128 __fpmp2_to_quad(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  return static_cast<__fpmp_fp128>(__x_hi) + static_cast<__fpmp_fp128>(__x_lo);
}
#  endif // _CCCL_FPMP_FP128_ENABLE == 1
#else // _CCCL_FPMP_USE_LIB

// -- fp32 (single precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_double(const double __x, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_mid_from_double(const double __x, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_low_from_double(const double __x, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_high_from_double(const double __x, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_int(const int32_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_uint(const uint32_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_ll(const int64_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_from_ull(const uint64_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp32mp2_to_double(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp32mp2_mid_to_double(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp32mp2_low_to_double(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp32mp2_high_to_double(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL float __fp32mp2_to_float(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int32_t __fp32mp2_to_int(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint32_t __fp32mp2_to_uint(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int64_t __fp32mp2_to_ll(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint64_t __fp32mp2_to_ull(const float __x_hi, const float __x_lo) noexcept;

// -- fp64 (double precision) built-in declarations --
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_double(const double __x, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_mid_from_double(const double __x, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_low_from_double(const double __x, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_high_from_double(const double __x, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_int(const int32_t __i, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_uint(const uint32_t __i, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_ll(const int64_t __i, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_from_ull(const uint64_t __i, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp64mp2_to_double(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp64mp2_mid_to_double(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp64mp2_low_to_double(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double __fp64mp2_high_to_double(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL float __fp64mp2_to_float(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int32_t __fp64mp2_to_int(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint32_t __fp64mp2_to_uint(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int64_t __fp64mp2_to_ll(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint64_t __fp64mp2_to_ull(const double __x_hi, const double __x_lo) noexcept;
#  if _CCCL_FPMP_FP128_ENABLE == 1
_CCCL_FPMP_FP128_BUILTIN_DECL void
__fp64mp2_from_quad(const __fpmp_fp128 __x, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_FP128_BUILTIN_DECL __fpmp_fp128 __fp64mp2_to_quad(const double __x_hi, const double __x_lo) noexcept;
#  endif // _CCCL_FPMP_FP128_ENABLE == 1

// -- type-generic template declarations (dispatch to fp32/fp64) --
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_double(const double __x, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_mid_from_double(const double __x, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_low_from_double(const double __x, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_high_from_double(const double __x, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_int(const int32_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_uint(const uint32_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_ll(const int64_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_ull(const uint64_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline double __fpmp2_to_double(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline double __fpmp2_mid_to_double(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline double __fpmp2_low_to_double(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline double __fpmp2_high_to_double(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline float __fpmp2_to_float(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline int32_t __fpmp2_to_int(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline uint32_t __fpmp2_to_uint(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline int64_t __fpmp2_to_ll(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_HOST_DEVICE_API inline uint64_t __fpmp2_to_ull(const _Tp __x_hi, const _Tp __x_lo) noexcept;
#  if _CCCL_FPMP_FP128_ENABLE == 1
template <typename _Tp>
_CCCL_FPMP_FP128_API inline void __fpmp2_from_quad(const __fpmp_fp128 __x, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_FPMP_FP128_API inline __fpmp_fp128 __fpmp2_to_quad(const _Tp __x_hi, const _Tp __x_lo) noexcept;
#  endif // _CCCL_FPMP_FP128_ENABLE == 1

// -- fp32 template specializations --
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_double<float>(const double __x, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_mid_from_double<float>(const double __x, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_mid_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_low_from_double<float>(const double __x, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_low_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_high_from_double<float>(const double __x, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_high_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_int<float>(const int32_t __i, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_from_int(__i, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_uint<float>(const uint32_t __i, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_from_uint(__i, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_ll<float>(const int64_t __i, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_from_ll(__i, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_ull<float>(const uint64_t __i, float* __res_hi, float* __res_lo) noexcept
{
  ::cuda::__fp32mp2_from_ull(__i, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp2_to_double<float>(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::__fp32mp2_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp2_mid_to_double<float>(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::__fp32mp2_mid_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp2_low_to_double<float>(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::__fp32mp2_low_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp2_high_to_double<float>(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::__fp32mp2_high_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline float __fpmp2_to_float<float>(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::__fp32mp2_to_float(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline int32_t __fpmp2_to_int<float>(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::__fp32mp2_to_int(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline uint32_t __fpmp2_to_uint<float>(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::__fp32mp2_to_uint(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline int64_t __fpmp2_to_ll<float>(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::__fp32mp2_to_ll(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline uint64_t __fpmp2_to_ull<float>(const float __x_hi, const float __x_lo) noexcept
{
  return ::cuda::__fp32mp2_to_ull(__x_hi, __x_lo);
}

// -- fp64 template specializations --
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_from_double<double>(const double __x, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_mid_from_double<double>(const double __x, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_mid_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_low_from_double<double>(const double __x, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_low_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_high_from_double<double>(const double __x, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_high_from_double(__x, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_from_int<double>(const int32_t __i, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_from_int(__i, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_from_uint<double>(const uint32_t __i, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_from_uint(__i, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void __fpmp2_from_ll<double>(const int64_t __i, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_from_ll(__i, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline void
__fpmp2_from_ull<double>(const uint64_t __i, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_from_ull(__i, __res_hi, __res_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp2_to_double<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp2_mid_to_double<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_mid_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp2_low_to_double<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_low_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline double __fpmp2_high_to_double<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_high_to_double(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline float __fpmp2_to_float<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_to_float(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline int32_t __fpmp2_to_int<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_to_int(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline uint32_t __fpmp2_to_uint<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_to_uint(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline int64_t __fpmp2_to_ll<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_to_ll(__x_hi, __x_lo);
}
template <>
_CCCL_HOST_DEVICE_API inline uint64_t __fpmp2_to_ull<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_to_ull(__x_hi, __x_lo);
}
#  if _CCCL_FPMP_FP128_ENABLE == 1
template <>
_CCCL_FPMP_FP128_API inline void
__fpmp2_from_quad<double>(const __fpmp_fp128 __x, double* __res_hi, double* __res_lo) noexcept
{
  ::cuda::__fp64mp2_from_quad(__x, __res_hi, __res_lo);
}
template <>
_CCCL_FPMP_FP128_API inline __fpmp_fp128 __fpmp2_to_quad<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::cuda::__fp64mp2_to_quad(__x_hi, __x_lo);
}
#  endif // _CCCL_FPMP_FP128_ENABLE == 1

#endif // _CCCL_FPMP_USE_LIB
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_IMPL_CVT_H
