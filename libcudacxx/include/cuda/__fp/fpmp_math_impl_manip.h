//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_MANIP_H
#define _CUDA___FP_FPMP_MATH_IMPL_MANIP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_manip.h - fpmp2 floating-point manipulation (frexp, ldexp, modf, scalbn, ilogb, logb, nextafter,
   copysign, fabs)
    ==================================================================================================
    Per-family math implementation split out of <cuda/__fp/fpmp_math.h>. Carries the
    dedicated fp32mp2 kernels and the fp64mp2 (<double>) specializations for this
    family. Shared helpers, constants and the fp128 scaffolding live in
    <cuda/__fp/fpmp_math_impl.h>, which this header includes.
*/

#include <cuda/__fp/fpmp_math_impl.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * --------------------------------------------------------------------
 * ldexp(x, n) = x * 2^n  (fp32mp2 dedicated)
 * --------------------------------------------------------------------
 * Replaces the legacy fp64 round-trip with an all-fp32 implementation:
 * the 2^n factor is built directly from its biased fp32 exponent via
 * bit-cast and multiplied into x.  No native double is ever touched,
 * so the routine runs at full fp32 throughput on GPUs where fp64 is
 * 1:32 or worse.
 *
 * Algorithm -- 3-piece split (vs. the 2-piece used by
 * __fp32mp2_ldexp2_internal for the bounded-n exp2/exp10 callers):
 *
 *   2^n  =  2^k * 2^k * 2^(n - 2k),  k = n/3
 *
 * Each factor's biased exponent is clamped to [1, 254] (fp32 normal
 * range), so every scale is finite and strictly positive.  This
 * keeps the mul chain IEEE-clean for the special inputs that show up
 * in `ldexp` corner cases:
 *
 *   ldexp(0,   n) ->  +-0    (0 * finite positive = +-0, never NaN)
 *   ldexp(+-inf, n) ->  +-inf    (inf * finite positive = inf)
 *   ldexp(NaN, n) ->  NaN  (NaN propagates)
 *   ldexp(x,   n) -> +-inf    when |n| is huge enough to overflow fp32
 *   ldexp(x,   n) -> +-0    when |n| is huge enough to underflow fp32
 *
 * Saturation: n is pre-clamped to +-300.  Even a denormal x (|x| >=
 * 2^-149) scaled by 2^300 overflows to +inf, and any finite x scaled by
 * 2^-300 underflows to +-0 -- so further-out n values would produce the
 * same result and saturating is exact, not lossy.
 *
 * The fp64mp2 branch (compile-time-selected when FpType == double)
 * forwards to `::ldexp(double, int)`; fp64 hardware handles this in
 * one instruction and there is no fp64 cost concern on machines that
 * have fp64mp2 enabled.
 * --------------------------------------------------------------------
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void
__fpmp2_ldexp(const _FpType __x_hi, const _FpType __x_lo, int __n, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  if constexpr (__fpmp2_is_fp32_v<_FpType>)
  {
    using ffloat = fp32mp2_low;

    /* Saturate |n| to +-300.  Any |n| larger than this is provably
     * monotone in the final result (overflow or underflow) for
     * every finite fp32 input, so clamping does not lose info. */
    if (__n > 300)
    {
      __n = 300;
    }
    if (__n < -300)
    {
      __n = -300;
    }

    const int __k = __n / 3;
    int __ek1     = 127 + __k;
    int __ek2     = 127 + __k;
    int __ek3     = 127 + (__n - 2 * __k);
    if (__ek1 < 1)
    {
      __ek1 = 1;
    }
    if (__ek2 < 1)
    {
      __ek2 = 1;
    }
    if (__ek3 < 1)
    {
      __ek3 = 1;
    }
    if (__ek1 > 254)
    {
      __ek1 = 254;
    }
    if (__ek2 > 254)
    {
      __ek2 = 254;
    }
    if (__ek3 > 254)
    {
      __ek3 = 254;
    }
    const float __s1 = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek1) << 23);
    const float __s2 = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek2) << 23);
    const float __s3 = ::cuda::std::bit_cast<float>(static_cast<unsigned>(__ek3) << 23);

    const ffloat __result = ffloat(__x_hi, __x_lo) * __s1 * __s2 * __s3;

    *__res_hi = __result.hi();
    *__res_lo = __result.lo();
  }
  else
  {
    /* fp64mp2 path: forward to libm `::ldexp` via the existing
     * double round-trip.  An explicit __fpmp2_ldexp<double>
     * specialization is also provided later for symmetry with
     * the rest of the fp64mp2 math surface; this else branch
     * exists so the primary template is well-formed for any
     * `FpType` and stays compilable in isolation. */
    using mp2_t      = fpmp2<_FpType>;
    const double __r = ::ldexp(static_cast<double>(mp2_t(__x_hi, __x_lo)), __n);
    mp2_t __result(__r);
    *__res_hi = __result.hi();
    *__res_lo = __result.lo();
  }
}

/*
 * --------------------------------------------------------------------
 * scalbn(x, n) = x * FLT_RADIX^n  (fp32mp2 dedicated)
 * --------------------------------------------------------------------
 * For IEEE 754 binary formats (FLT_RADIX == 2 -- true on every
 * platform we target) `scalbn(x, n)` is bit-identical to
 * `ldexp(x, n)`, so we simply forward to the dedicated fp32mp2
 * ldexp implementation above and avoid duplicating the 3-piece
 * bit-cast scaling logic.  No fp64 round-trip is required.
 * --------------------------------------------------------------------
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void
__fpmp2_scalbn(const _FpType __x_hi, const _FpType __x_lo, int __n, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  __fpmp2_ldexp<_FpType>(__x_hi, __x_lo, __n, __res_hi, __res_lo);
}

/*
 * ============================================================================
 * Floating-Point Manipulation and Min/Max
 * ============================================================================
 * Dedicated implementations of fabs / fmin / fmax / min / max for both fp32mp2 and
 * fp64mp2 that operate directly on the (hi, lo) pair without going through
 * a lossy intermediate `double` conversion.  The same template body is
 * correct for FpType == float and FpType == double, so no separate fp64
 * specialization is needed.
 *
 * Conventions follow C99/IEEE-754:
 *   - fabs preserves NaN payload and turns -0 into +0 (via ::fabs on hi).
 *   - fmin/fmax treat NaN as missing data: if exactly one operand is NaN,
 *     the non-NaN one is returned; if both are NaN, NaN is returned.
 *   - min/max keep std::min/std::max branch semantics:
 *     ties/unordered select the first argument.
 * ============================================================================
 */

/*
 * fabs: |x|.  For a normalized (hi, lo) pair the value's sign is the sign
 * of `hi`, and `|hi| > |lo|`, so flipping both components when `hi` is
 * negative yields the absolute value while preserving the residual `lo`
 * exactly.  We use ::fabs on `hi` to get IEEE-correct handling of -0 and
 * NaN, and use the sign of the original `hi` to decide whether to flip
 * `lo`.
 */
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void
__fpmp2_fabs(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
{
  *__res_hi = ::fabs(__x_hi);
  *__res_lo = (__x_hi < _FpType(0)) ? -__x_lo : __x_lo;
}

/* log2, log10, exp2, exp10, expm1: dedicated fp32mp2 implementations
 * live in the dedicated math section above (composed over the
 * dedicated fp32mp2 log/exp with ln(2)/ln(10) constants in fp32mp2;
 * expm1 has a small-|x| Taylor branch).  fp64mp2 specializations are
 * declared below via _CCCL_FPMP_CALL_FP64MP2_MATH.  exp10 had a hand
 * fallback even on fp32mp2 prior to the dedicated implementation;
 * that version is superseded. */
_CCCL_FPMP_MATH_PLACEHOLDER_1A(logb)
_CCCL_FPMP_MATH_PLACEHOLDER_2A(copysign)
_CCCL_FPMP_MATH_PLACEHOLDER_2A(nextafter)
_CCCL_FPMP_MATH_PLACEHOLDER_1A_RETINT(ilogb)
/* ldexp, scalbn: dedicated fp32mp2 implementations live in the
 * dedicated math section above (bit-cast 3-piece base-2 scaling,
 * no fp64 round-trip).  scalbn forwards to ldexp since FLT_RADIX
 * is 2 on every IEEE 754 platform we support.  fp64mp2 paths stay
 * on the explicit double specializations further below for
 * symmetry. */
_CCCL_FPMP_MATH_PLACEHOLDER_FP_LINT(scalbln)

// frexp: extract mantissa and exponent
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void
__fpmp2_frexp(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo, int* __nptr) noexcept
{
  using mp2_t = fpmp2<_FpType>;
  double __r  = ::frexp(static_cast<double>(mp2_t(__x_hi, __x_lo)), __nptr);
  mp2_t __result(__r);
  *__res_hi = __result.hi();
  *__res_lo = __result.lo();
}

// modf: break into integer and fractional parts
template <typename _FpType = float>
_CCCL_FPMP_CORE_API void __fpmp2_modf(
  const _FpType __x_hi,
  const _FpType __x_lo,
  _FpType* __res_hi,
  _FpType* __res_lo,
  _FpType* __iptr_hi,
  _FpType* __iptr_lo) noexcept
{
  using mp2_t = fpmp2<_FpType>;
  double __ipart;
  double __r = ::modf(static_cast<double>(mp2_t(__x_hi, __x_lo)), &__ipart);
  mp2_t __result(__r), __iresult(__ipart);
  *__res_hi  = __result.hi();
  *__res_lo  = __result.lo();
  *__iptr_hi = __iresult.hi();
  *__iptr_lo = __iresult.lo();
}
template <>
_CCCL_API inline void
__fpmp2_logb<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fpmp2_from_double(::logb(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_copysign<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fpmp2_from_double(
    ::copysign(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_nextafter<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fpmp2_from_double(
    ::nextafter(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);
}
template <>
_CCCL_API inline int __fpmp2_ilogb<double>(const double __x_hi, const double __x_lo) noexcept
{
  return ::ilogb(__fpmp2_to_double(__x_hi, __x_lo));
}
template <>
_CCCL_API inline void
__fpmp2_ldexp<double>(const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept
{
  __fpmp2_from_double(::ldexp(__fpmp2_to_double(__x_hi, __x_lo), __n), __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_scalbn<double>(const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept
{
  __fpmp2_from_double(::scalbn(__fpmp2_to_double(__x_hi, __x_lo), __n), __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_scalbln<double>(
  const double __x_hi, const double __x_lo, long int __n, double* __res_hi, double* __res_lo) noexcept
{
  __fpmp2_from_double(::scalbln(__fpmp2_to_double(__x_hi, __x_lo), __n), __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_frexp<double>(
  const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo, int* __nptr) noexcept
{
  __fpmp2_from_double(::frexp(__fpmp2_to_double(__x_hi, __x_lo), __nptr), __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_modf<double>(
  const double __x_hi,
  const double __x_lo,
  double* __res_hi,
  double* __res_lo,
  double* __iptr_hi,
  double* __iptr_lo) noexcept
{
  double __ip;
  __fpmp2_from_double(::modf(__fpmp2_to_double(__x_hi, __x_lo), &__ip), __res_hi, __res_lo);
  __fpmp2_from_double(__ip, __iptr_hi, __iptr_lo);
}

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_MANIP_H
