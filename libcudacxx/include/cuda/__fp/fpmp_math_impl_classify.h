//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_CLASSIFY_H
#define _CUDA___FP_FPMP_MATH_IMPL_CLASSIFY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_classify.h - fpmp2 classification and comparison (isfinite/isinf/isnan/signbit, fmax/fmin/max/min,
   fdim)
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
 * fmax: max(x, y).  Lexicographic comparison on (hi, lo) -- valid because
 * normalized fpmp2 inputs satisfy |lo| < ulp(hi)/2, so `x > y` iff
 * `x_hi > y_hi || (x_hi == y_hi && x_lo > y_lo)`.  NaN handling follows
 * C99/IEEE-754-2008: if exactly one operand is NaN, return the other.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_fmax(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const bool __x_is_nan = __fpmp_internal_isnan(__x_hi);
  const bool __y_is_nan = __fpmp_internal_isnan(__y_hi);
  if (__x_is_nan && !__y_is_nan)
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
    return;
  }
  if (__y_is_nan && !__x_is_nan)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }
  const bool __x_greater = (__x_hi > __y_hi) || (__x_hi == __y_hi && __x_lo > __y_lo);
  if (__x_greater)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
  }
  else
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
  }
}

/*
 * fmin: min(x, y).  Mirror image of fmax.
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_fmin(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const bool __x_is_nan = __fpmp_internal_isnan(__x_hi);
  const bool __y_is_nan = __fpmp_internal_isnan(__y_hi);
  if (__x_is_nan && !__y_is_nan)
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
    return;
  }
  if (__y_is_nan && !__x_is_nan)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
    return;
  }
  const bool __x_less = (__x_hi < __y_hi) || (__x_hi == __y_hi && __x_lo < __y_lo);
  if (__x_less)
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
  }
  else
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
  }
}

/*
 * max: std::max-like selection for fpmp2 values.  Uses the same
 * lexicographic ordering as fmax, but keeps std::max semantics:
 * return y only when x < y; otherwise return x (ties/unordered -> x).
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_max(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const bool __x_less = (__x_hi < __y_hi) || (__x_hi == __y_hi && __x_lo < __y_lo);
  if (__x_less)
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
  }
  else
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
  }
}

/*
 * min: std::min-like selection for fpmp2 values.  Uses the same
 * lexicographic ordering as fmin, but keeps std::min semantics:
 * return y only when y < x; otherwise return x (ties/unordered -> x).
 */
template <typename _FpType>
_CCCL_FPMP_CORE_API void __fpmp2_min(
  const _FpType __x_hi,
  const _FpType __x_lo,
  const _FpType __y_hi,
  const _FpType __y_lo,
  _FpType* __res_hi,
  _FpType* __res_lo) noexcept
{
  const bool __y_less = (__y_hi < __x_hi) || (__y_hi == __x_hi && __y_lo < __x_lo);
  if (__y_less)
  {
    *__res_hi = __y_hi;
    *__res_lo = __y_lo;
  }
  else
  {
    *__res_hi = __x_hi;
    *__res_lo = __x_lo;
  }
}
_CCCL_FPMP_MATH_PLACEHOLDER_2A(fdim)

// Classification and sign functions
template <typename _FpType>
_CCCL_FPMP_CORE_API int __fpmp2_isfinite(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  (void) __x_lo;
  return (std::isfinite) (static_cast<double>(__x_hi));
}

template <typename _FpType>
_CCCL_FPMP_CORE_API int __fpmp2_isinf(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  (void) __x_lo;
  return (std::isinf) (static_cast<double>(__x_hi));
}

template <typename _FpType>
_CCCL_FPMP_CORE_API int __fpmp2_isnan(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  (void) __x_lo;
  return (std::isnan) (static_cast<double>(__x_hi));
}

template <typename _FpType>
_CCCL_FPMP_CORE_API int __fpmp2_signbit(const _FpType __x_hi, const _FpType __x_lo) noexcept
{
  (void) __x_lo;
  return (std::signbit) (static_cast<double>(__x_hi));
}
template <>
_CCCL_API inline void __fpmp2_fdim<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fpmp2_from_double(::fdim(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);
}
template <>
_CCCL_API inline int __fpmp2_isfinite<double>(const double __x_hi, const double __x_lo) noexcept
{
  (void) __x_lo;
  return (std::isfinite) (__x_hi);
}
template <>
_CCCL_API inline int __fpmp2_isinf<double>(const double __x_hi, const double __x_lo) noexcept
{
  (void) __x_lo;
  return (std::isinf) (__x_hi);
}
template <>
_CCCL_API inline int __fpmp2_isnan<double>(const double __x_hi, const double __x_lo) noexcept
{
  (void) __x_lo;
  return (std::isnan) (__x_hi);
}
template <>
_CCCL_API inline int __fpmp2_signbit<double>(const double __x_hi, const double __x_lo) noexcept
{
  (void) __x_lo;
  return (std::signbit) (__x_hi);
}

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_CLASSIFY_H
