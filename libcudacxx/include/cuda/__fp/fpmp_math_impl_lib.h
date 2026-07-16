//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_IMPL_LIB_H
#define _CUDA___FP_FPMP_MATH_IMPL_LIB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
    fpmp_math_impl_lib.h - fpmp2 math declarations for library (precompiled-kernel) mode
    ==================================================================================================
    Active only when _CCCL_FPMP_USE_LIB is defined. Declares the low-level
    fp32mp2/fp64mp2 kernels (the __fp32mp2_ and __fp64mp2_ entry points) provided
    by the compiled fpmp library together with the generic __fpmp2_ template
    wrappers (and their float/double specializations) that forward to those
    kernels. In header-only mode this file expands to nothing -- the kernels are
    provided by the per-family fpmp_math_impl_<family>.h headers instead.

    Included (under _CCCL_FPMP_USE_LIB) by <cuda/__fp/fpmp_math.h>.
*/

#include <cuda/__fp/fpmp.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if (defined _CCCL_FPMP_USE_LIB)
/*
 * ============================================================================
 * Library mode - fp32mp2 declarations
 * ============================================================================
 */
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_exp(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_log(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_log2(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_log10(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_log1p(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_pow(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_cbrt(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_sin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_cos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_sincos(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_asin(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_acos(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_atan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_atan2(
  const float __y_hi,
  const float __y_lo,
  const float __x_hi,
  const float __x_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_sinh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_cosh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_tanh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_erf(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_erfc(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_normcdfinv(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_icdf32(uint32_t __x, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_icdf64(uint64_t __x, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_acosh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_asinh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_atanh(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_tan(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_exp2(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_exp10(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_expm1(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_logb(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_ceil(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_floor(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_trunc(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_round(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_rint(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_nearbyint(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_fabs(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_lgamma(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_tgamma(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_j0(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_j1(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_y0(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_y1(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_cyl_bessel_i0(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_cyl_bessel_i1(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_sinpi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_cospi(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_normcdf(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_rcbrt(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_erfcinv(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_erfinv(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_erfcx(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_boys_f0(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_norm3d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_norm4d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  const float __d_hi,
  const float __d_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_rnorm3d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_rnorm4d(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  const float __d_hi,
  const float __d_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fmax(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fmin(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_max(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_min(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fmod(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_remainder(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_hypot(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_copysign(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fdim(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_nextafter(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_rhypot(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_remquo(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo,
  int* __quo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp32mp2_ilogb(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL long long int __fp32mp2_llrint(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL long long int __fp32mp2_llround(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL long int __fp32mp2_lrint(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL long int __fp32mp2_lround(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp32mp2_isfinite(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp32mp2_isinf(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp32mp2_isnan(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp32mp2_signbit(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_ldexp(const float __x_hi, const float __x_lo, int __n, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_scalbn(const float __x_hi, const float __x_lo, int __n, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_scalbln(const float __x_hi, const float __x_lo, long int __n, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_jn(int __n, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_yn(int __n, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp32mp2_frexp(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, int* __nptr) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_modf(
  const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, float* __iptr_hi, float* __iptr_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp32mp2_sincospi(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept;

/*
 * ============================================================================
 * Library mode - fp64mp2 declarations
 * ============================================================================
 */
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_exp(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_log(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_log2(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_log10(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_log1p(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_pow(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_cbrt(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_sin(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_cos(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_sincos(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_asin(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_acos(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_atan(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_atan2(
  const double __y_hi,
  const double __y_lo,
  const double __x_hi,
  const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_sinh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_cosh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_tanh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_erf(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_erfc(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_normcdfinv(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_acosh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_asinh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_atanh(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_tan(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_exp2(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_exp10(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_expm1(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_logb(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_ceil(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_floor(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_trunc(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_round(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_rint(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_nearbyint(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_fabs(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_lgamma(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_tgamma(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_j0(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_j1(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_y0(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_y1(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_cyl_bessel_i0(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_cyl_bessel_i1(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_sinpi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_cospi(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_normcdf(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_rcbrt(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_erfcinv(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_erfinv(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_erfcx(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_boys_f0(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_norm3d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_norm4d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  const double __d_hi,
  const double __d_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_rnorm3d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_rnorm4d(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  const double __d_hi,
  const double __d_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fmax(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fmin(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_max(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_min(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fmod(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_remainder(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_hypot(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_copysign(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fdim(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_nextafter(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_rhypot(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_remquo(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo,
  int* __quo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp64mp2_ilogb(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL long long int __fp64mp2_llrint(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL long long int __fp64mp2_llround(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL long int __fp64mp2_lrint(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL long int __fp64mp2_lround(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp64mp2_isfinite(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp64mp2_isinf(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp64mp2_isnan(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int __fp64mp2_signbit(const double __x_hi, const double __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_ldexp(const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_scalbn(const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_scalbln(const double __x_hi, const double __x_lo, long int __n, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_jn(int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_yn(int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void
__fp64mp2_frexp(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo, int* __nptr) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_modf(
  const double __x_hi,
  const double __x_lo,
  double* __res_hi,
  double* __res_lo,
  double* __iptr_hi,
  double* __iptr_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void __fp64mp2_sincospi(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept;

/*
 * ============================================================================
 * Template declarations and float specializations
 * ============================================================================
 */
template <typename _Tp>
_CCCL_API inline void __fpmp2_exp(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_log(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_log2(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_log10(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_log1p(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_pow(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_cbrt(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_sin(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_cos(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void
__fpmp2_sincos(const _Tp __x_hi, const _Tp __x_lo, _Tp* __sin_hi, _Tp* __sin_lo, _Tp* __cos_hi, _Tp* __cos_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_asin(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_acos(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_atan(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_atan2(
  const _Tp __y_hi, const _Tp __y_lo, const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_sinh(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_cosh(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_tanh(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_erf(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_erfc(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_normcdfinv(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_acosh(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_asinh(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_atanh(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_tan(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_exp2(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_exp10(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_expm1(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_logb(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_ceil(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_floor(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_trunc(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_round(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_rint(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_nearbyint(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_fabs(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_lgamma(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_tgamma(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_j0(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_j1(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_y0(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_y1(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_cyl_bessel_i0(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_cyl_bessel_i1(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_sinpi(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_cospi(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_normcdf(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_rcbrt(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_erfcinv(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_erfinv(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_erfcx(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_boys_f0(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_norm3d(
  const _Tp __a_hi,
  const _Tp __a_lo,
  const _Tp __b_hi,
  const _Tp __b_lo,
  const _Tp __c_hi,
  const _Tp __c_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_norm4d(
  const _Tp __a_hi,
  const _Tp __a_lo,
  const _Tp __b_hi,
  const _Tp __b_lo,
  const _Tp __c_hi,
  const _Tp __c_lo,
  const _Tp __d_hi,
  const _Tp __d_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_rnorm3d(
  const _Tp __a_hi,
  const _Tp __a_lo,
  const _Tp __b_hi,
  const _Tp __b_lo,
  const _Tp __c_hi,
  const _Tp __c_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_rnorm4d(
  const _Tp __a_hi,
  const _Tp __a_lo,
  const _Tp __b_hi,
  const _Tp __b_lo,
  const _Tp __c_hi,
  const _Tp __c_lo,
  const _Tp __d_hi,
  const _Tp __d_lo,
  _Tp* __res_hi,
  _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_fmax(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_fmin(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_max(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_min(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_fmod(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_remainder(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_hypot(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_copysign(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_fdim(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_nextafter(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_rhypot(
  const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_remquo(
  const _Tp __x_hi,
  const _Tp __x_lo,
  const _Tp __y_hi,
  const _Tp __y_lo,
  _Tp* __res_hi,
  _Tp* __res_lo,
  int* __quo) noexcept;
template <typename _Tp>
_CCCL_API inline int __fpmp2_ilogb(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline long long int __fpmp2_llrint(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline long long int __fpmp2_llround(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline long int __fpmp2_lrint(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline long int __fpmp2_lround(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline int __fpmp2_isfinite(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline int __fpmp2_isinf(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline int __fpmp2_isnan(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline int __fpmp2_signbit(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_ldexp(const _Tp __x_hi, const _Tp __x_lo, int __n, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_scalbn(const _Tp __x_hi, const _Tp __x_lo, int __n, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void
__fpmp2_scalbln(const _Tp __x_hi, const _Tp __x_lo, long int __n, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_jn(int __n, const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_yn(int __n, const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void
__fpmp2_frexp(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo, int* __nptr) noexcept;
template <typename _Tp>
_CCCL_API inline void
__fpmp2_modf(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo, _Tp* __iptr_hi, _Tp* __iptr_lo) noexcept;
template <typename _Tp>
_CCCL_API inline void __fpmp2_sincospi(
  const _Tp __x_hi, const _Tp __x_lo, _Tp* __sin_hi, _Tp* __sin_lo, _Tp* __cos_hi, _Tp* __cos_lo) noexcept;

_CCCL_API inline void __fpmp2_icdf(uint32_t __x, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_icdf32(__x, __res_hi, __res_lo);
}
_CCCL_API inline void __fpmp2_icdf(uint64_t __x, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_icdf64(__x, __res_hi, __res_lo);
}

// Float (fp32) template specializations
template <>
_CCCL_API inline void
__fpmp2_exp<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_exp(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_log<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_log(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_log2<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_log2(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_log10<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_log10(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_log1p<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_log1p(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_pow<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_pow(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cbrt<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_cbrt(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_sin<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_sin(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cos<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_cos(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_sincos<float>(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  __fp32mp2_sincos(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo);
}
template <>
_CCCL_API inline void
__fpmp2_asin<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_asin(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_acos<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_acos(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_atan<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_atan(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_atan2<float>(
  const float __y_hi,
  const float __y_lo,
  const float __x_hi,
  const float __x_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_atan2(__y_hi, __y_lo, __x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_sinh<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_sinh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cosh<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_cosh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_tanh<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_tanh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erf<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_erf(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erfc<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_erfc(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_normcdfinv<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_normcdfinv(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_acosh<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_acosh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_asinh<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_asinh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_atanh<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_atanh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_tan<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_tan(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_exp2<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_exp2(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_exp10<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_exp10(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_expm1<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_expm1(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_logb<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_logb(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_ceil<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_ceil(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_floor<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_floor(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_trunc<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_trunc(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_round<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_round(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_rint<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_rint(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_nearbyint<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_nearbyint(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_fabs<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_fabs(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_lgamma<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_lgamma(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_tgamma<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_tgamma(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_j0<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_j0(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_j1<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_j1(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_y0<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_y0(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_y1<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_y1(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cyl_bessel_i0<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_cyl_bessel_i0(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cyl_bessel_i1<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_cyl_bessel_i1(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_sinpi<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_sinpi(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cospi<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_cospi(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_normcdf<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_normcdf(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_rcbrt<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_rcbrt(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erfcinv<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_erfcinv(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erfinv<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_erfinv(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erfcx<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_erfcx(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_boys_f0<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_boys_f0(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_norm3d<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_norm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_norm4d<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  const float __d_hi,
  const float __d_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_norm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_rnorm3d<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_rnorm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_rnorm4d<float>(
  const float __a_hi,
  const float __a_lo,
  const float __b_hi,
  const float __b_lo,
  const float __c_hi,
  const float __c_lo,
  const float __d_hi,
  const float __d_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_rnorm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_fmax<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_fmax(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_fmin<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_fmin(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_max<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_max(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_min<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_min(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_fmod<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_fmod(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_remainder<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_remainder(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_hypot<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_hypot(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_copysign<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_copysign(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_fdim<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_fdim(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_nextafter<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_nextafter(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_rhypot<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo) noexcept
{
  __fp32mp2_rhypot(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_remquo<float>(
  const float __x_hi,
  const float __x_lo,
  const float __y_hi,
  const float __y_lo,
  float* __res_hi,
  float* __res_lo,
  int* __quo) noexcept
{
  __fp32mp2_remquo(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, __quo);
}
template <>
_CCCL_API inline int __fpmp2_ilogb<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_ilogb(__x_hi, __x_lo);
}
template <>
_CCCL_API inline long long int __fpmp2_llrint<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_llrint(__x_hi, __x_lo);
}
template <>
_CCCL_API inline long long int __fpmp2_llround<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_llround(__x_hi, __x_lo);
}
template <>
_CCCL_API inline long int __fpmp2_lrint<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_lrint(__x_hi, __x_lo);
}
template <>
_CCCL_API inline long int __fpmp2_lround<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_lround(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int __fpmp2_isfinite<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_isfinite(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int __fpmp2_isinf<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_isinf(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int __fpmp2_isnan<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_isnan(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int __fpmp2_signbit<float>(const float __x_hi, const float __x_lo) noexcept
{
  return __fp32mp2_signbit(__x_hi, __x_lo);
}
template <>
_CCCL_API inline void
__fpmp2_ldexp<float>(const float __x_hi, const float __x_lo, int __n, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_ldexp(__x_hi, __x_lo, __n, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_scalbn<float>(const float __x_hi, const float __x_lo, int __n, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_scalbn(__x_hi, __x_lo, __n, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_scalbln<float>(const float __x_hi, const float __x_lo, long int __n, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_scalbln(__x_hi, __x_lo, __n, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_jn<float>(int __n, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_jn(__n, __x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_yn<float>(int __n, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept
{
  __fp32mp2_yn(__n, __x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_frexp<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, int* __nptr) noexcept
{
  __fp32mp2_frexp(__x_hi, __x_lo, __res_hi, __res_lo, __nptr);
}
template <>
_CCCL_API inline void __fpmp2_modf<float>(
  const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, float* __iptr_hi, float* __iptr_lo) noexcept
{
  __fp32mp2_modf(__x_hi, __x_lo, __res_hi, __res_lo, __iptr_hi, __iptr_lo);
}
template <>
_CCCL_API inline void __fpmp2_sincospi<float>(
  const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept
{
  __fp32mp2_sincospi(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo);
}

/*
 * ============================================================================
 * Double (fp64) template specializations
 * ============================================================================
 */
template <>
_CCCL_API inline void
__fpmp2_exp<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_exp(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_log<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_log(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_log2<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_log2(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_log10<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_log10(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_log1p<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_log1p(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_pow<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_pow(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cbrt<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_cbrt(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_sin<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_sin(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cos<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_cos(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_sincos<double>(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept
{
  __fp64mp2_sincos(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo);
}
template <>
_CCCL_API inline void
__fpmp2_asin<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_asin(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_acos<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_acos(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_atan<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_atan(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_atan2<double>(
  const double __y_hi,
  const double __y_lo,
  const double __x_hi,
  const double __x_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_atan2(__y_hi, __y_lo, __x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_sinh<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_sinh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cosh<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_cosh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_tanh<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_tanh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erf<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_erf(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erfc<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_erfc(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_normcdfinv<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_normcdfinv(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_acosh<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_acosh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_asinh<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_asinh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_atanh<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_atanh(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_tan<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_tan(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_exp2<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_exp2(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_exp10<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_exp10(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_expm1<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_expm1(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_logb<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_logb(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_ceil<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_ceil(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_floor<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_floor(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_trunc<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_trunc(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_round<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_round(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_rint<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_rint(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_nearbyint<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_nearbyint(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_fabs<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_fabs(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_lgamma<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_lgamma(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_tgamma<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_tgamma(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_j0<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_j0(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_j1<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_j1(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_y0<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_y0(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_y1<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_y1(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cyl_bessel_i0<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_cyl_bessel_i0(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cyl_bessel_i1<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_cyl_bessel_i1(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_sinpi<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_sinpi(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_cospi<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_cospi(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_normcdf<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_normcdf(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_rcbrt<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_rcbrt(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erfcinv<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_erfcinv(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erfinv<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_erfinv(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_erfcx<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_erfcx(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_boys_f0<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_boys_f0(__x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_norm3d<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_norm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_norm4d<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  const double __d_hi,
  const double __d_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_norm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_rnorm3d<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_rnorm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_rnorm4d<double>(
  const double __a_hi,
  const double __a_lo,
  const double __b_hi,
  const double __b_lo,
  const double __c_hi,
  const double __c_lo,
  const double __d_hi,
  const double __d_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_rnorm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_fmax<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_fmax(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_fmin<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_fmin(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_max<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_max(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_min<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_min(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_fmod<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_fmod(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_remainder<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_remainder(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_hypot<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_hypot(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
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
  __fp64mp2_copysign(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
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
  __fp64mp2_fdim(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
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
  __fp64mp2_nextafter(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_rhypot<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo) noexcept
{
  __fp64mp2_rhypot(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_remquo<double>(
  const double __x_hi,
  const double __x_lo,
  const double __y_hi,
  const double __y_lo,
  double* __res_hi,
  double* __res_lo,
  int* __quo) noexcept
{
  __fp64mp2_remquo(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, __quo);
}
template <>
_CCCL_API inline int __fpmp2_ilogb<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_ilogb(__x_hi, __x_lo);
}
template <>
_CCCL_API inline long long int __fpmp2_llrint<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_llrint(__x_hi, __x_lo);
}
template <>
_CCCL_API inline long long int __fpmp2_llround<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_llround(__x_hi, __x_lo);
}
template <>
_CCCL_API inline long int __fpmp2_lrint<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_lrint(__x_hi, __x_lo);
}
template <>
_CCCL_API inline long int __fpmp2_lround<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_lround(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int __fpmp2_isfinite<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_isfinite(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int __fpmp2_isinf<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_isinf(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int __fpmp2_isnan<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_isnan(__x_hi, __x_lo);
}
template <>
_CCCL_API inline int __fpmp2_signbit<double>(const double __x_hi, const double __x_lo) noexcept
{
  return __fp64mp2_signbit(__x_hi, __x_lo);
}
template <>
_CCCL_API inline void
__fpmp2_ldexp<double>(const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_ldexp(__x_hi, __x_lo, __n, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_scalbn<double>(const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_scalbn(__x_hi, __x_lo, __n, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_scalbln<double>(
  const double __x_hi, const double __x_lo, long int __n, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_scalbln(__x_hi, __x_lo, __n, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_jn<double>(int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_jn(__n, __x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void
__fpmp2_yn<double>(int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
{
  __fp64mp2_yn(__n, __x_hi, __x_lo, __res_hi, __res_lo);
}
template <>
_CCCL_API inline void __fpmp2_frexp<double>(
  const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo, int* __nptr) noexcept
{
  __fp64mp2_frexp(__x_hi, __x_lo, __res_hi, __res_lo, __nptr);
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
  __fp64mp2_modf(__x_hi, __x_lo, __res_hi, __res_lo, __iptr_hi, __iptr_lo);
}
template <>
_CCCL_API inline void __fpmp2_sincospi<double>(
  const double __x_hi,
  const double __x_lo,
  double* __sin_hi,
  double* __sin_lo,
  double* __cos_hi,
  double* __cos_lo) noexcept
{
  __fp64mp2_sincospi(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo);
}

#endif // _CCCL_FPMP_USE_LIB
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_IMPL_LIB_H
