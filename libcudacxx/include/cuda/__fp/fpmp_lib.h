//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_LIB_H
#define _CUDA___FP_FPMP_LIB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/**
 * @file fpmp_lib.h
 * @brief Core function APIs for FPMP multi-precision floating-point library
 *
 * This header provides the core function APIs for the FPMP library for multi-precision
 * floating-point arithmetic using double-float (fp32mp2) and double-double (fp64mp2)
 * representations. It defines:
 *
 * - Conversion operation APIs (float, double, integer conversions)
 * - Arithmetic operation APIs (add, sub, mul, div, acc, fma, mad)
 * - Accuracy-specific variants (low, mid, high; default == mid)
 * - Comparison operation APIs (eq, ne, lt, gt, le, ge)
 * - Square root and reciprocal square root APIs
 * - Negation and renormalization APIs
 * - Bit cast APIs
 * - Atomic operation APIs (CUDA device only)
 * - Math function APIs (exp, log, sin, cos, pow, etc.)
 *
 * Note: Warp shuffle helpers for fpmp2 pairs (__shfl_sync, __shfl_xor_sync,
 * __shfl_down_sync, __shfl_up_sync) are intentionally not part of the library
 * API. They are header-only inline templates over fpmp2_t<FpType, met>
 * provided by fpmp_math.h and have no extern "C" entry point here, because
 * their body is just two CUDA scalar shuffle intrinsics with nothing to
 * outline.
 *
 * Built-in naming convention (fp32mp2):
 *   __nv_fp32mp2_<op>           — default (== mid)
 *   __nv_fp32mp2_low_<op>       — low accuracy (minimal renormalization)
 *   __nv_fp32mp2_mid_<op>       — mid accuracy (default, Dekker-based)
 *   __nv_fp32mp2_high_<op>      — high accuracy (conservative normalization)
 *
 * Built-in naming convention (fp64mp2, when FPMP_FP64MP2_ENABLE=1):
 *   __nv_fp64mp2_<op>           — default (== mid)
 *   __nv_fp64mp2_low_<op>       — low accuracy
 *   __nv_fp64mp2_mid_<op>       — mid accuracy (default)
 *   __nv_fp64mp2_high_<op>      — high accuracy
 *
 * where <op> is the operation (add, sub, mul, div, sqrt, fma, mad, exp, etc.)
 *
 * The APIs are designed to work across both host and device code through
 * appropriate decorators.
 */

#include <cuda/__fp/fpmp_common.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

// ============================================================================
// fp32mp2 (double-float) built-in APIs
// ============================================================================

/*
 * Conversion operations
 */
/**
 * @brief Conversion functions between fp32mp2 and standard types
 *
 * Functions:
 * - __nv_fp32mp2_from_double: Convert from double to fp32mp2 (hi, lo)
 * - __nv_fp32mp2_from_int:   Convert from int32_t to fp32mp2
 * - __nv_fp32mp2_from_uint:  Convert from uint32_t to fp32mp2
 * - __nv_fp32mp2_from_ll:    Convert from int64_t to fp32mp2
 * - __nv_fp32mp2_from_ull:   Convert from uint64_t to fp32mp2
 * - __nv_fp32mp2_to_double:  Convert from fp32mp2 to double
 * - __nv_fp32mp2_to_float:   Convert from fp32mp2 to float
 * - __nv_fp32mp2_to_int:     Convert from fp32mp2 to int32_t
 * - __nv_fp32mp2_to_uint:    Convert from fp32mp2 to uint32_t
 * - __nv_fp32mp2_to_ll:      Convert from fp32mp2 to int64_t
 * - __nv_fp32mp2_to_ull:     Convert from fp32mp2 to uint64_t
 */
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_double (const double x, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_int    (const int32_t i, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_uint   (const uint32_t i, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_ll     (const int64_t i, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_ull    (const uint64_t i, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ double   __nv_fp32mp2_to_double   (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ float    __nv_fp32mp2_to_float    (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ int32_t  __nv_fp32mp2_to_int      (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ uint32_t __nv_fp32mp2_to_uint     (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ int64_t  __nv_fp32mp2_to_ll       (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ uint64_t __nv_fp32mp2_to_ull      (const float x_hi, const float x_lo);

/*
 * Addition operations
 */
/**
 * @brief Double-float addition with accuracy variants
 *
 * Each function takes two fp32mp2 operands (x_hi,x_lo) and (y_hi,y_lo)
 * and returns their sum as fp32mp2 (res_hi, res_lo).
 *
 * - __nv_fp32mp2_add:          Default (Dekker-based) addition
 * - __nv_fp32mp2_low_add:     Fast addition (no renormalization)
 * - __nv_fp32mp2_high_add: Accurate (Thall-based) addition
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_add      (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_mid_add  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_low_add  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_high_add (const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);

/*
 * Subtraction operations
 */
/**
 * @brief Double-float subtraction with accuracy variants
 *
 * Each function takes two fp32mp2 operands and returns their difference (x-y).
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sub      (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_mid_sub  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_low_sub  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_high_sub (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);

/*
 * Accumulate operations (single-component addition to multi-precision)
 */
/**
 * @brief Optimized single-component accumulation
 *
 * Adds a single float value c to an fp32mp2 accumulator (acc_hi, acc_lo).
 * More efficient than full mp2+mp2 addition (saves ~6 operations).
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_acc      (const float c, float* acc_hi, float* acc_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_mid_acc  (const float c, float* acc_hi, float* acc_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_low_acc  (const float c, float* acc_hi, float* acc_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_high_acc (const float c, float* acc_hi, float* acc_lo);

/*
 * Multiplication operations
 */
/**
 * @brief Double-float multiplication with accuracy variants
 *
 * Each function takes two fp32mp2 operands and returns their product.
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_mul      (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_mid_mul  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_low_mul  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
#if __FPMP_USE_ACCURATE_MUL__ == 1
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_high_mul (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
#endif

/*
 * Renormalization operations
 */
/**
 * @brief Renormalize an fp32mp2 value
 *
 * Ensures the two-float representation maintains strict ordering
 * and non-overlapping components.
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_renormalize (const float x_hi, const float x_lo, float* res_hi, float* res_lo);

/*
 * Division operations
 */
/**
 * @brief Double-float division with accuracy variants
 *
 * Each function takes two fp32mp2 operands and returns their quotient.
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_div      (const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_mid_div  (const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_low_div  (const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);
#if __FPMP_USE_ACCURATE_DIV__ == 1
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_high_div (const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);
#endif

/*
 * Square root & reciprocal square root operations
 */
/**
 * @brief Double-float square root and reciprocal square root
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sqrt  (const float a_hi, const float a_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rsqrt (const float a_hi, const float a_lo, float* res_hi, float* res_lo);

/*
 * Fused multiply-add operations
 */
/**
 * @brief Double-float fused multiply-add and multiply-add
 *
 * Computes (x*y)+z in extended precision.
 * - __nv_fp32mp2_fma:          Fused multiply-add (default)
 * - __nv_fp32mp2_low_fma:     Fast fused multiply-add
 * - __nv_fp32mp2_high_fma: Accurate fused multiply-add (cross-term error tracking)
 * - __nv_fp32mp2_mad:          Multiply-add (default: fast mul + default add)
 * - __nv_fp32mp2_low_mad:     Fast multiply-add (fast mul + fast add)
 * - __nv_fp32mp2_high_mad: Accurate multiply-add (default mul + accurate add)
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fma      (const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_mid_fma  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_low_fma  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_high_fma (const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_mad      (const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_mid_mad  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_low_mad  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_high_mad (const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);

/*
 * Negation operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_neg (const float x_hi, const float x_lo, float* res_hi, float* res_lo);

/*
 * Comparison operations
 */
/**
 * @brief Comparison functions for fp32mp2 values
 *
 * - __nv_fp32mp2_cmp_eq: Equality comparison (x == y)
 * - __nv_fp32mp2_cmp_ne: Inequality comparison (x != y)
 * - __nv_fp32mp2_cmp_lt: Less than comparison (x < y)
 * - __nv_fp32mp2_cmp_gt: Greater than comparison (x > y)
 * - __nv_fp32mp2_cmp_le: Less than or equal comparison (x <= y)
 * - __nv_fp32mp2_cmp_ge: Greater than or equal comparison (x >= y)
 */
__FPMP_BUILTIN_DECL__ bool __nv_fp32mp2_cmp_eq (const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp32mp2_cmp_ne (const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp32mp2_cmp_lt (const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp32mp2_cmp_gt (const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp32mp2_cmp_le (const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp32mp2_cmp_ge (const float x_hi, const float x_lo, const float y_hi, const float y_lo);

/*
 * Bit cast operations
 */
/**
 * @brief Bit-cast fp32mp2 to 64-bit integer (IEEE-754 format)
 */
__FPMP_BUILTIN_DECL__ uint64_t __nv_fp32mp2_bit_cast (const float x_hi, const float x_lo);

/*
 * Atomic operations (CUDA device only)
 */
#ifdef __CUDACC__
/**
 * @brief Atomic addition and subtraction for fp32mp2 (CUDA device only)
 */
__FPMP_BUILTIN_DEVICE_DECL__ void __nv_fp32mp2_atomicAdd (float* address_hi, float* address_lo, const float addition_hi, const float addition_lo, float* old_hi, float* old_lo);
__FPMP_BUILTIN_DEVICE_DECL__ void __nv_fp32mp2_atomicSub (float* address_hi, float* address_lo, const float val_hi, const float val_lo, float* old_hi, float* old_lo);
#endif // __CUDACC__

// ============================================================================
// fp64mp2 (double-double) built-in APIs
// ============================================================================

#if FPMP_FP64MP2_ENABLE == 1

/*
 * __fpmp_fp128 conversion operations (when available)
 */
#if FPMP_FP128_ENABLE == 1
__FPMP_BUILTIN_DECL__ void          __nv_fp64mp2_from_quad (const __fpmp_fp128 x, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ __fpmp_fp128  __nv_fp64mp2_to_quad   (const double x_hi, const double x_lo);
#endif // FPMP_FP128_ENABLE == 1

/*
 * Conversion operations
 */
/**
 * @brief Conversion functions between fp64mp2 and standard types
 */
__FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_double (const double x, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_int    (const int32_t i, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_uint   (const uint32_t i, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_ll     (const int64_t i, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_ull    (const uint64_t i, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ double   __nv_fp64mp2_to_double   (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ float    __nv_fp64mp2_to_float    (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ int32_t  __nv_fp64mp2_to_int      (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ uint32_t __nv_fp64mp2_to_uint     (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ int64_t  __nv_fp64mp2_to_ll       (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ uint64_t __nv_fp64mp2_to_ull      (const double x_hi, const double x_lo);

/*
 * Addition operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_add      (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_mid_add  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_low_add  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_high_add (const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);

/*
 * Subtraction operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sub      (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_mid_sub  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_low_sub  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_high_sub (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);

/*
 * Accumulate operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_acc      (const double c, double* acc_hi, double* acc_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_mid_acc  (const double c, double* acc_hi, double* acc_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_low_acc  (const double c, double* acc_hi, double* acc_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_high_acc (const double c, double* acc_hi, double* acc_lo);

/*
 * Multiplication operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_mul      (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_mid_mul  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_low_mul  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
#if __FPMP_USE_ACCURATE_MUL__ == 1
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_high_mul (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
#endif

/*
 * Renormalization operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_renormalize (const double x_hi, const double x_lo, double* res_hi, double* res_lo);

/*
 * Division operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_div      (const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_mid_div  (const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_low_div  (const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);
#if __FPMP_USE_ACCURATE_DIV__ == 1
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_high_div (const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);
#endif

/*
 * Square root & reciprocal square root operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sqrt  (const double a_hi, const double a_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rsqrt (const double a_hi, const double a_lo, double* res_hi, double* res_lo);

/*
 * Fused multiply-add operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fma      (const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_mid_fma  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_low_fma  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_high_fma (const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_mad      (const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_mid_mad  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_low_mad  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_high_mad (const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);

/*
 * Negation operations
 */
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_neg (const double x_hi, const double x_lo, double* res_hi, double* res_lo);

/*
 * Comparison operations
 */
__FPMP_BUILTIN_DECL__ bool __nv_fp64mp2_cmp_eq (const double x_hi, const double x_lo, const double y_hi, const double y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp64mp2_cmp_ne (const double x_hi, const double x_lo, const double y_hi, const double y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp64mp2_cmp_lt (const double x_hi, const double x_lo, const double y_hi, const double y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp64mp2_cmp_gt (const double x_hi, const double x_lo, const double y_hi, const double y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp64mp2_cmp_le (const double x_hi, const double x_lo, const double y_hi, const double y_lo);
__FPMP_BUILTIN_DECL__ bool __nv_fp64mp2_cmp_ge (const double x_hi, const double x_lo, const double y_hi, const double y_lo);

/*
 * Bit cast operations
 */
__FPMP_BUILTIN_DECL__ uint64_t __nv_fp64mp2_bit_cast (const double x_hi, const double x_lo);

/*
 * Atomic operations (CUDA device only)
 */
#ifdef __CUDACC__
__FPMP_BUILTIN_DEVICE_DECL__ void __nv_fp64mp2_atomicAdd (double* address_hi, double* address_lo, const double addition_hi, const double addition_lo, double* old_hi, double* old_lo);
__FPMP_BUILTIN_DEVICE_DECL__ void __nv_fp64mp2_atomicSub (double* address_hi, double* address_lo, const double val_hi, const double val_lo, double* old_hi, double* old_lo);
#endif // __CUDACC__


#endif // FPMP_FP64MP2_ENABLE == 1

/*
 * Math function APIs
 */
 /**
 * @brief Transcendental and special math functions
 *
 * Dedicated fp32mp2 optimizations currently include:
 * - exp, log, cbrt, rcbrt
 * - sin, cos, sincos
 * - erf, erfc, normcdfinv, icdf
 * - ceil, floor, trunc, round
 * - fabs, fmax, fmin, max, min
 *
 * Notes:
 * - rint / nearbyint currently use fallback-style implementations.
 * - fp64mp2 math functions are provided via the same API surface.
 *
 * Exponential/Logarithmic:
 * - exp, log, log2, log10, log1p, exp2, exp10, expm1, logb
 *
 * Power/Root:
 * - pow, cbrt, rcbrt
 *
 * Trigonometric:
 * - sin, cos, tan, sincos, sinpi, cospi, sincospi
 * - asin, acos, atan, atan2
 *
 * Hyperbolic:
 * - sinh, cosh, tanh, acosh, asinh, atanh
 *
 * Error/Probability:
 * - erf, erfc, normcdfinv, normcdf, icdf
 *
 * Gamma:
 * - lgamma, tgamma
 *
 * Bessel:
 * - j0, j1, jn, y0, y1, yn
 *
 * Rounding:
 * - ceil, floor, trunc, round, rint, nearbyint
 * - ilogb, lrint, lround, llrint, llround
 *
 * Floating-point manipulation:
 * - fabs, copysign, ldexp, scalbn, scalbln, frexp, modf, nextafter
 *
 * Min/Max/Difference/Remainder:
 * - fmax, fmin, max, min, fdim, fmod, remainder, remquo
 *
 * Distance:
 * - hypot, rhypot
 *
 * Classification:
 * - isfinite, isinf, isnan, signbit
 */
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_exp    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_log    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_log2   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_log10  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_log1p  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_pow    (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cbrt   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sin    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cos    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sincos (const float x_hi, const float x_lo, float* sin_hi, float* sin_lo, float* cos_hi, float* cos_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_asin   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_acos   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_atan   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_atan2  (const float y_hi, const float y_lo, const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sinh   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cosh   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_tanh   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erf    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erfc   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_normcdfinv (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_icdf32     (uint32_t x, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_icdf64     (uint64_t x, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_acosh      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_asinh      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_atanh      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_tan        (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_exp2       (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_exp10      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_expm1      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_logb       (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_ceil       (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_floor      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_trunc      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_round      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rint       (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_nearbyint  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fabs       (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_lgamma     (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_tgamma     (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_j0         (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_j1         (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_y0         (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_y1         (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cyl_bessel_i0(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cyl_bessel_i1(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erfcinv     (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erfinv      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erfcx       (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_boys_f0     (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_norm3d      (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_norm4d      (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, const float d_hi, const float d_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rnorm3d     (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rnorm4d     (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, const float d_hi, const float d_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sinpi      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cospi      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sincospi   (const float x_hi, const float x_lo, float* sin_hi, float* sin_lo, float* cos_hi, float* cos_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_normcdf    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rcbrt      (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fmax       (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fmin       (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_max        (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_min        (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fmod       (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_remainder  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_hypot      (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_copysign   (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fdim       (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_nextafter  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rhypot     (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_remquo     (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo, int* quo);
__FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_ilogb      (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ long long int __nv_fp32mp2_llrint  (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ long long int __nv_fp32mp2_llround (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ long int __nv_fp32mp2_lrint   (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ long int __nv_fp32mp2_lround  (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_isfinite   (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_isinf      (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_isnan      (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_signbit    (const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_ldexp      (const float x_hi, const float x_lo, int n, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_scalbn     (const float x_hi, const float x_lo, int n, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_scalbln    (const float x_hi, const float x_lo, long int n, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_jn         (int n, const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_yn         (int n, const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_frexp      (const float x_hi, const float x_lo, float* res_hi, float* res_lo, int* nptr);
__FPMP_BUILTIN_DECL__ void __nv_fp32mp2_modf       (const float x_hi, const float x_lo, float* res_hi, float* res_lo, float* iptr_hi, float* iptr_lo);
 #if FPMP_FP64MP2_ENABLE == 1
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_exp    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_log    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_log2   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_log10  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_log1p  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_pow    (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cbrt   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sin    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cos    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sincos (const double x_hi, const double x_lo, double* sin_hi, double* sin_lo, double* cos_hi, double* cos_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_asin   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_acos   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_atan   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_atan2  (const double y_hi, const double y_lo, const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sinh   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cosh   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_tanh   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erf    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erfc   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_normcdfinv (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_acosh      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_asinh      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_atanh      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_tan        (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_exp2       (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_exp10      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_expm1      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_logb       (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_ceil       (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_floor      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_trunc      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_round      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rint       (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_nearbyint  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fabs       (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_lgamma     (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_tgamma     (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_j0         (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_j1         (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_y0         (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_y1         (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cyl_bessel_i0(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cyl_bessel_i1(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erfcinv     (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erfinv      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erfcx       (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_boys_f0     (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_norm3d      (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_norm4d      (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, const double d_hi, const double d_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rnorm3d     (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rnorm4d     (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, const double d_hi, const double d_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sinpi      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cospi      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sincospi   (const double x_hi, const double x_lo, double* sin_hi, double* sin_lo, double* cos_hi, double* cos_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_normcdf    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rcbrt      (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fmax       (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fmin       (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_max        (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_min        (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fmod       (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_remainder  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_hypot      (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_copysign   (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fdim       (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_nextafter  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rhypot     (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_remquo     (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo, int* quo);
__FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_ilogb      (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ long long int __nv_fp64mp2_llrint  (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ long long int __nv_fp64mp2_llround (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ long int __nv_fp64mp2_lrint   (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ long int __nv_fp64mp2_lround  (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_isfinite   (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_isinf      (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_isnan      (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_signbit    (const double x_hi, const double x_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_ldexp      (const double x_hi, const double x_lo, int n, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_scalbn     (const double x_hi, const double x_lo, int n, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_scalbln    (const double x_hi, const double x_lo, long int n, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_jn         (int n, const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_yn         (int n, const double x_hi, const double x_lo, double* res_hi, double* res_lo);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_frexp      (const double x_hi, const double x_lo, double* res_hi, double* res_lo, int* nptr);
__FPMP_BUILTIN_DECL__ void __nv_fp64mp2_modf       (const double x_hi, const double x_lo, double* res_hi, double* res_lo, double* iptr_hi, double* iptr_lo);
#endif // FPMP_FP64MP2_ENABLE == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_LIB_H
