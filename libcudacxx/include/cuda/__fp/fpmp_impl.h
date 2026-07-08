//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_IMPL_H
#define _CUDA___FP_FPMP_IMPL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/*
    fpmp_impl.h - Core Multi-Precision Arithmetic Operations
    ======================================================================================================
    This header provides the core low-level (C-style) API for fpmp2 arithmetic.
    It implements fundamental operations using error-free transformations and supports multiple accuracy
    levels (low/mid/high; default == mid). The same templates are used for float-based (fp32mp2) and, when enabled,
    double-based (fp64mp2) variants.
    
    Supported Operations:
    -------------------------------------------------------------------------
    - Type Conversions:
        * __fpmp2_from_double, __fpmp2_from_int, __fpmp2_from_uint
        * __fpmp2_from_ll, __fpmp2_from_ull
        * __fpmp2_to_double, __fpmp2_to_float
        * __fpmp2_to_int, __fpmp2_to_uint, __fpmp2_to_ll, __fpmp2_to_ull
        * __fpmp2_from_double supports CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP (integer-only, no FP64)
        * __fpmp2_to_double supports CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE (integer-only, no FP64)
    
    - Basic Arithmetic:
        * Addition: __fpmp2_add, __fpmp2_low_add, __fpmp2_high_add
        * Subtraction: __fpmp2_sub, __fpmp2_low_sub, __fpmp2_high_sub
        * Accumulate: __fpmp2_acc, __fpmp2_low_acc, __fpmp2_high_acc (optimized single-component add)
        * Multiplication: __fpmp2_mul, __fpmp2_low_mul, __fpmp2_high_mul (if _CCCL_FPMP_USE_ACCURATE_MUL == 1)
        * Division: __fpmp2_div, __fpmp2_low_div, __fpmp2_high_div (if _CCCL_FPMP_USE_ACCURATE_DIV == 1)
        * Negation: __fpmp2_neg
        * Renormalization: __fpmp2_renormalize
    
    - Advanced Operations:
        * Square Root: __fpmp2_sqrt (Newton-Raphson iteration)
        * Reciprocal Square Root: __fpmp2_rsqrt (Karp-Markstein algorithm)
        * Fused Multiply-Add: __fpmp2_fma, __fpmp2_low_fma
        * Multiply-Add with Rounding: __fpmp2_mad
    
    - Comparison Operations:
        * __fpmp2_cmp_eq, __fpmp2_cmp_ne
        * __fpmp2_cmp_lt, __fpmp2_cmp_gt
        * __fpmp2_cmp_le, __fpmp2_cmp_ge
    
    - Utility Operations:
        * __fpmp2_bit_cast : IEEE-754 format bit representation
    
    - Atomic Operations (CUDA device only):
        * __fpmp2_atomicAdd : Atomic addition with CAS loop
        * __fpmp2_atomicSub : Atomic subtraction with CAS loop
    
    Implementation Details:
    -------------------------------------------------------------------------
    - Uses Dekker's error-free transformation algorithms (2Sum, 2Mult)
    - Supports three accuracy levels: low, mid (default), and high
    - Template-based for both float (fp32) and double (fp64) precision
    - Provides both inline implementations and library declarations
    - All operations maintain the (hi, lo) representation invariant
    - __fpmp2_from_double uses integer bit manipulation when CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP == 1
    
    Accuracy Levels:
    -------------------------------------------------------------------------
    - mid (Dekker, default): Balanced accuracy and performance
    - low: Minimizes renormalization steps for speed
    - high (Thall): Maximum precision with additional refinement steps
    - (Optional) FPAN-style normalization for addition in high mode (compile-time selection)
    
    Reference Papers:
    -------------------------------------------------------------------------
    [1] Dekker, T. (1971). A floating-point technique for extending available precision.
    [2] Karp & Markstein (1997). High Precision Division and Square Root. ACM TOMS.
    [3] Thall, A. Extended-Precision Floating-Point Numbers for GPU Computation.
    [4] Nagai et al. (2008). Fast Quadruple Precision Arithmetic Library. ICCS '08.
    [5] Fukuda et al. (2010). FPAN: A Fast Pairwise Addition Normalization Algorithm. SC '10.
*/

#include <cuda/__fp/fpmp_common.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

#if !(defined _CCCL_FPMP_USE_LIB)
/*********************************************************************
 * Built-in functions
 *********************************************************************/
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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_from_double (const double __x, 
                                                 _FpType*     __res_hi, 
                                                 _FpType*     __res_lo) noexcept 
    {
#if _CCCL_FPMP_USE_OPT_FROM_DOUBLE == 1
        if constexpr (::cuda::std::is_same_v<_FpType, float>)
        {
            uint64_t __dbits = __fpmp_internal_bit_cast<uint64_t>(__x);
            uint32_t __sign  = (uint32_t)(__dbits >> 63);
            uint32_t __d_exp = (uint32_t)((__dbits >> 52) & 0x7FFU);
            uint64_t __mant  = __dbits & 0x000FFFFFFFFFFFFFULL;

            // hi biased exponent in float space: f_exp = (d_exp - 1023) + 127.
            int32_t __f_exp = (int32_t)__d_exp - 896;

            // Fallback for: zero/denormal double (d_exp == 0), float underflow
            // (f_exp <= 0), and float overflow / Inf / NaN (f_exp >= 255).
            // Defers to the standard cast for these edge cases; lo is flushed.
            if (__d_exp == 0 || (__f_exp <= 0 || __f_exp >= 255)) 
            {
                *__res_hi = (float)__x;
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
                uint32_t __hi_round = (((uint32_t)(__mant >> 28)) + 1U) >> 1;
                uint32_t __hi_bits  = (__sign << 31) | (((uint32_t)__f_exp << 23) + __hi_round);
                *__res_hi = __fpmp_internal_bit_cast<float>(__hi_bits);

                // Encode the residual as a signed 32-bit integer with the
                // round bit placed at the sign position. Bottom 32 bits of
                // mant are bits [31:0]; shifting left by 3 in 32-bit
                // arithmetic discards bits [31:29] (already absorbed into
                // hi) and places bit 28 (the round bit) at bit 31. When the
                // round bit is 1 (hi was rounded up), rsd is negative in
                // two's complement, exactly representing the signed residual
                // x - hi at mantissa scale * 2^3.
                int32_t __rsd = (int32_t)((uint32_t)__mant << 3);

                // Convert rsd to float with round-to-nearest-even (default for
                // host int->float and CUDA cvt.rn.f32.s32). Then scale:
                //   * 2^-55  : undoes the << 3 (-3) and the mantissa-position
                //              offset (-52) to recover residual at unit scale.
                //   * scale  : 2^(f_exp - 127) with the sign of x. Both
                //              multiplications are exact (powers of two).
                float __scale = __fpmp_internal_bit_cast<float>((__sign << 31) | ((uint32_t)__f_exp << 23));
                *__res_lo = (static_cast<float>(__rsd) * 0x1p-55f) * __scale;

                // Fast2Sum to enforce canonical form fl(hi+lo) == hi.
                // Required because round-to-nearest on r can leave |lo|
                // exactly at ulp(hi)/2; if hi has an odd low mantissa bit,
                // fl(hi+lo) would otherwise round away from hi.
                *__res_hi = __fpmp_fast_two_sum(*__res_hi, *__res_lo, __res_lo);
            }
        }
        else if constexpr (::cuda::std::is_same_v<_FpType, double>)
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
            // cast-based split.
            *__res_hi = static_cast<_FpType>(__x);
            *__res_lo = static_cast<_FpType>(__x - static_cast<double>(*__res_hi));
        }
#else // !_CCCL_FPMP_USE_OPT_FROM_DOUBLE == 1
        if constexpr (::cuda::std::is_same_v<_FpType, double>)
        {
            // FpType == double (fp64mp2): trivial split, see comment above.
            *__res_hi = __x;
            *__res_lo = 0.0;
        }
        else
        {
            // Non-optimized path: two FP64 operations (cast + subtract).
            *__res_hi = static_cast<_FpType>(__x);
            *__res_lo = static_cast<_FpType>(__x - static_cast<double>(*__res_hi));
        }
#endif // !_CCCL_FPMP_USE_OPT_FROM_DOUBLE == 1
    } // __fpmp2_from_double

    // int -> (hi, lo) conversions
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_from_int (const int32_t __i, 
                                                      _FpType*       __res_hi, 
                                                      _FpType*       __res_lo) noexcept   
    {

        *__res_hi = __fpmp_int2fp_rz<_FpType>(__i);
        *__res_lo = __fpmp_int2fp_rz<_FpType>(__i - __fpmp_fp2int_rz(*__res_hi));
    }

    // uint -> (hi, lo) conversions
    // Note: Use signed arithmetic to compute residual, since __fpmp_fp2uint_rz(*res_hi)
    // might be larger than i when rounding direction differs
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_from_uint (const uint32_t __i, 
                                                       _FpType*        __res_hi, 
                                                       _FpType*        __res_lo) noexcept 
    {

        *__res_hi = __fpmp_uint2fp_rz<_FpType>(__i);
        // Compute residual using signed arithmetic to handle case where hi rounds up
        int32_t __residual = static_cast<int32_t>(__i) - static_cast<int32_t>(__fpmp_fp2uint_rz(*__res_hi));
        *__res_lo = __fpmp_int2fp_rz<_FpType>(__residual);
    }   

    // ll -> (hi, lo) conversions
    // With __fpmp_ll2fp_rz properly rounding toward zero, hi is always <= i for positive i
    // and >= i for negative i, so __fpmp_fp2ll_rz(hi) is always representable as int64_t.
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_from_ll (const int64_t __i, 
                                                     _FpType*       __res_hi, 
                                                     _FpType*       __res_lo) noexcept 
    {

        *__res_hi = __fpmp_ll2fp_rz<_FpType>(__i);
        *__res_lo = __fpmp_ll2fp_rz<_FpType>(__i - __fpmp_fp2ll_rz(*__res_hi));
    }

    // ull -> (hi, lo) conversions
    // With ull2fp_rz properly rounding toward zero, hi <= i always,
    // so the residual i - hi is always non-negative.
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_from_ull (const uint64_t __i, 
                                                      _FpType*        __res_hi, 
                                                      _FpType*        __res_lo) noexcept 
    {

        *__res_hi = __fpmp_ull2fp_rz<_FpType>(__i);
        // Residual is always non-negative and fits in int64_t (< 2^53 for double)
        uint64_t __residual = __i - __fpmp_fp2ull_rz(*__res_hi);
        *__res_lo = __fpmp_ull2fp_rz<_FpType>(__residual);
    }

    // (hi, lo) -> double conversions
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
    //   Cold-path (FP64 fallback) covers cases where the FP32 scale would
    //   overflow / be ill-defined:
    //     - fexp_a == 0      : hi is +/-0 or subnormal
    //     - fexp_a in [1,51] : 2^(179 - fexp_a) overflows float (need biased
    //                          exponent <= 254 ⇒ fexp_a >= 52)
    //     - fexp_a == 0xFF   : hi is +/-Inf or NaN
    //   Both ends of this range are extremely rare for typical fp32mp2 data.
    //
    // Non-optimized path:
    //   static_cast<double>(x_hi) + static_cast<double>(x_lo)
    //   (2x F2D + 1x DADD = 3 FP64 operations)
    //
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  double __fpmp2_to_double (const _FpType __x_hi_in,
                                                 const _FpType __x_lo_in) noexcept
    { 
#if _CCCL_FPMP_USE_OPT_TO_DOUBLE == 1
        if constexpr (::cuda::std::is_same_v<_FpType, float>)
        {
            // Renormalize input to canonical form. See "Step 0" in the
            // function's docstring for why this is required for the integer
            // path to be safe on non-canonical pairs (e.g. produced by
            // add_fast / long FAST accumulator chains).
            float __x_lo;
            float __x_hi = __fpmp_two_sum(__x_hi_in, __x_lo_in, &__x_lo);

            uint32_t __hi_bits = __fpmp_internal_bit_cast<uint32_t>(__x_hi);
            uint32_t __sign_a  = __hi_bits >> 31;
            uint32_t __fexp_a  = (__hi_bits >> 23) & 0xFFU;
            uint32_t __fmant_a = __hi_bits & 0x7FFFFFU;

            // Cold fallback for hi outside the FP32-scale-representable range:
            // zero/subnormal, very tiny normal (fexp_a in [1, 51]), Inf, NaN.
            if (__fexp_a < 52U || __fexp_a == 0xFFU)
                return static_cast<double>(__x_hi) + static_cast<double>(__x_lo);

            // scale = (sign_a ? -1 : +1) * 2^(179 - fexp_a). Always a normal
            // float for fexp_a in [52, 254] (biased scale_exp = 306 - fexp_a
            // in [52, 254]). Sign of hi baked in so the signed r below is
            // already lo's contribution relative to hi.
            float __scale = __fpmp_internal_bit_cast<float>(
                (__sign_a << 31) | ((306U - __fexp_a) << 23));

            // r exactly represents lo at hi's mantissa scale (signed). For
            // canonical fp32mp2 (|lo| <= ulp(hi)/2) the multiplication is
            // exact (power-of-two scaling) and |r| <= 2^28.
            int32_t __r = __fpmp_fp2int_rn(__x_lo * __scale);

            // M = (hi's 53-bit mantissa with implicit 1, at bit 52)
            //     + (signed lo contribution at the same scale).
            // Range: [2^52 - 2^28, 2^53 - 2^29 + 2^28]. Always positive.
            int64_t __M = (int64_t)(((uint64_t)(0x800000U | __fmant_a)) << 29)
                      + (int64_t)__r;

            // Subtraction can borrow at most one bit (|r| << 2^52), so the
            // implicit-1 lands at bit 52 (no shift) or bit 51 (shift up by 1).
            // Single conditional shift, no __clzll on the critical path.
            uint64_t __Mu         = (uint64_t)__M;
            uint64_t __need_shift = ((__Mu >> 52) & 1ULL) ^ 1ULL;
            __Mu <<= __need_shift;

            return __fpmp_internal_bit_cast<double>(
                ((uint64_t)__sign_a << 63)
              | ((uint64_t)(__fexp_a + 896U - (uint32_t)__need_shift) << 52)
              | (__Mu & 0x000FFFFFFFFFFFFFULL));
        }
        else
        {
            return static_cast<double>(__x_hi_in) + static_cast<double>(__x_lo_in);
        }
#else
        return static_cast<double>(__x_hi_in) + static_cast<double>(__x_lo_in);
#endif
    }

    // (hi, lo) -> float conversions (returns the sum as single FpType)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  _FpType __fpmp2_to_float (const _FpType __x_hi, 
                                                 const _FpType __x_lo) noexcept  
    { 
        return __x_hi + __x_lo; 
    }

    // (hi, lo) -> int conversions
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  int32_t __fpmp2_to_int (const _FpType __x_hi, 
                                               const _FpType __x_lo) noexcept  
    { 

        _FpType __abs_hi    = __fpmp_internal_fabs(__x_hi);
        // Check threshold BEFORE computing sum - for large values, addition loses precision
        // 2^24 for float, 2^53 for double
        _FpType __threshold = ::cuda::std::is_same_v<_FpType, float> ? 0x1.0p24f : 
                                                                       0x1.0p53;  
        if (__abs_hi < __threshold)
        {
            // Small value: use round-toward-zero addition
            _FpType __res = __fpmp_add_rz(__x_hi, __x_lo);
            return __fpmp_fp2int_rz(__res); 
        }
        else 
        {
            // Large value: use integer addition to preserve exactness
            int32_t __hi_int = __fpmp_fp2int_rz(__x_hi);
            int32_t __lo_int = __fpmp_fp2int_rz(__x_lo);
            return __hi_int + __lo_int;
        }
    } // __fpmp2_to_int

    // (hi, lo) -> uint conversions
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  uint32_t __fpmp2_to_uint (const _FpType __x_hi, 
                                                 const _FpType __x_lo) noexcept  
    { 

        // Check threshold BEFORE computing sum
        // 2^24 for float, 2^53 for double
        _FpType __threshold = ::cuda::std::is_same_v<_FpType, float> ? 0x1.0p24f : 
                                                                       0x1.0p53;  
        if (__x_hi < __threshold)
        {
            // Small value: use round-toward-zero addition
            _FpType __res = __fpmp_add_rz(__x_hi, __x_lo);
            return __fpmp_fp2uint_rz(__res);
        }
        else 
        {
            // Large value: use integer addition to preserve exactness
            uint32_t __hi_uint = __fpmp_fp2uint_rz(__x_hi);
            int32_t __lo_int   = __fpmp_fp2int_rz(__x_lo);
            return __hi_uint + __lo_int;
        }
    } // __fpmp2_to_uint

    // (hi, lo) -> ll conversions
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  int64_t __fpmp2_to_ll (const _FpType __x_hi, 
                                                      const _FpType __x_lo) noexcept  
    { 

        _FpType __abs_hi    = __fpmp_internal_fabs(__x_hi);
        // Check threshold BEFORE computing sum
        // 2^24 for float, 2^53 for double
        _FpType __threshold = ::cuda::std::is_same_v<_FpType, float> ? 0x1.0p24f : 
                                                                0x1.0p53;  
        if (__abs_hi < __threshold)
        {
            // Small value: use round-toward-zero addition
            _FpType __res = __fpmp_add_rz(__x_hi, __x_lo);
            return __fpmp_fp2ll_rz(__res);
        }
        else 
        {
            // Large value: use integer addition to preserve exactness
            int64_t __hi_ll = __fpmp_fp2ll_rz(__x_hi);
            int64_t __lo_ll = __fpmp_fp2ll_rz(__x_lo);
            return __hi_ll + __lo_ll;
        }
    } // __fpmp2_to_ll

    // (hi, lo) -> ull conversions
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  uint64_t __fpmp2_to_ull (const _FpType __x_hi, 
                                                const _FpType __x_lo) noexcept 
    { 

        // Check threshold BEFORE computing sum
        // 2^24 for float, 2^53 for double
        _FpType __threshold = ::cuda::std::is_same_v<_FpType, float> ? 0x1.0p24f : 
                                                                       0x1.0p53;  
        if (__x_hi < __threshold)
        {
            // Small value: use round-toward-zero addition
            _FpType __res = __fpmp_add_rz(__x_hi, __x_lo);
            return __fpmp_fp2ull_rz(__res);
        }
        else 
        {
            // Large value: use integer addition to preserve exactness
            uint64_t __hi_ull = __fpmp_fp2ull_rz(__x_hi);
            int64_t __lo_ll   = __fpmp_fp2ll_rz(__x_lo);
            return __hi_ull + __lo_ll;
        }
    } // __fpmp2_to_ull

    /*
    * --------------------------------------------------------------------
    * Re-normalization operations
    * --------------------------------------------------------------------
    */
    // Renormalize a multi-precision (double-float) number
    // to ensure that the hi and lo parts are non-overlapping
    // This is useful for fast mode to ensure that the result is accurate
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_renormalize (const _FpType __x_hi, 
                                                const _FpType __x_lo, 
                                                _FpType*      __res_hi, 
                                                _FpType*      __res_lo) noexcept
    {

        *__res_hi = __fpmp_fast_two_sum(__x_hi, __x_lo, __res_lo);
    }

    /*
    * --------------------------------------------------------------------
    * Addition operations
    * --------------------------------------------------------------------
    */
    /*
    * Fast addition operation
    * This is a simple addition operation with no normalization.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_low_add (const _FpType __x_hi, 
                                            const _FpType __x_lo, 
                                            const _FpType __y_hi, 
                                            const _FpType __y_lo, 
                                            _FpType*      __res_hi, 
                                            _FpType*      __res_lo) noexcept
    {

        _FpType __r_hi, __r_lo;

        // Add high parts using general 2-Sum (no magnitude assumption)
        __r_hi = __fpmp_two_sum(__x_hi, __y_hi, &__r_lo);
        // Add low parts
        __r_lo = __fpmp_add_rn(__fpmp_add_rn(__x_lo, __y_lo), __r_lo);

        *__res_hi = __r_hi;
        *__res_lo = __r_lo;
    } // __fpmp2_low_add

    /*
    * Dekker addition operation
    * This is classic split and error accumulation addition operation with normalization.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_add (const _FpType __x_hi, 
                                        const _FpType __x_lo, 
                                        const _FpType __y_hi, 
                                        const _FpType __y_lo, 
                                        _FpType*      __res_hi, 
                                        _FpType*      __res_lo) noexcept
    {


        _FpType __r_lo_refine;
        _FpType __r_hi, __r_lo;

        // Add high parts using general 2-Sum (no magnitude assumption)
        __r_hi        = __fpmp_two_sum(__x_hi, __y_hi, &__r_lo);
        // Add low parts
        __r_lo_refine = __fpmp_add_rn(__fpmp_add_rn(__x_lo, __y_lo), __r_lo);
        // Normalize:
        *__res_hi     = __fpmp_fast_two_sum(__r_hi, __r_lo_refine, __res_lo);
    } // __fpmp2_add
    
    /*
    * FPAN-style accurate addition (compile-time selectable normalization strategy)
    * Optimized for instruction-level parallelism with branch-free structure.
    * Algorithm:
    *   (s_h, s_l) = TwoSum(a_hi, b_hi)   // Level 1 (parallel)
    *   (t_h, t_l) = TwoSum(a_lo, b_lo)   // Level 1 (parallel)
    *   c = s_l + t_h                      // Level 2: merge middle terms
    *   (v_h, v_l) = Fast2Sum(s_h, c)      // Level 3: normalize
    *   w = t_l + v_l                      // Level 4: absorb error
    *   (r_h, r_l) = Fast2Sum(v_h, w)      // Level 5: final normalize
    * 
    * Total: 20 ops, Critical path: 14 ops (vs Thall's 17 ops sequential)
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_nv_fpmp2_add_fpan (const _FpType __a_hi, 
                                                         const _FpType __a_lo, 
                                                         const _FpType __b_hi, 
                                                         const _FpType __b_lo, 
                                                         _FpType*      __res_hi, 
                                                         _FpType*      __res_lo) noexcept
    {

        
        // Level 1: Two independent 2Sums - can execute in parallel
        // Inline two_sum for a_hi + b_hi to help compiler see independence
        _FpType __s_h   = __fpmp_add_rn(__a_hi, __b_hi);
        _FpType __s_a   = __fpmp_sub_rn(__s_h, __b_hi);
        _FpType __s_b   = __fpmp_sub_rn(__s_h, __s_a);
        
        // Inline two_sum for a_lo + b_lo (parallel with above)
        _FpType __t_h   = __fpmp_add_rn(__a_lo, __b_lo);
        _FpType __t_a   = __fpmp_sub_rn(__t_h, __b_lo);
        _FpType __t_b   = __fpmp_sub_rn(__t_h, __t_a);
        
        // Complete the error calculations (can interleave)
        _FpType __s_da  = __fpmp_sub_rn(__a_hi, __s_a);
        _FpType __s_db  = __fpmp_sub_rn(__b_hi, __s_b);
        _FpType __s_l   = __fpmp_add_rn(__s_da, __s_db);
        
        _FpType __t_da  = __fpmp_sub_rn(__a_lo, __t_a);
        _FpType __t_db  = __fpmp_sub_rn(__b_lo, __t_b);
        _FpType __t_l   = __fpmp_add_rn(__t_da, __t_db);
        
        // Level 2: Merge middle terms
        _FpType __c     = __fpmp_add_rn(__s_l, __t_h);
        
        // Level 3: First normalization (Fast2Sum since |s_h| >= |c| typically)
        _FpType __v_h   = __fpmp_add_rn(__s_h, __c);
        _FpType __v_tmp = __fpmp_sub_rn(__v_h, __s_h);
        _FpType __v_l   = __fpmp_sub_rn(__c, __v_tmp);
        
        // Level 4: Absorb remaining error
        _FpType __w     = __fpmp_add_rn(__t_l, __v_l);
        
        // Level 5: Final normalization
        *__res_hi      = __fpmp_add_rn(__v_h, __w);
        _FpType __r_tmp = __fpmp_sub_rn(*__res_hi, __v_h);
        *__res_lo      = __fpmp_sub_rn(__w, __r_tmp);
    } // __fpmp2_add_fpan
    
    /*
    * Thall addition operation via expansion series
    * This implementation is based on: Andrew Thall, Extended-Precision
    * Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
    * from http://andrewthall.org/papers/df64_qf128.pdf.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_nv_fpmp2_add_exp (const _FpType __a_hi, 
                                                        const _FpType __a_lo, 
                                                        const _FpType __b_hi, 
                                                        const _FpType __b_lo, 
                                                        _FpType*      __res_hi, 
                                                        _FpType*      __res_lo) noexcept
    {

        _FpType __t1, __t2, __t3, __t4, __t5, __e;
        __t1 = __fpmp_add_rn (__a_hi, __b_hi);
        __t2 = __fpmp_sub_rn (__t1, __a_hi);
        __t3 = __fpmp_add_rn (__fpmp_add_rn (__a_hi, __fpmp_sub_rn(__t2,__t1)), __fpmp_sub_rn (__b_hi, __t2));
        __t4 = __fpmp_add_rn (__a_lo, __b_lo);
        __t2 = __fpmp_sub_rn (__t4, __a_lo);
        __t5 = __fpmp_add_rn (__fpmp_add_rn (__a_lo, __fpmp_sub_rn(__t2,__t4)), __fpmp_sub_rn (__b_lo, __t2));
        __t3 = __fpmp_add_rn (__t3, __t4);
        __t4 = __fpmp_add_rn (__t1, __t3);
        __t3 = __fpmp_add_rn (__fpmp_sub_rn(__t1,__t4), __t3);
        __t3 = __fpmp_add_rn (__t3, __t5);
        __e  = __fpmp_add_rn (__t4, __t3);

        *__res_lo = __fpmp_add_rn (__fpmp_sub_rn(__t4, __e), __t3);
        *__res_hi = __e;
    } // __fpmp2_high_add

#define _CCCL_FPMP_FPAN_METHOD

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_high_add (const _FpType __a_hi, 
                                             const _FpType __a_lo, 
                                             const _FpType __b_hi, 
                                             const _FpType __b_lo, 
                                             _FpType*      __res_hi, 
                                             _FpType*      __res_lo) noexcept
    {
#if defined   _CCCL_FPMP_FPAN_METHOD
        __internal_nv_fpmp2_add_fpan (__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
#else
        __internal_nv_fpmp2_add_exp  (__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo);
#endif
    }
    
    /*
    * --------------------------------------------------------------------
    * Subtraction operations
    * --------------------------------------------------------------------
    */
    /*
    * Fast subtraction operation
    * This is a simple subtraction operation with no normalization.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_low_sub (const _FpType __x_hi, 
                                            const _FpType __x_lo, 
                                            const _FpType __y_hi, 
                                            const _FpType __y_lo, 
                                            _FpType*      __res_hi, 
                                            _FpType*      __res_lo) noexcept
    {
        __fpmp2_low_add(__x_hi, __x_lo, -__y_hi, -__y_lo, __res_hi, __res_lo);
    }
    /*
    * Classic split and error accumulation subtraction operation
    * This is a Dekker subtraction operation with normalization.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_sub (const _FpType __x_hi, 
                                        const _FpType __x_lo, 
                                        const _FpType __y_hi, 
                                        const _FpType __y_lo, 
                                        _FpType*      __res_hi, 
                                        _FpType*      __res_lo) noexcept
    {
        __fpmp2_add(__x_hi, __x_lo, -__y_hi, -__y_lo, __res_hi, __res_lo);
    }
    /*
    * Thall accurate subtraction operation
    * This implementation is based on: Andrew Thall, Extended-Precision
    * Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
    * from http://andrewthall.org/papers/df64_qf128.pdf.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_high_sub(const _FpType __x_hi, 
                                             const _FpType __x_lo, 
                                             const _FpType __y_hi, 
                                             const _FpType __y_lo, 
                                             _FpType*      __res_hi, 
                                             _FpType*      __res_lo) noexcept
    {
        __fpmp2_high_add(__x_hi, __x_lo, -__y_hi, -__y_lo, __res_hi, __res_lo);
    }

    /*
    * --------------------------------------------------------------------
    * Accumulate operations (single-component addition to multi-precision)
    * --------------------------------------------------------------------
    * These functions efficiently accumulate a single-precision value into
    * a multi-precision (hi, lo) pair. More efficient than full mp2+mp2 
    * addition since only one 2Sum is needed (the contribution has lo=0).
    *
    * Algorithm (Dekker-style):
    *   (new_hi, err) = 2Sum(acc_hi, c)      // Add c to high part
    *   new_lo = acc_lo + err                 // Accumulate error into low part
    *   (res_hi, res_lo) = Fast2Sum(new_hi, new_lo)  // Normalize
    *
    * This saves ~6 operations vs full addition (no 2Sum for low parts).
    */
    
    /*
    * Fast accumulate: no final normalization
    * Result may have overlapping hi/lo components until renormalized.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_low_acc (const _FpType __c, 
                                            _FpType*      __acc_hi, 
                                            _FpType*      __acc_lo) noexcept
    {

        _FpType __err;
        // Add c to high part with error capture
        _FpType __new_hi = __fpmp_two_sum(*__acc_hi, __c, &__err);
        // Accumulate error into low part (no normalization)
        *__acc_hi = __new_hi;
        *__acc_lo = __fpmp_add_rn(*__acc_lo, __err);
    }
    
    /*
    * Default accumulate: Dekker-style with normalization
    * Result is properly normalized (non-overlapping hi/lo).
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_acc (const _FpType __c, 
                                        _FpType*      __acc_hi, 
                                        _FpType*      __acc_lo) noexcept
    {

        _FpType __err;
        // Add c to high part with error capture
        _FpType __new_hi = __fpmp_two_sum(*__acc_hi, __c, &__err);
        // Combine error with existing low part
        _FpType __new_lo = __fpmp_add_rn(*__acc_lo, __err);
        // Normalize result
        *__acc_hi = __fpmp_fast_two_sum(__new_hi, __new_lo, __acc_lo);
    }
    
    /*
    * Accurate accumulate: Full error propagation (FPAN-style)
    * Provides maximum precision by properly ordering all error terms.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_high_acc (const _FpType __c, 
                                             _FpType*      __acc_hi, 
                                             _FpType*      __acc_lo) noexcept
    {

        _FpType __err;
        // Add c to high part with error capture
        _FpType __s_hi = __fpmp_two_sum(*__acc_hi, __c, &__err);
        // Add error to low part
        _FpType __t    = __fpmp_add_rn(*__acc_lo, __err);
        // First normalization
        _FpType __v_hi = __fpmp_add_rn(__s_hi, __t);
        _FpType __v_tmp = __fpmp_sub_rn(__v_hi, __s_hi);
        _FpType __v_lo = __fpmp_sub_rn(__t, __v_tmp);
        // Final normalization
        *__acc_hi = __fpmp_add_rn(__v_hi, __v_lo);
        _FpType __r_tmp = __fpmp_sub_rn(*__acc_hi, __v_hi);
        *__acc_lo = __fpmp_sub_rn(__v_lo, __r_tmp);
    }

    /*
    * --------------------------------------------------------------------
    * Multiplication operations
    * --------------------------------------------------------------------
    */
    /*
    * Fast multiplication operation
    * This is a simple multiplication operation with no normalization.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_low_mul (const _FpType __x_hi, 
                                             const _FpType __x_lo, 
                                             const _FpType __y_hi, 
                                             const _FpType __y_lo, 
                                             _FpType*      __res_hi, 
                                             _FpType*      __res_lo) noexcept  
    { 

        _FpType __t_hi = __fpmp_mul_rn (__x_hi, __y_hi);
        _FpType __t_lo = __fpmp_fma_rn (__x_hi, __y_hi, -__t_hi);
        __t_lo        = __fpmp_fma_rn (__x_lo, __y_lo, __t_lo);
        __t_lo        = __fpmp_fma_rn (__x_hi, __y_lo, __t_lo); 
        __t_lo        = __fpmp_fma_rn (__x_lo, __y_hi, __t_lo);

        *__res_hi = __t_hi;
        *__res_lo = __t_lo;
    } // __fpmp2_low_mul

    /*
    * Dekker multiplication operation
    * This is a Dekker multiplication operation with normalization.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_mul (const _FpType __x_hi, 
                                        const _FpType __x_lo, 
                                        const _FpType __y_hi, 
                                        const _FpType __y_lo, 
                                        _FpType*      __res_hi, 
                                        _FpType*      __res_lo) noexcept
    {

        _FpType __p1, __p2, __c_hi, __c_lo, __res_hi_tmp, __res_lo_tmp;
        __c_hi = __fpmp_two_mult_fma(__x_hi, __y_hi, &__c_lo);
        __p1   = __fpmp_mul_rn(__x_hi, __y_lo);
        __p2   = __fpmp_mul_rn(__x_lo, __y_hi);
        __c_lo = __fpmp_add_rn(__c_lo, __fpmp_add_rn(__p1, __p2));
        // Normalize:
        __res_hi_tmp = __fpmp_fast_two_sum(__c_hi, __c_lo, &__res_lo_tmp);

        *__res_hi = __res_hi_tmp;
        *__res_lo = __res_lo_tmp;
    } // __fpmp2_mul

#if _CCCL_FPMP_USE_ACCURATE_MUL == 1
    /*
    * Dekker multiplication with branch-free conditional scaling
    * ===========================================================
    * 
    * This implementation uses the standard Dekker multiplication algorithm
    * with branch-free conditional scaling to handle subnormal results accurately.
    *
    * The standard Dekker algorithm fails when the product approaches the
    * subnormal range because the error term from two_mult_fma loses precision
    * due to gradual underflow. This implementation detects such cases using
    * bit manipulation and applies scaling to ensure all intermediate computations
    * happen in the normal range where error-free transformations are exact.
    *
    * ALGORITHM:
    *   1. Compute conditional scale factor based on operand exponents (branch-free).
    *   2. Scale first operand if product would be small.
    *   3. Perform standard Dekker multiplication.
    *   4. Scale result back with inverse factor.
    *
    * CONDITIONAL SCALE COMPUTATION (branch-free):
    *   - Extract sum of exponents from x_hi and y_hi.
    *   - If sum < threshold: scale = 2^64 (float) or 2^512 (double), else scale = 1.0
    *   - Use bit manipulation to select between scale values without branches.
    *
    * PERFORMANCE:
    *   - Normal case: scale = 1.0, minimal overhead (MUL by 1.0 is fast).
    *   - Subnormal case: full scaling applied for accuracy.
    *   - Branch-free: no GPU warp divergence.
    *   - Overhead vs __fpmp2_mul: ~6 integer ops + 4 MUL (often identity) + 1 fast_two_sum.
    *
    * REFERENCE:
    *   Dekker, T. (1971). A floating-point technique for extending available precision.
    *   Conditional scaling adapted from QD library techniques.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_high_mul (const _FpType __x_hi, 
                                             const _FpType __x_lo, 
                                             const _FpType __y_hi, 
                                             const _FpType __y_lo, 
                                             _FpType*      __res_hi, 
                                             _FpType*      __res_lo) noexcept
    {

        
        // Type-specific constants for conditional scaling
        using UintType = ::cuda::std::conditional_t<::cuda::std::is_same_v<_FpType, float>, uint32_t, uint64_t>;
        
        constexpr int __exp_bits   = ::cuda::std::is_same_v<_FpType, float> ? 8 : 11;
        constexpr int __mant_bits  = ::cuda::std::is_same_v<_FpType, float> ? 23 : 52;
        constexpr int __exp_bias   = ::cuda::std::is_same_v<_FpType, float> ? 127 : 1023;
        constexpr UintType __exp_mask = ((UintType(1) << __exp_bits) - 1) << __mant_bits;
        
        // Threshold: if combined exponent < this, we need scaling
        // For float: scale_shift=64, threshold=190 (2*127-64)
        // For double: scale_shift=512, threshold=1534 (2*1023-512)
        constexpr int __scale_shift = ::cuda::std::is_same_v<_FpType, float> ? 64 : 512;
        constexpr int __exp_threshold = 2 * __exp_bias - __scale_shift;
        
        // Scale factors
        constexpr _FpType __scale_up   = ::cuda::std::is_same_v<_FpType, float> ? _FpType(0x1.0p64f)  : _FpType(0x1.0p512);
        constexpr _FpType __scale_down = ::cuda::std::is_same_v<_FpType, float> ? _FpType(0x1.0p-64f) : _FpType(0x1.0p-512);
        
        // Extract exponents and compute conditional scale (branch-free)
        UintType __x_bits = __fpmp_internal_bit_cast<UintType>(__x_hi);
        UintType __y_bits = __fpmp_internal_bit_cast<UintType>(__y_hi);
        int __x_exp = static_cast<int>((__x_bits & __exp_mask) >> __mant_bits);
        int __y_exp = static_cast<int>((__y_bits & __exp_mask) >> __mant_bits);
        int __result_exp = __x_exp + __y_exp;
        
        // Create mask: -1 (all 1s) if needs scaling, 0 otherwise
        int __needs_scale = (__result_exp - __exp_threshold) >> 31;
        
        // Select scale factor using bit manipulation (branch-free)
        UintType __scale_up_bits   = __fpmp_internal_bit_cast<UintType>(__scale_up);
        UintType __one_bits        = __fpmp_internal_bit_cast<UintType>(_FpType(1.0));
        UintType __scale_bits      = (__scale_up_bits & UintType(__needs_scale)) | (__one_bits & UintType(~__needs_scale));
        _FpType __scale             = __fpmp_internal_bit_cast<_FpType>(__scale_bits);
        
        UintType __scale_down_bits = __fpmp_internal_bit_cast<UintType>(__scale_down);
        UintType __inv_scale_bits  = (__scale_down_bits & UintType(__needs_scale)) | (__one_bits & UintType(~__needs_scale));
        _FpType __inv_scale         = __fpmp_internal_bit_cast<_FpType>(__inv_scale_bits);
        
        // Scale first operand
        _FpType __a_hi = __fpmp_mul_rn(__x_hi, __scale);
        _FpType __a_lo = __fpmp_mul_rn(__x_lo, __scale);
        
        // Standard Dekker multiplication
        _FpType __c_lo;
        _FpType __c_hi = __fpmp_two_mult_fma(__a_hi, __y_hi, &__c_lo);
        _FpType __p1   = __fpmp_mul_rn(__a_hi, __y_lo);
        _FpType __p2   = __fpmp_mul_rn(__a_lo, __y_hi);
        __c_lo        = __fpmp_add_rn(__c_lo, __fpmp_add_rn(__p1, __p2));
        
        // Normalize
        _FpType __r_lo;
        _FpType __r_hi = __fpmp_fast_two_sum(__c_hi, __c_lo, &__r_lo);
        
        // Scale back
        __r_hi = __fpmp_mul_rn(__r_hi, __inv_scale);
        __r_lo = __fpmp_mul_rn(__r_lo, __inv_scale);
        
        // Final normalization to ensure (hi, lo) invariant after scaling
        *__res_hi = __fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
    } // __fpmp2_high_mul
#endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1

    /*
    * --------------------------------------------------------------------
    * Division operations
    * --------------------------------------------------------------------
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_low_div (const _FpType __a_hi, 
                                            const _FpType __a_lo, 
                                            const _FpType __b_hi, 
                                            const _FpType __b_lo, 
                                            _FpType*      __res_hi, 
                                            _FpType*      __res_lo) noexcept
    {

        // Get an estimate from *this->hi:
        _FpType __recip_hi = __fpmp_rcp_rn(__b_hi);

        // Do a Newton-Rhapson iteration:
        // This line can break for some uninvestigated reason,
        // Use the one below:
        //recip_hi = recip_hi*(2.0 - (x.get_hi())*recip_hi);
        _FpType __two = static_cast<_FpType>(2.0);
        __recip_hi = __fpmp_fma_rn(-__b_hi*__recip_hi, __recip_hi, __two*__recip_hi);

        _FpType __recip2_hi = __recip_hi*__recip_hi;
        _FpType __recip2_lo = __fpmp_fma_rn(__recip_hi, __recip_hi, -__recip2_hi);

        // recip^2 * this->(hi/lo), Dekker multiplication:
        _FpType __mul_hi = __recip2_hi*(__b_hi);
        _FpType __mul_lo = __fpmp_fma_rn(__recip2_hi, (__b_hi), -__mul_hi);
        __mul_lo       += (__recip2_hi*(__b_lo) + __recip2_lo*(__b_hi));

        // Our answer is now 2*recip_hi + mul_hi + mul_lo
        _FpType __final_recip_hi = __two*__recip_hi - __mul_hi;
        _FpType __final_recip_lo = __two*__recip_hi - __fpmp_add_rn(__final_recip_hi, __mul_hi);
        __final_recip_lo       -= __mul_lo;

        // Multiply the reciprocal by the numerator
        __fpmp2_low_mul(__a_hi, __a_lo, __final_recip_hi, __final_recip_lo, __res_hi, __res_lo);
    } // __fpmp2_low_div


    /* Compute high-accuracy quotient, using Newton-
    Raphson iteration. Derived from: T. Nagai, H. Yoshida, H. Kuroda, Y. Kanada.
    Fast Quadruple Precision Arithmetic Library on Parallel Computer SR11000/J2.
    In Proceedings of the 8th International Conference on Computational Science,
    ICCS '08, Part I, pp. 446-455.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_div (const _FpType __a_hi, 
                                        const _FpType __a_lo, 
                                        const _FpType __b_hi, 
                                        const _FpType __b_lo, 
                                        _FpType*      __res_hi, 
                                        _FpType*      __res_lo) noexcept
    {

        _FpType __t_hi, __t_lo;
        _FpType __e, __r;
        __r          = __fpmp_rcp_rn(__b_hi);
        __t_hi       = __fpmp_mul_rn (__a_hi, __r);
        __e          = __fpmp_fma_rn (__b_hi, -__t_hi, __a_hi);
        __t_hi       = __fpmp_fma_rn (__r, __e, __t_hi);
        __t_lo       = __fpmp_fma_rn (__b_hi, -__t_hi, __a_hi);
        __t_lo       = __fpmp_add_rn (__a_lo, __t_lo);
        __t_lo       = __fpmp_fma_rn (__b_lo, -__t_hi, __t_lo);
        __e          = __fpmp_mul_rn (__r, __t_lo);
        __t_lo       = __fpmp_fma_rn (__b_hi, -__e, __t_lo);
        __t_lo       = __fpmp_fma_rn (__r, __t_lo, __e);
        __e          = __fpmp_add_rn (__t_hi, __t_lo);

        *__res_lo    = __fpmp_add_rn (__t_hi - __e, __t_lo);
        *__res_hi    = __e;
    } // __fpmp2_div

#if _CCCL_FPMP_USE_ACCURATE_DIV == 1
    /*
    * --------------------------------------------------------------------
    * Accurate Division with Conditional Scaling
    * --------------------------------------------------------------------
    * This implementation handles division when operands are near the 
    * denormal range by using branch-free conditional scaling.
    *
    * For division a/b:
    *   - If 'a' is small (near denormal), intermediate computations may
    *     lose precision due to denormal arithmetic
    *   - If 'b' is small, the reciprocal computation is affected
    *   - The result exponent is approximately exp(a) - exp(b)
    *
    * Strategy:
    *   1. Check if either operand is in the "danger zone" (small exponent)
    *   2. Scale the small operand up before division
    *   3. Perform division using the Nagai et al. algorithm
    *   4. Scale the result back
    *
    * Reference:
    *   Nagai, Yoshida, Kuroda, Kanada (2008). Fast Quadruple Precision
    *   Arithmetic Library on Parallel Computer SR11000/J2.
    *   Conditional scaling adapted from QD library techniques.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_high_div (const _FpType __a_hi, 
                                             const _FpType __a_lo, 
                                             const _FpType __b_hi, 
                                             const _FpType __b_lo, 
                                             _FpType*      __res_hi, 
                                             _FpType*      __res_lo) noexcept
    {

        
        // Type-specific constants for conditional scaling
        using UintType = ::cuda::std::conditional_t<::cuda::std::is_same_v<_FpType, float>, uint32_t, uint64_t>;
        
        constexpr int __exp_bits   = ::cuda::std::is_same_v<_FpType, float> ? 8 : 11;
        constexpr int __mant_bits  = ::cuda::std::is_same_v<_FpType, float> ? 23 : 52;
        constexpr UintType __exp_mask = ((UintType(1) << __exp_bits) - 1) << __mant_bits;
        
        // Threshold for scaling: exponent < threshold means we should scale up
        // For float: if exp < 32 (value < 2^-95), scale up by 2^64
        // For double: if exp < 64 (value < 2^-959), scale up by 2^512
        constexpr int __exp_threshold_low = ::cuda::std::is_same_v<_FpType, float> ? 32 : 64;
        
        // Scale factors
        constexpr _FpType __scale_up   = ::cuda::std::is_same_v<_FpType, float> ? _FpType(0x1.0p64f)  : _FpType(0x1.0p512);
        constexpr _FpType __scale_down = ::cuda::std::is_same_v<_FpType, float> ? _FpType(0x1.0p-64f) : _FpType(0x1.0p-512);
        
        // Extract exponents
        UintType __a_bits = __fpmp_internal_bit_cast<UintType>(__a_hi);
        UintType __b_bits = __fpmp_internal_bit_cast<UintType>(__b_hi);
        int __a_exp = static_cast<int>((__a_bits & __exp_mask) >> __mant_bits);
        int __b_exp = static_cast<int>((__b_bits & __exp_mask) >> __mant_bits);
        
        // Branch-free: create mask for whether 'a' needs scaling
        // needs_scale_a = -1 if a_exp < exp_threshold_low, 0 otherwise
        int __needs_scale_a = (__a_exp - __exp_threshold_low) >> 31;
        
        // Branch-free: create mask for whether 'b' needs scaling
        // needs_scale_b = -1 if b_exp < exp_threshold_low, 0 otherwise
        int __needs_scale_b = (__b_exp - __exp_threshold_low) >> 31;
        
        // Select scale factors for 'a' (branch-free)
        UintType __scale_up_bits = __fpmp_internal_bit_cast<UintType>(__scale_up);
        UintType __one_bits      = __fpmp_internal_bit_cast<UintType>(_FpType(1.0));
        UintType __scale_a_bits  = (__scale_up_bits & UintType(__needs_scale_a)) | (__one_bits & UintType(~__needs_scale_a));
        _FpType __scale_a         = __fpmp_internal_bit_cast<_FpType>(__scale_a_bits);
        
        // Select scale factors for 'b' (branch-free)
        UintType __scale_b_bits  = (__scale_up_bits & UintType(__needs_scale_b)) | (__one_bits & UintType(~__needs_scale_b));
        _FpType __scale_b         = __fpmp_internal_bit_cast<_FpType>(__scale_b_bits);
        
        // Scale operands
        _FpType __sa_hi = __fpmp_mul_rn(__a_hi, __scale_a);
        _FpType __sa_lo = __fpmp_mul_rn(__a_lo, __scale_a);
        _FpType __sb_hi = __fpmp_mul_rn(__b_hi, __scale_b);
        _FpType __sb_lo = __fpmp_mul_rn(__b_lo, __scale_b);
        
        // Perform division on scaled operands using Nagai et al. algorithm
        _FpType __t_hi, __t_lo;
        _FpType __e, __r;
        __r          = __fpmp_rcp_rn(__sb_hi);
        __t_hi       = __fpmp_mul_rn (__sa_hi, __r);
        __e          = __fpmp_fma_rn (__sb_hi, -__t_hi, __sa_hi);
        __t_hi       = __fpmp_fma_rn (__r, __e, __t_hi);
        __t_lo       = __fpmp_fma_rn (__sb_hi, -__t_hi, __sa_hi);
        __t_lo       = __fpmp_add_rn (__sa_lo, __t_lo);
        __t_lo       = __fpmp_fma_rn (__sb_lo, -__t_hi, __t_lo);
        __e          = __fpmp_mul_rn (__r, __t_lo);
        __t_lo       = __fpmp_fma_rn (__sb_hi, -__e, __t_lo);
        __t_lo       = __fpmp_fma_rn (__r, __t_lo, __e);
        __e          = __fpmp_add_rn (__t_hi, __t_lo);
        
        _FpType __r_hi = __e;
        _FpType __r_lo = __fpmp_add_rn (__t_hi - __e, __t_lo);
        
        // Compute result scale factor: inv_scale = scale_b / scale_a
        // If a was scaled up, result should be scaled down
        // If b was scaled up, result should be scaled up (since we divided by larger b)
        UintType __scale_down_bits = __fpmp_internal_bit_cast<UintType>(__scale_down);
        
        // For 'a' scaling: if we scaled a up, scale result down
        UintType __inv_scale_a_bits = (__scale_down_bits & UintType(__needs_scale_a)) | (__one_bits & UintType(~__needs_scale_a));
        _FpType __inv_scale_a        = __fpmp_internal_bit_cast<_FpType>(__inv_scale_a_bits);
        
        // For 'b' scaling: if we scaled b up, scale result up (compensate)
        UintType __comp_scale_b_bits = (__scale_up_bits & UintType(__needs_scale_b)) | (__one_bits & UintType(~__needs_scale_b));
        _FpType __comp_scale_b        = __fpmp_internal_bit_cast<_FpType>(__comp_scale_b_bits);
        
        // Combined scale factor
        _FpType __final_scale = __fpmp_mul_rn(__inv_scale_a, __comp_scale_b);
        
        // Scale result back
        __r_hi = __fpmp_mul_rn(__r_hi, __final_scale);
        __r_lo = __fpmp_mul_rn(__r_lo, __final_scale);
        
        // Final normalization to ensure (hi, lo) invariant after scaling
        *__res_hi = __fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
    } // __fpmp2_high_div
#endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1

    /*
    * --------------------------------------------------------------------
    * Square root & reciprocal square root operations
    * --------------------------------------------------------------------
    */
    /*
    iteration based on equation 4 from a paper by Alan Karp and Peter Markstein,
    High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4, December
    1997, pp. 561-589.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_rsqrt (const _FpType __a_hi, 
                                           const _FpType __a_lo, 
                                           _FpType*      __res_hi, 
                                           _FpType*      __res_lo) noexcept
    {

        _FpType __z_hi, __z_lo;
        _FpType __r, __s, __e;
        _FpType __one = static_cast<_FpType>(1.0);
        _FpType __half = static_cast<_FpType>(0.5);
        __r    = __fpmp_rsqrt_rn(__a_hi);
        __e    = __fpmp_mul_rn (__a_hi, __r);
        __s    = __fpmp_fma_rn (__e, -__r, __one);
        __e    = __fpmp_fma_rn (__a_hi, __r, -__e);
        __s    = __fpmp_fma_rn (__e, -__r, __s);
        __e    = __fpmp_mul_rn (__a_lo, __r);
        __s    = __fpmp_fma_rn (__e, -__r, __s);
        __e    = __fpmp_mul_rn (__half, __r);
        __z_hi = __fpmp_mul_rn (__e, __s);
        __z_lo = __fpmp_fma_rn (__e, __s, -__z_hi);
        __s    = __fpmp_add_rn (__r, __z_hi);
        __r    = __fpmp_add_rn (__r, -__s);
        __r    = __fpmp_add_rn (__r, __z_hi);
        __r    = __fpmp_add_rn (__r, __z_lo);
        __e    = __fpmp_add_rn (__s, __r);
        __z_lo = __fpmp_add_rn (__s - __e, __r);
        __z_hi = __e;

        *__res_hi = __z_hi;
        *__res_lo = __z_lo;
    } // __fpmp2_rsqrt

    /* Compute high-accuracy square root. Newton-Raphson
    iteration based on equation 4 from a paper by Alan Karp and Peter Markstein,
    High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4, December
    1997, pp. 561-589.
    */    
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_sqrt (const _FpType __a_hi, 
                                          const _FpType __a_lo, 
                                          _FpType*      __res_hi, 
                                          _FpType*      __res_lo) noexcept
    { 

        _FpType __t_hi, __t_lo, __tmp_lo;
        _FpType __e, __y, __s, __r;
        _FpType __zero = static_cast<_FpType>(0.0);
        _FpType __half = static_cast<_FpType>(0.5);
        __r = __fpmp_rsqrt_rn(__a_hi);
        if (__a_hi == __zero) __r = __zero;
        __y           = __fpmp_mul_rn (__a_hi, __r);
        __s           = __fpmp_fma_rn (__y, -__y, __a_hi);
        __r           = __fpmp_mul_rn (__half, __r);
        __e           = __fpmp_add_rn (__s, __a_lo);
        __tmp_lo      = __fpmp_add_rn (__s - __e, __a_lo);
        __t_hi        = __fpmp_mul_rn (__r, __e);
        __t_lo        = __fpmp_fma_rn (__r, __e, -__t_hi);
        __t_lo        = __fpmp_fma_rn (__r, __tmp_lo, __t_lo);
        __r           = __fpmp_add_rn (__y, __t_hi);
        __s           = __fpmp_add_rn (__y - __r, __t_hi);
        __s           = __fpmp_add_rn (__s, __t_lo);
        __e           = __fpmp_add_rn (__r, __s);

        *__res_lo    = __fpmp_add_rn (__r - __e, __s);
        *__res_hi    = __e;
    } // __fpmp2_sqrt

    /* Compute fast fused multiply-add: x*y+z  (16 ops, no normalization)
        Uses hardware FMA for the main term (single rounding), then recovers
        the exact error via the Boldo-Muller EFT:
          x_hi*y_hi = p + q  (exact, via two_mult_fma)
          p + z_hi  = s + t  (exact, via two_sum)
          => error  = (s - r_hi) + t + q
        where (s - r_hi) is exact by the Boldo-Muller theorem.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_low_fma (const _FpType __x_hi, 
                                             const _FpType __x_lo, 
                                             const _FpType __y_hi, 
                                             const _FpType __y_lo, 
                                             const _FpType __z_hi, 
                                             const _FpType __z_lo, 
                                             _FpType*      __res_hi, 
                                             _FpType*      __res_lo) noexcept
    { 

        
        _FpType __r_hi = __fpmp_fma_rn(__x_hi, __y_hi, __z_hi);
        
        _FpType __q;
        _FpType __p = __fpmp_two_mult_fma(__x_hi, __y_hi, &__q);
        _FpType __t;
        _FpType __s = __fpmp_two_sum(__p, __z_hi, &__t);
        _FpType __r_lo = __fpmp_add_rn(__fpmp_sub_rn(__s, __r_hi), __fpmp_add_rn(__t, __q));
        
        __r_lo = __fpmp_fma_rn(__x_hi, __y_lo, __r_lo);
        __r_lo = __fpmp_fma_rn(__x_lo, __y_hi, __r_lo);
        __r_lo = __fpmp_fma_rn(__x_lo, __y_lo, __r_lo);
        __r_lo = __fpmp_add_rn(__r_lo, __z_lo);

        *__res_hi = __r_hi;
        *__res_lo = __r_lo;
    } // __fpmp2_low_fma

    /* Compute high-accuracy fused multiply-add: x*y+z
        Uses hardware FMA for the main term (single rounding), then recovers
        the exact error via the Boldo-Muller EFT:
          x_hi*y_hi = p + q  (exact, via two_mult_fma)
          p + z_hi  = s + t  (exact, via two_sum)
          => error  = (s - r_hi) + t + q
        where (s - r_hi) is exact by the Boldo-Muller theorem.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_fma (const _FpType __x_hi, 
                                         const _FpType __x_lo, 
                                         const _FpType __y_hi, 
                                         const _FpType __y_lo, 
                                         const _FpType __z_hi, 
                                         const _FpType __z_lo, 
                                         _FpType*      __res_hi, 
                                         _FpType*      __res_lo) noexcept
    { 

        
        // Hardware FMA: x_hi*y_hi + z_hi with single rounding (optimal)
        _FpType __r_hi = __fpmp_fma_rn(__x_hi, __y_hi, __z_hi);
        
        // Exact error recovery for the main FMA
        _FpType __q;
        _FpType __p = __fpmp_two_mult_fma(__x_hi, __y_hi, &__q);
        _FpType __t;
        _FpType __s = __fpmp_two_sum(__p, __z_hi, &__t);
        _FpType __r_lo = __fpmp_add_rn(__fpmp_sub_rn(__s, __r_hi), __fpmp_add_rn(__t, __q));
        
        // Cross terms and remaining contributions
        __r_lo = __fpmp_fma_rn(__x_hi, __y_lo, __r_lo);
        __r_lo = __fpmp_fma_rn(__x_lo, __y_hi, __r_lo);
        __r_lo = __fpmp_fma_rn(__x_lo, __y_lo, __r_lo);
        __r_lo = __fpmp_add_rn(__r_lo, __z_lo);
        
        // Normalize
        *__res_hi = __fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
    } // __fpmp2_fma

    /* Compute accurate fused multiply-add: x*y+z
        Same EFT-based main term as __fpmp2_fma, but cross terms are
        computed exactly via two_mult_fma and accumulated with two_sum
        error tracking. This avoids precision loss when cross terms are
        of similar magnitude to r_lo (e.g. catastrophic cancellation in
        the main term).
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_high_fma (const _FpType __x_hi, 
                                              const _FpType __x_lo, 
                                              const _FpType __y_hi, 
                                              const _FpType __y_lo, 
                                              const _FpType __z_hi, 
                                              const _FpType __z_lo, 
                                              _FpType*      __res_hi, 
                                              _FpType*      __res_lo) noexcept
    { 
        _FpType __r_hi = __fpmp_fma_rn(__x_hi, __y_hi, __z_hi);
        
        _FpType __q;
        _FpType __p = __fpmp_two_mult_fma(__x_hi, __y_hi, &__q);
        _FpType __t;
        _FpType __s = __fpmp_two_sum(__p, __z_hi, &__t);
        _FpType __r_lo = __fpmp_add_rn(__fpmp_sub_rn(__s, __r_hi), __fpmp_add_rn(__t, __q));
        
        _FpType __c1_lo;
        _FpType __c1_hi = __fpmp_two_mult_fma(__x_hi, __y_lo, &__c1_lo);
        
        _FpType __c2_lo;
        _FpType __c2_hi = __fpmp_two_mult_fma(__x_lo, __y_hi, &__c2_lo);
        
        _FpType __cross_err;
        _FpType __cross = __fpmp_two_sum(__c1_hi, __c2_hi, &__cross_err);
        
        _FpType __acc_err;
        __r_lo = __fpmp_two_sum(__r_lo, __cross, &__acc_err);
        
        _FpType __residual = __fpmp_add_rn(__acc_err, __fpmp_add_rn(__cross_err, __fpmp_add_rn(__c1_lo, __c2_lo)));
        __residual = __fpmp_fma_rn(__x_lo, __y_lo, __residual);
        __residual = __fpmp_add_rn(__residual, __z_lo);
        
        __r_lo = __fpmp_add_rn(__r_lo, __residual);
        
        *__res_hi = __fpmp_fast_two_sum(__r_hi, __r_lo, __res_lo);
    } // __fpmp2_high_fma

    /*
    * --------------------------------------------------------------------
    * Fused multiply-add with rounding operations
    * --------------------------------------------------------------------
    */
    // multiply-add with rounding (default: fast mul + default add)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_mad (const _FpType __x_hi, 
                                         const _FpType __x_lo, 
                                         const _FpType __y_hi, 
                                         const _FpType __y_lo, 
                                         const _FpType __z_hi, 
                                         const _FpType __z_lo, 
                                         _FpType*      __res_hi, 
                                         _FpType*      __res_lo) noexcept
    { 
        _FpType __t_hi, __t_lo;
        __fpmp2_low_mul(__x_hi, __x_lo, __y_hi, __y_lo, &__t_hi, &__t_lo);
        __fpmp2_add(__t_hi, __t_lo, __z_hi, __z_lo, __res_hi, __res_lo);
    }

    // multiply-add fast (fast mul + fast add)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_low_mad (const _FpType __x_hi, 
                                             const _FpType __x_lo, 
                                             const _FpType __y_hi, 
                                             const _FpType __y_lo, 
                                             const _FpType __z_hi, 
                                             const _FpType __z_lo, 
                                             _FpType*      __res_hi, 
                                             _FpType*      __res_lo) noexcept
    { 
        _FpType __t_hi, __t_lo;
        __fpmp2_low_mul(__x_hi, __x_lo, __y_hi, __y_lo, &__t_hi, &__t_lo);
        __fpmp2_low_add(__t_hi, __t_lo, __z_hi, __z_lo, __res_hi, __res_lo);
    }

    // multiply-add accurate (default mul + accurate add)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_high_mad (const _FpType __x_hi, 
                                              const _FpType __x_lo, 
                                              const _FpType __y_hi, 
                                              const _FpType __y_lo, 
                                              const _FpType __z_hi, 
                                              const _FpType __z_lo, 
                                              _FpType*      __res_hi, 
                                              _FpType*      __res_lo) noexcept
    { 
        _FpType __t_hi, __t_lo;
        __fpmp2_mul(__x_hi, __x_lo, __y_hi, __y_lo, &__t_hi, &__t_lo);
        __fpmp2_high_add(__t_hi, __t_lo, __z_hi, __z_lo, __res_hi, __res_lo);
    }

    /*
    * --------------------------------------------------------------------
    * Negation operations
    * --------------------------------------------------------------------
    */
    // negation
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  void __fpmp2_neg (const _FpType __x_hi, 
                                         const _FpType __x_lo, 
                                         _FpType*      __res_hi, 
                                         _FpType*      __res_lo) noexcept
    { 
        *__res_hi = -__x_hi;
        *__res_lo = -__x_lo;
    }

    /*
    * --------------------------------------------------------------------
    * Comparison operations
    * --------------------------------------------------------------------
    */
    // == comparison
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  bool __fpmp2_cmp_eq (const _FpType __x_hi, 
                                            const _FpType __x_lo, 
                                            const _FpType __y_hi, 
                                            const _FpType __y_lo) noexcept 
    { 
        return __x_hi == __y_hi && __x_lo == __y_lo;
    }

    // != comparison
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  bool __fpmp2_cmp_ne(const _FpType __x_hi, 
                                           const _FpType __x_lo, 
                                           const _FpType __y_hi, 
                                           const _FpType __y_lo) noexcept 
    { 
        return __x_hi != __y_hi || __x_lo != __y_lo;
    }

    // < comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  bool __fpmp2_cmp_lt (const _FpType __x_hi, 
                                            const _FpType __x_lo, 
                                            const _FpType __y_hi, 
                                            const _FpType __y_lo) noexcept 
    { 
        return __x_hi < __y_hi || (__x_hi == __y_hi && __x_lo < __y_lo);
    }

    // > comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  bool __fpmp2_cmp_gt (const _FpType __x_hi, 
                                            const _FpType __x_lo, 
                                            const _FpType __y_hi, 
                                            const _FpType __y_lo) noexcept 
    { 
        return __x_hi > __y_hi || (__x_hi == __y_hi && __x_lo > __y_lo);
    }

    // <= comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  bool __fpmp2_cmp_le (const _FpType __x_hi, 
                                            const _FpType __x_lo, 
                                            const _FpType __y_hi, 
                                            const _FpType __y_lo) noexcept 
    { 
        return __x_hi < __y_hi || (__x_hi == __y_hi && __x_lo <= __y_lo);
    }

    // >= comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  bool __fpmp2_cmp_ge (const _FpType __x_hi, 
                                            const _FpType __x_lo, 
                                            const _FpType __y_hi, 
                                            const _FpType __y_lo) noexcept 
    { 
        return __x_hi > __y_hi || (__x_hi == __y_hi && __x_lo >= __y_lo);
    }

    /*
    * --------------------------------------------------------------------
    * Bit cast operations (IEEE-754 format)
    * --------------------------------------------------------------------
    */
    // bit_cast to IEEE-754 format bits
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API  uint64_t __fpmp2_bit_cast (const _FpType __x_hi, 
                                                  const _FpType __x_lo) noexcept
    { 

        double __d = __fpmp2_to_double(__x_hi, __x_lo);
        return __fpmp_internal_bit_cast<uint64_t>(__d);
    }

    /*
    * --------------------------------------------------------------------
    * Atomic operations (CUDA device only)
    * --------------------------------------------------------------------
    */
  #ifdef __CUDACC__
    /*
    * --------------------------------------------------------------------
    * Atomic operations - Primary template declarations
    * --------------------------------------------------------------------
    */
    // Primary template declarations (specialized for float and double below)
    template<typename _FpType>
    _CCCL_DEVICE_API inline void __fpmp2_atomicAdd (_FpType*      __address_hi, 
                                                    _FpType*      __address_lo, 
                                                    const _FpType __addition_hi, 
                                                    const _FpType __addition_lo, 
                                                    _FpType*      __old_hi, 
                                                    _FpType*      __old_lo) noexcept;

    template<typename _FpType>
    _CCCL_DEVICE_API inline void __fpmp2_atomicSub (_FpType*      __address_hi, 
                                                    _FpType*      __address_lo, 
                                                    const _FpType __val_hi, 
                                                    const _FpType __val_lo, 
                                                    _FpType*      __old_hi, 
                                                    _FpType*      __old_lo) noexcept;

    /*
    * --------------------------------------------------------------------
    * Atomic operations - Float (fp32) specializations
    * --------------------------------------------------------------------
    */
    // atomicAdd for float (fp32mp2): Uses 64-bit atomicCAS
    // Two floats = 64 bits fits in unsigned long long int
    // Returns the old value before the addition
    template<>
    _CCCL_DEVICE_API inline void __fpmp2_atomicAdd<float>  (float*       __address_hi, 
                                                            float*       __address_lo, 
                                                            const float  __addition_hi, 
                                                            const float  __addition_lo, 
                                                            float*       __old_hi, 
                                                            float*       __old_lo) noexcept
    {

        
        // Treat the two floats as a single 64-bit value for atomic operations
        // The address must be 8-byte aligned (guaranteed by alignas(2*alignof(float)) in the class)
        static_assert(sizeof(float) * 2 == sizeof(unsigned long long int), 
                      "Two floats must equal one unsigned long long int"); 
        
        unsigned long long int* __address_as_ull = reinterpret_cast<unsigned long long int*>(__address_hi);
        unsigned long long int __old             = *__address_as_ull;
        unsigned long long int __assumed;
        
        // Use the atomicCAS loop with retries to ensure atomicity
        do 
        {
            __assumed = __old;
            
            // Extract old values from the 64-bit integer
            uint32_t __old_hi_bits = static_cast<uint32_t>(__assumed & 0xFFFFFFFFULL);
            uint32_t __old_lo_bits = static_cast<uint32_t>((__assumed >> 32) & 0xFFFFFFFFULL);
            float __old_hi_val = __fpmp_internal_bit_cast<float>(__old_hi_bits);
            float __old_lo_val = __fpmp_internal_bit_cast<float>(__old_lo_bits);
            
            // Perform addition based on method
            float __new_hi, __new_lo;
            __fpmp2_high_add(__old_hi_val, __old_lo_val, __addition_hi, __addition_lo, &__new_hi, &__new_lo);
            
            // Pack new values into a 64-bit integer
            uint32_t __new_hi_bits = __fpmp_internal_bit_cast<uint32_t>(__new_hi);
            uint32_t __new_lo_bits = __fpmp_internal_bit_cast<uint32_t>(__new_lo);
            unsigned long long int __new_ull = static_cast<unsigned long long int>(__new_hi_bits) | 
                                            (static_cast<unsigned long long int>(__new_lo_bits) << 32);
            
            __old = atomicCAS(__address_as_ull, __assumed, __new_ull);
        } while (__assumed != __old);
        
        // Return old value - extract from the final 'old' value
        uint32_t __old_hi_bits = static_cast<uint32_t>(__old & 0xFFFFFFFFULL);
        uint32_t __old_lo_bits = static_cast<uint32_t>((__old >> 32) & 0xFFFFFFFFULL);
        *__old_hi = __fpmp_internal_bit_cast<float>(__old_hi_bits);
        *__old_lo = __fpmp_internal_bit_cast<float>(__old_lo_bits);
    }

    // atomicSub for float: Uses negation and atomicAdd
    template<>
    _CCCL_DEVICE_API inline void __fpmp2_atomicSub<float> (float*       __address_hi, 
                                                           float*       __address_lo, 
                                                           const float  __val_hi, 
                                                           const float  __val_lo, 
                                                           float*       __old_hi, 
                                                           float*       __old_lo) noexcept
    {
        // Negate the value and call atomicAdd with the same method
        __fpmp2_atomicAdd<float>(__address_hi, __address_lo, -__val_hi, -__val_lo, __old_hi, __old_lo);
    }

    /*
    * --------------------------------------------------------------------
    * Atomic operations - Double (fp64) specializations
    * --------------------------------------------------------------------
    */
    // atomicAdd for double (fp64mp2): Uses 128-bit atomicCAS
    // Two doubles = 128 bits requires ulonglong2 and sm_90+ (Hopper architecture)
    // Returns the old value before the addition
    template<>
    _CCCL_DEVICE_API inline void __fpmp2_atomicAdd<double> (double*       __address_hi, 
                                                            double*       __address_lo, 
                                                            const double  __addition_hi, 
                                                            const double  __addition_lo, 
                                                            double*       __old_hi, 
                                                            double*       __old_lo) noexcept
    {
      #if __CUDA_ARCH__ >= 900

        
        // Treat the two doubles as a single 128-bit value for atomic operations
        // The address must be 16-byte aligned for 128-bit atomics
        static_assert(sizeof(double) * 2 == sizeof(ulonglong2), 
                      "Two doubles must equal one ulonglong2 (128 bits)"); 
        
        ulonglong2* __address_as_ull2 = reinterpret_cast<ulonglong2*>(__address_hi);
        ulonglong2 __old              = *__address_as_ull2;
        ulonglong2 __assumed;
        
        // Use the atomicCAS loop with retries to ensure atomicity
        do 
        {
            __assumed = __old;
            
            // Extract old values from the 128-bit structure
            double __old_hi_val = __fpmp_internal_bit_cast<double>(__assumed.x);
            double __old_lo_val = __fpmp_internal_bit_cast<double>(__assumed.y);
            
            // Perform addition based on method
            double __new_hi, __new_lo;
            __fpmp2_high_add(__old_hi_val, __old_lo_val, __addition_hi, __addition_lo, &__new_hi, &__new_lo);

            // Pack new values into a 128-bit structure
            ulonglong2 __new_ull2;
            __new_ull2.x = __fpmp_internal_bit_cast<unsigned long long int>(__new_hi);
            __new_ull2.y = __fpmp_internal_bit_cast<unsigned long long int>(__new_lo);
            
            // 128-bit atomicCAS available on sm_90+
            __old = atomicCAS(__address_as_ull2, __assumed, __new_ull2);
        } while (__assumed.x != __old.x || __assumed.y != __old.y);
        
        // Return old value - extract from the final 'old' value
        *__old_hi = __fpmp_internal_bit_cast<double>(__old.x);
        *__old_lo = __fpmp_internal_bit_cast<double>(__old.y);
      #else
        // 128-bit atomicCAS requires sm_90+ (Hopper architecture)
        // On older architectures, this is a no-op stub
        // Runtime checks should prevent this code path from being executed
        (void)__address_hi; (void)__address_lo;
        (void)__addition_hi; (void)__addition_lo;
        // Return the current values unchanged
        *__old_hi = *__address_hi;
        *__old_lo = *__address_lo;
      #endif
    }

    // atomicSub for double: Uses negation and atomicAdd
    template<>
    _CCCL_DEVICE_API inline void __fpmp2_atomicSub<double> (double*       __address_hi, 
                                                            double*       __address_lo, 
                                                            const double  __val_hi, 
                                                            const double  __val_lo, 
                                                            double*       __old_hi, 
                                                            double*       __old_lo) noexcept
    {
        // Negate the value and call atomicAdd with the same method
        __fpmp2_atomicAdd<double>(__address_hi, __address_lo, -__val_hi, -__val_lo, __old_hi, __old_lo);
    }

  #endif // __CUDACC__

  // __fpmp_fp128 operations (only for FpType == double)
  // available only for CUDA architectures >= 1000 or when _CCCL_FPMP_FP128_ENABLE is defined
  #if _CCCL_FPMP_FP128_ENABLE == 1
    template<typename _FpType = double>
    constexpr _CCCL_TRIVIAL_API  void __fpmp2_from_quad  (const __fpmp_fp128 __x, 
                                                          _FpType*           __res_hi, 
                                                          _FpType*           __res_lo) noexcept 
    {
        *__res_hi = static_cast<_FpType>(__x);
        *__res_lo = static_cast<_FpType>(__x - static_cast<__fpmp_fp128>(*__res_hi));
    }

    template<typename _FpType = double>
    _CCCL_TRIVIAL_API  __fpmp_fp128 __fpmp2_to_quad  (const _FpType __x_hi, 
                                                      const _FpType __x_lo) noexcept 
    {
        return static_cast<__fpmp_fp128>(__x_hi) + static_cast<__fpmp_fp128>(__x_lo);
    }
   #endif // _CCCL_FPMP_FP128_ENABLE == 1    

  #else // _CCCL_FPMP_USE_LIB

/*
 * ============================================================================
 * Single Precision (fp32) Multi-Precision Operations
 * ============================================================================
 */
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_from_double(const double __x, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_from_int(const int32_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_from_uint(const uint32_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_from_ll(const int64_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_from_ull(const uint64_t __i, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL double   __fp32mp2_to_double(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL float    __fp32mp2_to_float(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int32_t  __fp32mp2_to_int(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint32_t __fp32mp2_to_uint(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL int64_t  __fp32mp2_to_ll(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint64_t __fp32mp2_to_ull(const float __x_hi, const float __x_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_add(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_mid_add(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_low_add(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_high_add(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_sub(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_mid_sub(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_low_sub(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_high_sub(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_acc(const float __c, float* __acc_hi, float* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_mid_acc(const float __c, float* __acc_hi, float* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_low_acc(const float __c, float* __acc_hi, float* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_high_acc(const float __c, float* __acc_hi, float* __acc_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_mul(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_mid_mul(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_low_mul(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
#if _CCCL_FPMP_USE_ACCURATE_MUL == 1
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_high_mul(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
#endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_renormalize(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_div(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_mid_div(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_low_div(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept;
#if _CCCL_FPMP_USE_ACCURATE_DIV == 1
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_high_div(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept;
#endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_sqrt(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_rsqrt(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_mad(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_mid_mad(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_low_mad(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_high_mad(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_fma(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_mid_fma(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_low_fma(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_high_fma(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL void     __fp32mp2_neg(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool     __fp32mp2_cmp_eq(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool     __fp32mp2_cmp_ne(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool     __fp32mp2_cmp_lt(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool     __fp32mp2_cmp_gt(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool     __fp32mp2_cmp_le(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL bool     __fp32mp2_cmp_ge(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept;
_CCCL_FPMP_BUILTIN_DECL uint64_t __fp32mp2_bit_cast(const float __x_hi, const float __x_lo) noexcept;
#ifdef __CUDACC__
_CCCL_FPMP_BUILTIN_DEVICE_DECL void __fp32mp2_atomicAdd(float* __address_hi, float* __address_lo, const float __addition_hi, const float __addition_lo, float* __old_hi, float* __old_lo) noexcept;
_CCCL_FPMP_BUILTIN_DEVICE_DECL void __fp32mp2_atomicSub(float* __address_hi, float* __address_lo, const float __val_hi, const float __val_lo, float* __old_hi, float* __old_lo) noexcept;
#endif // __CUDACC__

    /*
    * ============================================================================
    * Double Precision (fp64) Multi-Precision Operations
    * ============================================================================
    */
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_from_double(const double __x, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_from_int(const int32_t __i, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_from_uint(const uint32_t __i, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_from_ll(const int64_t __i, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_from_ull(const uint64_t __i, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL double   __fp64mp2_to_double(const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL float    __fp64mp2_to_float(const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int32_t  __fp64mp2_to_int(const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL uint32_t __fp64mp2_to_uint(const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int64_t  __fp64mp2_to_ll(const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL uint64_t __fp64mp2_to_ull(const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_add(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_mid_add(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_low_add(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_high_add(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_sub(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_mid_sub(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_low_sub(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_high_sub(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_acc(const double __c, double* __acc_hi, double* __acc_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_mid_acc(const double __c, double* __acc_hi, double* __acc_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_low_acc(const double __c, double* __acc_hi, double* __acc_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_high_acc(const double __c, double* __acc_hi, double* __acc_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_mul(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_mid_mul(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_low_mul(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
#if _CCCL_FPMP_USE_ACCURATE_MUL == 1
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_high_mul(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
#endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_renormalize(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_div(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_mid_div(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_low_div(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept;
#if _CCCL_FPMP_USE_ACCURATE_DIV == 1
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_high_div(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept;
#endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_sqrt(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_rsqrt(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_mad(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_mid_mad(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_low_mad(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_high_mad(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_fma(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_mid_fma(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_low_fma(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_high_fma(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void     __fp64mp2_neg(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL bool     __fp64mp2_cmp_eq(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL bool     __fp64mp2_cmp_ne(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL bool     __fp64mp2_cmp_lt(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL bool     __fp64mp2_cmp_gt(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL bool     __fp64mp2_cmp_le(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL bool     __fp64mp2_cmp_ge(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL uint64_t __fp64mp2_bit_cast(const double __x_hi, const double __x_lo) noexcept;
    #ifdef __CUDACC__
    _CCCL_FPMP_BUILTIN_DEVICE_DECL void __fp64mp2_atomicAdd(double* __address_hi, double* __address_lo, const double __addition_hi, const double __addition_lo, double* __old_hi, double* __old_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DEVICE_DECL void __fp64mp2_atomicSub(double* __address_hi, double* __address_lo, const double __val_hi, const double __val_lo, double* __old_hi, double* __old_lo) noexcept;
    #endif // __CUDACC__
    #if _CCCL_FPMP_FP128_ENABLE == 1
    _CCCL_FPMP_BUILTIN_DECL void         __fp64mp2_from_quad(const __fpmp_fp128 __x, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL __fpmp_fp128 __fp64mp2_to_quad(const double __x_hi, const double __x_lo) noexcept;
    #endif // _CCCL_FPMP_FP128_ENABLE == 1


/*
 * ============================================================================
 * Template Wrappers for Type-Generic Function Dispatch
 * ============================================================================
 * 
 * These templates provide a unified interface that automatically dispatches
 * to the appropriate fp32mp2 or fp64mp2 implementation based on the template
 * parameter type T (float or double).
 * ============================================================================
 */
// Template declarations - type-generic prototypes

template<typename _Tp> _CCCL_API inline void     __fpmp2_from_double(const double __x, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_from_int(const int32_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_from_uint(const uint32_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_from_ll(const int64_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_from_ull(const uint64_t __i, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline double   __fpmp2_to_double(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template<typename _Tp> _CCCL_API inline float    __fpmp2_to_float(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template<typename _Tp> _CCCL_API inline int32_t  __fpmp2_to_int(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template<typename _Tp> _CCCL_API inline uint32_t __fpmp2_to_uint(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template<typename _Tp> _CCCL_API inline int64_t  __fpmp2_to_ll(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template<typename _Tp> _CCCL_API inline uint64_t __fpmp2_to_ull(const _Tp __x_hi, const _Tp __x_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_add(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_mid_add(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_low_add(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_high_add(const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_sub(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_mid_sub(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_low_sub(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_high_sub(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_acc(const _Tp __c, _Tp* __acc_hi, _Tp* __acc_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_mid_acc(const _Tp __c, _Tp* __acc_hi, _Tp* __acc_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_low_acc(const _Tp __c, _Tp* __acc_hi, _Tp* __acc_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_high_acc(const _Tp __c, _Tp* __acc_hi, _Tp* __acc_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_mul(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_mid_mul(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_low_mul(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
#if _CCCL_FPMP_USE_ACCURATE_MUL == 1
template<typename T> _CCCL_API inline void     __fpmp2_high_mul(const T __x_hi, const T __x_lo, const T __y_hi, const T __y_lo, T* __res_hi, T* __res_lo) noexcept;
#endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
template<typename _Tp> _CCCL_API inline void     __fpmp2_renormalize(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_div(const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_mid_div(const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_low_div(const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
#if _CCCL_FPMP_USE_ACCURATE_DIV == 1
template<typename T> _CCCL_API inline void     __fpmp2_high_div(const T __a_hi, const T __a_lo, const T __b_hi, const T __b_lo, T* __res_hi, T* __res_lo) noexcept;
#endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
template<typename _Tp> _CCCL_API inline void     __fpmp2_sqrt(const _Tp __a_hi, const _Tp __a_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_rsqrt(const _Tp __a_hi, const _Tp __a_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_mad(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, const _Tp __z_hi, const _Tp __z_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_mid_mad(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, const _Tp __z_hi, const _Tp __z_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_low_mad(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, const _Tp __z_hi, const _Tp __z_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_high_mad(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, const _Tp __z_hi, const _Tp __z_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_fma(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, const _Tp __z_hi, const _Tp __z_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_mid_fma(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, const _Tp __z_hi, const _Tp __z_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_low_fma(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, const _Tp __z_hi, const _Tp __z_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_high_fma(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, const _Tp __z_hi, const _Tp __z_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_fma_exp(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, const _Tp __z_hi, const _Tp __z_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline void     __fpmp2_neg(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline bool     __fpmp2_cmp_eq(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template<typename _Tp> _CCCL_API inline bool     __fpmp2_cmp_ne(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template<typename _Tp> _CCCL_API inline bool     __fpmp2_cmp_lt(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template<typename _Tp> _CCCL_API inline bool     __fpmp2_cmp_gt(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template<typename _Tp> _CCCL_API inline bool     __fpmp2_cmp_le(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template<typename _Tp> _CCCL_API inline bool     __fpmp2_cmp_ge(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo) noexcept;
template<typename _Tp> _CCCL_API inline uint64_t __fpmp2_bit_cast(const _Tp __x_hi, const _Tp __x_lo) noexcept;
#ifdef __CUDACC__
template<typename _Tp> _CCCL_DEVICE_API inline void __fpmp2_atomicAdd(_Tp* __address_hi, _Tp* __address_lo, const _Tp __addition_hi, const _Tp __addition_lo, _Tp* __old_hi, _Tp* __old_lo) noexcept;
template<typename _Tp> _CCCL_DEVICE_API inline void __fpmp2_atomicSub(_Tp* __address_hi, _Tp* __address_lo, const _Tp __val_hi, const _Tp __val_lo, _Tp* __old_hi, _Tp* __old_lo) noexcept;
#endif
#if _CCCL_FPMP_FP128_ENABLE == 1
template<typename _Tp> _CCCL_API inline void         __fpmp2_from_quad(const __fpmp_fp128 __x, _Tp* __res_hi, _Tp* __res_lo) noexcept;
template<typename _Tp> _CCCL_API inline __fpmp_fp128 __fpmp2_to_quad(const _Tp __x_hi, const _Tp __x_lo) noexcept;
#endif // _CCCL_FPMP_FP128_ENABLE == 1
/*
 * ============================================================================
 * Float (fp32) Template Specializations
 * ============================================================================
 * 
 * These specializations map the generic __fpmp2_* template functions
 * to the concrete __fp32mp2_* implementations for float types.
 * ============================================================================
 */
template<> _CCCL_API inline void     __fpmp2_from_double<float>(const double __x, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_from_double(__x, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_from_int<float>(const int32_t __i, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_from_int(__i, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_from_uint<float>(const uint32_t __i, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_from_uint(__i, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_from_ll<float>(const int64_t __i, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_from_ll(__i, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_from_ull<float>(const uint64_t __i, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_from_ull(__i, __res_hi, __res_lo); }
template<> _CCCL_API inline double   __fpmp2_to_double<float>(const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_to_double(__x_hi, __x_lo); }
template<> _CCCL_API inline float    __fpmp2_to_float<float>(const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_to_float(__x_hi, __x_lo); }
template<> _CCCL_API inline int32_t  __fpmp2_to_int<float>(const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_to_int(__x_hi, __x_lo); }
template<> _CCCL_API inline uint32_t __fpmp2_to_uint<float>(const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_to_uint(__x_hi, __x_lo); }
template<> _CCCL_API inline int64_t  __fpmp2_to_ll<float>(const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_to_ll(__x_hi, __x_lo); }
template<> _CCCL_API inline uint64_t __fpmp2_to_ull<float>(const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_to_ull(__x_hi, __x_lo); }
template<> _CCCL_API inline void     __fpmp2_add<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_mid_add<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_mid_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_low_add<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_low_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_high_add<float>(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_high_add(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_sub<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_mid_sub<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_mid_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_low_sub<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_low_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_high_sub<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_high_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_acc<float>(const float __c, float* __acc_hi, float* __acc_lo) noexcept { __fp32mp2_acc(__c, __acc_hi, __acc_lo); }
template<> _CCCL_API inline void     __fpmp2_mid_acc<float>(const float __c, float* __acc_hi, float* __acc_lo) noexcept { __fp32mp2_mid_acc(__c, __acc_hi, __acc_lo); }
template<> _CCCL_API inline void     __fpmp2_low_acc<float>(const float __c, float* __acc_hi, float* __acc_lo) noexcept { __fp32mp2_low_acc(__c, __acc_hi, __acc_lo); }
template<> _CCCL_API inline void     __fpmp2_high_acc<float>(const float __c, float* __acc_hi, float* __acc_lo) noexcept { __fp32mp2_high_acc(__c, __acc_hi, __acc_lo); }
template<> _CCCL_API inline void     __fpmp2_mul<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_mid_mul<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_mid_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_low_mul<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_low_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
#if _CCCL_FPMP_USE_ACCURATE_MUL == 1
template<> _CCCL_API inline void     __fpmp2_high_mul<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_high_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
#endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
template<> _CCCL_API inline void     __fpmp2_renormalize<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_renormalize(__x_hi, __x_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_div<float>(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_mid_div<float>(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_mid_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_low_div<float>(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_low_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
#if _CCCL_FPMP_USE_ACCURATE_DIV == 1
template<> _CCCL_API inline void     __fpmp2_high_div<float>(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_high_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
#endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
template<> _CCCL_API inline void     __fpmp2_sqrt<float>(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_sqrt(__a_hi, __a_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_rsqrt<float>(const float __a_hi, const float __a_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_rsqrt(__a_hi, __a_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_mad<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_mid_mad<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_mid_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_low_mad<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_low_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_high_mad<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_high_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_fma<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_mid_fma<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_mid_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_low_fma<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_low_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_high_fma<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, const float __z_hi, const float __z_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_high_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline void     __fpmp2_neg<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_neg(__x_hi, __x_lo, __res_hi, __res_lo); }
template<> _CCCL_API inline bool     __fpmp2_cmp_eq<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept { return __fp32mp2_cmp_eq(__x_hi, __x_lo, __y_hi, __y_lo); }
template<> _CCCL_API inline bool     __fpmp2_cmp_ne<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept { return __fp32mp2_cmp_ne(__x_hi, __x_lo, __y_hi, __y_lo); }
template<> _CCCL_API inline bool     __fpmp2_cmp_lt<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept { return __fp32mp2_cmp_lt(__x_hi, __x_lo, __y_hi, __y_lo); }
template<> _CCCL_API inline bool     __fpmp2_cmp_gt<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept { return __fp32mp2_cmp_gt(__x_hi, __x_lo, __y_hi, __y_lo); }
template<> _CCCL_API inline bool     __fpmp2_cmp_le<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept { return __fp32mp2_cmp_le(__x_hi, __x_lo, __y_hi, __y_lo); }
template<> _CCCL_API inline bool     __fpmp2_cmp_ge<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo) noexcept { return __fp32mp2_cmp_ge(__x_hi, __x_lo, __y_hi, __y_lo); }
template<> _CCCL_API inline uint64_t __fpmp2_bit_cast<float>(const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_bit_cast(__x_hi, __x_lo); }
#ifdef __CUDACC__
template<> _CCCL_DEVICE_API inline void __fpmp2_atomicAdd<float>(float* __address_hi, float* __address_lo, const float __addition_hi, const float __addition_lo, float* __old_hi, float* __old_lo) noexcept { __fp32mp2_atomicAdd(__address_hi, __address_lo, __addition_hi, __addition_lo, __old_hi, __old_lo); }
template<> _CCCL_DEVICE_API inline void __fpmp2_atomicSub<float>(float* __address_hi, float* __address_lo, const float __val_hi, const float __val_lo, float* __old_hi, float* __old_lo) noexcept { __fp32mp2_atomicSub(__address_hi, __address_lo, __val_hi, __val_lo, __old_hi, __old_lo); }
#endif // __CUDACC__

    /*
    * ============================================================================
    * Double (fp64) Template Specializations
    * ============================================================================
    * 
    * These specializations map the generic __fpmp2_* template functions
    * to the concrete __fp64mp2_* implementations for double types.
    * ============================================================================
    */
    template<> _CCCL_API inline void     __fpmp2_from_double<double>(const double __x, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_from_double(__x, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_from_int<double>(const int32_t __i, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_from_int(__i, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_from_uint<double>(const uint32_t __i, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_from_uint(__i, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_from_ll<double>(const int64_t __i, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_from_ll(__i, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_from_ull<double>(const uint64_t __i, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_from_ull(__i, __res_hi, __res_lo); }
    template<> _CCCL_API inline double   __fpmp2_to_double<double>(const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_to_double(__x_hi, __x_lo); }
    template<> _CCCL_API inline float    __fpmp2_to_float<double>(const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_to_float(__x_hi, __x_lo); }
    template<> _CCCL_API inline int32_t  __fpmp2_to_int<double>(const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_to_int(__x_hi, __x_lo); }
    template<> _CCCL_API inline uint32_t __fpmp2_to_uint<double>(const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_to_uint(__x_hi, __x_lo); }
    template<> _CCCL_API inline int64_t  __fpmp2_to_ll<double>(const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_to_ll(__x_hi, __x_lo); }
    template<> _CCCL_API inline uint64_t __fpmp2_to_ull<double>(const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_to_ull(__x_hi, __x_lo); }
    template<> _CCCL_API inline void     __fpmp2_add<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_mid_add<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_mid_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_low_add<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_low_add(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_high_add<double>(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_high_add(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_sub<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_mid_sub<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_mid_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_low_sub<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_low_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_high_sub<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_high_sub(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_acc<double>(const double __c, double* __acc_hi, double* __acc_lo) noexcept { __fp64mp2_acc(__c, __acc_hi, __acc_lo); }
    template<> _CCCL_API inline void     __fpmp2_mid_acc<double>(const double __c, double* __acc_hi, double* __acc_lo) noexcept { __fp64mp2_mid_acc(__c, __acc_hi, __acc_lo); }
    template<> _CCCL_API inline void     __fpmp2_low_acc<double>(const double __c, double* __acc_hi, double* __acc_lo) noexcept { __fp64mp2_low_acc(__c, __acc_hi, __acc_lo); }
    template<> _CCCL_API inline void     __fpmp2_high_acc<double>(const double __c, double* __acc_hi, double* __acc_lo) noexcept { __fp64mp2_high_acc(__c, __acc_hi, __acc_lo); }
    template<> _CCCL_API inline void     __fpmp2_mul<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_mid_mul<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_mid_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_low_mul<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_low_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
#if _CCCL_FPMP_USE_ACCURATE_MUL == 1
    template<> _CCCL_API inline void     __fpmp2_high_mul<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_high_mul(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
#endif // _CCCL_FPMP_USE_ACCURATE_MUL == 1
    template<> _CCCL_API inline void     __fpmp2_renormalize<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_renormalize(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_div<double>(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_mid_div<double>(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_mid_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_low_div<double>(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_low_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
#if _CCCL_FPMP_USE_ACCURATE_DIV == 1
    template<> _CCCL_API inline void     __fpmp2_high_div<double>(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_high_div(__a_hi, __a_lo, __b_hi, __b_lo, __res_hi, __res_lo); }
#endif // _CCCL_FPMP_USE_ACCURATE_DIV == 1
    template<> _CCCL_API inline void     __fpmp2_sqrt<double>(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_sqrt(__a_hi, __a_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_rsqrt<double>(const double __a_hi, const double __a_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_rsqrt(__a_hi, __a_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_mad<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_mid_mad<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_mid_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_low_mad<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_low_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_high_mad<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_high_mad(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_fma<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_mid_fma<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_mid_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_low_fma<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_low_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_high_fma<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, const double __z_hi, const double __z_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_high_fma(__x_hi, __x_lo, __y_hi, __y_lo, __z_hi, __z_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void     __fpmp2_neg<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_neg(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline bool     __fpmp2_cmp_eq<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept { return __fp64mp2_cmp_eq(__x_hi, __x_lo, __y_hi, __y_lo); }
    template<> _CCCL_API inline bool     __fpmp2_cmp_ne<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept { return __fp64mp2_cmp_ne(__x_hi, __x_lo, __y_hi, __y_lo); }
    template<> _CCCL_API inline bool     __fpmp2_cmp_lt<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept { return __fp64mp2_cmp_lt(__x_hi, __x_lo, __y_hi, __y_lo); }
    template<> _CCCL_API inline bool     __fpmp2_cmp_gt<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept { return __fp64mp2_cmp_gt(__x_hi, __x_lo, __y_hi, __y_lo); }
    template<> _CCCL_API inline bool     __fpmp2_cmp_le<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept { return __fp64mp2_cmp_le(__x_hi, __x_lo, __y_hi, __y_lo); }
    template<> _CCCL_API inline bool     __fpmp2_cmp_ge<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo) noexcept { return __fp64mp2_cmp_ge(__x_hi, __x_lo, __y_hi, __y_lo); }
    template<> _CCCL_API inline uint64_t __fpmp2_bit_cast<double>(const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_bit_cast(__x_hi, __x_lo); }
    #ifdef __CUDACC__
    template<> _CCCL_DEVICE_API inline void __fpmp2_atomicAdd<double>(double* __address_hi, double* __address_lo, const double __addition_hi, const double __addition_lo, double* __old_hi, double* __old_lo) noexcept { __fp64mp2_atomicAdd(__address_hi, __address_lo, __addition_hi, __addition_lo, __old_hi, __old_lo); }
    template<> _CCCL_DEVICE_API inline void __fpmp2_atomicSub<double>(double* __address_hi, double* __address_lo, const double __val_hi, const double __val_lo, double* __old_hi, double* __old_lo) noexcept { __fp64mp2_atomicSub(__address_hi, __address_lo, __val_hi, __val_lo, __old_hi, __old_lo); }
    #endif // __CUDACC__
    #if _CCCL_FPMP_FP128_ENABLE == 1
    template<> _CCCL_API inline void         __fpmp2_from_quad<double>(const __fpmp_fp128 __x, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_from_quad(__x, __res_hi, __res_lo); }
    template<> _CCCL_API inline __fpmp_fp128 __fpmp2_to_quad<double>(const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_to_quad(__x_hi, __x_lo); }
    #endif // _CCCL_FPMP_FP128_ENABLE == 1


#endif // _CCCL_FPMP_USE_LIB

// NOTE: the freestanding fpmp2 atomics (atomicAdd/atomicSub) and warp-shuffle
// helpers live in <cuda/__fp/fpmp.h>, after the fpmp2 class definition, since
// they are public class-dependent free functions rather than internal impl.

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_IMPL_H