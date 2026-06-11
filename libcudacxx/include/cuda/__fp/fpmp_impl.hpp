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
    fpmp_impl.hpp - Core Multi-Precision Arithmetic Operations
    ======================================================================================================
    This header provides the core low-level (C-style) API for fpmp2 arithmetic.
    It implements fundamental operations using error-free transformations and supports multiple accuracy
    levels (low/mid/high; default == mid). The same templates are used for float-based (fp32mp2) and, when enabled,
    double-based (fp64mp2) variants.
    
    Supported Operations:
    -------------------------------------------------------------------------
    - Type Conversions:
        * __nv_fpmp2_from_double, __nv_fpmp2_from_int, __nv_fpmp2_from_uint
        * __nv_fpmp2_from_ll, __nv_fpmp2_from_ull
        * __nv_fpmp2_to_double, __nv_fpmp2_to_float
        * __nv_fpmp2_to_int, __nv_fpmp2_to_uint, __nv_fpmp2_to_ll, __nv_fpmp2_to_ull
        * __nv_fpmp2_from_double supports FPMP_OPTIMIZED_DOUBLE_TO_FPMP (integer-only, no FP64)
        * __nv_fpmp2_to_double supports FPMP_OPTIMIZED_FPMP_TO_DOUBLE (integer-only, no FP64)
    
    - Basic Arithmetic:
        * Addition: __nv_fpmp2_add, __nv_fpmp2_low_add, __nv_fpmp2_high_add
        * Subtraction: __nv_fpmp2_sub, __nv_fpmp2_low_sub, __nv_fpmp2_high_sub
        * Accumulate: __nv_fpmp2_acc, __nv_fpmp2_low_acc, __nv_fpmp2_high_acc (optimized single-component add)
        * Multiplication: __nv_fpmp2_mul, __nv_fpmp2_low_mul, __nv_fpmp2_high_mul (if __FPMP_USE_ACCURATE_MUL__ == 1)
        * Division: __nv_fpmp2_div, __nv_fpmp2_low_div, __nv_fpmp2_high_div (if __FPMP_USE_ACCURATE_DIV__ == 1)
        * Negation: __nv_fpmp2_neg
        * Renormalization: __nv_fpmp2_renormalize
    
    - Advanced Operations:
        * Square Root: __nv_fpmp2_sqrt (Newton-Raphson iteration)
        * Reciprocal Square Root: __nv_fpmp2_rsqrt (Karp-Markstein algorithm)
        * Fused Multiply-Add: __nv_fpmp2_fma, __nv_fpmp2_low_fma
        * Multiply-Add with Rounding: __nv_fpmp2_mad
    
    - Comparison Operations:
        * __nv_fpmp2_cmp_eq, __nv_fpmp2_cmp_ne
        * __nv_fpmp2_cmp_lt, __nv_fpmp2_cmp_gt
        * __nv_fpmp2_cmp_le, __nv_fpmp2_cmp_ge
    
    - Utility Operations:
        * __nv_fpmp2_bit_cast : IEEE-754 format bit representation
    
    - Atomic Operations (CUDA device only):
        * __nv_fpmp2_atomicAdd : Atomic addition with CAS loop
        * __nv_fpmp2_atomicSub : Atomic subtraction with CAS loop
    
    Implementation Details:
    -------------------------------------------------------------------------
    - Uses Dekker's error-free transformation algorithms (2Sum, 2Mult)
    - Supports three accuracy levels: low, mid (default), and high
    - Template-based for both float (fp32) and double (fp64) precision
    - Provides both inline implementations and library declarations
    - All operations maintain the (hi, lo) representation invariant
    - __nv_fpmp2_from_double uses integer bit manipulation when FPMP_OPTIMIZED_DOUBLE_TO_FPMP == 1
    
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

#include <cuda/__fp/fpmp_common.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

#if !(defined __FPMP_USE_LIB__)
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
    // __nv_fpmp2_from_double: Convert double → fpmp2 (hi, lo) pair
    // -----------------------------------------------------------------------
    // Splits a 64-bit double into two FpType components such that:
    //   x ≈ hi + lo    (with hi carrying the leading bits, lo the remainder)
    //
    // When FPMP_OPTIMIZED_DOUBLE_TO_FPMP == 1 and FpType == float:
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
    // When FPMP_OPTIMIZED_DOUBLE_TO_FPMP == 0 or FpType != float:
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
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_from_double (const double x, 
                                                         FpType*      res_hi, 
                                                         FpType*      res_lo) 
    {
#if __FPMP_USE_OPT_FROM_DOUBLE__ == 1
        if constexpr (std::is_same<FpType, float>::value)
        {
            uint64_t dbits = fpmp::internal_bit_cast<uint64_t>(x);
            uint32_t sign  = (uint32_t)(dbits >> 63);
            uint32_t d_exp = (uint32_t)((dbits >> 52) & 0x7FFU);
            uint64_t mant  = dbits & 0x000FFFFFFFFFFFFFULL;

            // hi biased exponent in float space: f_exp = (d_exp - 1023) + 127.
            int32_t f_exp = (int32_t)d_exp - 896;

            // Fallback for: zero/denormal double (d_exp == 0), float underflow
            // (f_exp <= 0), and float overflow / Inf / NaN (f_exp >= 255).
            // Defers to the standard cast for these edge cases; lo is flushed.
            if (d_exp == 0 || (f_exp <= 0 || f_exp >= 255)) 
            {
                *res_hi = (float)x;
                *res_lo = 0.0f;
            }
            else 
            {
                // hi mantissa: top 23 explicit bits with round-to-nearest,
                // ties away from zero. (((mant >> 28) + 1) >> 1) takes the
                // top 24 bits of mant, adds 1 at the round position, then
                // drops it. The carry can ripple into the exponent (when
                // hi_round == 0x800000), so we use '+' (not '|') to merge
                // hi_round into the shifted exponent field.
                uint32_t hi_round = (((uint32_t)(mant >> 28)) + 1U) >> 1;
                uint32_t hi_bits  = (sign << 31) | (((uint32_t)f_exp << 23) + hi_round);
                *res_hi = fpmp::internal_bit_cast<float>(hi_bits);

                // Encode the residual as a signed 32-bit integer with the
                // round bit placed at the sign position. Bottom 32 bits of
                // mant are bits [31:0]; shifting left by 3 in 32-bit
                // arithmetic discards bits [31:29] (already absorbed into
                // hi) and places bit 28 (the round bit) at bit 31. When the
                // round bit is 1 (hi was rounded up), rsd is negative in
                // two's complement, exactly representing the signed residual
                // x - hi at mantissa scale * 2^3.
                int32_t rsd = (int32_t)((uint32_t)mant << 3);

                // Convert rsd to float with round-to-nearest-even (default for
                // host int->float and CUDA cvt.rn.f32.s32). Then scale:
                //   * 2^-55  : undoes the << 3 (-3) and the mantissa-position
                //              offset (-52) to recover residual at unit scale.
                //   * scale  : 2^(f_exp - 127) with the sign of x. Both
                //              multiplications are exact (powers of two).
                float scale = fpmp::internal_bit_cast<float>((sign << 31) | ((uint32_t)f_exp << 23));
                *res_lo = (static_cast<float>(rsd) * 0x1p-55f) * scale;

                // Fast2Sum to enforce canonical form fl(hi+lo) == hi.
                // Required because round-to-nearest on r can leave |lo|
                // exactly at ulp(hi)/2; if hi has an odd low mantissa bit,
                // fl(hi+lo) would otherwise round away from hi.
                *res_hi = fpmp::fast_two_sum(*res_hi, *res_lo, res_lo);
            }
        }
        else if constexpr (std::is_same<FpType, double>::value)
        {
            // FpType == double (fp64mp2): the cast-based split below would
            // compute (double)(x - (double)x) == 0.0 and the compiler folds
            // it; spell that out at the source level so the intent is
            // explicit and the lo store is guaranteed not to depend on any
            // FP64 instruction.
            *res_hi = x;
            *res_lo = 0.0;
        }
        else
        {
            // Generic fallback for any future non-float, non-double FpType:
            // cast-based split.
            *res_hi = static_cast<FpType>(x);
            *res_lo = static_cast<FpType>(x - static_cast<double>(*res_hi));
        }
#else // !__FPMP_USE_OPT_FROM_DOUBLE__ == 1
        if constexpr (std::is_same<FpType, double>::value)
        {
            // FpType == double (fp64mp2): trivial split, see comment above.
            *res_hi = x;
            *res_lo = 0.0;
        }
        else
        {
            // Non-optimized path: two FP64 operations (cast + subtract).
            *res_hi = static_cast<FpType>(x);
            *res_lo = static_cast<FpType>(x - static_cast<double>(*res_hi));
        }
#endif // !__FPMP_USE_OPT_FROM_DOUBLE__ == 1
    } // __nv_fpmp2_from_double

    // int -> (hi, lo) conversions
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_from_int (const int32_t i, 
                                                      FpType*       res_hi, 
                                                      FpType*       res_lo)   
    {

        *res_hi = fpmp::int2fp_rz<FpType>(i);
        *res_lo = fpmp::int2fp_rz<FpType>(i - fpmp::fp2int_rz(*res_hi));
    }

    // uint -> (hi, lo) conversions
    // Note: Use signed arithmetic to compute residual, since fpmp::fp2uint_rz(*res_hi)
    // might be larger than i when rounding direction differs
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_from_uint (const uint32_t i, 
                                                       FpType*        res_hi, 
                                                       FpType*        res_lo) 
    {

        *res_hi = fpmp::uint2fp_rz<FpType>(i);
        // Compute residual using signed arithmetic to handle case where hi rounds up
        int32_t residual = static_cast<int32_t>(i) - static_cast<int32_t>(fpmp::fp2uint_rz(*res_hi));
        *res_lo = fpmp::int2fp_rz<FpType>(residual);
    }   

    // ll -> (hi, lo) conversions
    // With ll2fp_rz properly rounding toward zero, hi is always <= i for positive i
    // and >= i for negative i, so fpmp::fp2ll_rz(hi) is always representable as int64_t.
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_from_ll (const int64_t i, 
                                                     FpType*       res_hi, 
                                                     FpType*       res_lo) 
    {

        *res_hi = fpmp::ll2fp_rz<FpType>(i);
        *res_lo = fpmp::ll2fp_rz<FpType>(i - fpmp::fp2ll_rz(*res_hi));
    }

    // ull -> (hi, lo) conversions
    // With ull2fp_rz properly rounding toward zero, hi <= i always,
    // so the residual i - hi is always non-negative.
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_from_ull (const uint64_t i, 
                                                      FpType*        res_hi, 
                                                      FpType*        res_lo) 
    {

        *res_hi = fpmp::ull2fp_rz<FpType>(i);
        // Residual is always non-negative and fits in int64_t (< 2^53 for double)
        uint64_t residual = i - fpmp::fp2ull_rz(*res_hi);
        *res_lo = fpmp::ull2fp_rz<FpType>(residual);
    }

    // (hi, lo) -> double conversions
    //
    // Optimized path (__FPMP_USE_OPT_TO_DOUBLE__ == 1):
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
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  double __nv_fpmp2_to_double (const FpType x_hi_in,
                                                         const FpType x_lo_in)
    { 
#if __FPMP_USE_OPT_TO_DOUBLE__ == 1
        if constexpr (std::is_same<FpType, float>::value)
        {
            // Renormalize input to canonical form. See "Step 0" in the
            // function's docstring for why this is required for the integer
            // path to be safe on non-canonical pairs (e.g. produced by
            // add_fast / long FAST accumulator chains).
            float x_lo;
            float x_hi = fpmp::two_sum(x_hi_in, x_lo_in, &x_lo);

            uint32_t hi_bits = fpmp::internal_bit_cast<uint32_t>(x_hi);
            uint32_t sign_a  = hi_bits >> 31;
            uint32_t fexp_a  = (hi_bits >> 23) & 0xFFU;
            uint32_t fmant_a = hi_bits & 0x7FFFFFU;

            // Cold fallback for hi outside the FP32-scale-representable range:
            // zero/subnormal, very tiny normal (fexp_a in [1, 51]), Inf, NaN.
            if (fexp_a < 52U || fexp_a == 0xFFU)
                return static_cast<double>(x_hi) + static_cast<double>(x_lo);

            // scale = (sign_a ? -1 : +1) * 2^(179 - fexp_a). Always a normal
            // float for fexp_a in [52, 254] (biased scale_exp = 306 - fexp_a
            // in [52, 254]). Sign of hi baked in so the signed r below is
            // already lo's contribution relative to hi.
            float scale = fpmp::internal_bit_cast<float>(
                (sign_a << 31) | ((306U - fexp_a) << 23));

            // r exactly represents lo at hi's mantissa scale (signed). For
            // canonical fp32mp2 (|lo| <= ulp(hi)/2) the multiplication is
            // exact (power-of-two scaling) and |r| <= 2^28.
            int32_t r = fpmp::fp2int_rn(x_lo * scale);

            // M = (hi's 53-bit mantissa with implicit 1, at bit 52)
            //     + (signed lo contribution at the same scale).
            // Range: [2^52 - 2^28, 2^53 - 2^29 + 2^28]. Always positive.
            int64_t M = (int64_t)(((uint64_t)(0x800000U | fmant_a)) << 29)
                      + (int64_t)r;

            // Subtraction can borrow at most one bit (|r| << 2^52), so the
            // implicit-1 lands at bit 52 (no shift) or bit 51 (shift up by 1).
            // Single conditional shift, no __clzll on the critical path.
            uint64_t Mu         = (uint64_t)M;
            uint64_t need_shift = ((Mu >> 52) & 1ULL) ^ 1ULL;
            Mu <<= need_shift;

            return fpmp::internal_bit_cast<double>(
                ((uint64_t)sign_a << 63)
              | ((uint64_t)(fexp_a + 896U - (uint32_t)need_shift) << 52)
              | (Mu & 0x000FFFFFFFFFFFFFULL));
        }
        else
        {
            return static_cast<double>(x_hi_in) + static_cast<double>(x_lo_in);
        }
#else
        return static_cast<double>(x_hi_in) + static_cast<double>(x_lo_in);
#endif
    }

    // (hi, lo) -> float conversions (returns the sum as single FpType)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  FpType __nv_fpmp2_to_float (const FpType x_hi, 
                                                        const FpType x_lo)  
    { 
        return x_hi + x_lo; 
    }

    // (hi, lo) -> int conversions
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  int32_t __nv_fpmp2_to_int (const FpType x_hi, 
                                                       const FpType x_lo)  
    { 

        FpType abs_hi    = fpmp::internal_fabs(x_hi);
        // Check threshold BEFORE computing sum - for large values, addition loses precision
        // 2^24 for float, 2^53 for double
        FpType threshold = std::is_same<FpType, float>::value ? 0x1.0p24f : 
                                                                0x1.0p53;  
        if (abs_hi < threshold)
        {
            // Small value: use round-toward-zero addition
            FpType res = fpmp::add_rz(x_hi, x_lo);
            return fpmp::fp2int_rz(res); 
        }
        else 
        {
            // Large value: use integer addition to preserve exactness
            int32_t hi_int = fpmp::fp2int_rz(x_hi);
            int32_t lo_int = fpmp::fp2int_rz(x_lo);
            return hi_int + lo_int;
        }
    } // __nv_fpmp2_to_int

    // (hi, lo) -> uint conversions
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  uint32_t __nv_fpmp2_to_uint (const FpType x_hi, 
                                                         const FpType x_lo)  
    { 

        // Check threshold BEFORE computing sum
        // 2^24 for float, 2^53 for double
        FpType threshold = std::is_same<FpType, float>::value ? 0x1.0p24f : 
                                                                0x1.0p53;  
        if (x_hi < threshold)
        {
            // Small value: use round-toward-zero addition
            FpType res = fpmp::add_rz(x_hi, x_lo);
            return fpmp::fp2uint_rz(res);
        }
        else 
        {
            // Large value: use integer addition to preserve exactness
            uint32_t hi_uint = fpmp::fp2uint_rz(x_hi);
            int32_t lo_int   = fpmp::fp2int_rz(x_lo);
            return hi_uint + lo_int;
        }
    } // __nv_fpmp2_to_uint

    // (hi, lo) -> ll conversions
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  int64_t __nv_fpmp2_to_ll (const FpType x_hi, 
                                                      const FpType x_lo)  
    { 

        FpType abs_hi    = fpmp::internal_fabs(x_hi);
        // Check threshold BEFORE computing sum
        // 2^24 for float, 2^53 for double
        FpType threshold = std::is_same<FpType, float>::value ? 0x1.0p24f : 
                                                                0x1.0p53;  
        if (abs_hi < threshold)
        {
            // Small value: use round-toward-zero addition
            FpType res = fpmp::add_rz(x_hi, x_lo);
            return fpmp::fp2ll_rz(res);
        }
        else 
        {
            // Large value: use integer addition to preserve exactness
            int64_t hi_ll = fpmp::fp2ll_rz(x_hi);
            int64_t lo_ll = fpmp::fp2ll_rz(x_lo);
            return hi_ll + lo_ll;
        }
    } // __nv_fpmp2_to_ll

    // (hi, lo) -> ull conversions
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  uint64_t __nv_fpmp2_to_ull (const FpType x_hi, 
                                                        const FpType x_lo) 
    { 

        // Check threshold BEFORE computing sum
        // 2^24 for float, 2^53 for double
        FpType threshold = std::is_same<FpType, float>::value ? 0x1.0p24f : 
                                                                0x1.0p53;  
        if (x_hi < threshold)
        {
            // Small value: use round-toward-zero addition
            FpType res = fpmp::add_rz(x_hi, x_lo);
            return fpmp::fp2ull_rz(res);
        }
        else 
        {
            // Large value: use integer addition to preserve exactness
            uint64_t hi_ull = fpmp::fp2ull_rz(x_hi);
            int64_t lo_ll   = fpmp::fp2ll_rz(x_lo);
            return hi_ull + lo_ll;
        }
    } // __nv_fpmp2_to_ull

    /*
    * --------------------------------------------------------------------
    * Re-normalization operations
    * --------------------------------------------------------------------
    */
    // Renormalize a multi-precision (double-float) number
    // to ensure that the hi and lo parts are non-overlapping
    // This is useful for fast mode to ensure that the result is accurate
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_renormalize (const FpType x_hi, 
                                                        const FpType x_lo, 
                                                        FpType*      res_hi, 
                                                        FpType*      res_lo)
    {

        *res_hi = fpmp::fast_two_sum(x_hi, x_lo, res_lo);
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
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_low_add (const FpType x_hi, 
                                                     const FpType x_lo, 
                                                     const FpType y_hi, 
                                                     const FpType y_lo, 
                                                     FpType*      res_hi, 
                                                     FpType*      res_lo)
    {

        FpType r_hi, r_lo;

        // Add high parts using general 2-Sum (no magnitude assumption)
        r_hi = fpmp::two_sum(x_hi, y_hi, &r_lo);
        // Add low parts
        r_lo = fpmp::add_rn(fpmp::add_rn(x_lo, y_lo), r_lo);

        *res_hi = r_hi;
        *res_lo = r_lo;
    } // __nv_fpmp2_low_add

    /*
    * Dekker addition operation
    * This is classic split and error accumulation addition operation with normalization.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_add (const FpType x_hi, 
                                                const FpType x_lo, 
                                                const FpType y_hi, 
                                                const FpType y_lo, 
                                                FpType*      res_hi, 
                                                FpType*      res_lo)
    {


        FpType r_lo_refine;
        FpType r_hi, r_lo;

        // Add high parts using general 2-Sum (no magnitude assumption)
        r_hi        = fpmp::two_sum(x_hi, y_hi, &r_lo);
        // Add low parts
        r_lo_refine = fpmp::add_rn(fpmp::add_rn(x_lo, y_lo), r_lo);
        // Normalize:
        *res_hi     = fpmp::fast_two_sum(r_hi, r_lo_refine, res_lo);
    } // __nv_fpmp2_add
    
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
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_nv_fpmp2_add_fpan (const FpType a_hi, 
                                                     const FpType a_lo, 
                                                     const FpType b_hi, 
                                                     const FpType b_lo, 
                                                     FpType*      res_hi, 
                                                     FpType*      res_lo)
    {

        
        // Level 1: Two independent 2Sums - can execute in parallel
        // Inline two_sum for a_hi + b_hi to help compiler see independence
        FpType s_h   = fpmp::add_rn(a_hi, b_hi);
        FpType s_a   = fpmp::sub_rn(s_h, b_hi);
        FpType s_b   = fpmp::sub_rn(s_h, s_a);
        
        // Inline two_sum for a_lo + b_lo (parallel with above)
        FpType t_h   = fpmp::add_rn(a_lo, b_lo);
        FpType t_a   = fpmp::sub_rn(t_h, b_lo);
        FpType t_b   = fpmp::sub_rn(t_h, t_a);
        
        // Complete the error calculations (can interleave)
        FpType s_da  = fpmp::sub_rn(a_hi, s_a);
        FpType s_db  = fpmp::sub_rn(b_hi, s_b);
        FpType s_l   = fpmp::add_rn(s_da, s_db);
        
        FpType t_da  = fpmp::sub_rn(a_lo, t_a);
        FpType t_db  = fpmp::sub_rn(b_lo, t_b);
        FpType t_l   = fpmp::add_rn(t_da, t_db);
        
        // Level 2: Merge middle terms
        FpType c     = fpmp::add_rn(s_l, t_h);
        
        // Level 3: First normalization (Fast2Sum since |s_h| >= |c| typically)
        FpType v_h   = fpmp::add_rn(s_h, c);
        FpType v_tmp = fpmp::sub_rn(v_h, s_h);
        FpType v_l   = fpmp::sub_rn(c, v_tmp);
        
        // Level 4: Absorb remaining error
        FpType w     = fpmp::add_rn(t_l, v_l);
        
        // Level 5: Final normalization
        *res_hi      = fpmp::add_rn(v_h, w);
        FpType r_tmp = fpmp::sub_rn(*res_hi, v_h);
        *res_lo      = fpmp::sub_rn(w, r_tmp);
    } // __nv_fpmp2_add_fpan
    
    /*
    * Thall addition operation via expansion series
    * This implementation is based on: Andrew Thall, Extended-Precision
    * Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
    * from http://andrewthall.org/papers/df64_qf128.pdf.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_nv_fpmp2_add_exp (const FpType a_hi, 
                                                             const FpType a_lo, 
                                                             const FpType b_hi, 
                                                             const FpType b_lo, 
                                                             FpType*      res_hi, 
                                                             FpType*      res_lo)
    {

        FpType t1, t2, t3, t4, t5, e;
        t1 = fpmp::add_rn (a_hi, b_hi);
        t2 = fpmp::sub_rn (t1, a_hi);
        t3 = fpmp::add_rn (fpmp::add_rn (a_hi, fpmp::sub_rn(t2,t1)), fpmp::sub_rn (b_hi, t2));
        t4 = fpmp::add_rn (a_lo, b_lo);
        t2 = fpmp::sub_rn (t4, a_lo);
        t5 = fpmp::add_rn (fpmp::add_rn (a_lo, fpmp::sub_rn(t2,t4)), fpmp::sub_rn (b_lo, t2));
        t3 = fpmp::add_rn (t3, t4);
        t4 = fpmp::add_rn (t1, t3);
        t3 = fpmp::add_rn (fpmp::sub_rn(t1,t4), t3);
        t3 = fpmp::add_rn (t3, t5);
        e  = fpmp::add_rn (t4, t3);

        *res_lo = fpmp::add_rn (fpmp::sub_rn(t4, e), t3);
        *res_hi = e;
    } // __nv_fpmp2_high_add

#define __FPMP_FPAN_METHOD__

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_high_add (const FpType a_hi, 
                                                         const FpType a_lo, 
                                                         const FpType b_hi, 
                                                         const FpType b_lo, 
                                                         FpType*      res_hi, 
                                                         FpType*      res_lo)
    {
#if defined   __FPMP_FPAN_METHOD__
        __internal_nv_fpmp2_add_fpan (a_hi, a_lo, b_hi, b_lo, res_hi, res_lo);
#else
        __internal_nv_fpmp2_add_exp  (a_hi, a_lo, b_hi, b_lo, res_hi, res_lo);
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
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_low_sub (const FpType x_hi, 
                                                     const FpType x_lo, 
                                                     const FpType y_hi, 
                                                     const FpType y_lo, 
                                                     FpType*      res_hi, 
                                                     FpType*      res_lo)
    {
        __nv_fpmp2_low_add(x_hi, x_lo, -y_hi, -y_lo, res_hi, res_lo);
    }
    /*
    * Classic split and error accumulation subtraction operation
    * This is a Dekker subtraction operation with normalization.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_sub (const FpType x_hi, 
                                                const FpType x_lo, 
                                                const FpType y_hi, 
                                                const FpType y_lo, 
                                                FpType*      res_hi, 
                                                FpType*      res_lo)
    {
        __nv_fpmp2_add(x_hi, x_lo, -y_hi, -y_lo, res_hi, res_lo);
    }
    /*
    * Thall accurate subtraction operation
    * This implementation is based on: Andrew Thall, Extended-Precision
    * Floating-Point Numbers for GPU Computation. Retrieved on 7/12/2011
    * from http://andrewthall.org/papers/df64_qf128.pdf.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_high_sub(const FpType x_hi, 
                                                         const FpType x_lo, 
                                                         const FpType y_hi, 
                                                         const FpType y_lo, 
                                                         FpType*      res_hi, 
                                                         FpType*      res_lo)
    {
        __nv_fpmp2_high_add(x_hi, x_lo, -y_hi, -y_lo, res_hi, res_lo);
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
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_low_acc (const FpType c, 
                                                     FpType*      acc_hi, 
                                                     FpType*      acc_lo)
    {

        FpType err;
        // Add c to high part with error capture
        FpType new_hi = fpmp::two_sum(*acc_hi, c, &err);
        // Accumulate error into low part (no normalization)
        *acc_hi = new_hi;
        *acc_lo = fpmp::add_rn(*acc_lo, err);
    }
    
    /*
    * Default accumulate: Dekker-style with normalization
    * Result is properly normalized (non-overlapping hi/lo).
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_acc (const FpType c, 
                                                FpType*      acc_hi, 
                                                FpType*      acc_lo)
    {

        FpType err;
        // Add c to high part with error capture
        FpType new_hi = fpmp::two_sum(*acc_hi, c, &err);
        // Combine error with existing low part
        FpType new_lo = fpmp::add_rn(*acc_lo, err);
        // Normalize result
        *acc_hi = fpmp::fast_two_sum(new_hi, new_lo, acc_lo);
    }
    
    /*
    * Accurate accumulate: Full error propagation (FPAN-style)
    * Provides maximum precision by properly ordering all error terms.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_high_acc (const FpType c, 
                                                         FpType*      acc_hi, 
                                                         FpType*      acc_lo)
    {

        FpType err;
        // Add c to high part with error capture
        FpType s_hi = fpmp::two_sum(*acc_hi, c, &err);
        // Add error to low part
        FpType t    = fpmp::add_rn(*acc_lo, err);
        // First normalization
        FpType v_hi = fpmp::add_rn(s_hi, t);
        FpType v_tmp = fpmp::sub_rn(v_hi, s_hi);
        FpType v_lo = fpmp::sub_rn(t, v_tmp);
        // Final normalization
        *acc_hi = fpmp::add_rn(v_hi, v_lo);
        FpType r_tmp = fpmp::sub_rn(*acc_hi, v_hi);
        *acc_lo = fpmp::sub_rn(v_lo, r_tmp);
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
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_low_mul (const FpType x_hi, 
                                                      const FpType x_lo, 
                                                      const FpType y_hi, 
                                                      const FpType y_lo, 
                                                      FpType*      res_hi, 
                                                      FpType*      res_lo)  
    { 

        FpType t_hi = fpmp::mul_rn (x_hi, y_hi);
        FpType t_lo = fpmp::fma_rn (x_hi, y_hi, -t_hi);
        t_lo        = fpmp::fma_rn (x_lo, y_lo, t_lo);
        t_lo        = fpmp::fma_rn (x_hi, y_lo, t_lo); 
        t_lo        = fpmp::fma_rn (x_lo, y_hi, t_lo);

        *res_hi = t_hi;
        *res_lo = t_lo;
    } // __nv_fpmp2_low_mul

    /*
    * Dekker multiplication operation
    * This is a Dekker multiplication operation with normalization.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_mul (const FpType x_hi, 
                                                const FpType x_lo, 
                                                const FpType y_hi, 
                                                const FpType y_lo, 
                                                FpType*      res_hi, 
                                                FpType*      res_lo)
    {

        FpType p1, p2, c_hi, c_lo, res_hi_tmp, res_lo_tmp;
        c_hi = fpmp::two_mult_fma(x_hi, y_hi, &c_lo);
        p1   = fpmp::mul_rn(x_hi, y_lo);
        p2   = fpmp::mul_rn(x_lo, y_hi);
        c_lo = fpmp::add_rn(c_lo, fpmp::add_rn(p1, p2));
        // Normalize:
        res_hi_tmp = fpmp::fast_two_sum(c_hi, c_lo, &res_lo_tmp);

        *res_hi = res_hi_tmp;
        *res_lo = res_lo_tmp;
    } // __nv_fpmp2_mul

#if __FPMP_USE_ACCURATE_MUL__ == 1
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
    *   - Overhead vs __nv_fpmp2_mul: ~6 integer ops + 4 MUL (often identity) + 1 fast_two_sum.
    *
    * REFERENCE:
    *   Dekker, T. (1971). A floating-point technique for extending available precision.
    *   Conditional scaling adapted from QD library techniques.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_high_mul (const FpType x_hi, 
                                                          const FpType x_lo, 
                                                          const FpType y_hi, 
                                                          const FpType y_lo, 
                                                          FpType*      res_hi, 
                                                          FpType*      res_lo)
    {

        
        // Type-specific constants for conditional scaling
        using UintType = typename std::conditional<std::is_same<FpType, float>::value, uint32_t, uint64_t>::type;
        
        constexpr int exp_bits   = std::is_same<FpType, float>::value ? 8 : 11;
        constexpr int mant_bits  = std::is_same<FpType, float>::value ? 23 : 52;
        constexpr int exp_bias   = std::is_same<FpType, float>::value ? 127 : 1023;
        constexpr UintType exp_mask = ((UintType(1) << exp_bits) - 1) << mant_bits;
        
        // Threshold: if combined exponent < this, we need scaling
        // For float: scale_shift=64, threshold=190 (2*127-64)
        // For double: scale_shift=512, threshold=1534 (2*1023-512)
        constexpr int scale_shift = std::is_same<FpType, float>::value ? 64 : 512;
        constexpr int exp_threshold = 2 * exp_bias - scale_shift;
        
        // Scale factors
        constexpr FpType scale_up   = std::is_same<FpType, float>::value ? FpType(0x1.0p64f)  : FpType(0x1.0p512);
        constexpr FpType scale_down = std::is_same<FpType, float>::value ? FpType(0x1.0p-64f) : FpType(0x1.0p-512);
        
        // Extract exponents and compute conditional scale (branch-free)
        UintType x_bits = fpmp::internal_bit_cast<UintType>(x_hi);
        UintType y_bits = fpmp::internal_bit_cast<UintType>(y_hi);
        int x_exp = static_cast<int>((x_bits & exp_mask) >> mant_bits);
        int y_exp = static_cast<int>((y_bits & exp_mask) >> mant_bits);
        int result_exp = x_exp + y_exp;
        
        // Create mask: -1 (all 1s) if needs scaling, 0 otherwise
        int needs_scale = (result_exp - exp_threshold) >> 31;
        
        // Select scale factor using bit manipulation (branch-free)
        UintType scale_up_bits   = fpmp::internal_bit_cast<UintType>(scale_up);
        UintType one_bits        = fpmp::internal_bit_cast<UintType>(FpType(1.0));
        UintType scale_bits      = (scale_up_bits & UintType(needs_scale)) | (one_bits & UintType(~needs_scale));
        FpType scale             = fpmp::internal_bit_cast<FpType>(scale_bits);
        
        UintType scale_down_bits = fpmp::internal_bit_cast<UintType>(scale_down);
        UintType inv_scale_bits  = (scale_down_bits & UintType(needs_scale)) | (one_bits & UintType(~needs_scale));
        FpType inv_scale         = fpmp::internal_bit_cast<FpType>(inv_scale_bits);
        
        // Scale first operand
        FpType a_hi = fpmp::mul_rn(x_hi, scale);
        FpType a_lo = fpmp::mul_rn(x_lo, scale);
        
        // Standard Dekker multiplication
        FpType c_lo;
        FpType c_hi = fpmp::two_mult_fma(a_hi, y_hi, &c_lo);
        FpType p1   = fpmp::mul_rn(a_hi, y_lo);
        FpType p2   = fpmp::mul_rn(a_lo, y_hi);
        c_lo        = fpmp::add_rn(c_lo, fpmp::add_rn(p1, p2));
        
        // Normalize
        FpType r_lo;
        FpType r_hi = fpmp::fast_two_sum(c_hi, c_lo, &r_lo);
        
        // Scale back
        r_hi = fpmp::mul_rn(r_hi, inv_scale);
        r_lo = fpmp::mul_rn(r_lo, inv_scale);
        
        // Final normalization to ensure (hi, lo) invariant after scaling
        *res_hi = fpmp::fast_two_sum(r_hi, r_lo, res_lo);
    } // __nv_fpmp2_high_mul
#endif // __FPMP_USE_ACCURATE_MUL__ == 1

    /*
    * --------------------------------------------------------------------
    * Division operations
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_low_div (const FpType a_hi, 
                                                     const FpType a_lo, 
                                                     const FpType b_hi, 
                                                     const FpType b_lo, 
                                                     FpType*      res_hi, 
                                                     FpType*      res_lo)
    {

        // Get an estimate from *this->hi:
        FpType recip_hi = fpmp::rcp_rn(b_hi);

        // Do a Newton-Rhapson iteration:
        // This line can break for some uninvestigated reason,
        // Use the one below:
        //recip_hi = recip_hi*(2.0 - (x.get_hi())*recip_hi);
        FpType two = static_cast<FpType>(2.0);
        recip_hi = fpmp::fma_rn(-b_hi*recip_hi, recip_hi, two*recip_hi);

        FpType recip2_hi = recip_hi*recip_hi;
        FpType recip2_lo = fpmp::fma_rn(recip_hi, recip_hi, -recip2_hi);

        // recip^2 * this->(hi/lo), Dekker multiplication:
        FpType mul_hi = recip2_hi*(b_hi);
        FpType mul_lo = fpmp::fma_rn(recip2_hi, (b_hi), -mul_hi);
        mul_lo       += (recip2_hi*(b_lo) + recip2_lo*(b_hi));

        // Our answer is now 2*recip_hi + mul_hi + mul_lo
        FpType final_recip_hi = two*recip_hi - mul_hi;
        FpType final_recip_lo = two*recip_hi - fpmp::add_rn(final_recip_hi, mul_hi);
        final_recip_lo       -= mul_lo;

        // Multiply the reciprocal by the numerator
        __nv_fpmp2_low_mul(a_hi, a_lo, final_recip_hi, final_recip_lo, res_hi, res_lo);
    } // __nv_fpmp2_low_div


    /* Compute high-accuracy quotient, using Newton-
    Raphson iteration. Derived from: T. Nagai, H. Yoshida, H. Kuroda, Y. Kanada.
    Fast Quadruple Precision Arithmetic Library on Parallel Computer SR11000/J2.
    In Proceedings of the 8th International Conference on Computational Science,
    ICCS '08, Part I, pp. 446-455.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_div (const FpType a_hi, 
                                                const FpType a_lo, 
                                                const FpType b_hi, 
                                                const FpType b_lo, 
                                                FpType*      res_hi, 
                                                FpType*      res_lo)
    {

        FpType t_hi, t_lo;
        FpType e, r;
        r          = fpmp::rcp_rn(b_hi);
        t_hi       = fpmp::mul_rn (a_hi, r);
        e          = fpmp::fma_rn (b_hi, -t_hi, a_hi);
        t_hi       = fpmp::fma_rn (r, e, t_hi);
        t_lo       = fpmp::fma_rn (b_hi, -t_hi, a_hi);
        t_lo       = fpmp::add_rn (a_lo, t_lo);
        t_lo       = fpmp::fma_rn (b_lo, -t_hi, t_lo);
        e          = fpmp::mul_rn (r, t_lo);
        t_lo       = fpmp::fma_rn (b_hi, -e, t_lo);
        t_lo       = fpmp::fma_rn (r, t_lo, e);
        e          = fpmp::add_rn (t_hi, t_lo);

        *res_lo    = fpmp::add_rn (t_hi - e, t_lo);
        *res_hi    = e;
    } // __nv_fpmp2_div

#if __FPMP_USE_ACCURATE_DIV__ == 1
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
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_high_div (const FpType a_hi, 
                                                         const FpType a_lo, 
                                                         const FpType b_hi, 
                                                         const FpType b_lo, 
                                                         FpType*      res_hi, 
                                                         FpType*      res_lo)
    {

        
        // Type-specific constants for conditional scaling
        using UintType = typename std::conditional<std::is_same<FpType, float>::value, uint32_t, uint64_t>::type;
        
        constexpr int exp_bits   = std::is_same<FpType, float>::value ? 8 : 11;
        constexpr int mant_bits  = std::is_same<FpType, float>::value ? 23 : 52;
        constexpr UintType exp_mask = ((UintType(1) << exp_bits) - 1) << mant_bits;
        
        // Threshold for scaling: exponent < threshold means we should scale up
        // For float: if exp < 32 (value < 2^-95), scale up by 2^64
        // For double: if exp < 64 (value < 2^-959), scale up by 2^512
        constexpr int exp_threshold_low = std::is_same<FpType, float>::value ? 32 : 64;
        
        // Scale factors
        constexpr FpType scale_up   = std::is_same<FpType, float>::value ? FpType(0x1.0p64f)  : FpType(0x1.0p512);
        constexpr FpType scale_down = std::is_same<FpType, float>::value ? FpType(0x1.0p-64f) : FpType(0x1.0p-512);
        
        // Extract exponents
        UintType a_bits = fpmp::internal_bit_cast<UintType>(a_hi);
        UintType b_bits = fpmp::internal_bit_cast<UintType>(b_hi);
        int a_exp = static_cast<int>((a_bits & exp_mask) >> mant_bits);
        int b_exp = static_cast<int>((b_bits & exp_mask) >> mant_bits);
        
        // Branch-free: create mask for whether 'a' needs scaling
        // needs_scale_a = -1 if a_exp < exp_threshold_low, 0 otherwise
        int needs_scale_a = (a_exp - exp_threshold_low) >> 31;
        
        // Branch-free: create mask for whether 'b' needs scaling
        // needs_scale_b = -1 if b_exp < exp_threshold_low, 0 otherwise
        int needs_scale_b = (b_exp - exp_threshold_low) >> 31;
        
        // Select scale factors for 'a' (branch-free)
        UintType scale_up_bits = fpmp::internal_bit_cast<UintType>(scale_up);
        UintType one_bits      = fpmp::internal_bit_cast<UintType>(FpType(1.0));
        UintType scale_a_bits  = (scale_up_bits & UintType(needs_scale_a)) | (one_bits & UintType(~needs_scale_a));
        FpType scale_a         = fpmp::internal_bit_cast<FpType>(scale_a_bits);
        
        // Select scale factors for 'b' (branch-free)
        UintType scale_b_bits  = (scale_up_bits & UintType(needs_scale_b)) | (one_bits & UintType(~needs_scale_b));
        FpType scale_b         = fpmp::internal_bit_cast<FpType>(scale_b_bits);
        
        // Scale operands
        FpType sa_hi = fpmp::mul_rn(a_hi, scale_a);
        FpType sa_lo = fpmp::mul_rn(a_lo, scale_a);
        FpType sb_hi = fpmp::mul_rn(b_hi, scale_b);
        FpType sb_lo = fpmp::mul_rn(b_lo, scale_b);
        
        // Perform division on scaled operands using Nagai et al. algorithm
        FpType t_hi, t_lo;
        FpType e, r;
        r          = fpmp::rcp_rn(sb_hi);
        t_hi       = fpmp::mul_rn (sa_hi, r);
        e          = fpmp::fma_rn (sb_hi, -t_hi, sa_hi);
        t_hi       = fpmp::fma_rn (r, e, t_hi);
        t_lo       = fpmp::fma_rn (sb_hi, -t_hi, sa_hi);
        t_lo       = fpmp::add_rn (sa_lo, t_lo);
        t_lo       = fpmp::fma_rn (sb_lo, -t_hi, t_lo);
        e          = fpmp::mul_rn (r, t_lo);
        t_lo       = fpmp::fma_rn (sb_hi, -e, t_lo);
        t_lo       = fpmp::fma_rn (r, t_lo, e);
        e          = fpmp::add_rn (t_hi, t_lo);
        
        FpType r_hi = e;
        FpType r_lo = fpmp::add_rn (t_hi - e, t_lo);
        
        // Compute result scale factor: inv_scale = scale_b / scale_a
        // If a was scaled up, result should be scaled down
        // If b was scaled up, result should be scaled up (since we divided by larger b)
        UintType scale_down_bits = fpmp::internal_bit_cast<UintType>(scale_down);
        
        // For 'a' scaling: if we scaled a up, scale result down
        UintType inv_scale_a_bits = (scale_down_bits & UintType(needs_scale_a)) | (one_bits & UintType(~needs_scale_a));
        FpType inv_scale_a        = fpmp::internal_bit_cast<FpType>(inv_scale_a_bits);
        
        // For 'b' scaling: if we scaled b up, scale result up (compensate)
        UintType comp_scale_b_bits = (scale_up_bits & UintType(needs_scale_b)) | (one_bits & UintType(~needs_scale_b));
        FpType comp_scale_b        = fpmp::internal_bit_cast<FpType>(comp_scale_b_bits);
        
        // Combined scale factor
        FpType final_scale = fpmp::mul_rn(inv_scale_a, comp_scale_b);
        
        // Scale result back
        r_hi = fpmp::mul_rn(r_hi, final_scale);
        r_lo = fpmp::mul_rn(r_lo, final_scale);
        
        // Final normalization to ensure (hi, lo) invariant after scaling
        *res_hi = fpmp::fast_two_sum(r_hi, r_lo, res_lo);
    } // __nv_fpmp2_high_div
#endif // __FPMP_USE_ACCURATE_DIV__ == 1

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
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_rsqrt (const FpType a_hi, 
                                                   const FpType a_lo, 
                                                   FpType*      res_hi, 
                                                   FpType*      res_lo)
    {

        FpType z_hi, z_lo;
        FpType r, s, e;
        FpType one = static_cast<FpType>(1.0);
        FpType half = static_cast<FpType>(0.5);
        r    = fpmp::rsqrt_rn(a_hi);
        e    = fpmp::mul_rn (a_hi, r);
        s    = fpmp::fma_rn (e, -r, one);
        e    = fpmp::fma_rn (a_hi, r, -e);
        s    = fpmp::fma_rn (e, -r, s);
        e    = fpmp::mul_rn (a_lo, r);
        s    = fpmp::fma_rn (e, -r, s);
        e    = fpmp::mul_rn (half, r);
        z_hi = fpmp::mul_rn (e, s);
        z_lo = fpmp::fma_rn (e, s, -z_hi);
        s    = fpmp::add_rn (r, z_hi);
        r    = fpmp::add_rn (r, -s);
        r    = fpmp::add_rn (r, z_hi);
        r    = fpmp::add_rn (r, z_lo);
        e    = fpmp::add_rn (s, r);
        z_lo = fpmp::add_rn (s - e, r);
        z_hi = e;

        *res_hi = z_hi;
        *res_lo = z_lo;
    } // __nv_fpmp2_rsqrt

    /* Compute high-accuracy square root. Newton-Raphson
    iteration based on equation 4 from a paper by Alan Karp and Peter Markstein,
    High Precision Division and Square Root, ACM TOMS, vol. 23, no. 4, December
    1997, pp. 561-589.
    */    
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_sqrt (const FpType a_hi, 
                                                  const FpType a_lo, 
                                                  FpType*      res_hi, 
                                                  FpType*      res_lo)
    { 

        FpType t_hi, t_lo, tmp_lo;
        FpType e, y, s, r;
        FpType zero = static_cast<FpType>(0.0);
        FpType half = static_cast<FpType>(0.5);
        r = fpmp::rsqrt_rn(a_hi);
        if (a_hi == zero) r = zero;
        y           = fpmp::mul_rn (a_hi, r);
        s           = fpmp::fma_rn (y, -y, a_hi);
        r           = fpmp::mul_rn (half, r);
        e           = fpmp::add_rn (s, a_lo);
        tmp_lo      = fpmp::add_rn (s - e, a_lo);
        t_hi        = fpmp::mul_rn (r, e);
        t_lo        = fpmp::fma_rn (r, e, -t_hi);
        t_lo        = fpmp::fma_rn (r, tmp_lo, t_lo);
        r           = fpmp::add_rn (y, t_hi);
        s           = fpmp::add_rn (y - r, t_hi);
        s           = fpmp::add_rn (s, t_lo);
        e           = fpmp::add_rn (r, s);

        *res_lo    = fpmp::add_rn (r - e, s);
        *res_hi    = e;
    } // __nv_fpmp2_sqrt

    /* Compute fast fused multiply-add: x*y+z  (16 ops, no normalization)
        Uses hardware FMA for the main term (single rounding), then recovers
        the exact error via the Boldo-Muller EFT:
          x_hi*y_hi = p + q  (exact, via two_mult_fma)
          p + z_hi  = s + t  (exact, via two_sum)
          => error  = (s - r_hi) + t + q
        where (s - r_hi) is exact by the Boldo-Muller theorem.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_low_fma (const FpType x_hi, 
                                                      const FpType x_lo, 
                                                      const FpType y_hi, 
                                                      const FpType y_lo, 
                                                      const FpType z_hi, 
                                                      const FpType z_lo, 
                                                      FpType*      res_hi, 
                                                      FpType*      res_lo)
    { 

        
        FpType r_hi = fpmp::fma_rn(x_hi, y_hi, z_hi);
        
        FpType q;
        FpType p = fpmp::two_mult_fma(x_hi, y_hi, &q);
        FpType t;
        FpType s = fpmp::two_sum(p, z_hi, &t);
        FpType r_lo = fpmp::add_rn(fpmp::sub_rn(s, r_hi), fpmp::add_rn(t, q));
        
        r_lo = fpmp::fma_rn(x_hi, y_lo, r_lo);
        r_lo = fpmp::fma_rn(x_lo, y_hi, r_lo);
        r_lo = fpmp::fma_rn(x_lo, y_lo, r_lo);
        r_lo = fpmp::add_rn(r_lo, z_lo);

        *res_hi = r_hi;
        *res_lo = r_lo;
    } // __nv_fpmp2_low_fma

    /* Compute high-accuracy fused multiply-add: x*y+z
        Uses hardware FMA for the main term (single rounding), then recovers
        the exact error via the Boldo-Muller EFT:
          x_hi*y_hi = p + q  (exact, via two_mult_fma)
          p + z_hi  = s + t  (exact, via two_sum)
          => error  = (s - r_hi) + t + q
        where (s - r_hi) is exact by the Boldo-Muller theorem.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_fma (const FpType x_hi, 
                                                 const FpType x_lo, 
                                                 const FpType y_hi, 
                                                 const FpType y_lo, 
                                                 const FpType z_hi, 
                                                 const FpType z_lo, 
                                                 FpType*      res_hi, 
                                                 FpType*      res_lo)
    { 

        
        // Hardware FMA: x_hi*y_hi + z_hi with single rounding (optimal)
        FpType r_hi = fpmp::fma_rn(x_hi, y_hi, z_hi);
        
        // Exact error recovery for the main FMA
        FpType q;
        FpType p = fpmp::two_mult_fma(x_hi, y_hi, &q);
        FpType t;
        FpType s = fpmp::two_sum(p, z_hi, &t);
        FpType r_lo = fpmp::add_rn(fpmp::sub_rn(s, r_hi), fpmp::add_rn(t, q));
        
        // Cross terms and remaining contributions
        r_lo = fpmp::fma_rn(x_hi, y_lo, r_lo);
        r_lo = fpmp::fma_rn(x_lo, y_hi, r_lo);
        r_lo = fpmp::fma_rn(x_lo, y_lo, r_lo);
        r_lo = fpmp::add_rn(r_lo, z_lo);
        
        // Normalize
        *res_hi = fpmp::fast_two_sum(r_hi, r_lo, res_lo);
    } // __nv_fpmp2_fma

    /* Compute accurate fused multiply-add: x*y+z
        Same EFT-based main term as __nv_fpmp2_fma, but cross terms are
        computed exactly via two_mult_fma and accumulated with two_sum
        error tracking. This avoids precision loss when cross terms are
        of similar magnitude to r_lo (e.g. catastrophic cancellation in
        the main term).
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_high_fma (const FpType x_hi, 
                                                          const FpType x_lo, 
                                                          const FpType y_hi, 
                                                          const FpType y_lo, 
                                                          const FpType z_hi, 
                                                          const FpType z_lo, 
                                                          FpType*      res_hi, 
                                                          FpType*      res_lo)
    { 
        FpType r_hi = fpmp::fma_rn(x_hi, y_hi, z_hi);
        
        FpType q;
        FpType p = fpmp::two_mult_fma(x_hi, y_hi, &q);
        FpType t;
        FpType s = fpmp::two_sum(p, z_hi, &t);
        FpType r_lo = fpmp::add_rn(fpmp::sub_rn(s, r_hi), fpmp::add_rn(t, q));
        
        FpType c1_lo;
        FpType c1_hi = fpmp::two_mult_fma(x_hi, y_lo, &c1_lo);
        
        FpType c2_lo;
        FpType c2_hi = fpmp::two_mult_fma(x_lo, y_hi, &c2_lo);
        
        FpType cross_err;
        FpType cross = fpmp::two_sum(c1_hi, c2_hi, &cross_err);
        
        FpType acc_err;
        r_lo = fpmp::two_sum(r_lo, cross, &acc_err);
        
        FpType residual = fpmp::add_rn(acc_err, fpmp::add_rn(cross_err, fpmp::add_rn(c1_lo, c2_lo)));
        residual = fpmp::fma_rn(x_lo, y_lo, residual);
        residual = fpmp::add_rn(residual, z_lo);
        
        r_lo = fpmp::add_rn(r_lo, residual);
        
        *res_hi = fpmp::fast_two_sum(r_hi, r_lo, res_lo);
    } // __nv_fpmp2_high_fma

    /*
    * --------------------------------------------------------------------
    * Fused multiply-add with rounding operations
    * --------------------------------------------------------------------
    */
    // multiply-add with rounding (default: fast mul + default add)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_mad (const FpType x_hi, 
                                                 const FpType x_lo, 
                                                 const FpType y_hi, 
                                                 const FpType y_lo, 
                                                 const FpType z_hi, 
                                                 const FpType z_lo, 
                                                 FpType*      res_hi, 
                                                 FpType*      res_lo)
    { 
        FpType t_hi, t_lo;
        __nv_fpmp2_low_mul(x_hi, x_lo, y_hi, y_lo, &t_hi, &t_lo);
        __nv_fpmp2_add(t_hi, t_lo, z_hi, z_lo, res_hi, res_lo);
    }

    // multiply-add fast (fast mul + fast add)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_low_mad (const FpType x_hi, 
                                                      const FpType x_lo, 
                                                      const FpType y_hi, 
                                                      const FpType y_lo, 
                                                      const FpType z_hi, 
                                                      const FpType z_lo, 
                                                      FpType*      res_hi, 
                                                      FpType*      res_lo)
    { 
        FpType t_hi, t_lo;
        __nv_fpmp2_low_mul(x_hi, x_lo, y_hi, y_lo, &t_hi, &t_lo);
        __nv_fpmp2_low_add(t_hi, t_lo, z_hi, z_lo, res_hi, res_lo);
    }

    // multiply-add accurate (default mul + accurate add)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_high_mad (const FpType x_hi, 
                                                          const FpType x_lo, 
                                                          const FpType y_hi, 
                                                          const FpType y_lo, 
                                                          const FpType z_hi, 
                                                          const FpType z_lo, 
                                                          FpType*      res_hi, 
                                                          FpType*      res_lo)
    { 
        FpType t_hi, t_lo;
        __nv_fpmp2_mul(x_hi, x_lo, y_hi, y_lo, &t_hi, &t_lo);
        __nv_fpmp2_high_add(t_hi, t_lo, z_hi, z_lo, res_hi, res_lo);
    }

    /*
    * --------------------------------------------------------------------
    * Negation operations
    * --------------------------------------------------------------------
    */
    // negation
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  void __nv_fpmp2_neg (const FpType x_hi, 
                                                 const FpType x_lo, 
                                                 FpType*      res_hi, 
                                                 FpType*      res_lo)
    { 
        *res_hi = -x_hi;
        *res_lo = -x_lo;
    }

    /*
    * --------------------------------------------------------------------
    * Comparison operations
    * --------------------------------------------------------------------
    */
    // == comparison
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  bool __nv_fpmp2_cmp_eq (const FpType x_hi, 
                                                    const FpType x_lo, 
                                                    const FpType y_hi, 
                                                    const FpType y_lo) 
    { 
        return x_hi == y_hi && x_lo == y_lo;
    }

    // != comparison
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  bool __nv_fpmp2_cmp_ne(const FpType x_hi, 
                                                   const FpType x_lo, 
                                                   const FpType y_hi, 
                                                   const FpType y_lo) 
    { 
        return x_hi != y_hi || x_lo != y_lo;
    }

    // < comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  bool __nv_fpmp2_cmp_lt (const FpType x_hi, 
                                                    const FpType x_lo, 
                                                    const FpType y_hi, 
                                                    const FpType y_lo) 
    { 
        return x_hi < y_hi || (x_hi == y_hi && x_lo < y_lo);
    }

    // > comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  bool __nv_fpmp2_cmp_gt (const FpType x_hi, 
                                                    const FpType x_lo, 
                                                    const FpType y_hi, 
                                                    const FpType y_lo) 
    { 
        return x_hi > y_hi || (x_hi == y_hi && x_lo > y_lo);
    }

    // <= comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  bool __nv_fpmp2_cmp_le (const FpType x_hi, 
                                                    const FpType x_lo, 
                                                    const FpType y_hi, 
                                                    const FpType y_lo) 
    { 
        return x_hi < y_hi || (x_hi == y_hi && x_lo <= y_lo);
    }

    // >= comparison (assumes normalized inputs where |lo| < ulp(hi)/2)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  bool __nv_fpmp2_cmp_ge (const FpType x_hi, 
                                                    const FpType x_lo, 
                                                    const FpType y_hi, 
                                                    const FpType y_lo) 
    { 
        return x_hi > y_hi || (x_hi == y_hi && x_lo >= y_lo);
    }

    /*
    * --------------------------------------------------------------------
    * Bit cast operations (IEEE-754 format)
    * --------------------------------------------------------------------
    */
    // bit_cast to IEEE-754 format bits
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__  uint64_t __nv_fpmp2_bit_cast (const FpType x_hi, 
                                                          const FpType x_lo)
    { 

        double d = __nv_fpmp2_to_double(x_hi, x_lo);
        return fpmp::internal_bit_cast<uint64_t>(d);
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
    template<typename FpType>
    __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicAdd (FpType*      address_hi, 
                                                        FpType*      address_lo, 
                                                        const FpType addition_hi, 
                                                        const FpType addition_lo, 
                                                        FpType*      old_hi, 
                                                        FpType*      old_lo);

    template<typename FpType>
    __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicSub (FpType*      address_hi, 
                                                        FpType*      address_lo, 
                                                        const FpType val_hi, 
                                                        const FpType val_lo, 
                                                        FpType*      old_hi, 
                                                        FpType*      old_lo);

    /*
    * --------------------------------------------------------------------
    * Atomic operations - Float (fp32) specializations
    * --------------------------------------------------------------------
    */
    // atomicAdd for float (fp32mp2): Uses 64-bit atomicCAS
    // Two floats = 64 bits fits in unsigned long long int
    // Returns the old value before the addition
    template<>
    __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicAdd<float>       (float*       address_hi, 
                                                                     float*       address_lo, 
                                                                     const float  addition_hi, 
                                                                     const float  addition_lo, 
                                                                     float*       old_hi, 
                                                                     float*       old_lo)
    {

        
        // Treat the two floats as a single 64-bit value for atomic operations
        // The address must be 8-byte aligned (guaranteed by alignas(2*alignof(float)) in the class)
        static_assert(sizeof(float) * 2 == sizeof(unsigned long long int), 
                      "Two floats must equal one unsigned long long int"); 
        
        unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address_hi);
        unsigned long long int old             = *address_as_ull;
        unsigned long long int assumed;
        
        // Use the atomicCAS loop with retries to ensure atomicity
        do 
        {
            assumed = old;
            
            // Extract old values from the 64-bit integer
            uint32_t old_hi_bits = static_cast<uint32_t>(assumed & 0xFFFFFFFFULL);
            uint32_t old_lo_bits = static_cast<uint32_t>((assumed >> 32) & 0xFFFFFFFFULL);
            float old_hi_val = fpmp::internal_bit_cast<float>(old_hi_bits);
            float old_lo_val = fpmp::internal_bit_cast<float>(old_lo_bits);
            
            // Perform addition based on method
            float new_hi, new_lo;
            __nv_fpmp2_high_add(old_hi_val, old_lo_val, addition_hi, addition_lo, &new_hi, &new_lo);
            
            // Pack new values into a 64-bit integer
            uint32_t new_hi_bits = fpmp::internal_bit_cast<uint32_t>(new_hi);
            uint32_t new_lo_bits = fpmp::internal_bit_cast<uint32_t>(new_lo);
            unsigned long long int new_ull = static_cast<unsigned long long int>(new_hi_bits) | 
                                            (static_cast<unsigned long long int>(new_lo_bits) << 32);
            
            old = atomicCAS(address_as_ull, assumed, new_ull);
        } while (assumed != old);
        
        // Return old value - extract from the final 'old' value
        uint32_t old_hi_bits = static_cast<uint32_t>(old & 0xFFFFFFFFULL);
        uint32_t old_lo_bits = static_cast<uint32_t>((old >> 32) & 0xFFFFFFFFULL);
        *old_hi = fpmp::internal_bit_cast<float>(old_hi_bits);
        *old_lo = fpmp::internal_bit_cast<float>(old_lo_bits);
    }

    // atomicSub for float: Uses negation and atomicAdd
    template<>
    __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicSub<float>       (float*       address_hi, 
                                                                     float*       address_lo, 
                                                                     const float  val_hi, 
                                                                     const float  val_lo, 
                                                                     float*       old_hi, 
                                                                     float*       old_lo)
    {
        // Negate the value and call atomicAdd with the same method
        __nv_fpmp2_atomicAdd<float>(address_hi, address_lo, -val_hi, -val_lo, old_hi, old_lo);
    }

  #if (FPMP_FP64MP2_ENABLE == 1)
    /*
    * --------------------------------------------------------------------
    * Atomic operations - Double (fp64) specializations
    * --------------------------------------------------------------------
    */
    // atomicAdd for double (fp64mp2): Uses 128-bit atomicCAS
    // Two doubles = 128 bits requires ulonglong2 and sm_90+ (Hopper architecture)
    // Returns the old value before the addition
    template<>
    __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicAdd<double>       (double*       address_hi, 
                                                                      double*       address_lo, 
                                                                      const double  addition_hi, 
                                                                      const double  addition_lo, 
                                                                      double*       old_hi, 
                                                                      double*       old_lo)
    {
      #if __CUDA_ARCH__ >= 900

        
        // Treat the two doubles as a single 128-bit value for atomic operations
        // The address must be 16-byte aligned for 128-bit atomics
        static_assert(sizeof(double) * 2 == sizeof(ulonglong2), 
                      "Two doubles must equal one ulonglong2 (128 bits)"); 
        
        ulonglong2* address_as_ull2 = reinterpret_cast<ulonglong2*>(address_hi);
        ulonglong2 old              = *address_as_ull2;
        ulonglong2 assumed;
        
        // Use the atomicCAS loop with retries to ensure atomicity
        do 
        {
            assumed = old;
            
            // Extract old values from the 128-bit structure
            double old_hi_val = fpmp::internal_bit_cast<double>(assumed.x);
            double old_lo_val = fpmp::internal_bit_cast<double>(assumed.y);
            
            // Perform addition based on method
            double new_hi, new_lo;
            __nv_fpmp2_high_add(old_hi_val, old_lo_val, addition_hi, addition_lo, &new_hi, &new_lo);

            // Pack new values into a 128-bit structure
            ulonglong2 new_ull2;
            new_ull2.x = fpmp::internal_bit_cast<unsigned long long int>(new_hi);
            new_ull2.y = fpmp::internal_bit_cast<unsigned long long int>(new_lo);
            
            // 128-bit atomicCAS available on sm_90+
            old = atomicCAS(address_as_ull2, assumed, new_ull2);
        } while (assumed.x != old.x || assumed.y != old.y);
        
        // Return old value - extract from the final 'old' value
        *old_hi = fpmp::internal_bit_cast<double>(old.x);
        *old_lo = fpmp::internal_bit_cast<double>(old.y);
      #else
        // 128-bit atomicCAS requires sm_90+ (Hopper architecture)
        // On older architectures, this is a no-op stub
        // Runtime checks should prevent this code path from being executed
        (void)address_hi; (void)address_lo;
        (void)addition_hi; (void)addition_lo;
        // Return the current values unchanged
        *old_hi = *address_hi;
        *old_lo = *address_lo;
      #endif
    }

    // atomicSub for double: Uses negation and atomicAdd
    template<>
    __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicSub<double>       (double*       address_hi, 
                                                                      double*       address_lo, 
                                                                      const double  val_hi, 
                                                                      const double  val_lo, 
                                                                      double*       old_hi, 
                                                                      double*       old_lo)
    {
        // Negate the value and call atomicAdd with the same method
        __nv_fpmp2_atomicAdd<double>(address_hi, address_lo, -val_hi, -val_lo, old_hi, old_lo);
    }
  #endif // FPMP_FP64MP2_ENABLE == 1

  #endif // __CUDACC__

  // __fpmp_fp128 operations (only for FpType == double)
  // available only for CUDA architectures >= 1000 or when FPMP_FP128_ENABLE is defined
  #if FPMP_FP128_ENABLE == 1
    template<typename FpType = double>
    constexpr __FPMP_INTERNAL_DECL__  void __nv_fpmp2_from_quad  (const __fpmp_fp128 x, 
                                                        FpType*      res_hi, 
                                                        FpType*      res_lo) 
    {
        *res_hi = static_cast<FpType>(x);
        *res_lo = static_cast<FpType>(x - static_cast<__fpmp_fp128>(*res_hi));
    }

    template<typename FpType = double>
    __FPMP_INTERNAL_DECL__  __fpmp_fp128 __nv_fpmp2_to_quad  (const FpType x_hi, 
                                                            const FpType x_lo) 
    {
        return static_cast<__fpmp_fp128>(x_hi) + static_cast<__fpmp_fp128>(x_lo);
    }
   #endif // FPMP_FP128_ENABLE == 1    

  #else // __FPMP_USE_LIB__

/*
 * ============================================================================
 * Single Precision (fp32) Multi-Precision Operations
 * ============================================================================
 */
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_double(const double x, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_int(const int32_t i, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_uint(const uint32_t i, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_ll(const int64_t i, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_from_ull(const uint64_t i, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ double   __nv_fp32mp2_to_double(const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ float    __nv_fp32mp2_to_float(const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ int32_t  __nv_fp32mp2_to_int(const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ uint32_t __nv_fp32mp2_to_uint(const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ int64_t  __nv_fp32mp2_to_ll(const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ uint64_t __nv_fp32mp2_to_ull(const float x_hi, const float x_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_add(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_mid_add(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_low_add(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_high_add(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_sub(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_mid_sub(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_low_sub(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_high_sub(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_acc(const float c, float* acc_hi, float* acc_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_mid_acc(const float c, float* acc_hi, float* acc_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_low_acc(const float c, float* acc_hi, float* acc_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_high_acc(const float c, float* acc_hi, float* acc_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_mul(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_mid_mul(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_low_mul(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
#if __FPMP_USE_ACCURATE_MUL__ == 1
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_high_mul(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
#endif // __FPMP_USE_ACCURATE_MUL__ == 1
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_renormalize(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_div(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_mid_div(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_low_div(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);
#if __FPMP_USE_ACCURATE_DIV__ == 1
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_high_div(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo);
#endif // __FPMP_USE_ACCURATE_DIV__ == 1
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_sqrt(const float a_hi, const float a_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_rsqrt(const float a_hi, const float a_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_mad(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_mid_mad(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_low_mad(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_high_mad(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_fma(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_mid_fma(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_low_fma(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_high_fma(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ void     __nv_fp32mp2_neg(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
__FPMP_BUILTIN_DECL__ bool     __nv_fp32mp2_cmp_eq(const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool     __nv_fp32mp2_cmp_ne(const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool     __nv_fp32mp2_cmp_lt(const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool     __nv_fp32mp2_cmp_gt(const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool     __nv_fp32mp2_cmp_le(const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ bool     __nv_fp32mp2_cmp_ge(const float x_hi, const float x_lo, const float y_hi, const float y_lo);
__FPMP_BUILTIN_DECL__ uint64_t __nv_fp32mp2_bit_cast(const float x_hi, const float x_lo);
#ifdef __CUDACC__
__FPMP_BUILTIN_DEVICE_DECL__ void __nv_fp32mp2_atomicAdd(float* address_hi, float* address_lo, const float addition_hi, const float addition_lo, float* old_hi, float* old_lo);
__FPMP_BUILTIN_DEVICE_DECL__ void __nv_fp32mp2_atomicSub(float* address_hi, float* address_lo, const float val_hi, const float val_lo, float* old_hi, float* old_lo);
#endif // __CUDACC__

#if FPMP_FP64MP2_ENABLE == 1
    /*
    * ============================================================================
    * Double Precision (fp64) Multi-Precision Operations
    * ============================================================================
    */
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_double(const double x, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_int(const int32_t i, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_uint(const uint32_t i, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_ll(const int64_t i, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_from_ull(const uint64_t i, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ double   __nv_fp64mp2_to_double(const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ float    __nv_fp64mp2_to_float(const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ int32_t  __nv_fp64mp2_to_int(const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ uint32_t __nv_fp64mp2_to_uint(const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ int64_t  __nv_fp64mp2_to_ll(const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ uint64_t __nv_fp64mp2_to_ull(const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_add(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_mid_add(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_low_add(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_high_add(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_sub(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_mid_sub(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_low_sub(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_high_sub(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_acc(const double c, double* acc_hi, double* acc_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_mid_acc(const double c, double* acc_hi, double* acc_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_low_acc(const double c, double* acc_hi, double* acc_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_high_acc(const double c, double* acc_hi, double* acc_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_mul(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_mid_mul(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_low_mul(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
#if __FPMP_USE_ACCURATE_MUL__ == 1
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_high_mul(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
#endif // __FPMP_USE_ACCURATE_MUL__ == 1
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_renormalize(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_div(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_mid_div(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_low_div(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);
#if __FPMP_USE_ACCURATE_DIV__ == 1
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_high_div(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo);
#endif // __FPMP_USE_ACCURATE_DIV__ == 1
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_sqrt(const double a_hi, const double a_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_rsqrt(const double a_hi, const double a_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_mad(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_mid_mad(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_low_mad(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_high_mad(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_fma(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_mid_fma(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_low_fma(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_high_fma(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void     __nv_fp64mp2_neg(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ bool     __nv_fp64mp2_cmp_eq(const double x_hi, const double x_lo, const double y_hi, const double y_lo);
    __FPMP_BUILTIN_DECL__ bool     __nv_fp64mp2_cmp_ne(const double x_hi, const double x_lo, const double y_hi, const double y_lo);
    __FPMP_BUILTIN_DECL__ bool     __nv_fp64mp2_cmp_lt(const double x_hi, const double x_lo, const double y_hi, const double y_lo);
    __FPMP_BUILTIN_DECL__ bool     __nv_fp64mp2_cmp_gt(const double x_hi, const double x_lo, const double y_hi, const double y_lo);
    __FPMP_BUILTIN_DECL__ bool     __nv_fp64mp2_cmp_le(const double x_hi, const double x_lo, const double y_hi, const double y_lo);
    __FPMP_BUILTIN_DECL__ bool     __nv_fp64mp2_cmp_ge(const double x_hi, const double x_lo, const double y_hi, const double y_lo);
    __FPMP_BUILTIN_DECL__ uint64_t __nv_fp64mp2_bit_cast(const double x_hi, const double x_lo);
    #ifdef __CUDACC__
    __FPMP_BUILTIN_DEVICE_DECL__ void __nv_fp64mp2_atomicAdd(double* address_hi, double* address_lo, const double addition_hi, const double addition_lo, double* old_hi, double* old_lo);
    __FPMP_BUILTIN_DEVICE_DECL__ void __nv_fp64mp2_atomicSub(double* address_hi, double* address_lo, const double val_hi, const double val_lo, double* old_hi, double* old_lo);
    #endif // __CUDACC__
    #if FPMP_FP128_ENABLE == 1
    __FPMP_BUILTIN_DECL__ void         __nv_fp64mp2_from_quad(const __fpmp_fp128 x, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ __fpmp_fp128 __nv_fp64mp2_to_quad(const double x_hi, const double x_lo);
    #endif // FPMP_FP128_ENABLE == 1

#endif // FPMP_FP64MP2_ENABLE == 1

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

template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_from_double(const double x, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_from_int(const int32_t i, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_from_uint(const uint32_t i, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_from_ll(const int64_t i, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_from_ull(const uint64_t i, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ double   __nv_fpmp2_to_double(const T x_hi, const T x_lo);
template<typename T> __FPMP_API_DECL__ float    __nv_fpmp2_to_float(const T x_hi, const T x_lo);
template<typename T> __FPMP_API_DECL__ int32_t  __nv_fpmp2_to_int(const T x_hi, const T x_lo);
template<typename T> __FPMP_API_DECL__ uint32_t __nv_fpmp2_to_uint(const T x_hi, const T x_lo);
template<typename T> __FPMP_API_DECL__ int64_t  __nv_fpmp2_to_ll(const T x_hi, const T x_lo);
template<typename T> __FPMP_API_DECL__ uint64_t __nv_fpmp2_to_ull(const T x_hi, const T x_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_add(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_mid_add(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_low_add(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_high_add(const T a_hi, const T a_lo, const T b_hi, const T b_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_sub(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_mid_sub(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_low_sub(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_high_sub(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_acc(const T c, T* acc_hi, T* acc_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_mid_acc(const T c, T* acc_hi, T* acc_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_low_acc(const T c, T* acc_hi, T* acc_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_high_acc(const T c, T* acc_hi, T* acc_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_mul(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_mid_mul(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_low_mul(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
#if __FPMP_USE_ACCURATE_MUL__ == 1
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_high_mul(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
#endif // __FPMP_USE_ACCURATE_MUL__ == 1
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_renormalize(const T x_hi, const T x_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_div(const T a_hi, const T a_lo, const T b_hi, const T b_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_mid_div(const T a_hi, const T a_lo, const T b_hi, const T b_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_low_div(const T a_hi, const T a_lo, const T b_hi, const T b_lo, T* res_hi, T* res_lo);
#if __FPMP_USE_ACCURATE_DIV__ == 1
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_high_div(const T a_hi, const T a_lo, const T b_hi, const T b_lo, T* res_hi, T* res_lo);
#endif // __FPMP_USE_ACCURATE_DIV__ == 1
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_sqrt(const T a_hi, const T a_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_rsqrt(const T a_hi, const T a_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_mad(const T x_hi, const T x_lo, const T y_hi, const T y_lo, const T z_hi, const T z_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_mid_mad(const T x_hi, const T x_lo, const T y_hi, const T y_lo, const T z_hi, const T z_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_low_mad(const T x_hi, const T x_lo, const T y_hi, const T y_lo, const T z_hi, const T z_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_high_mad(const T x_hi, const T x_lo, const T y_hi, const T y_lo, const T z_hi, const T z_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_fma(const T x_hi, const T x_lo, const T y_hi, const T y_lo, const T z_hi, const T z_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_mid_fma(const T x_hi, const T x_lo, const T y_hi, const T y_lo, const T z_hi, const T z_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_low_fma(const T x_hi, const T x_lo, const T y_hi, const T y_lo, const T z_hi, const T z_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_high_fma(const T x_hi, const T x_lo, const T y_hi, const T y_lo, const T z_hi, const T z_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_fma_exp(const T x_hi, const T x_lo, const T y_hi, const T y_lo, const T z_hi, const T z_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ void     __nv_fpmp2_neg(const T x_hi, const T x_lo, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_eq(const T x_hi, const T x_lo, const T y_hi, const T y_lo);
template<typename T> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_ne(const T x_hi, const T x_lo, const T y_hi, const T y_lo);
template<typename T> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_lt(const T x_hi, const T x_lo, const T y_hi, const T y_lo);
template<typename T> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_gt(const T x_hi, const T x_lo, const T y_hi, const T y_lo);
template<typename T> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_le(const T x_hi, const T x_lo, const T y_hi, const T y_lo);
template<typename T> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_ge(const T x_hi, const T x_lo, const T y_hi, const T y_lo);
template<typename T> __FPMP_API_DECL__ uint64_t __nv_fpmp2_bit_cast(const T x_hi, const T x_lo);
#ifdef __CUDACC__
template<typename T> __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicAdd(T* address_hi, T* address_lo, const T addition_hi, const T addition_lo, T* old_hi, T* old_lo);
template<typename T> __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicSub(T* address_hi, T* address_lo, const T val_hi, const T val_lo, T* old_hi, T* old_lo);
#endif
#if FPMP_FP128_ENABLE == 1
template<typename T> __FPMP_API_DECL__ void         __nv_fpmp2_from_quad(const __fpmp_fp128 x, T* res_hi, T* res_lo);
template<typename T> __FPMP_API_DECL__ __fpmp_fp128 __nv_fpmp2_to_quad(const T x_hi, const T x_lo);
#endif // FPMP_FP128_ENABLE == 1
/*
 * ============================================================================
 * Float (fp32) Template Specializations
 * ============================================================================
 * 
 * These specializations map the generic __nv_fpmp2_* template functions
 * to the concrete __nv_fp32mp2_* implementations for float types.
 * ============================================================================
 */
template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_double<float>(const double x, float* res_hi, float* res_lo) { __nv_fp32mp2_from_double(x, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_int<float>(const int32_t i, float* res_hi, float* res_lo) { __nv_fp32mp2_from_int(i, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_uint<float>(const uint32_t i, float* res_hi, float* res_lo) { __nv_fp32mp2_from_uint(i, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_ll<float>(const int64_t i, float* res_hi, float* res_lo) { __nv_fp32mp2_from_ll(i, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_ull<float>(const uint64_t i, float* res_hi, float* res_lo) { __nv_fp32mp2_from_ull(i, res_hi, res_lo); }
template<> __FPMP_API_DECL__ double   __nv_fpmp2_to_double<float>(const float x_hi, const float x_lo) { return __nv_fp32mp2_to_double(x_hi, x_lo); }
template<> __FPMP_API_DECL__ float    __nv_fpmp2_to_float<float>(const float x_hi, const float x_lo) { return __nv_fp32mp2_to_float(x_hi, x_lo); }
template<> __FPMP_API_DECL__ int32_t  __nv_fpmp2_to_int<float>(const float x_hi, const float x_lo) { return __nv_fp32mp2_to_int(x_hi, x_lo); }
template<> __FPMP_API_DECL__ uint32_t __nv_fpmp2_to_uint<float>(const float x_hi, const float x_lo) { return __nv_fp32mp2_to_uint(x_hi, x_lo); }
template<> __FPMP_API_DECL__ int64_t  __nv_fpmp2_to_ll<float>(const float x_hi, const float x_lo) { return __nv_fp32mp2_to_ll(x_hi, x_lo); }
template<> __FPMP_API_DECL__ uint64_t __nv_fpmp2_to_ull<float>(const float x_hi, const float x_lo) { return __nv_fp32mp2_to_ull(x_hi, x_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_add<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_add(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_add<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_mid_add(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_add<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_low_add(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_add<float>(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_high_add(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_sub<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_sub(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_sub<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_mid_sub(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_sub<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_low_sub(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_sub<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_high_sub(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_acc<float>(const float c, float* acc_hi, float* acc_lo) { __nv_fp32mp2_acc(c, acc_hi, acc_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_acc<float>(const float c, float* acc_hi, float* acc_lo) { __nv_fp32mp2_mid_acc(c, acc_hi, acc_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_acc<float>(const float c, float* acc_hi, float* acc_lo) { __nv_fp32mp2_low_acc(c, acc_hi, acc_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_acc<float>(const float c, float* acc_hi, float* acc_lo) { __nv_fp32mp2_high_acc(c, acc_hi, acc_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_mul<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_mul(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_mul<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_mid_mul(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_mul<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_low_mul(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
#if __FPMP_USE_ACCURATE_MUL__ == 1
template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_mul<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_high_mul(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
#endif // __FPMP_USE_ACCURATE_MUL__ == 1
template<> __FPMP_API_DECL__ void     __nv_fpmp2_renormalize<float>(const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_renormalize(x_hi, x_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_div<float>(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_div(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_div<float>(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_mid_div(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_div<float>(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_low_div(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
#if __FPMP_USE_ACCURATE_DIV__ == 1
template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_div<float>(const float a_hi, const float a_lo, const float b_hi, const float b_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_high_div(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
#endif // __FPMP_USE_ACCURATE_DIV__ == 1
template<> __FPMP_API_DECL__ void     __nv_fpmp2_sqrt<float>(const float a_hi, const float a_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_sqrt(a_hi, a_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_rsqrt<float>(const float a_hi, const float a_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_rsqrt(a_hi, a_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_mad<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_mad(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_mad<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_mid_mad(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_mad<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_low_mad(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_mad<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_high_mad(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_fma<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_fma(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_fma<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_mid_fma(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_fma<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_low_fma(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_fma<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, const float z_hi, const float z_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_high_fma(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ void     __nv_fpmp2_neg<float>(const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_neg(x_hi, x_lo, res_hi, res_lo); }
template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_eq<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo) { return __nv_fp32mp2_cmp_eq(x_hi, x_lo, y_hi, y_lo); }
template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_ne<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo) { return __nv_fp32mp2_cmp_ne(x_hi, x_lo, y_hi, y_lo); }
template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_lt<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo) { return __nv_fp32mp2_cmp_lt(x_hi, x_lo, y_hi, y_lo); }
template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_gt<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo) { return __nv_fp32mp2_cmp_gt(x_hi, x_lo, y_hi, y_lo); }
template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_le<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo) { return __nv_fp32mp2_cmp_le(x_hi, x_lo, y_hi, y_lo); }
template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_ge<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo) { return __nv_fp32mp2_cmp_ge(x_hi, x_lo, y_hi, y_lo); }
template<> __FPMP_API_DECL__ uint64_t __nv_fpmp2_bit_cast<float>(const float x_hi, const float x_lo) { return __nv_fp32mp2_bit_cast(x_hi, x_lo); }
#ifdef __CUDACC__
template<> __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicAdd<float>(float* address_hi, float* address_lo, const float addition_hi, const float addition_lo, float* old_hi, float* old_lo) { __nv_fp32mp2_atomicAdd(address_hi, address_lo, addition_hi, addition_lo, old_hi, old_lo); }
template<> __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicSub<float>(float* address_hi, float* address_lo, const float val_hi, const float val_lo, float* old_hi, float* old_lo) { __nv_fp32mp2_atomicSub(address_hi, address_lo, val_hi, val_lo, old_hi, old_lo); }
#endif // __CUDACC__

#if FPMP_FP64MP2_ENABLE == 1
    /*
    * ============================================================================
    * Double (fp64) Template Specializations
    * ============================================================================
    * 
    * These specializations map the generic __nv_fpmp2_* template functions
    * to the concrete __nv_fp64mp2_* implementations for double types.
    * ============================================================================
    */
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_double<double>(const double x, double* res_hi, double* res_lo) { __nv_fp64mp2_from_double(x, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_int<double>(const int32_t i, double* res_hi, double* res_lo) { __nv_fp64mp2_from_int(i, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_uint<double>(const uint32_t i, double* res_hi, double* res_lo) { __nv_fp64mp2_from_uint(i, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_ll<double>(const int64_t i, double* res_hi, double* res_lo) { __nv_fp64mp2_from_ll(i, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_from_ull<double>(const uint64_t i, double* res_hi, double* res_lo) { __nv_fp64mp2_from_ull(i, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ double   __nv_fpmp2_to_double<double>(const double x_hi, const double x_lo) { return __nv_fp64mp2_to_double(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ float    __nv_fpmp2_to_float<double>(const double x_hi, const double x_lo) { return __nv_fp64mp2_to_float(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int32_t  __nv_fpmp2_to_int<double>(const double x_hi, const double x_lo) { return __nv_fp64mp2_to_int(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ uint32_t __nv_fpmp2_to_uint<double>(const double x_hi, const double x_lo) { return __nv_fp64mp2_to_uint(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int64_t  __nv_fpmp2_to_ll<double>(const double x_hi, const double x_lo) { return __nv_fp64mp2_to_ll(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ uint64_t __nv_fpmp2_to_ull<double>(const double x_hi, const double x_lo) { return __nv_fp64mp2_to_ull(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_add<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_add(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_add<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_mid_add(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_add<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_low_add(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_add<double>(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_high_add(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_sub<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_sub(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_sub<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_mid_sub(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_sub<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_low_sub(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_sub<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_high_sub(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_acc<double>(const double c, double* acc_hi, double* acc_lo) { __nv_fp64mp2_acc(c, acc_hi, acc_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_acc<double>(const double c, double* acc_hi, double* acc_lo) { __nv_fp64mp2_mid_acc(c, acc_hi, acc_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_acc<double>(const double c, double* acc_hi, double* acc_lo) { __nv_fp64mp2_low_acc(c, acc_hi, acc_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_acc<double>(const double c, double* acc_hi, double* acc_lo) { __nv_fp64mp2_high_acc(c, acc_hi, acc_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_mul<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_mul(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_mul<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_mid_mul(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_mul<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_low_mul(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
#if __FPMP_USE_ACCURATE_MUL__ == 1
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_mul<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_high_mul(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
#endif // __FPMP_USE_ACCURATE_MUL__ == 1
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_renormalize<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_renormalize(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_div<double>(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_div(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_div<double>(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_mid_div(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_div<double>(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_low_div(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
#if __FPMP_USE_ACCURATE_DIV__ == 1
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_div<double>(const double a_hi, const double a_lo, const double b_hi, const double b_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_high_div(a_hi, a_lo, b_hi, b_lo, res_hi, res_lo); }
#endif // __FPMP_USE_ACCURATE_DIV__ == 1
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_sqrt<double>(const double a_hi, const double a_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_sqrt(a_hi, a_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_rsqrt<double>(const double a_hi, const double a_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_rsqrt(a_hi, a_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_mad<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_mad(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_mad<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_mid_mad(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_mad<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_low_mad(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_mad<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_high_mad(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_fma<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_fma(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_mid_fma<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_mid_fma(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_low_fma<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_low_fma(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_high_fma<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, const double z_hi, const double z_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_high_fma(x_hi, x_lo, y_hi, y_lo, z_hi, z_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void     __nv_fpmp2_neg<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_neg(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_eq<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo) { return __nv_fp64mp2_cmp_eq(x_hi, x_lo, y_hi, y_lo); }
    template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_ne<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo) { return __nv_fp64mp2_cmp_ne(x_hi, x_lo, y_hi, y_lo); }
    template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_lt<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo) { return __nv_fp64mp2_cmp_lt(x_hi, x_lo, y_hi, y_lo); }
    template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_gt<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo) { return __nv_fp64mp2_cmp_gt(x_hi, x_lo, y_hi, y_lo); }
    template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_le<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo) { return __nv_fp64mp2_cmp_le(x_hi, x_lo, y_hi, y_lo); }
    template<> __FPMP_API_DECL__ bool     __nv_fpmp2_cmp_ge<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo) { return __nv_fp64mp2_cmp_ge(x_hi, x_lo, y_hi, y_lo); }
    template<> __FPMP_API_DECL__ uint64_t __nv_fpmp2_bit_cast<double>(const double x_hi, const double x_lo) { return __nv_fp64mp2_bit_cast(x_hi, x_lo); }
    #ifdef __CUDACC__
    template<> __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicAdd<double>(double* address_hi, double* address_lo, const double addition_hi, const double addition_lo, double* old_hi, double* old_lo) { __nv_fp64mp2_atomicAdd(address_hi, address_lo, addition_hi, addition_lo, old_hi, old_lo); }
    template<> __FPMP_API_DEVICE_DECL__ void __nv_fpmp2_atomicSub<double>(double* address_hi, double* address_lo, const double val_hi, const double val_lo, double* old_hi, double* old_lo) { __nv_fp64mp2_atomicSub(address_hi, address_lo, val_hi, val_lo, old_hi, old_lo); }
    #endif // __CUDACC__
    #if FPMP_FP128_ENABLE == 1
    template<> __FPMP_API_DECL__ void         __nv_fpmp2_from_quad<double>(const __fpmp_fp128 x, double* res_hi, double* res_lo) { __nv_fp64mp2_from_quad(x, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ __fpmp_fp128 __nv_fpmp2_to_quad<double>(const double x_hi, const double x_lo) { return __nv_fp64mp2_to_quad(x_hi, x_lo); }
    #endif // FPMP_FP128_ENABLE == 1

#endif // FPMP_FP64MP2_ENABLE == 1

#endif // __FPMP_USE_LIB__

/*
 * ============================================================================
 * Freestanding Atomic Operations for fpmp2_t
 * ============================================================================
 * These are CUDA-style freestanding atomic functions that work with the
 * fpmp2_t class. They are placed outside the USE_LIB conditional to
 * work in both inline and library modes.
 * ============================================================================
 */
#ifdef __CUDACC__

// Forward declaration of the class template (defined in fpmp.hpp)
template <typename FpType, fpmp2_accuracy met> class fpmp2_t;

// atomicAdd: Atomic addition for fpmp2_t
// Returns the old value before the addition
template <typename FpType, fpmp2_accuracy met>
__FPMP_API_DEVICE_DECL__ fpmp2_t<FpType, met> atomicAdd(fpmp2_t<FpType, met>* address, 
                                                              const fpmp2_t<FpType, met>& val)
{
    fpmp2_t<FpType, met> result;
    // Class layout: alignas(2*alignof(FpType)) with mp2_hi at offset 0, mp2_lo at offset sizeof(FpType)
    FpType* addr_hi = reinterpret_cast<FpType*>(address);
    FpType* addr_lo = addr_hi + 1;
    FpType* res_hi  = reinterpret_cast<FpType*>(&result);
    FpType* res_lo  = res_hi + 1;
  #if defined(__FPMP_USE_LIB__)
    // In library mode, call the library function directly
    if constexpr (std::is_same<FpType, float>::value) {
        __nv_fp32mp2_atomicAdd(addr_hi, addr_lo, val.hi(), val.lo(), res_hi, res_lo);
    } 
    #if FPMP_FP64MP2_ENABLE == 1
    else if constexpr (std::is_same<FpType, double>::value) {
        __nv_fp64mp2_atomicAdd(addr_hi, addr_lo, val.hi(), val.lo(), res_hi, res_lo);
    }
    #endif
  #else
    __nv_fpmp2_atomicAdd(addr_hi, addr_lo, val.hi(), val.lo(), res_hi, res_lo);
  #endif
    return result;
}

// atomicSub: Atomic subtraction for fpmp2_t
// Returns the old value before the subtraction
template <typename FpType, fpmp2_accuracy met>
__FPMP_API_DEVICE_DECL__ fpmp2_t<FpType, met> atomicSub(fpmp2_t<FpType, met>* address, 
                                                              const fpmp2_t<FpType, met>& val)
{
    fpmp2_t<FpType, met> result;
    // Class layout: alignas(2*alignof(FpType)) with mp2_hi at offset 0, mp2_lo at offset sizeof(FpType)
    FpType* addr_hi = reinterpret_cast<FpType*>(address);
    FpType* addr_lo = addr_hi + 1;
    FpType* res_hi  = reinterpret_cast<FpType*>(&result);
    FpType* res_lo  = res_hi + 1;
  #if defined(__FPMP_USE_LIB__)
    // In library mode, call the library function directly
    if constexpr (std::is_same<FpType, float>::value) {
        __nv_fp32mp2_atomicSub(addr_hi, addr_lo, val.hi(), val.lo(), res_hi, res_lo);
    } 
    #if FPMP_FP64MP2_ENABLE == 1
    else if constexpr (std::is_same<FpType, double>::value) {
        __nv_fp64mp2_atomicSub(addr_hi, addr_lo, val.hi(), val.lo(), res_hi, res_lo);
    }
    #endif
  #else
    __nv_fpmp2_atomicSub(addr_hi, addr_lo, val.hi(), val.lo(), res_hi, res_lo);
  #endif
    return result;
}

#endif // __CUDACC__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_IMPL_H