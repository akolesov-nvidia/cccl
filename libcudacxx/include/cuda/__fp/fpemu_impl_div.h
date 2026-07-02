//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_DIV_H
#define _CUDA___FP_FPEMU_IMPL_DIV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_ddiv_impl.hpp
 * @brief Implementation of double-precision division operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of double-precision division operations for the FPEMU library.
 * It includes:
 *
 * - Division functions for fp64emu_t
 * - Division operators for fp64emu_t
 * - Division functions to other types
 *
 * The division functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

#include <cuda/__fp/fpemu_common.h>
#include <cuda/__fp/fpemu_impl_utils.h>
#include <cuda/__fp/fpemu_impl_unpack.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

namespace impl
{
    // ========================================================================
    // Native fp64 division.
    //
    // Split sign/exp/mantissa, normalize subnormals, form a 32-bit reciprocal of the divisor
    // significand, then refine the quotient with fixed-point integer remainder
    // arithmetic and round/pack. We use the fp32 reciprocal builtin (1/x in float) and a single
    // Newton-Raphson step, which meets/exceeds the accuracy the remainder
    // refinement needs for correctly-rounded results.
    // ========================================================================

    /// @brief Approximation of floor(2^63 / b32) for b32 in [2^31, 2^32).
    ///        Seeded by the fp32 reciprocal builtin, refined by one Newton step.
    __FPEMU_INTERNAL_DECL__ uint32_t __nv_internal_fp64emu_div_recip32 (uint32_t __b32)
    {
        // fp32 seed: interpret b32 as bf = b32 / 2^31 in [1, 2); 1/bf in (0.5, 1].
        // r ~ (1/bf) * 2^32 = 2^63 / b32.
        float    __bf = (float)__b32 * (1.0f / 2147483648.0f);   // 1/2^31
        // Fast fp32 reciprocal seed. The Newton step below refines it and the
        // final trim guarantees a strict underestimate, so the low accuracy of
        // the SFU approximation (rcp.approx) is acceptable here.
    #if defined(__CUDA_ARCH__)
        float    __rf;
        asm ("rcp.approx.ftz.f32 %0, %1;" : "=f"(__rf) : "f"(__bf));
    #else
        float    __rf = 1.0f / __bf;                             // host fallback
    #endif
        uint64_t __r  = (uint64_t)(__rf * 4294967296.0f);        // rf * 2^32

        if (__r < 0x80000000ULL) __r = 0x80000000ULL;
        if (__r > 0xFFFFFFFFULL) __r = 0xFFFFFFFFULL;

        // One Newton-Raphson step: r <- r + r*(2^63 - b32*r)/2^63.
        uint64_t __prod = (uint64_t)__b32 * __r;                   // ~2^63
        int64_t  __e    = (int64_t)(__FPEMU_SIGN_64__ - __prod);   // 2^63 - prod
        uint64_t __ae   = (uint64_t)(__e < 0 ? -__e : __e);
        uint64_t __lo   = __r * __ae;                              // low 64 bits of r*ae
        uint64_t __hi   = __nv_internal_fp64emu_mulhi64(__r, __ae);
        uint64_t __corr = (__hi << 1) | (__lo >> 63);              // (r*ae) >> 63
        __r = (__e < 0) ? (__r - __corr) : (__r + __corr);
        if (__r > 0xFFFFFFFFULL) __r = 0xFFFFFFFFULL;

        // The division algorithm requires a strict underestimate of 2^63/b32: 
        // if the estimate overshoots floor(2^63/b32), sig32Z exceeds the true quotient
        // and the unsigned remainder underflows. The fp32 seed + Newton step is
        // accurate to +/-1 ULP, so trim any positive overshoot.
        while ((uint64_t)__b32 * (uint32_t)__r > __FPEMU_SIGN_64__) --__r;   // > 2^63
        return (uint32_t)__r;
    } // __nv_internal_fp64emu_div_recip32

    /// @brief True if the bit pattern encodes a NaN.
    __FPEMU_INTERNAL_DECL__ bool __nv_internal_fp64emu_div_is_nan (uint64_t __ui)
    {
        return ((~__ui & __FPEMU_EXP_64__) == 0) && ( __ui & __FPEMU_MANT_64__);
    } // __nv_internal_fp64emu_div_is_nan

    /// Propagate NaN. Precise (8086) propagation is only done for correctly-rounded
    /// accuracy; other modes return a cheap NaN (ui64_a | ui64_b is always a NaN when at
    /// least one operand is a NaN) to keep the implementation light.
    template<fp64emu_accuracy _Acc = fp64emu_accuracy::high>
    __FPEMU_INTERNAL_DECL__ uint64_t __nv_internal_fp64emu_div_propagate_nan (uint64_t __ui64_a, 
                                                                              uint64_t __ui64_b)
    {
        if constexpr (_Acc != fp64emu_accuracy::high)
        {
            return __ui64_a | __ui64_b;
        }

        bool     __is_sig_nan_a = ((__ui64_a & __FPEMU_QNAN_64__) == __FPEMU_EXP_64__) && (__ui64_a & __FPEMU_SNAN_PAYLOAD_64__);
        bool     __is_sig_nan_b = ((__ui64_b & __FPEMU_QNAN_64__) == __FPEMU_EXP_64__) && (__ui64_b & __FPEMU_SNAN_PAYLOAD_64__);
        uint64_t __nonsig64_a   = __ui64_a | __FPEMU_QNAN_BIT_64__;
        uint64_t __nonsig64_b   = __ui64_b | __FPEMU_QNAN_BIT_64__;

        if (__is_sig_nan_a && !__is_sig_nan_b) return __nv_internal_fp64emu_div_is_nan(__ui64_b) ? __nonsig64_b : __nonsig64_a;
        if (__is_sig_nan_b && !__is_sig_nan_a) return __nv_internal_fp64emu_div_is_nan(__ui64_a) ? __nonsig64_a : __nonsig64_b;
        // Both signaling or neither signaling: return the larger-magnitude NaN.
        uint64_t __mag64_a = __ui64_a & __FPEMU_ABS_64__;
        uint64_t __mag64_b = __ui64_b & __FPEMU_ABS_64__;
        if (__mag64_a < __mag64_b) return __nonsig64_b;
        if (__mag64_b < __mag64_a) return __nonsig64_a;
        return (__nonsig64_a < __nonsig64_b) ? __nonsig64_a : __nonsig64_b;
    } // __nv_internal_fp64emu_div_propagate_nan

    // Forward declaration: the unpacked divide core is defined below, but the
    // packed wrapper references it for the packed-via-unpacked (testing) path.
    template<fp64emu_accuracy _Acc>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_ddiv_unpacked(fpbits64_unpacked_t __x,
                                                            fpbits64_unpacked_t __y);

    /**
     * @brief Divide two fpbits64_t
     * 
     * This function divides two fpbits64_t.
     * 
     * @param x The first fpbits64_t
     * @param y The second fpbits64_t
     * @return The result of the division
     */
    template<fpemu::rounding _Rm   = fpemu::rounding::def, 
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_ddiv(fpbits64_t __x, 
                                          fpbits64_t __y)
    {
    #if (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
        // Packed-via-unpacked (testing): pack(ddiv_unpacked(unpack(x), unpack(y))).
        // The ddiv_unpacked core handles special operands and method selection; the
        // universal unpack/pack are the shared prologue/epilogue. Rounding is applied
        // only at pack, so the packed builtins keep their per-mode behavior.
        {
            fpbits64_unpacked_t __a = __nv_internal_fp64emu_unpack(__x);
            fpbits64_unpacked_t __b = __nv_internal_fp64emu_unpack(__y);
            fpbits64_unpacked_t __r = __nv_internal_fp64emu_ddiv_unpacked<_Acc>(__a, __b);
            return __nv_internal_fp64emu_pack<_Rm>(__r);
        }
    #else
        const uint64_t __ui64_a = (uint64_t)__x;
        const uint64_t __ui64_b = (uint64_t)__y;

        bool     __sign_a = (__ui64_a >> 63) != 0;
        int32_t  __exp_a  = (int32_t)((__ui64_a >> 52) & 0x7FF);
        uint64_t __mant_a = __ui64_a & __FPEMU_MANT_64__;

        bool     __sign_b = (__ui64_b >> 63) != 0;
        int32_t  __exp_b  = (int32_t)((__ui64_b >> 52) & 0x7FF);
        uint64_t __mant_b = __ui64_b & __FPEMU_MANT_64__;

        bool     __sign_z = __sign_a ^ __sign_b;

        // -------- special operands (NaN / Inf / zero) --------
        if (__exp_a == 0x7FF)
        {
            if (__mant_a) return (fpbits64_t)__nv_internal_fp64emu_div_propagate_nan<_Acc>(__ui64_a, __ui64_b);
            if (__exp_b == 0x7FF)
            {
                if (__mant_b) return (fpbits64_t)__nv_internal_fp64emu_div_propagate_nan<_Acc>(__ui64_a, __ui64_b);
                else        return (fpbits64_t)__FPEMU_DEFNAN_64__;                          // inf / inf -> NaN
            }
            return (fpbits64_t)(((uint64_t)__sign_z << 63) | __FPEMU_INF_64__);              // inf / finite
        }
        if (__exp_b == 0x7FF)
        {
            if (__mant_b) return (fpbits64_t)__nv_internal_fp64emu_div_propagate_nan<_Acc>(__ui64_a, __ui64_b);
            else        return (fpbits64_t)((uint64_t)__sign_z << 63);                         // finite / inf -> 0
        }

        // -------- subnormals & division by zero --------
        if (!__exp_b)
        {
            if (!__mant_b)
            {
                if (!__exp_a && !__mant_a) return (fpbits64_t)__FPEMU_DEFNAN_64__;                   // 0 / 0 -> NaN
                else                  return (fpbits64_t)(((uint64_t)__sign_z << 63) | __FPEMU_INF_64__); // x / 0 -> inf
            }
            int __mant_b_shft = fpemu::__internal_clzll((int64_t)__mant_b) - 11;
            // normalize subnormal b
            __exp_b  = 1 - __mant_b_shft;
            __mant_b = __mant_b << __mant_b_shft;
        }
        if (!__exp_a)
        {
            if (!__mant_a) return (fpbits64_t)((uint64_t)__sign_z << 63);            // 0 / x -> 0
            int __mant_a_shft = fpemu::__internal_clzll((int64_t)__mant_a) - 11;
            // normalize subnormal a
            __exp_a  = 1 - __mant_a_shft;
            __mant_a = __mant_a << __mant_a_shft;
        }

        // -------- fixed-point reciprocal division --------
        int32_t __exp_z = __exp_a - __exp_b + 0x3FE;

        __mant_a |= __FPEMU_HIDDEN_64__;
        __mant_b |= __FPEMU_HIDDEN_64__;

        if (__mant_a < __mant_b) { --__exp_z; __mant_a <<= 11; } else { __mant_a <<= 10; }

        __mant_b <<= 11;

        uint32_t __recip32   = __nv_internal_fp64emu_div_recip32((uint32_t)(__mant_b >> 32)) - 2;
        uint32_t __mant32_z  = (uint32_t)(((uint64_t)(uint32_t)(__mant_a >> 32) * (uint64_t)__recip32) >> 32);
        uint32_t __mant32_z2 = __mant32_z << 1;
        uint64_t __rem64     = ((__mant_a - (uint64_t)__mant32_z2 * (uint32_t)(__mant_b >> 32)) << 28) - (uint64_t)__mant32_z2 * ((uint32_t)__mant_b >> 4);
        uint32_t __q32       = (uint32_t)(((uint64_t)(uint32_t)(__rem64 >> 32) * (uint64_t)__recip32) >> 32) + 4;
        uint64_t __mant64_z  = ((uint64_t)__mant32_z << 32) + ((uint64_t)__q32 << 4);

        // Refine if the quotient is close to a rounding boundary (exact remainder).
        if ((__mant64_z & 0x1FF) < (4u << 4))
        {
            __q32     &= ~7u;
            __mant64_z &= ~(uint64_t)0x7F;
            __mant32_z2 = __q32 << 1;

            __rem64 = ((__rem64 - (uint64_t)__mant32_z2 * (uint32_t)(__mant_b >> 32)) << 28) - (uint64_t)__mant32_z2 * ((uint32_t)__mant_b >> 4);

            if (__rem64 & __FPEMU_SIGN_64__) { __mant64_z -= 1 << 7; }
            else { if (__rem64) __mant64_z |= 1; }
        }

        return __nv_internal_fp64emu_round_pack<_Rm>(__sign_z, __exp_z, __mant64_z);
    #endif // __FPEMU_PACKED_VIA_UNPACKED__
    } // __nv_internal_fp64emu_ddiv

    /**
     * @brief Divide two fpbits64_unpacked_t
     * 
     * This function divides two fpbits64_unpacked_t.
     * 
     * @param x The first fpbits64_unpacked_t
     * @param y The second fpbits64_unpacked_t
     * @return The result of the division
     */
    template<fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_ddiv_unpacked(fpbits64_unpacked_t __x, 
                                                          fpbits64_unpacked_t __y)
    {
        // ---- True unpacked divide -------------------------------------------
        // Operates directly on the fully-accurate unpacked operands (no operand
        // pack, no legacy packed kernel). The full unpack has already normalized
        // denormals and encoded inf/nan in the exponent band, so the significand
        // is mantissa>>EXTRA_BITS (implicit bit at 52) and the exponent is the
        // IEEE-biased value -- no subnormal renormalization needed. The proven
        // fixed-point reciprocal quotient is computed exactly as the packed core,
        // then expressed on the universal unpacked scale (implicit bit at 61, a
        // sticky LSB) so the full pack does the single correctly-rounded
        // finalization (subnormal / overflow / rounding). Division has no def/fast
        // arithmetic variant -- the quotient is correctly rounded for every method.
        constexpr int32_t __NAN_EXP = 0x0007ff00;
        constexpr int32_t __INF_EXP = 0x00007ff0;

        const int32_t __exp_x  = (int32_t)__x.exponent;
        const int32_t __exp_y  = (int32_t)__y.exponent;
        const bool    __sign_z = ((__x.sign != 0) ^ (__y.sign != 0));
        const uint64_t __sign_bit = (uint64_t)__sign_z << 63;

        const bool __nan_x  = (__exp_x == __NAN_EXP), __nan_y  = (__exp_y == __NAN_EXP);
        const bool __inf_x  = (__exp_x == __INF_EXP), __inf_y  = (__exp_y == __INF_EXP);
        const bool __zero_x = (__x.mantissa == 0),  __zero_y = (__y.mantissa == 0);

        // Special operands: build the canonical packed result and unpack it (rare,
        // off the hot path -- no arithmetic round trip).
        if (__nan_x || __nan_y)
            return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_DEFNAN_64__);
        if (__inf_x)
        {
            if (__inf_y) return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_DEFNAN_64__);     // inf/inf
            return __nv_internal_fp64emu_unpack((fpbits64_t)(__sign_bit | __FPEMU_INF_64__));      // inf/finite
        }
        if (__inf_y)
            return __nv_internal_fp64emu_unpack((fpbits64_t)__sign_bit);                           // finite/inf -> 0
        if (__zero_y)
        {
            if (__zero_x) return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_DEFNAN_64__);    // 0/0
            return __nv_internal_fp64emu_unpack((fpbits64_t)(__sign_bit | __FPEMU_INF_64__));      // x/0
        }
        if (__zero_x)
            return __nv_internal_fp64emu_unpack((fpbits64_t)__sign_bit);                           // 0/finite -> 0

        // ---- finite / finite : fixed-point reciprocal division --------------
        uint64_t __mant_a = __x.mantissa >> EXTRA_BITS;   // 53-bit significand, implicit bit at 52
        uint64_t __mant_b = __y.mantissa >> EXTRA_BITS;
        int32_t  __exp_z  = __exp_x - __exp_y + 0x3FE;

        if (__mant_a < __mant_b) { --__exp_z; __mant_a <<= 11; } else { __mant_a <<= 10; }
        __mant_b <<= 11;

        uint32_t __recip32   = __nv_internal_fp64emu_div_recip32((uint32_t)(__mant_b >> 32)) - 2;
        uint32_t __mant32_z  = (uint32_t)(((uint64_t)(uint32_t)(__mant_a >> 32) * (uint64_t)__recip32) >> 32);
        uint32_t __mant32_z2 = __mant32_z << 1;
        uint64_t __rem64     = ((__mant_a - (uint64_t)__mant32_z2 * (uint32_t)(__mant_b >> 32)) << 28) - (uint64_t)__mant32_z2 * ((uint32_t)__mant_b >> 4);
        uint32_t __q32       = (uint32_t)(((uint64_t)(uint32_t)(__rem64 >> 32) * (uint64_t)__recip32) >> 32) + 4;
        uint64_t __mant64_z  = ((uint64_t)__mant32_z << 32) + ((uint64_t)__q32 << 4);

        // Refine if the quotient is close to a rounding boundary (exact remainder).
        if ((__mant64_z & 0x1FF) < (4u << 4))
        {
            __q32      &= ~7u;
            __mant64_z &= ~(uint64_t)0x7F;
            __mant32_z2 = __q32 << 1;
            __rem64 = ((__rem64 - (uint64_t)__mant32_z2 * (uint32_t)(__mant_b >> 32)) << 28) - (uint64_t)__mant32_z2 * ((uint32_t)__mant_b >> 4);
            if (__rem64 & __FPEMU_SIGN_64__) { __mant64_z -= 1 << 7; }
            else { if (__rem64) __mant64_z |= 1; }
        }

        // round_pack expects the leading significand bit at 62 and exp == biased-1.
        // The universal unpacked scale puts the implicit bit at 61 with EXTRA_BITS
        // round bits and exponent == IEEE-biased (== exp_z + 1); shift the leading
        // bit down one place (preserving the dropped bit as sticky) and let the
        // full pack round + emit subnormal / saturate to inf.
        fpbits64_unpacked_t __r;
        __r.sign     = __sign_z ? (1u << 31) : 0u;
        __r.exponent = (uint32_t)(__exp_z + 1);
        __r.mantissa = (__mant64_z >> 1) | (__mant64_z & 1);
        return __r;
    } // __nv_internal_fp64emu_ddiv_unpacked
} // namespace impl

// ============================================================================
// Builtin declarations/implementations for division operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ddiv_rn (fpbits64_t __x, fpbits64_t __y) { return impl::__nv_internal_fp64emu_ddiv<fpemu::rounding::rn, fp64emu_accuracy::high>(__x, __y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ddiv_rz (fpbits64_t __x, fpbits64_t __y) { return impl::__nv_internal_fp64emu_ddiv<fpemu::rounding::rz, fp64emu_accuracy::high>(__x, __y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ddiv_ru (fpbits64_t __x, fpbits64_t __y) { return impl::__nv_internal_fp64emu_ddiv<fpemu::rounding::ru, fp64emu_accuracy::high>(__x, __y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ddiv_rd (fpbits64_t __x, fpbits64_t __y) { return impl::__nv_internal_fp64emu_ddiv<fpemu::rounding::rd, fp64emu_accuracy::high>(__x, __y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_ddiv_rn (fpbits64_t __x, fpbits64_t __y) { return impl::__nv_internal_fp64emu_ddiv<fpemu::rounding::rn, fp64emu_accuracy::high>(__x, __y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_ddiv_rn      (fpbits64_t __x, fpbits64_t __y) { return impl::__nv_internal_fp64emu_ddiv<fpemu::rounding::rn, fp64emu_accuracy::mid>(__x, __y); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_ddiv_rn     (fpbits64_t __x, fpbits64_t __y) { return impl::__nv_internal_fp64emu_ddiv<fpemu::rounding::rn, fp64emu_accuracy::low>(__x, __y); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_ddiv          (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y) { return impl::__nv_internal_fp64emu_ddiv_unpacked<fp64emu_accuracy::high>(__x, __y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_ddiv (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y) { return impl::__nv_internal_fp64emu_ddiv_unpacked<fp64emu_accuracy::high>(__x, __y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_ddiv      (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y) { return impl::__nv_internal_fp64emu_ddiv_unpacked<fp64emu_accuracy::mid>(__x, __y); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_ddiv     (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y) { return impl::__nv_internal_fp64emu_ddiv_unpacked<fp64emu_accuracy::low>(__x, __y); }
#endif
#else
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ddiv_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ddiv_rz (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ddiv_ru (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ddiv_rd (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_ddiv_rn (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_ddiv_rn      (fpbits64_t x, fpbits64_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_ddiv_rn     (fpbits64_t x, fpbits64_t y);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_ddiv          (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_ddiv (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_ddiv      (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_ddiv     (fpbits64_unpacked_t x, fpbits64_unpacked_t y);
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_DIV_HPP__ (builtins)

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_DDIV_API_MERGED__)
#define __FPEMU_DDIV_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_ddiv_api.hpp)
// ============================================================================

    // Default API implementation
    template<fp64emu_accuracy _Acc> __FPEMU_HOST_DEVICE_DECL__ static fp64emu_t<_Acc> operator/ (const fp64emu_t<_Acc>& __x, 
                                                                        const fp64emu_t<_Acc>& __y)
    {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_high_ddiv_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)      { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_mid_ddiv_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)     { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_low_ddiv_rn(__x.bits, __y.bits)); }
        else                                             { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_ddiv_rn(__x.bits, __y.bits)); }
    } // operator /


    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __ddiv_rn (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_high_ddiv_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)     { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_low_ddiv_rn(__x.bits, __y.bits)); }
        else                                             { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_mid_ddiv_rn(__x.bits, __y.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __ddiv_rz (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_ddiv_rz(__x.bits, __y.bits)); }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __ddiv_ru (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_ddiv_ru(__x.bits, __y.bits)); }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __ddiv_rd (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_ddiv_rd(__x.bits, __y.bits)); } 

#if __FPEMU_UNPACKED__ == 1

    // Operator/ for unpacked division
    template<fp64emu_accuracy _Acc>
    __FPEMU_DEVICE_DECL__ static fp64emu_unpacked_t<_Acc> operator/ (const fp64emu_unpacked_t<_Acc>& __x, 
                                                                            const fp64emu_unpacked_t<_Acc>& __y)
    {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_high_ddiv(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)      { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_mid_ddiv(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_low_ddiv(__x.bits, __y.bits)); }
        else                                             { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_ddiv(__x.bits, __y.bits)); }
    } // operator/


    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<_Acc> __ddiv_rn (const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_high_ddiv(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)     { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_low_ddiv(__x.bits, __y.bits)); }
        else                                             { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_mid_ddiv(__x.bits, __y.bits)); }
    }


#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_DIV_HPP__
