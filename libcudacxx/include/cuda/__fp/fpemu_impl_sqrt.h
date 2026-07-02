//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_SQRT_H
#define _CUDA___FP_FPEMU_IMPL_SQRT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_dsqrt_impl.hpp
 * @brief Implementation of double-precision square root operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of double-precision square root operations for the FPEMU library.
 * It includes:
 *
 * - Square root functions for fp64emu_t
 * - Square root operators for fp64emu_t
 * - Square root functions to other types
 *
 * The square root functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

#include <cuda/__fp/fpemu_common.h>
#include <cuda/__fp/fpemu_impl_utils.h>
#include <cuda/__fp/fpemu_impl_unpack.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

#if !(defined (__CUDA_ARCH__))
extern "C" double sqrt(double __x);
extern "C" float  sqrtf(float __x);   // host seed for the reciprocal-sqrt builtin
#endif

namespace impl
{
    // ========================================================================
    // Native fp64 square root.
    //
    // Split sign/exp/mantissa, normalize subnormals, form a 32-bit reciprocal
    // square root of the significand, then refine with fixed-point integer
    // remainder arithmetic and round/pack. The reciprocal-sqrt seed comes from
    // the fp32 rsqrt builtin (rsqrt.approx on device, 1/sqrtf on host) and is
    // refined by a single Newton step plus a trim that guarantees a strict
    // underestimate, which the remainder refinement needs for correctly-rounded
    // results.
    // ========================================================================

    /// @brief Approximation of 2^47 / sqrt(a / 2^odd_exp), a in [2^31, 2^32),
    ///        in [2^31, 2^32). Seeded by the fp32 rsqrt builtin, refined by one
    ///        Newton step, then trimmed to a strict underestimate so that the
    ///        derived root stays a lower bound (keeps the remainder unsigned).
    __FPEMU_INTERNAL_DECL__ uint32_t __nv_internal_fp64emu_sqrt_recip_sqrt32 (uint32_t __odd_exp, 
                                                                              uint32_t __a)
    {
        // Seed: r ~ 2^47 * rsqrt(a / 2^odd_exp). Halving the radicand for the
        // odd-exponent case folds the sqrt(2) factor into the same 2^47 scale.
        float    __af = __odd_exp ? (float)__a * 0.5f : (float)__a;
    #if defined(__CUDA_ARCH__)
        float    __rf;
        asm ("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(__rf) : "f"(__af));
    #else
        float    __rf = 1.0f / sqrtf(__af);                      // host fallback
    #endif
        uint64_t __r  = (uint64_t)(__rf * 140737488355328.0f);   // * 2^47

        if (__r < 0x80000000ULL) __r = 0x80000000ULL;
        if (__r > 0xFFFFFFFFULL) __r = 0xFFFFFFFFULL;

        // One Newton step (reciprocal-sqrt): r <- r*(3K - a*r^2)/(2K), K = 2^(94+odd).
        uint64_t __r2      = __r * __r;                              // < 2^64
        uint64_t __a_r2_hi = __nv_internal_fp64emu_mulhi64((uint64_t)__a, __r2);   // a*r^2 >> 64
        uint64_t __a_r2_lo = (uint64_t)__a * __r2;                 // low 64 bits (wraps)
        uint64_t __k3_hi   = 3ULL << (30 + __odd_exp);            // (3 * 2^(94+odd)) >> 64
        uint64_t __t_lo    = 0ULL - __a_r2_lo;
        uint64_t __borrow  = (__a_r2_lo != 0) ? 1u : 0u;
        uint64_t __t_hi    = __k3_hi - __a_r2_hi - __borrow;          // t = 3K - a*r^2 (128-bit)
        // r' = (r * t) >> (95 + odd); pre-shift t by 33 so the product fits 128 bits.
        uint64_t __t_s    = (__t_hi << 31) | (__t_lo >> 33);
        uint64_t __pr_hi  = __nv_internal_fp64emu_mulhi64(__r, __t_s);
        uint64_t __pr_lo  = __r * __t_s;
        __r = (__pr_hi << (2 - __odd_exp)) | (__pr_lo >> (62 + __odd_exp));
        if (__r < 0x80000000ULL) __r = 0x80000000ULL;
        if (__r > 0xFFFFFFFFULL) __r = 0xFFFFFFFFULL;

        // Trim to a strict underestimate of 2^47*2^(odd/2)/sqrt(a): a*r^2 <= 2^(94+odd).
        while (__r > 0x80000000ULL)
        {
            uint64_t __rr2 = __r * __r;
            uint64_t __hi  = __nv_internal_fp64emu_mulhi64((uint64_t)__a, __rr2);
            uint64_t __lo  = (uint64_t)__a * __rr2;
            uint64_t __thr = 1ULL << (30 + __odd_exp);
            if ((__hi > __thr) || (__hi == __thr && __lo != 0)) --__r; else break;
        }

        return (uint32_t)__r;
    } // __nv_internal_fp64emu_sqrt_recip_sqrt32

    // Forward declaration: the unpacked sqrt core is defined below, but the packed
    // wrapper references it for the packed-via-unpacked (testing) path.
    template<fp64emu_accuracy _Acc>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_dsqrt_unpacked(fpbits64_unpacked_t __x);

    template<fpemu::rounding    _Rm  = fpemu::rounding::def, 
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__ fpbits64_t __nv_internal_fp64emu_dsqrt(fpbits64_t __x)
    {
    #if (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
        // Packed-via-unpacked (testing): pack(dsqrt_unpacked(unpack(x))). The
        // dsqrt_unpacked core handles special operands and method selection; the
        // universal unpack/pack are the shared prologue/epilogue. Rounding is
        // applied only at pack, preserving the packed builtins' per-mode behavior.
        {
            fpbits64_unpacked_t __a = __nv_internal_fp64emu_unpack(__x);
            fpbits64_unpacked_t __r = __nv_internal_fp64emu_dsqrt_unpacked<_Acc>(__a);
            return __nv_internal_fp64emu_pack<_Rm>(__r);
        }
    #else

        const uint64_t __ui64_x = (uint64_t)__x;

        bool           __sign_x = (__ui64_x >> 63) != 0;
        int32_t        __exp_x  = (int32_t)((__ui64_x >> 52) & 0x7FF);
        uint64_t       __mant_x = __ui64_x & __FPEMU_MANT_64__;

        // -------- special operands (NaN / Inf / negative / zero) --------
        if (__exp_x == 0x7FF)
        {
            if (__mant_x)  return (fpbits64_t)(__ui64_x | __FPEMU_QNAN_BIT_64__);   // NaN -> quiet NaN
            if (!__sign_x) return (fpbits64_t)__ui64_x;                            // +inf -> +inf
            return (fpbits64_t)__FPEMU_DEFNAN_64__;                          // sqrt(-inf) -> NaN
        }
        if (__sign_x)
        {
            if (!(__exp_x | (int32_t)(__mant_x != 0))) return (fpbits64_t)__ui64_x;   // -0 -> -0
            return (fpbits64_t)__FPEMU_DEFNAN_64__;                          // sqrt(negative) -> NaN
        }
        if (!__exp_x)
        {
            if (!__mant_x) return (fpbits64_t)__ui64_x;                             // +0 -> +0

            int __mant_shft = fpemu::__internal_clzll((int64_t)__mant_x) - 11;           // normalize subnormal

            __exp_x   = 1 - __mant_shft;
            __mant_x  = __mant_x << __mant_shft;
        }

        // -------- fixed-point reciprocal-sqrt root --------
        int32_t __exp_z = ((__exp_x - 0x3FF) >> 1) + 0x3FE;
        __exp_x        &= 1;
        __mant_x       |= __FPEMU_HIDDEN_64__;

        uint32_t __mant32_x  = (uint32_t)(__mant_x >> 21);
        uint32_t __rcp32     = __nv_internal_fp64emu_sqrt_recip_sqrt32((uint32_t)__exp_x, __mant32_x);
        uint32_t __mant32_z  = (uint32_t)(((uint64_t)__mant32_x * __rcp32) >> 32);

        if (__exp_x) { __mant_x <<= 8; __mant32_z >>= 1; } else { __mant_x <<= 9; }

        uint64_t __rem64    = __mant_x - (uint64_t)__mant32_z * __mant32_z;
        uint32_t __q32      = (uint32_t)(((uint32_t)(__rem64 >> 2) * (uint64_t)__rcp32) >> 32);
        // form mantissa: (1 << 52) + (mant32_z << 21) + (q32 << 3)
        uint64_t __mant64_z = ((uint64_t)__mant32_z << 32 | (1u << 5)) + ((uint64_t)__q32 << 3);

        // Refine if the root is close to a rounding boundary (exact remainder).
        if ((__mant64_z & 0x1FF) < 0x22)
        {
            __mant64_z               &= ~(uint64_t)0x3F;
            uint64_t __mant64_z_shftd = __mant64_z >> 6;
            __rem64                   = (__mant_x << 52) - __mant64_z_shftd * __mant64_z_shftd;

            if (__rem64 & __FPEMU_SIGN_64__) { --__mant64_z; }
            else { if (__rem64) __mant64_z |= 1; }
        }

        return __nv_internal_fp64emu_round_pack<_Rm>(false, __exp_z, __mant64_z);
    #endif // __FPEMU_PACKED_VIA_UNPACKED__
    } // __nv_internal_fp64emu_dsqrt

    // Unpacked operation
    template<fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_dsqrt_unpacked(fpbits64_unpacked_t __x)
    {
        // ---- True unpacked square root --------------------------------------
        // Operates directly on the fully-accurate unpacked operand (no operand
        // pack, no legacy packed kernel). The full unpack already normalized
        // denormals and encoded inf/nan, so the significand is mantissa>>EXTRA_BITS
        // (implicit bit at 52) and the exponent is the IEEE-biased value (same as
        // the packed core's post-normalization exponent, so the parity bit and the
        // halved result exponent are identical). The proven fixed-point
        // reciprocal-sqrt root is computed exactly as the packed core, then
        // expressed on the universal unpacked scale (implicit bit at 61, sticky LSB)
        // so the full pack performs the single correctly-rounded finalization.
        // sqrt is correctly rounded for every accuracy level (no mid/low variant).
        constexpr int32_t __NAN_EXP = 0x0007ff00;
        constexpr int32_t __INF_EXP = 0x00007ff0;

        const int32_t __exp_x  = (int32_t)__x.exponent;
        const bool    __sign_x = (__x.sign != 0);
        const bool    __zero_x = (__x.mantissa == 0);

        // Special operands (canonical packed result, then unpack -- rare path).
        if (__exp_x == __NAN_EXP)
            return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_DEFNAN_64__);
        if (__exp_x == __INF_EXP)
        {
            if (!__sign_x) return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_INF_64__); // +inf -> +inf
            return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_DEFNAN_64__);           // sqrt(-inf) -> NaN
        }
        if (__sign_x)
        {
            if (__zero_x) return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_SIGN_64__);  // -0 -> -0
            return __nv_internal_fp64emu_unpack((fpbits64_t)__FPEMU_DEFNAN_64__);            // sqrt(negative) -> NaN
        }
        if (__zero_x)
            return __nv_internal_fp64emu_unpack((fpbits64_t)0);                              // +0 -> +0

        // ---- finite positive : fixed-point reciprocal-sqrt root -------------
        int32_t  __exp_z  = ((__exp_x - 0x3FF) >> 1) + 0x3FE;
        int32_t  __odd    = __exp_x & 1;                    // exponent parity
        uint64_t __mant_x = __x.mantissa >> EXTRA_BITS;     // 53-bit significand, implicit bit at 52

        uint32_t __mant32_x = (uint32_t)(__mant_x >> 21);
        uint32_t __rcp32    = __nv_internal_fp64emu_sqrt_recip_sqrt32((uint32_t)__odd, __mant32_x);
        uint32_t __mant32_z = (uint32_t)(((uint64_t)__mant32_x * __rcp32) >> 32);

        if (__odd) { __mant_x <<= 8; __mant32_z >>= 1; } else { __mant_x <<= 9; }

        uint64_t __rem64    = __mant_x - (uint64_t)__mant32_z * __mant32_z;
        uint32_t __q32      = (uint32_t)(((uint32_t)(__rem64 >> 2) * (uint64_t)__rcp32) >> 32);
        uint64_t __mant64_z = ((uint64_t)__mant32_z << 32 | (1u << 5)) + ((uint64_t)__q32 << 3);

        // Refine if the root is close to a rounding boundary (exact remainder).
        if ((__mant64_z & 0x1FF) < 0x22)
        {
            __mant64_z               &= ~(uint64_t)0x3F;
            uint64_t __mant64_z_shftd = __mant64_z >> 6;
            __rem64                   = (__mant_x << 52) - __mant64_z_shftd * __mant64_z_shftd;
            if (__rem64 & __FPEMU_SIGN_64__) { --__mant64_z; }
            else { if (__rem64) __mant64_z |= 1; }
        }

        // Same conversion as the unpacked divide: leading bit 62 -> 61 (sticky
        // preserved), exponent biased-1 -> IEEE-biased; full pack rounds.
        fpbits64_unpacked_t __r;
        __r.sign     = 0u;                                // sqrt result is non-negative
        __r.exponent = (uint32_t)(__exp_z + 1);
        __r.mantissa = (__mant64_z >> 1) | (__mant64_z & 1);
        return __r;
    } // __nv_internal_fp64emu_dsqrt_unpacked

} // namespace impl

// ============================================================================
// Builtin declarations/implementations for sqrt operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rn (fpbits64_t __x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rn, fp64emu_accuracy::high>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rz (fpbits64_t __x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rz, fp64emu_accuracy::high>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_ru (fpbits64_t __x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::ru, fp64emu_accuracy::high>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rd (fpbits64_t __x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rd, fp64emu_accuracy::high>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dsqrt_rn (fpbits64_t __x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rn, fp64emu_accuracy::high>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsqrt_rn  (fpbits64_t __x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rn, fp64emu_accuracy::mid>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsqrt_rn  (fpbits64_t __x) { return impl::__nv_internal_fp64emu_dsqrt<fpemu::rounding::rn, fp64emu_accuracy::low>(__x); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dsqrt      (fpbits64_unpacked_t __x) { return impl::__nv_internal_fp64emu_dsqrt_unpacked<fp64emu_accuracy::high>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dsqrt (fpbits64_unpacked_t __x) { return impl::__nv_internal_fp64emu_dsqrt_unpacked<fp64emu_accuracy::high>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dsqrt  (fpbits64_unpacked_t __x) { return impl::__nv_internal_fp64emu_dsqrt_unpacked<fp64emu_accuracy::mid>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dsqrt  (fpbits64_unpacked_t __x) { return impl::__nv_internal_fp64emu_dsqrt_unpacked<fp64emu_accuracy::low>(__x); }
#endif
#else
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rz (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_ru (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_dsqrt_rd (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_dsqrt_rn (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_dsqrt_rn  (fpbits64_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_dsqrt_rn  (fpbits64_t x);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_dsqrt      (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_dsqrt (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_dsqrt  (fpbits64_unpacked_t x);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_dsqrt  (fpbits64_unpacked_t x);
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_SQRT_HPP__ (builtins)

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_DSQRT_API_MERGED__)
#define __FPEMU_DSQRT_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_dsqrt_api.hpp)
// ============================================================================


    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> sqrt (const fp64emu_t<_Acc>& __x) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_high_dsqrt_rn(__x.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_low_dsqrt_rn(__x.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_mid_dsqrt_rn(__x.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __dsqrt_rn (const fp64emu_t<_Acc>& __x) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_high_dsqrt_rn(__x.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_low_dsqrt_rn(__x.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_mid_dsqrt_rn(__x.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __dsqrt_rz (const fp64emu_t<_Acc>& __x) { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_dsqrt_rz(__x.bits)); }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __dsqrt_ru (const fp64emu_t<_Acc>& __x) { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_dsqrt_ru(__x.bits)); }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __dsqrt_rd (const fp64emu_t<_Acc>& __x) { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_dsqrt_rd(__x.bits)); } 

#if __FPEMU_UNPACKED__ == 1


    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<_Acc> sqrt (const fp64emu_unpacked_t<_Acc>& __x) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_high_dsqrt(__x.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_low_dsqrt(__x.bits)); }
        else                                               { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dsqrt(__x.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<_Acc> __dsqrt_rn (const fp64emu_unpacked_t<_Acc>& __x) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_high_dsqrt(__x.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_low_dsqrt(__x.bits)); }
        else                                               { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_mid_dsqrt(__x.bits)); }
    }


#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_SQRT_HPP__
