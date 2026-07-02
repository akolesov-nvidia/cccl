//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_FMA_H
#define _CUDA___FP_FPEMU_IMPL_FMA_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_impl_fma.h
 * @brief Implementation of fused multiply-add operations (FMA & MAD) for FPEMU floating point emulation library
 *
 * This header provides the implementation of fused multiply-add operations for the FPEMU library.
 * It includes:
 *   - Fused multiply-add functions for different accuracy and range configurations 
 *   - Special case handling for NaN, inf, zero, etc
 *
 * The implementation is designed to work across both host and device code
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
extern "C" double fma(double __x, double __y, double __z);
#endif

namespace impl
{
    /**
     * @brief Pure FMA core operating on the unpacked representation.
     *
     * Consumes/produces fpbits64_unpacked_t exactly as produced by the universal
     * impl::__nv_internal_fp64emu_unpack and consumed by impl::__nv_internal_fp64emu_pack.
     * Inputs carry a normalized mantissa (implicit bit set, denormals normalized)
     * with inf/nan encoded in the exponent band (around the 0x00007ff0 / 0x0007ff00
     * magics), so the floating-point class is read from the exponent rather than a
     * separate field. The returned value is the pre-rounding intermediate: a 64-bit
     * mantissa with a sticky LSB and an exponent that may be <= 0 (subnormal) or in
     * the inf/nan band. Final rounding, subnormal shifting and inf/saturate are the
     * job of pack. Templated on the rounding mode (sign-of-zero, rd handling) and
     * the method (product accuracy via __mul_128 and range-based special handling).
     */
    template<fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_unpacked_t __nv_internal_fp64emu_fma_unpacked(fpbits64_unpacked_t __a,
                                                           fpbits64_unpacked_t __b,
                                                           fpbits64_unpacked_t __c)
    {
        constexpr fp64emu_accuracy   __acc_forced = fp64emu_accuracy::__FPEMU_FMA_METHOD__;
        constexpr fp64emu_accuracy   __acc_used   = (__acc_forced != fp64emu_accuracy::unset) ? __acc_forced : _Acc;
        // Unpacked cores always run on the fully-accurate full-range unpack/pack
        // boundary (method-independent): the inf/nan folds stay live and underflow
        // flows to the full pack (no FTZ, correct subnormal + min_normal round-up).

        // Inf/Nan exponent magics produced by the universal unpack.
        constexpr uint32_t __INF_EXP = 0x00007ff0u;
        constexpr int32_t  __NAN_EXP = 0x0007ff00;

        fpbits64_unpacked_t __r;
        __uint128_t __mantissa_ab;
        fpemu::__uint64x2_t __ab_res;

        // MUL START:
        // Compute mantissa_ab - the product of a and b in 128-bit
        fpemu::__uint32x2_t __a_32x2 = fpemu::bit_cast<fpemu::__uint32x2_t>(__a.mantissa);
        fpemu::__uint32x2_t __b_32x2 = fpemu::bit_cast<fpemu::__uint32x2_t>(__b.mantissa);
        __ab_res = fpemu::bit_cast<fpemu::__uint64x2_t>(fpemu::__mul_128<__acc_used>(__a_32x2, __b_32x2));

        __mantissa_ab = fpemu::bit_cast<__uint128_t>(__ab_res);
        fpemu::__uint32x4_t __mantissa_ab32 = fpemu::bit_cast<fpemu::__uint32x4_t>( __mantissa_ab );

        // Exponents/signs (read with explicit signedness: the public field is
        // uint32 but the core needs signed arithmetic for subnormal exponents).
        int32_t __exponent_ab = (int32_t)__a.exponent + (int32_t)__b.exponent - (int32_t)fpemu::BIAS;
        int32_t __exponent_c  = (int32_t)__c.exponent;
        int32_t __sign_ab     = (int32_t)(__a.sign ^ __b.sign);
        int32_t __sign_c      = (int32_t)__c.sign;

        // Compute exponent_ab_new - the exponent of the product of a and b
        int __mul_nzeros = __mantissa_ab32.hi.x[1] < 0x08000000;
        int32_t __exponent_ab_new = __exponent_ab - __mul_nzeros + 1;
        // Shift mantissa_ab
        __mantissa_ab = __mantissa_ab << (11 - EXTRA_BITS + __mul_nzeros);
        // Compute mantissa_c - the mantissa of c
        __uint128_t __mantissa_c = __c.mantissa;
        // Compute mantissa_r - the result of the product of a and b and c
        __uint128_t __mantissa_r;

        {
            // Check if a or b is inf and c is inf and sign_ab != sign_c then return NaN
            if ((__a.exponent == __INF_EXP || __b.exponent == __INF_EXP) && __c.exponent == __INF_EXP && __sign_ab != __sign_c)
            {
                __exponent_ab_new = __NAN_EXP;
            }
        }

        // Check if exponent_ab_new is INF_ZERO then return NaN
        if (__exponent_ab_new == (int32_t)fpemu::INF_ZERO) 
        { 
            __exponent_ab_new = __NAN_EXP;
        }

        //ADD START:
        // Compute exponent_r - the larger of exponent_ab_new and exponent_c
        int32_t __exponent_r = __max_fp64emu( __exponent_ab_new, __exponent_c );

        // Compute delta_a and delta_b for mantissas shift
        int32_t __delta_a = __exponent_r - __exponent_ab_new;
        int32_t __delta_b = __exponent_r - __exponent_c;

        #ifndef __CUDA_ARCH__
        __delta_a = (__delta_a>127) ? 127 : __delta_a;
        __delta_b = (__delta_b>127) ? 127 : __delta_b;
        #endif
        // Shift mantissas with jam only (SoftFloat shiftRightJam*); round at pack
        __mantissa_ab = fpemu::__shr_128_jam(__mantissa_ab, __delta_a);
        __mantissa_c = fpemu::__shr_128_jam(__mantissa_c << 64, __delta_b);

        // Add or subtract mantissas
        uint32_t __sign_r = __sign_ab;
        if (__sign_ab == __sign_c) 
        {
            __mantissa_r = __mantissa_ab + __mantissa_c;
        }
        else if (__mantissa_ab == __mantissa_c)
        {
            __mantissa_r = 0;
        }
        else if ((__mantissa_ab > __mantissa_c))
        {
            __mantissa_r = __mantissa_ab - __mantissa_c;
        } 
        else // mantissa_ab < mantissa_c
        {
            __sign_r = __sign_c;
            __mantissa_r = __mantissa_c - __mantissa_ab;
        }

        if (__mantissa_r == 0)
        {
            // Exact cancellation -> zero. IEEE-754 6.3 would make this -0 under
            // round-toward-negative (rd); that rounding-dependent zero sign is
            // intentionally NOT honored here (the core is rounding-independent), so
            // the zero is -0 only when both effective signs are negative.
            __sign_r = __sign_ab & __sign_c;
        }

        // Normalize mantissa_r
        // use reinterpret_cast to avoid slowdown from bit_cast
        uint64_t *__m = reinterpret_cast<uint64_t*>( &__mantissa_r );
        int __nzeros = (__m[1] == 0)? (fpemu::__internal_clzll(__m[0] + 64)):
                                (fpemu::__internal_clzll(__m[1] << 1));

        // Shift mantissa_r
        __mantissa_r = (__nzeros == 0)? (__mantissa_r >> 1): 
                                    (__mantissa_r << (__nzeros - 1)); 

        fpemu::__uint64x2_t __mantissa_r64 = fpemu::bit_cast<fpemu::__uint64x2_t>(__mantissa_r);

        // The result class (inf vs finite-overflow) is recoverable from the
        // exponent band by pack: genuine infinities inherit the huge exponent of
        // their inf operand, finite results never reach it. So no class field is
        // written here; the inf-inf -> NaN and inf*0 -> NaN cases were already
        // folded into exponent_ab_new (NAN_EXP) above.
        __r.sign     = __sign_r;
        // +1 matches the unified pack's "mask the implicit bit" convention
        // (pack reconstitutes via exp-1); the +1/-1 cancel so packed FMA is
        // bit-exact with the legacy add-convention packer.
        __r.exponent = static_cast<uint32_t>(__exponent_r - __nzeros + 1);
        __r.mantissa = __mantissa_r64.x[1] | (__mantissa_r64.x[0] != 0);

        // The unpacked core runs on the full-range boundary: underflow (and the rare
        // top-subnormal -> min_normal round-up) flows to the full pack, which has the
        // complete mantissa and resolves the correct subnormal / min_normal for every
        // rounding mode. No FTZ here, so no rounding-dependent fix-up is needed.

        return __r;
    } // __nv_internal_fp64emu_fma_unpacked

    template<fpemu::rounding    _Rm  = fpemu::rounding::def, 
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    __FPEMU_INTERNAL_DECL__
    fpbits64_t __nv_internal_fp64emu_fma (fpbits64_t __x, 
                                          fpbits64_t __y, 
                                          fpbits64_t __z)
    {
        // Forced parameters for the fused multiply-add operation
        constexpr fp64emu_accuracy   __acc_forced = fp64emu_accuracy::__FPEMU_FMA_METHOD__;
        constexpr fp64emu_accuracy   __acc_used   = (__acc_forced != fp64emu_accuracy::unset) ? __acc_forced : _Acc;

        {
                {
                    // FMA = pack(fma_unpacked(unpack(x), unpack(y), unpack(z))). The fma_unpacked
                    // core selects accurate/def/fast internally; the universal full-range
                    // unpack/pack are the shared prologue/epilogue (def/fast are full-range here).
                    fpbits64_unpacked_t __a = __nv_internal_fp64emu_unpack(__x);
                    fpbits64_unpacked_t __b = __nv_internal_fp64emu_unpack(__y);
                    fpbits64_unpacked_t __c = __nv_internal_fp64emu_unpack(__z);
                    fpbits64_unpacked_t __r = __nv_internal_fp64emu_fma_unpacked<__acc_used>(__a, __b, __c);
                    fpbits64_t __result = __nv_internal_fp64emu_pack<_Rm>(__r);

                    if constexpr (_Rm == fpemu::rounding::rd)
                    {
                        // Exact cancellation (a*b + c == 0 with opposite effective signs)
                        // must yield -0 under round-toward-negative. The rounding-independent
                        // core packs an exact zero to +0 (r.mantissa == 0); a misaligned
                        // remainder can also surface as a tiny artifact. Both map to -0 here.
                        // A genuine underflow keeps a nonzero core mantissa and stays +0.
                        const bool __opposite_signs = ((__a.sign ^ __b.sign) != __c.sign);
                        const bool __exact_zero      = (__r.mantissa == 0) && ((__result << 1) == 0);
                        const bool __tiny_artifact   = (__result == UINT64_C(0x0000000100000000));
                        if (__opposite_signs && (__exact_zero || __tiny_artifact))
                        {
                            fpbits64_unpacked_t __zneg;
                            __zneg.sign     = 1U << 31;
                            __zneg.exponent = 0;
                            __zneg.mantissa = 0;
                            __result = __nv_internal_fp64emu_pack<_Rm>(__zneg);
                        }
                    }
                    return __result;
                }
        }
    } // __nv_internal_fp64emu_fma

} // namespace impl

// ============================================================================
// Builtin declarations/implementations for FMA operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rn          (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rn, fp64emu_accuracy::high>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rz          (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rz, fp64emu_accuracy::high>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_ru          (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::ru, fp64emu_accuracy::high>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rd          (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rd, fp64emu_accuracy::high>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_fma_rn (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rn, fp64emu_accuracy::high>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rn      (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rn, fp64emu_accuracy::mid>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rz      (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rz, fp64emu_accuracy::mid>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_ru      (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::ru, fp64emu_accuracy::mid>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rd      (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rd, fp64emu_accuracy::mid>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rn      (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rn, fp64emu_accuracy::low>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rz      (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rz, fp64emu_accuracy::low>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_ru      (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::ru, fp64emu_accuracy::low>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rd      (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) { return impl::__nv_internal_fp64emu_fma<fpemu::rounding::rd, fp64emu_accuracy::low>(__x, __y, __z); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_fma          (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y, fpbits64_unpacked_t __z) { return impl::__nv_internal_fp64emu_fma_unpacked<fp64emu_accuracy::high>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_fma     (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y, fpbits64_unpacked_t __z) { return impl::__nv_internal_fp64emu_fma_unpacked<fp64emu_accuracy::high>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_fma      (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y, fpbits64_unpacked_t __z) { return impl::__nv_internal_fp64emu_fma_unpacked<fp64emu_accuracy::mid>(__x, __y, __z); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_fma      (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y, fpbits64_unpacked_t __z) { return impl::__nv_internal_fp64emu_fma_unpacked<fp64emu_accuracy::low>(__x, __y, __z); }
#endif
#else
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rn          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rz          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_ru          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_fma_rd          (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_high_fma_rn     (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rn      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rz      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_ru      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_mid_fma_rd      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rn      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rz      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_ru      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_low_fma_rd      (fpbits64_t x, fpbits64_t y, fpbits64_t z);
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_fma      (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_fma (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_fma  (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_fma  (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z);
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_FMA_HPP__

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_FMA_API_MERGED__)
#define __FPEMU_FMA_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_fma_api.hpp)
// ============================================================================


    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> fma (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y, const fp64emu_t<_Acc>& __z) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_high_fma_rn(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_low_fma_rn(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_mid_fma_rn(__x.bits, __y.bits, __z.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __fma_rn (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y, const fp64emu_t<_Acc>& __z) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_high_fma_rn(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_low_fma_rn(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_mid_fma_rn(__x.bits, __y.bits, __z.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __fma_rz (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y, const fp64emu_t<_Acc>& __z) {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_fma_rz(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_mid_fma_rz(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_low_fma_rz(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_fma_rz(__x.bits, __y.bits, __z.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __fma_ru (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y, const fp64emu_t<_Acc>& __z) {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_fma_ru(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_mid_fma_ru(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_low_fma_ru(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_fma_ru(__x.bits, __y.bits, __z.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_t<_Acc> __fma_rd (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y, const fp64emu_t<_Acc>& __z) {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_fma_rd(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_mid_fma_rd(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_low_fma_rd(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_fma_rd(__x.bits, __y.bits, __z.bits)); }
    }

#if __FPEMU_UNPACKED__ == 1


    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<_Acc> fma (const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y, const fp64emu_unpacked_t<_Acc>& __z) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_high_fma(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_low_fma(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_mid_fma(__x.bits, __y.bits, __z.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    __FPEMU_API_DECL__ fp64emu_unpacked_t<_Acc> __fma_rn (const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y, const fp64emu_unpacked_t<_Acc>& __z) { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_high_fma(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_low_fma(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_mid_fma(__x.bits, __y.bits, __z.bits)); }
    }


#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_FMA_API_MERGED__
