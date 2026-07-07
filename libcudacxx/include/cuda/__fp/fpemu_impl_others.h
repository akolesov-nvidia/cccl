//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_OTHERS_H
#define _CUDA___FP_FPEMU_IMPL_OTHERS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_impl_others.h
 * @brief Implementation of MAD, DOT, and CMUL operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of other operations for the FPEMU library.
 * It includes:
 *   - MAD (Multiply-Add with intermediate rounding) functions for different accuracy and range configurations 
 *   - DOT (dot product) functions
 *   - CMUL (complex multiply) functions
 *   - Special case handling for NaN, inf, zero, etc
 *
 * The implementation is designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

 #define _CCCL_FP64EMU_USE_OPT_MAD_UNPACKED  1
 #define _CCCL_FP64EMU_USE_OPT_DOT_UNPACKED  1
 #define _CCCL_FP64EMU_USE_OPT_CMUL_UNPACKED 1

#include <cuda/__fp/fpemu_common.h>
#include <cuda/__fp/fpemu_impl_utils.h>
#include <cuda/__fp/fpemu_impl_unpack.h>
#include <cuda/__fp/fpemu_impl_mul.h>
#include <cuda/__fp/fpemu_impl_add.h>
#include <cuda/__fp/fpemu_impl_sub.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


    // MAD unpacked implementation
    template<fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API
    fpbits64_unpacked_t __internal_fp64emu_mad_unpacked (fpbits64_unpacked_t __x, 
                                                            fpbits64_unpacked_t __y, 
                                                            fpbits64_unpacked_t __z) noexcept
    {
        return __internal_fp64emu_dadd_unpacked<_Acc>(
               __internal_fp64emu_dmul_unpacked<_Acc>(__x, __y), __z);
    }

    // DOT unpacked implementation
    template<fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API
    fpbits64_unpacked_t __internal_fp64emu_dot_unpacked (fpbits64_unpacked_t __x1, 
                                                            fpbits64_unpacked_t __y1, 
                                                            fpbits64_unpacked_t __x2,
                                                            fpbits64_unpacked_t __y2) noexcept
    {
        return __internal_fp64emu_dadd_unpacked<_Acc>(
               __internal_fp64emu_dmul_unpacked<_Acc>(__x1, __x2), 
               __internal_fp64emu_dmul_unpacked<_Acc>(__y1, __y2));
    }

    // CMPLX MUL unpacked implementation
    // (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
    template<fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API
    void __internal_fp64emu_cmul_unpacked (fpbits64_unpacked_t  __x_re, 
                                              fpbits64_unpacked_t  __x_im, 
                                              fpbits64_unpacked_t  __y_re,
                                              fpbits64_unpacked_t  __y_im,
                                              fpbits64_unpacked_t& __r_re,
                                              fpbits64_unpacked_t& __r_im) noexcept
    {
        __r_re = __internal_fp64emu_dsub_unpacked<_Acc>(__internal_fp64emu_dmul_unpacked<_Acc>(__x_re, __y_re), 
                                                                    __internal_fp64emu_dmul_unpacked<_Acc>(__x_im, __y_im));
        __r_im = __internal_fp64emu_dadd_unpacked<_Acc>(__internal_fp64emu_dmul_unpacked<_Acc>(__x_re, __y_im), 
                                                                    __internal_fp64emu_dmul_unpacked<_Acc>(__x_im, __y_re));
        return;
    }

    // MAD implementation
    template<__fpemu_rounding _Rm   = __fpemu_rounding::def, 
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API
    fpbits64_t __internal_fp64emu_mad (fpbits64_t __x, 
                                        fpbits64_t __y, 
                                        fpbits64_t __z) noexcept
    {
        if constexpr (_Acc == fp64emu_accuracy::mid)
        {
            fpbits64_unpacked_t __x_unpacked = __internal_fp64emu_unpack(__x);
            fpbits64_unpacked_t __y_unpacked = __internal_fp64emu_unpack(__y);
            fpbits64_unpacked_t __z_unpacked = __internal_fp64emu_unpack(__z);

            fpbits64_unpacked_t __r_unpacked = __internal_fp64emu_mad_unpacked<_Acc>(__x_unpacked, 
                                                                                                 __y_unpacked, 
                                                                                                 __z_unpacked);
            return __internal_fp64emu_pack<_Rm>(__r_unpacked);
        }
        else
        {
            return  __internal_fp64emu_dadd<_Rm, _Acc>(
                    __internal_fp64emu_dmul<_Rm, _Acc>(__x, __y), __z);
        }
    }

    // DOT implementation
    template<__fpemu_rounding _Rm   = __fpemu_rounding::def, 
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API
    fpbits64_t __internal_fp64emu_dot (fpbits64_t __x1, 
                                          fpbits64_t __y1, 
                                          fpbits64_t __x2,
                                          fpbits64_t __y2) noexcept
    {
        if constexpr (_Acc == fp64emu_accuracy::mid)
        {
            fpbits64_unpacked_t __x1_unpacked = __internal_fp64emu_unpack(__x1);
            fpbits64_unpacked_t __y1_unpacked = __internal_fp64emu_unpack(__y1);
            fpbits64_unpacked_t __x2_unpacked = __internal_fp64emu_unpack(__x2);
            fpbits64_unpacked_t __y2_unpacked = __internal_fp64emu_unpack(__y2);

            fpbits64_unpacked_t __r_unpacked = __internal_fp64emu_dot_unpacked<_Acc>(__x1_unpacked, 
                                                                                                 __y1_unpacked, 
                                                                                                 __x2_unpacked,
                                                                                                 __y2_unpacked);
            fpbits64_t __r = __internal_fp64emu_pack<_Rm>(__r_unpacked);

            return __r;
        }
        else
        {
            fpbits64_t __r = __internal_fp64emu_dadd<_Rm, _Acc>(
                           __internal_fp64emu_dmul<_Rm, _Acc>(__x1, __x2),
                           __internal_fp64emu_dmul<_Rm, _Acc>(__y1, __y2));
            return __r;
        }
    }

    // CMUL implementation
    template<__fpemu_rounding _Rm   = __fpemu_rounding::def, 
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API
    void __internal_fp64emu_cmul (fpbits64_t  __x_re, 
                                     fpbits64_t  __x_im, 
                                     fpbits64_t  __y_re, 
                                     fpbits64_t  __y_im,
                                     fpbits64_t& __r_re,
                                     fpbits64_t& __r_im) noexcept
    {
        if constexpr (_Acc == fp64emu_accuracy::mid)
        {
            fpbits64_unpacked_t __x_re_unpacked = __internal_fp64emu_unpack(__x_re);
            fpbits64_unpacked_t __y_re_unpacked = __internal_fp64emu_unpack(__y_re);
            fpbits64_unpacked_t __x_im_unpacked = __internal_fp64emu_unpack(__x_im);
            fpbits64_unpacked_t __y_im_unpacked = __internal_fp64emu_unpack(__y_im);
            fpbits64_unpacked_t __r_re_unpacked;
            fpbits64_unpacked_t __r_im_unpacked;

            __internal_fp64emu_cmul_unpacked<_Acc>(__x_re_unpacked,
                                                                 __x_im_unpacked,
                                                                 __y_re_unpacked, 
                                                                 __y_im_unpacked,
                                                                 __r_re_unpacked,
                                                                 __r_im_unpacked);

            __r_re = __internal_fp64emu_pack<_Rm>(__r_re_unpacked);
            __r_im = __internal_fp64emu_pack<_Rm>(__r_im_unpacked);

            return;
        }
        else
        {
            fpbits64_t __r_re_y_re = __internal_fp64emu_dmul<_Rm, _Acc>(__x_re, __y_re);
            fpbits64_t __r_im_y_im = __internal_fp64emu_dmul<_Rm, _Acc>(__x_im, __y_im);
            fpbits64_t __r_re_y_im = __internal_fp64emu_dmul<_Rm, _Acc>(__x_re, __y_im);
            fpbits64_t __r_im_y_re = __internal_fp64emu_dmul<_Rm, _Acc>(__x_im, __y_re);

            __r_re = __internal_fp64emu_dsub<_Rm, _Acc>(__r_re_y_re, __r_im_y_im);
            __r_im = __internal_fp64emu_dadd<_Rm, _Acc>(__r_re_y_im, __r_im_y_re);

            return;
        }
    }

    _CCCL_TRIVIAL_API fpbits64_unpacked_t __internal_fp64emu_neg_unpacked (fpbits64_unpacked_t __x) noexcept 
    { 
       __x.sign = __invert_msb(__x.sign);
       return __x;
    }

    _CCCL_TRIVIAL_API fpbits64_t __internal_fp64emu_neg (fpbits64_t __x) noexcept 
    { 
        __uint32x2_t __t = __fpemu_bit_cast<__uint32x2_t>(__x);
        __t.x[1]              = __invert_msb(__t.x[1]);
        __x                   = __fpemu_bit_cast<uint64_t>(__t);
        return              __x;
    }



// ============================================================================
// Builtin declarations/implementations for MAD, DOT, CMUL, NEG operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)

// mad (packed)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mad_rn      (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) noexcept { return __internal_fp64emu_mad<__fpemu_rounding::rn, fp64emu_accuracy::high>(__x, __y, __z); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_high_mad_rn (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) noexcept { return __internal_fp64emu_mad<__fpemu_rounding::rn, fp64emu_accuracy::high>(__x, __y, __z); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_mad_rn  (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) noexcept { return __internal_fp64emu_mad<__fpemu_rounding::rn, fp64emu_accuracy::mid>(__x, __y, __z); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_mad_rn  (fpbits64_t __x, fpbits64_t __y, fpbits64_t __z) noexcept { return __internal_fp64emu_mad<__fpemu_rounding::rn, fp64emu_accuracy::low>(__x, __y, __z); }

// dot (packed)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dot_rn      (fpbits64_t __x1, fpbits64_t __y1, fpbits64_t __x2, fpbits64_t __y2) noexcept { return __internal_fp64emu_dot<__fpemu_rounding::rn, fp64emu_accuracy::high>(__x1, __y1, __x2, __y2); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_high_dot_rn (fpbits64_t __x1, fpbits64_t __y1, fpbits64_t __x2, fpbits64_t __y2) noexcept { return __internal_fp64emu_dot<__fpemu_rounding::rn, fp64emu_accuracy::high>(__x1, __y1, __x2, __y2); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dot_rn  (fpbits64_t __x1, fpbits64_t __y1, fpbits64_t __x2, fpbits64_t __y2) noexcept { return __internal_fp64emu_dot<__fpemu_rounding::rn, fp64emu_accuracy::mid>(__x1, __y1, __x2, __y2); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dot_rn  (fpbits64_t __x1, fpbits64_t __y1, fpbits64_t __x2, fpbits64_t __y2) noexcept { return __internal_fp64emu_dot<__fpemu_rounding::rn, fp64emu_accuracy::low>(__x1, __y1, __x2, __y2); }

// cmul (packed)
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_cmul_rn      (fpbits64_t __x_re, fpbits64_t __x_im, fpbits64_t __y_re, fpbits64_t __y_im, fpbits64_t& __r_re, fpbits64_t& __r_im) noexcept { __internal_fp64emu_cmul<__fpemu_rounding::rn, fp64emu_accuracy::mid>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im); }
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_high_cmul_rn (fpbits64_t __x_re, fpbits64_t __x_im, fpbits64_t __y_re, fpbits64_t __y_im, fpbits64_t& __r_re, fpbits64_t& __r_im) noexcept { __internal_fp64emu_cmul<__fpemu_rounding::rn, fp64emu_accuracy::high>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im); }
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_mid_cmul_rn  (fpbits64_t __x_re, fpbits64_t __x_im, fpbits64_t __y_re, fpbits64_t __y_im, fpbits64_t& __r_re, fpbits64_t& __r_im) noexcept { __internal_fp64emu_cmul<__fpemu_rounding::rn, fp64emu_accuracy::mid>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im); }
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_low_cmul_rn  (fpbits64_t __x_re, fpbits64_t __x_im, fpbits64_t __y_re, fpbits64_t __y_im, fpbits64_t& __r_re, fpbits64_t& __r_im) noexcept { __internal_fp64emu_cmul<__fpemu_rounding::rn, fp64emu_accuracy::low>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im); }

// neg (packed)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_neg (fpbits64_t __x) noexcept { return __internal_fp64emu_neg(__x); }

// mad (unpacked)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_mad      (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y, fpbits64_unpacked_t __z) noexcept { return __internal_fp64emu_mad_unpacked<fp64emu_accuracy::mid>(__x, __y, __z); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_high_mad (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y, fpbits64_unpacked_t __z) noexcept { return __internal_fp64emu_mad_unpacked<fp64emu_accuracy::high>(__x, __y, __z); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_mid_mad  (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y, fpbits64_unpacked_t __z) noexcept { return __internal_fp64emu_mad_unpacked<fp64emu_accuracy::mid>(__x, __y, __z); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_low_mad  (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y, fpbits64_unpacked_t __z) noexcept { return __internal_fp64emu_mad_unpacked<fp64emu_accuracy::low>(__x, __y, __z); }

// dot (unpacked)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_dot      (fpbits64_unpacked_t __x1, fpbits64_unpacked_t __y1, fpbits64_unpacked_t __x2, fpbits64_unpacked_t __y2) noexcept { return __internal_fp64emu_dot_unpacked<fp64emu_accuracy::mid>(__x1, __y1, __x2, __y2); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_high_dot (fpbits64_unpacked_t __x1, fpbits64_unpacked_t __y1, fpbits64_unpacked_t __x2, fpbits64_unpacked_t __y2) noexcept { return __internal_fp64emu_dot_unpacked<fp64emu_accuracy::high>(__x1, __y1, __x2, __y2); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_mid_dot  (fpbits64_unpacked_t __x1, fpbits64_unpacked_t __y1, fpbits64_unpacked_t __x2, fpbits64_unpacked_t __y2) noexcept { return __internal_fp64emu_dot_unpacked<fp64emu_accuracy::mid>(__x1, __y1, __x2, __y2); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_low_dot  (fpbits64_unpacked_t __x1, fpbits64_unpacked_t __y1, fpbits64_unpacked_t __x2, fpbits64_unpacked_t __y2) noexcept { return __internal_fp64emu_dot_unpacked<fp64emu_accuracy::low>(__x1, __y1, __x2, __y2); }

// cmul (unpacked)
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_cmul      (fpbits64_unpacked_t __x_re, fpbits64_unpacked_t __x_im, fpbits64_unpacked_t __y_re, fpbits64_unpacked_t __y_im, fpbits64_unpacked_t& __r_re, fpbits64_unpacked_t& __r_im) noexcept { __internal_fp64emu_cmul_unpacked<fp64emu_accuracy::mid>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im); }
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_high_cmul (fpbits64_unpacked_t __x_re, fpbits64_unpacked_t __x_im, fpbits64_unpacked_t __y_re, fpbits64_unpacked_t __y_im, fpbits64_unpacked_t& __r_re, fpbits64_unpacked_t& __r_im) noexcept { __internal_fp64emu_cmul_unpacked<fp64emu_accuracy::high>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im); }
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_mid_cmul  (fpbits64_unpacked_t __x_re, fpbits64_unpacked_t __x_im, fpbits64_unpacked_t __y_re, fpbits64_unpacked_t __y_im, fpbits64_unpacked_t& __r_re, fpbits64_unpacked_t& __r_im) noexcept { __internal_fp64emu_cmul_unpacked<fp64emu_accuracy::mid>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im); }
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_low_cmul  (fpbits64_unpacked_t __x_re, fpbits64_unpacked_t __x_im, fpbits64_unpacked_t __y_re, fpbits64_unpacked_t __y_im, fpbits64_unpacked_t& __r_re, fpbits64_unpacked_t& __r_im) noexcept { __internal_fp64emu_cmul_unpacked<fp64emu_accuracy::low>(__x_re, __x_im, __y_re, __y_im, __r_re, __r_im); }

// neg (unpacked)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_neg (fpbits64_unpacked_t __x) noexcept { return __internal_fp64emu_neg_unpacked(__x); }

#else // LTO mode - declarations only

// mad (packed)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mad_rn      (fpbits64_t x, fpbits64_t y, fpbits64_t z) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_high_mad_rn (fpbits64_t x, fpbits64_t y, fpbits64_t z) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_mad_rn  (fpbits64_t x, fpbits64_t y, fpbits64_t z) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_mad_rn  (fpbits64_t x, fpbits64_t y, fpbits64_t z) noexcept ;

// dot (packed)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dot_rn      (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_high_dot_rn (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dot_rn  (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dot_rn  (fpbits64_t x1, fpbits64_t y1, fpbits64_t x2, fpbits64_t y2) noexcept ;

// cmul (packed)
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_cmul_rn      (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_high_cmul_rn (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_mid_cmul_rn  (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_low_cmul_rn  (fpbits64_t x_re, fpbits64_t x_im, fpbits64_t y_re, fpbits64_t y_im, fpbits64_t& r_re, fpbits64_t& r_im) noexcept ;

// neg (packed)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_neg (fpbits64_t x) noexcept ;

// mad (unpacked)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_mad      (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_high_mad (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_mid_mad  (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_low_mad  (fpbits64_unpacked_t x, fpbits64_unpacked_t y, fpbits64_unpacked_t z) noexcept ;

// dot (unpacked)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_dot      (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_high_dot (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_mid_dot  (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_low_dot  (fpbits64_unpacked_t x1, fpbits64_unpacked_t y1, fpbits64_unpacked_t x2, fpbits64_unpacked_t y2) noexcept ;

// cmul (unpacked)
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_cmul      (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_high_cmul (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_mid_cmul  (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL void __fp64emu_unpacked_low_cmul  (fpbits64_unpacked_t x_re, fpbits64_unpacked_t x_im, fpbits64_unpacked_t y_re, fpbits64_unpacked_t y_im, fpbits64_unpacked_t& r_re, fpbits64_unpacked_t& r_im) noexcept ;

// neg (unpacked)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_neg (fpbits64_unpacked_t x) noexcept ;

#endif // _CCCL_FPEMU_INLINE

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_OTHERS_H

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_OTHERS_API_MERGED)
#define _CCCL_FPEMU_OTHERS_API_MERGED
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_others_api.hpp)
// ============================================================================

    // Unary negation operator - member function implementation
    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_t<_Acc> fp64emu_t<_Acc>::operator-() const noexcept
    {
        fp64emu_t __temp(*this);
        __temp.bits = __fp64emu_neg(__temp.bits);
        return __temp;
    }


    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_t<_Acc> mad (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y, const fp64emu_t<_Acc>& __z) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_high_mad_rn(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_low_mad_rn(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_mid_mad_rn(__x.bits, __y.bits, __z.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_t<_Acc> __mad_rn (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y, const fp64emu_t<_Acc>& __z) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_high_mad_rn(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_low_mad_rn(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_mid_mad_rn(__x.bits, __y.bits, __z.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_t<_Acc> dot (const fp64emu_t<_Acc>& __x1, const fp64emu_t<_Acc>& __y1, const fp64emu_t<_Acc>& __x2, const fp64emu_t<_Acc>& __y2) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_high_dot_rn(__x1.bits, __y1.bits, __x2.bits, __y2.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_low_dot_rn(__x1.bits, __y1.bits, __x2.bits, __y2.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_mid_dot_rn(__x1.bits, __y1.bits, __x2.bits, __y2.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    _CCCL_API void cmul (const fp64emu_t<_Acc>& __x_re, const fp64emu_t<_Acc>& __x_im, const fp64emu_t<_Acc>& __y_re, const fp64emu_t<_Acc>& __y_im, fp64emu_t<_Acc>& __r_re, fp64emu_t<_Acc>& __r_im) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { __fp64emu_high_cmul_rn(__x_re.bits, __x_im.bits, __y_re.bits, __y_im.bits, __r_re.bits, __r_im.bits); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { __fp64emu_low_cmul_rn(__x_re.bits, __x_im.bits, __y_re.bits, __y_im.bits, __r_re.bits, __r_im.bits); }
        else                                               { __fp64emu_mid_cmul_rn(__x_re.bits, __x_im.bits, __y_re.bits, __y_im.bits, __r_re.bits, __r_im.bits); }
    } 


    // Unary negation operator for unpacked - member function implementation
    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_unpacked_t<_Acc> fp64emu_unpacked_t<_Acc>::operator-() const noexcept
    {
        fp64emu_unpacked_t __temp(*this);
        __temp.bits = __fp64emu_unpacked_neg(__temp.bits);
        return __temp;
    }


    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_unpacked_t<_Acc> mad (const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y, const fp64emu_unpacked_t<_Acc>& __z) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_high_mad(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_low_mad(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_mid_mad(__x.bits, __y.bits, __z.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_unpacked_t<_Acc> __mad_rn (const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y, const fp64emu_unpacked_t<_Acc>& __z) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_high_mad(__x.bits, __y.bits, __z.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_low_mad(__x.bits, __y.bits, __z.bits)); }
        else                                               { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_mid_mad(__x.bits, __y.bits, __z.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_unpacked_t<_Acc> dot (const fp64emu_unpacked_t<_Acc>& __x1, const fp64emu_unpacked_t<_Acc>& __y1, const fp64emu_unpacked_t<_Acc>& __x2, const fp64emu_unpacked_t<_Acc>& __y2) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_high_dot(__x1.bits, __y1.bits, __x2.bits, __y2.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_low_dot(__x1.bits, __y1.bits, __x2.bits, __y2.bits)); }
        else                                               { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_mid_dot(__x1.bits, __y1.bits, __x2.bits, __y2.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    _CCCL_API void cmul (const fp64emu_unpacked_t<_Acc>& __x_re, const fp64emu_unpacked_t<_Acc>& __x_im, const fp64emu_unpacked_t<_Acc>& __y_re, const fp64emu_unpacked_t<_Acc>& __y_im, fp64emu_unpacked_t<_Acc>& __r_re, fp64emu_unpacked_t<_Acc>& __r_im) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { __fp64emu_unpacked_high_cmul(__x_re.bits, __x_im.bits, __y_re.bits, __y_im.bits, __r_re.bits, __r_im.bits); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { __fp64emu_unpacked_low_cmul(__x_re.bits, __x_im.bits, __y_re.bits, __y_im.bits, __r_re.bits, __r_im.bits); }
        else                                               { __fp64emu_unpacked_mid_cmul(__x_re.bits, __x_im.bits, __y_re.bits, __y_im.bits, __r_re.bits, __r_im.bits); }
    }



} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CCCL_FPEMU_OTHERS_API_MERGED
