//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_SUB_H
#define _CUDA___FP_FPEMU_IMPL_SUB_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/** 
 * @file fpemu_dsub_impl.hpp
 * @brief Implementation of double-precision subtraction operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of double-precision subtraction operations for the FPEMU library.
 * It includes:
 *
 * - Subtraction functions for fp64emu_t
 * - Subtraction operators for fp64emu_t
 * - Subtraction functions to other types
 *
 * The subtraction functions are designed to work across both host and device code
 * through appropriate decorators and provide bit-exact results matching hardware
 * floating point units.
 */

#include <cuda/__fp/fpemu_common.h>
#include <cuda/__fp/fpemu_impl_utils.h>
#include <cuda/__fp/fpemu_impl_unpack.h>
#include <cuda/__fp/fpemu_impl_add.h>
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


    /**
     * @brief Subtract two fpbits64_unpacked_t
     * 
     * This function subtracts two fpbits64_unpacked_t.
     * 
     * @param a The first fpbits64_unpacked_t
     * @param b The second fpbits64_unpacked_t
     * @return The result of the subtraction
     */
    template<fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API
    fpbits64_unpacked_t __internal_fp64emu_dsub_unpacked(fpbits64_unpacked_t __a, 
                                                            fpbits64_unpacked_t __b) noexcept
    {
        return __internal_fp64emu_dadd_unpacked<_Acc, true>(__a, __b);
    }

    /**
     * @brief Subtract two fpbits64_t
     * 
     * This function subtracts two fpbits64_t.
     * 
     * @param x The first fpbits64_t
     * @param y The second fpbits64_t
     * @return The result of the subtraction
     */
    template<__fpemu_rounding    _Rm  = __fpemu_rounding::def, 
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API
    fpbits64_t __internal_fp64emu_dsub(fpbits64_t __x, 
                                        fpbits64_t __y) noexcept
    {
        // Forced parameters for the subtraction operation
        constexpr fp64emu_accuracy   __acc_forced = fp64emu_accuracy::_CCCL_FPEMU_ADD_METHOD;
        constexpr fp64emu_accuracy   __acc_used   = (__acc_forced != fp64emu_accuracy::unset) ? __acc_forced : _Acc;

        {
            // Pass true to the dadd function to indicate that we are subtracting
            return __internal_fp64emu_dadd<_Rm, __acc_used, true>(__x, __y);
        }
    } // __internal_fp64emu_dsub


// ============================================================================
// Builtin declarations/implementations for subtraction operations
// ============================================================================
#if defined(_CCCL_FPEMU_INLINE)
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dsub_rn (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rn, fp64emu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dsub_rz (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rz, fp64emu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dsub_ru (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::ru, fp64emu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dsub_rd (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rd, fp64emu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_high_dsub_rn (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rn, fp64emu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dsub_rn  (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rn, fp64emu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dsub_rz  (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rz, fp64emu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dsub_ru  (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::ru, fp64emu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dsub_rd  (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rd, fp64emu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dsub_rn  (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rn, fp64emu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dsub_rz  (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rz, fp64emu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dsub_ru  (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::ru, fp64emu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dsub_rd  (fpbits64_t __x, fpbits64_t __y) noexcept { return __internal_fp64emu_dsub<__fpemu_rounding::rd, fp64emu_accuracy::low>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_dsub          (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y) noexcept { return __internal_fp64emu_dsub_unpacked<fp64emu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_high_dsub (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y) noexcept { return __internal_fp64emu_dsub_unpacked<fp64emu_accuracy::high>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_mid_dsub      (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y) noexcept { return __internal_fp64emu_dsub_unpacked<fp64emu_accuracy::mid>(__x, __y); }
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_low_dsub     (fpbits64_unpacked_t __x, fpbits64_unpacked_t __y) noexcept { return __internal_fp64emu_dsub_unpacked<fp64emu_accuracy::low>(__x, __y); }
#else
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dsub_rn (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dsub_rz (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dsub_ru (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_dsub_rd (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_high_dsub_rn (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dsub_rn  (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dsub_rz  (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dsub_ru  (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_mid_dsub_rd  (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dsub_rn  (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dsub_rz  (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dsub_ru  (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_t __fp64emu_low_dsub_rd  (fpbits64_t x, fpbits64_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_dsub      (fpbits64_unpacked_t x, fpbits64_unpacked_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_high_dsub (fpbits64_unpacked_t x, fpbits64_unpacked_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_mid_dsub  (fpbits64_unpacked_t x, fpbits64_unpacked_t y) noexcept ;
_CCCL_FPEMU_BUILTIN_DECL fpbits64_unpacked_t __fp64emu_unpacked_low_dsub  (fpbits64_unpacked_t x, fpbits64_unpacked_t y) noexcept ;
#endif // _CCCL_FPEMU_INLINE

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_SUB_H (builtins)

#if defined(_CCCL_FPEMU_API_CLASSES_DEFINED) && !defined(_CCCL_FPEMU_DSUB_API_MERGED)
#define _CCCL_FPEMU_DSUB_API_MERGED
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_dsub_api.hpp)
// ============================================================================

    // Default API implementation - binary subtraction operator
    template<fp64emu_accuracy _Acc> _CCCL_API static fp64emu_t<_Acc> operator- (const fp64emu_t<_Acc>& __x, 
                                                                        const fp64emu_t<_Acc>& __y) noexcept
    {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_high_dsub_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_mid_dsub_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_low_dsub_rn(__x.bits, __y.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_dsub_rn(__x.bits, __y.bits)); }
    } // operator-


    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_t<_Acc> __dsub_rn (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_high_dsub_rn(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_low_dsub_rn(__x.bits, __y.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_mid_dsub_rn(__x.bits, __y.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_t<_Acc> __dsub_rz (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) noexcept {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_dsub_rz(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_mid_dsub_rz(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_low_dsub_rz(__x.bits, __y.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_dsub_rz(__x.bits, __y.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_t<_Acc> __dsub_ru (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) noexcept {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_dsub_ru(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_mid_dsub_ru(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_low_dsub_ru(__x.bits, __y.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_dsub_ru(__x.bits, __y.bits)); }
    }
    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_t<_Acc> __dsub_rd (const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) noexcept {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_dsub_rd(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_mid_dsub_rd(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_low_dsub_rd(__x.bits, __y.bits)); }
        else                                               { return fp64emu_t<_Acc>(fpbits64_construct, __fp64emu_dsub_rd(__x.bits, __y.bits)); }
    }


    // Operator- for unpacked subtraction
    template<fp64emu_accuracy _Acc>
    _CCCL_DEVICE_API static fp64emu_unpacked_t<_Acc> operator- (const fp64emu_unpacked_t<_Acc>& __x, 
                                                                            const fp64emu_unpacked_t<_Acc>& __y) noexcept
    {
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_high_dsub(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_mid_dsub(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_low_dsub(__x.bits, __y.bits)); }
        else                                               { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_dsub(__x.bits, __y.bits)); }
    } // operator-


    template<fp64emu_accuracy _Acc>
    _CCCL_API fp64emu_unpacked_t<_Acc> __dsub_rn (const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y) noexcept { 
        if      constexpr (_Acc == fp64emu_accuracy::high) { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_high_dsub(__x.bits, __y.bits)); }
        else if constexpr (_Acc == fp64emu_accuracy::low)  { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_low_dsub(__x.bits, __y.bits)); }
        else                                               { return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __fp64emu_unpacked_mid_dsub(__x.bits, __y.bits)); }
    }



} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA___FP_FPEMU_IMPL_SUB_H
