//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_IMPL_CVT_H
#define _CUDA___FP_FPEMU_IMPL_CVT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file fpemu_impl_cvt.h
 * @brief Implementation of type conversion operations for FPEMU floating point emulation library
 *
 * This header provides the implementation of type conversion operations for the FPEMU library.
 * It includes:
 *
 * - Conversion functions from fp64emu_t to other types
 * - Conversion operators to other types
 * - Conversion functions to other types
 *
 * The conversion functions are designed to work across both host and device code
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
    // fp64 -> integer conversions (self-contained; no SoftFloat dependency).
    //
    // The rounding core mirrors SoftFloat-3e (significand jam-shift + a
    // round-increment selected by mode/sign), while the out-of-range and NaN
    // handling matches CUDA hardware saturating conversions
    // (__double2int_rd, __double2ll_ru, ...):
    //   NaN              -> integer indefinite (sign bit only):
    //                       0x80000000 (32-bit) / 0x8000000000000000 (64-bit)
    //   signed   +ovf    -> INT_MAX ;  -ovf -> INT_MIN
    //   unsigned  ovf    -> UINT_MAX;  any negative -> 0
    // ========================================================================

    // Right shift keeping a sticky (jam) bit: __nv_internal_fp64emu_shr_jam64,
    // shared with the divide/sqrt cores (see fpemu_impl_utils.h).

    /// @brief Round a 64-bit fixed-point significand (12 fractional bits) to int32 (CUDA saturation).
    template<fpemu::rounding _Rm>
    _CCCL_TRIVIAL_API int32_t __nv_internal_fp64emu_round_to_i32 (bool __sign, uint64_t __sig) noexcept
    {
        const int32_t __sat = __sign ? (int32_t)0x80000000 : (int32_t)0x7FFFFFFF;
        uint32_t __roundIncrement = 0x800;
        if constexpr (_Rm != fpemu::rounding::rn)
        {
            __roundIncrement = 0;
            const bool __toward_inf = __sign ? (_Rm == fpemu::rounding::rd)
                                         : (_Rm == fpemu::rounding::ru);
            if (__toward_inf) __roundIncrement = 0xFFF;
        }
        uint32_t __roundBits = (uint32_t)(__sig & 0xFFF);
        __sig += __roundIncrement;
        if (__sig & 0xFFFFF00000000000ULL) return __sat;

        uint32_t __sig32 = (uint32_t)(__sig >> 12);
        if constexpr (_Rm == fpemu::rounding::rn)
        {
            if (__roundBits == 0x800) __sig32 &= ~(uint32_t)1;
        }
        uint32_t __uz = __sign ? (uint32_t)(0u - __sig32) : __sig32;
        int32_t __z = (int32_t)__uz;
        if (__z && ((__z < 0) != __sign)) return __sat;
        return __z;
    }

    /// @brief Round a 64-bit fixed-point significand (12 fractional bits) to uint32 (CUDA saturation).
    template<fpemu::rounding _Rm>
    _CCCL_TRIVIAL_API uint32_t __nv_internal_fp64emu_round_to_ui32 (bool __sign, uint64_t __sig) noexcept
    {
        if (__sign) return 0; // any negative real saturates to 0

        uint32_t __roundIncrement = (_Rm == fpemu::rounding::rn) ? 0x800
                                : (_Rm == fpemu::rounding::ru) ? 0xFFF : 0;
        uint32_t __roundBits = (uint32_t)(__sig & 0xFFF);
        __sig += __roundIncrement;
        if (__sig & 0xFFFFF00000000000ULL) return 0xFFFFFFFFu;

        uint32_t __z = (uint32_t)(__sig >> 12);
        if constexpr (_Rm == fpemu::rounding::rn)
        {
            if (__roundBits == 0x800) __z &= ~(uint32_t)1;
        }
        return __z;
    }

    /// @brief Round (sig : sigExtra) to int64 (CUDA saturation).
    template<fpemu::rounding _Rm>
    _CCCL_TRIVIAL_API int64_t __nv_internal_fp64emu_round_to_i64 (bool __sign, uint64_t __sig, uint64_t __sigExtra) noexcept
    {
        const int64_t __sat = __sign ? (int64_t)0x8000000000000000ULL
                                 : (int64_t)0x7FFFFFFFFFFFFFFFULL;
        bool __increment;
        if constexpr (_Rm == fpemu::rounding::rn)
        {
            __increment = (__sigExtra >= 0x8000000000000000ULL);
        }
        else
        {
            const bool __toward_inf = __sign ? (_Rm == fpemu::rounding::rd)
                                         : (_Rm == fpemu::rounding::ru);
            __increment = (__sigExtra != 0) && __toward_inf;
        }
        if (__increment)
        {
            ++__sig;
            if (!__sig) return __sat;
            if constexpr (_Rm == fpemu::rounding::rn)
            {
                if (__sigExtra == 0x8000000000000000ULL) __sig &= ~(uint64_t)1;
            }
        }
        uint64_t __uz = __sign ? (uint64_t)(0ULL - __sig) : __sig;
        int64_t __z = (int64_t)__uz;
        if (__z && ((__z < 0) != __sign)) return __sat;
        return __z;
    }

    /// @brief Round (sig : sigExtra) to uint64 (CUDA saturation).
    template<fpemu::rounding _Rm>
    _CCCL_TRIVIAL_API uint64_t __nv_internal_fp64emu_round_to_ui64 (bool __sign, uint64_t __sig, uint64_t __sigExtra) noexcept
    {
        if (__sign) return 0; // any negative real saturates to 0

        bool __increment;
        if constexpr (_Rm == fpemu::rounding::rn)
        {
            __increment = (__sigExtra >= 0x8000000000000000ULL);
        }
        else
        {
            __increment = (_Rm == fpemu::rounding::ru) && (__sigExtra != 0);
        }
        if (__increment)
        {
            ++__sig;
            if (!__sig) return 0xFFFFFFFFFFFFFFFFULL;
            if constexpr (_Rm == fpemu::rounding::rn)
            {
                if (__sigExtra == 0x8000000000000000ULL) __sig &= ~(uint64_t)1;
            }
        }
        return __sig;
    }

    template<fpemu::rounding _Rm  = fpemu::rounding::rz>
    _CCCL_TRIVIAL_API  int32_t __nv_internal_fp64emu_fpbits64_to_int (fpbits64_t __x) noexcept
    {
        const bool    __sign = ((uint64_t)__x >> 63) != 0;
        const int32_t __exp  = (int32_t)(((uint64_t)__x >> FP64_MANT_BITS) & 0x7FF);
        uint64_t      __sig  = (uint64_t)__x & fpemu::MANTISSA_MASK;

        if (__exp == 0x7FF && __sig) return (int32_t)0x80000000; // NaN -> integer indefinite

        if (__exp) __sig |= __FPEMU_HIDDEN_64__;
        int32_t __shiftDist = 0x427 - __exp;
        if (__shiftDist > 0) __sig = __nv_internal_fp64emu_shr_jam64(__sig, (uint32_t)__shiftDist);
        return __nv_internal_fp64emu_round_to_i32<_Rm>(__sign, __sig);
    } // __nv_internal_fp64emu_fpbits64_to_int

    template<fpemu::rounding _Rm  = fpemu::rounding::rz>
    _CCCL_TRIVIAL_API  uint32_t __nv_internal_fp64emu_fpbits64_to_uint (fpbits64_t __x) noexcept
    {
        const bool    __sign = ((uint64_t)__x >> 63) != 0;
        const int32_t __exp  = (int32_t)(((uint64_t)__x >> FP64_MANT_BITS) & 0x7FF);
        uint64_t      __sig  = (uint64_t)__x & fpemu::MANTISSA_MASK;

        if (__exp == 0x7FF && __sig) return 0x80000000u; // NaN -> integer indefinite

        if (__exp) __sig |= __FPEMU_HIDDEN_64__;
        int32_t __shiftDist = 0x427 - __exp;
        if (__shiftDist > 0) __sig = __nv_internal_fp64emu_shr_jam64(__sig, (uint32_t)__shiftDist);
        return __nv_internal_fp64emu_round_to_ui32<_Rm>(__sign, __sig);
    } // __nv_internal_fp64emu_fpbits64_to_uint

    template<fpemu::rounding _Rm  = fpemu::rounding::rz>
    _CCCL_TRIVIAL_API  int64_t __nv_internal_fp64emu_fpbits64_to_ll (fpbits64_t __x) noexcept
    {
        const bool    __sign = ((uint64_t)__x >> 63) != 0;
        const int32_t __exp  = (int32_t)(((uint64_t)__x >> FP64_MANT_BITS) & 0x7FF);
        uint64_t      __sig  = (uint64_t)__x & fpemu::MANTISSA_MASK;

        if (__exp == 0x7FF && __sig) return (int64_t)0x8000000000000000ULL; // NaN -> integer indefinite

        if (__exp) __sig |= __FPEMU_HIDDEN_64__;
        int32_t __shiftDist = 0x433 - __exp;
        uint64_t __sig_int, __sig_extra;
        if (__shiftDist <= 0)
        {
            if (__shiftDist < -11) return __sign ? (int64_t)0x8000000000000000ULL
                                             : (int64_t)0x7FFFFFFFFFFFFFFFULL;
            __sig_int   = __sig << (-__shiftDist);
            __sig_extra = 0;
        }
        else if (__shiftDist < 64)
        {
            __sig_int   = __sig >> __shiftDist;
            __sig_extra = __sig << (-__shiftDist & 63);
        }
        else
        {
            __sig_int   = 0;
            __sig_extra = (__shiftDist == 64) ? __sig : (uint64_t)(__sig != 0);
        }
        return __nv_internal_fp64emu_round_to_i64<_Rm>(__sign, __sig_int, __sig_extra);
    } // __nv_internal_fp64emu_fpbits64_to_ll

    template<fpemu::rounding _Rm  = fpemu::rounding::rz>
    _CCCL_TRIVIAL_API  uint64_t __nv_internal_fp64emu_fpbits64_to_ull (fpbits64_t __x) noexcept
    {
        const bool    __sign = ((uint64_t)__x >> 63) != 0;
        const int32_t __exp  = (int32_t)(((uint64_t)__x >> FP64_MANT_BITS) & 0x7FF);
        uint64_t      __sig  = (uint64_t)__x & fpemu::MANTISSA_MASK;

        if (__exp == 0x7FF && __sig) return 0x8000000000000000ULL; // NaN -> integer indefinite

        if (__exp) __sig |= __FPEMU_HIDDEN_64__;
        int32_t __shiftDist = 0x433 - __exp;
        uint64_t __sig_int, __sig_extra;
        if (__shiftDist <= 0)
        {
            // Negative saturates to 0; positive out-of-range to UINT64_MAX.
            if (__shiftDist < -11) return __sign ? 0ULL : 0xFFFFFFFFFFFFFFFFULL;
            __sig_int   = __sig << (-__shiftDist);
            __sig_extra = 0;
        }
        else if (__shiftDist < 64)
        {
            __sig_int   = __sig >> __shiftDist;
            __sig_extra = __sig << (-__shiftDist & 63);
        }
        else
        {
            __sig_int   = 0;
            __sig_extra = (__shiftDist == 64) ? __sig : (uint64_t)(__sig != 0);
        }
        return __nv_internal_fp64emu_round_to_ui64<_Rm>(__sign, __sig_int, __sig_extra);
    } // __nv_internal_fp64emu_fpbits64_to_ull

    _CCCL_TRIVIAL_API  float __nv_internal_fp64emu_fpbits64_to_float (fpbits64_t __x) noexcept
    {
        uint64_t __bits = (uint64_t)__x;
        uint32_t __sign = (uint32_t)(__bits >> 63) << 31;
        int32_t  __exp  = (int32_t)((__bits >> FP64_MANT_BITS) & 0x7FF);
        uint64_t __frac = __bits & fpemu::MANTISSA_MASK;

        if (__exp == 0x7FF) 
        {
            if (__frac == 0)
                return fpemu::bit_cast<float>(__sign | 0x7F800000u);
            // NaN: preserve sign, set quiet bit, keep upper payload
            uint32_t __frac32 = (uint32_t)(__frac >> 29) | 0x00400000u;
            return fpemu::bit_cast<float>(__sign | 0x7F800000u | __frac32);
        }

        // Zero or subnormal double (below float range) → ±0
        if (__exp == 0)
            return fpemu::bit_cast<float>(__sign);

        int32_t  __e_f = __exp - (FP64_BIAS - FP32_BIAS);
        uint64_t __sig = (1ULL << FP64_MANT_BITS) | __frac;

        if (__e_f >= 0xFF)
            return fpemu::bit_cast<float>(__sign | 0x7F800000u);

        // Number of mantissa bits to discard: 52 - 23 = 29
        constexpr int32_t __DROP = FP64_MANT_BITS - FP32_MANT_BITS;

        if (__e_f > 0) 
        {
            uint64_t __half  = 1ULL << (__DROP - 1);
            uint64_t __trail = __sig & ((1ULL << __DROP) - 1);
            uint32_t __sig24 = (uint32_t)(__sig >> __DROP);

            if (__trail > __half || (__trail == __half && (__sig24 & 1))) 
            {
                __sig24++;
                if (__sig24 >> (FP32_MANT_BITS + 1)) 
                {
                    __sig24 >>= 1;
                    __e_f++;
                    if (__e_f >= 0xFF)
                        return fpemu::bit_cast<float>(__sign | 0x7F800000u);
                }
            }

            uint32_t __frac_f = __sig24 & ((1u << FP32_MANT_BITS) - 1);
            return fpemu::bit_cast<float>(__sign | ((uint32_t)__e_f << FP32_MANT_BITS) | __frac_f);
        } // if (e_f > 0)

        // Subnormal float output (e_f <= 0)
        int32_t __total_shift = __DROP + 1 - __e_f;

        if (__total_shift >= 54)
            return fpemu::bit_cast<float>(__sign);

        uint64_t __half  = 1ULL << (__total_shift - 1);
        uint64_t __trail = __sig & ((1ULL << __total_shift) - 1);
        uint32_t __sig_sub = (uint32_t)(__sig >> __total_shift);

        if (__trail > __half || (__trail == __half && (__sig_sub & 1)))
            __sig_sub++;

        // Overflow to 2^23 naturally becomes exponent=1, frac=0 (min normal)
        return fpemu::bit_cast<float>(__sign | __sig_sub);
    } // __nv_internal_fp64emu_fpbits64_to_float

    _CCCL_TRIVIAL_API fpbits64_t __nv_internal_fp64emu_float_to_fpbits64  (float __x) noexcept
    {
        uint32_t __bits = fpemu::bit_cast<uint32_t>(__x);
        uint64_t __sign = (uint64_t)(__bits >> 31) << 63;
        int32_t  __exp  = (int32_t)((__bits >> FP32_MANT_BITS) & 0xFF);
        uint32_t __frac = __bits & ((1u << FP32_MANT_BITS) - 1);

        if (__exp == 0xFF) 
        {
            if (__frac == 0)
                return (fpbits64_t)(__sign | ((uint64_t)0x7FF << FP64_MANT_BITS));
            // NaN: preserve sign, set quiet bit, widen payload
            uint64_t __d_frac = ((uint64_t)__frac << 29) | ((uint64_t)1 << (FP64_MANT_BITS - 1));
            return (fpbits64_t)(__sign | ((uint64_t)0x7FF << FP64_MANT_BITS) | __d_frac);
        }

        if (__exp == 0 && __frac == 0)
            return (fpbits64_t)__sign;

        // Subnormal float → normalize
        if (__exp == 0) 
        {
            int32_t __nz = fpemu::__internal_clz((int)__frac) - (32 - FP32_MANT_BITS - 1);
            __frac = (__frac << __nz) & ((1u << FP32_MANT_BITS) - 1);
            __exp  = 1 - __nz;
        }

        // Exact widening conversion
        uint64_t __d_exp  = (uint64_t)(__exp + (FP64_BIAS - FP32_BIAS));
        uint64_t __d_frac = (uint64_t)__frac << (FP64_MANT_BITS - FP32_MANT_BITS);
        return (fpbits64_t)(__sign | (__d_exp << FP64_MANT_BITS) | __d_frac);
    } // __nv_internal_fp64emu_float_to_fpbits64

    _CCCL_TRIVIAL_API 
    fpbits64_t __nv_internal_fp64emu_int_to_fpbits64  (int32_t __x) noexcept  
    { 
        if (__x == 0) return (fpbits64_t)0;

        uint64_t __sign  = (__x < 0) ? (1ULL << 63) : 0ULL;
        uint32_t __abs_x = (uint32_t)((__x < 0) ? -(int64_t)__x : (int64_t)__x);

        int32_t __nz        = fpemu::__internal_clz((int)__abs_x);
        uint64_t __exp      = (uint64_t)(FP64_BIAS + 31 - __nz);
        uint64_t __mantissa = ((uint64_t)__abs_x << (21 + __nz)) & fpemu::MANTISSA_MASK;

        return (fpbits64_t)(__sign | (__exp << FP64_MANT_BITS) | __mantissa);
    } // __nv_internal_fp64emu_int_to_fpbits64

    _CCCL_TRIVIAL_API 
    fpbits64_t __nv_internal_fp64emu_uint_to_fpbits64 (uint32_t __x) noexcept 
    { 
        if (__x == 0) return (fpbits64_t)0;

        int32_t __nz        = fpemu::__internal_clz((int)__x);
        uint64_t __exp      = (uint64_t)(FP64_BIAS + 31 - __nz);
        uint64_t __mantissa = ((uint64_t)__x << (21 + __nz)) & fpemu::MANTISSA_MASK;

        return (fpbits64_t)((__exp << FP64_MANT_BITS) | __mantissa);
    } // __nv_internal_fp64emu_uint_to_fpbits64

    _CCCL_TRIVIAL_API 
    fpbits64_t __nv_internal_fp64emu_ll_to_fpbits64   (int64_t __x) noexcept  
    { 
        if (__x == 0) return (fpbits64_t)0;

        uint64_t __sign = (__x < 0) ? (1ULL << 63) : 0ULL;
        uint64_t __absA = (__x < 0) ? -(uint64_t)__x : (uint64_t)__x;

        int32_t __nz = fpemu::__internal_clzll((int64_t)__absA);
        int32_t __exp = FP64_BIAS + 63 - __nz;

        if (__nz >= 11) 
        {
            // <= 53 significant bits: exact
            uint64_t __mantissa = (__absA << (__nz - 11)) & fpemu::MANTISSA_MASK;
            return (fpbits64_t)(__sign | ((uint64_t)__exp << FP64_MANT_BITS) | __mantissa);
        }

        // > 53 significant bits: round to nearest even
        int32_t __shift  = 11 - __nz;
        uint64_t __half  = 1ULL << (__shift - 1);
        uint64_t __trail = __absA & ((1ULL << __shift) - 1);
        uint64_t __sig53 = __absA >> __shift;

        if (__trail > __half || (__trail == __half && (__sig53 & 1))) 
        {
            __sig53++;
            if (__sig53 >> 53) { __sig53 >>= 1; __exp++; }
        }

        uint64_t __mantissa = __sig53 & fpemu::MANTISSA_MASK;
        return (fpbits64_t)(__sign | ((uint64_t)__exp << FP64_MANT_BITS) | __mantissa);
    } // __nv_internal_fp64emu_ll_to_fpbits64

    _CCCL_TRIVIAL_API 
    fpbits64_t __nv_internal_fp64emu_ull_to_fpbits64  (uint64_t __x) noexcept 
    {
        if (__x == 0) return (fpbits64_t)0;

        int32_t __nz = fpemu::__internal_clzll((int64_t)__x);
        int32_t __exp = FP64_BIAS + 63 - __nz;

        if (__nz >= 11) 
        {
            // <= 53 significant bits: exact
            uint64_t __mantissa = (__x << (__nz - 11)) & fpemu::MANTISSA_MASK;
            return (fpbits64_t)(((uint64_t)__exp << FP64_MANT_BITS) | __mantissa);
        }

        // > 53 significant bits: round to nearest even
        int32_t __shift  = 11 - __nz;
        uint64_t __half  = 1ULL << (__shift - 1);
        uint64_t __trail = __x & ((1ULL << __shift) - 1);
        uint64_t __sig53 = __x >> __shift;

        if (__trail > __half || (__trail == __half && (__sig53 & 1))) 
        {
            __sig53++;
            if (__sig53 >> 53) { __sig53 >>= 1; __exp++; }
        }

        uint64_t __mantissa = __sig53 & fpemu::MANTISSA_MASK;
        return (fpbits64_t)(((uint64_t)__exp << FP64_MANT_BITS) | __mantissa);
    } // __nv_internal_fp64emu_ull_to_fpbits64


    // fpbits64<->uint64 casts
    _CCCL_TRIVIAL_API uint64_t   __nv_internal_fp64emu_fpbits64_cast_ull (fpbits64_t __x) noexcept { return __x; }
    _CCCL_TRIVIAL_API fpbits64_t __nv_internal_fp64emu_ull_cast_fpbits64 (uint64_t __x) noexcept   { return fpbits64_t{__x}; }

    // double<->fpbits64 conversions
    _CCCL_TRIVIAL_API double     __nv_internal_fp64emu_fpbits64_to_double (fpbits64_t __x) noexcept { return fpemu::bit_cast<double>(__x); }
    _CCCL_TRIVIAL_API fpbits64_t __nv_internal_fp64emu_double_to_fpbits64 (double __x) noexcept
    { 
        return fpemu::bit_cast<fpbits64_t>(__x);
    }

#if __FPEMU_UNPACKED__ == 1

    _CCCL_TRIVIAL_API uint64_t   __nv_internal_fp64emu_fpbits64_unpacked_cast_ull (fpbits64_unpacked_t __x) noexcept 
    { 
        fpbits64_t __x_packed = __nv_internal_fp64emu_pack(__x);
        return __nv_internal_fp64emu_fpbits64_cast_ull(__x_packed);
    }

    _CCCL_TRIVIAL_API fpbits64_unpacked_t __nv_internal_fp64emu_ull_cast_fpbits64_unpacked (uint64_t __x) noexcept
    { 
        fpbits64_t __x_packed = __nv_internal_fp64emu_ull_cast_fpbits64(__x);
        return __nv_internal_fp64emu_unpack(__x_packed);
    }

    /**
     * @brief Convert a fpbits64_unpacked_t to a double
     * 
     * This function converts a fpbits64_unpacked_t to a double.
     * 
     * @param x The fpbits64_unpacked_t to convert
     * @return The converted double
     */
    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API double __nv_internal_fp64emu_fpbits64_unpacked_to_double (fpbits64_unpacked_t __x) noexcept 
    { 
        fpbits64_t __x_packed = __nv_internal_fp64emu_pack<_Rm>(__x);
        return __nv_internal_fp64emu_fpbits64_to_double(__x_packed); 
    }

    /**
     * @brief Convert a double to a fpbits64_unpacked_t
     * 
     * This function converts a double to a fpbits64_unpacked_t.
     * 
     * @param x The double to convert
     * @return The converted fpbits64_unpacked_t
     */
    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API fpbits64_unpacked_t __nv_internal_fp64emu_double_to_fpbits64_unpacked (double __x) noexcept
    { 
        fpbits64_t __x_packed = __nv_internal_fp64emu_double_to_fpbits64(__x);
        return __nv_internal_fp64emu_unpack(__x_packed);
    }

    // ------------------------------------------------------------------------
    // True unpacked -> integer conversions. Operate directly on the fully-accurate
    // unpacked fields (no operand pack): the full unpack already normalized
    // denormals (implicit bit at 61) and encoded inf/nan in the exponent band, so
    // the 53-bit significand is mantissa>>EXTRA_BITS and the exponent is the same
    // IEEE-biased value the packed converters consume. The shift/round/saturate
    // cores (shr_jam64 + round_to_*) are shared with the packed path, so results
    // match bit-for-bit (incl. NaN->indefinite and inf/overflow saturation).
    // ------------------------------------------------------------------------
    static constexpr int32_t __FP64EMU_CVT_NAN_EXP = 0x0007ff00;
    static constexpr int32_t __FP64EMU_CVT_INF_EXP = 0x00007ff0;

    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API  int32_t __nv_internal_fp64emu_fpbits64_unpacked_to_int (fpbits64_unpacked_t __x) noexcept
    {
        const bool    __sign = (__x.sign != 0);
        const int32_t __exp  = (int32_t)__x.exponent;
        if (__exp == __FP64EMU_CVT_NAN_EXP) return (int32_t)0x80000000;                       // NaN -> indefinite
        if (__exp == __FP64EMU_CVT_INF_EXP) return __sign ? (int32_t)0x80000000 : (int32_t)0x7FFFFFFF;
        uint64_t __sig = __x.mantissa >> EXTRA_BITS;   // 53-bit significand (implicit at 52), 0 for zero
        int32_t  __shiftDist = 0x427 - __exp;
        if (__shiftDist > 0) __sig = __nv_internal_fp64emu_shr_jam64(__sig, (uint32_t)__shiftDist);
        return __nv_internal_fp64emu_round_to_i32<_Rm>(__sign, __sig);
    }
    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API  uint32_t __nv_internal_fp64emu_fpbits64_unpacked_to_uint (fpbits64_unpacked_t __x) noexcept
    {
        const bool    __sign = (__x.sign != 0);
        const int32_t __exp  = (int32_t)__x.exponent;
        if (__exp == __FP64EMU_CVT_NAN_EXP) return 0x80000000u;                               // NaN -> indefinite
        if (__exp == __FP64EMU_CVT_INF_EXP) return __sign ? 0u : 0xFFFFFFFFu;
        uint64_t __sig = __x.mantissa >> EXTRA_BITS;
        int32_t  __shiftDist = 0x427 - __exp;
        if (__shiftDist > 0) __sig = __nv_internal_fp64emu_shr_jam64(__sig, (uint32_t)__shiftDist);
        return __nv_internal_fp64emu_round_to_ui32<_Rm>(__sign, __sig);
    }

    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API  int64_t __nv_internal_fp64emu_fpbits64_unpacked_to_ll (fpbits64_unpacked_t __x) noexcept
    {
        const bool    __sign = (__x.sign != 0);
        const int32_t __exp  = (int32_t)__x.exponent;
        if (__exp == __FP64EMU_CVT_NAN_EXP) return (int64_t)0x8000000000000000ULL;            // NaN -> indefinite
        if (__exp == __FP64EMU_CVT_INF_EXP) return __sign ? (int64_t)0x8000000000000000ULL
                                                      : (int64_t)0x7FFFFFFFFFFFFFFFULL;
        uint64_t __sig = __x.mantissa >> EXTRA_BITS;
        int32_t  __shiftDist = 0x433 - __exp;
        uint64_t __sig_int, __sig_extra;
        if (__shiftDist <= 0)
        {
            if (__shiftDist < -11) return __sign ? (int64_t)0x8000000000000000ULL
                                             : (int64_t)0x7FFFFFFFFFFFFFFFULL;
            __sig_int   = __sig << (-__shiftDist);
            __sig_extra = 0;
        }
        else if (__shiftDist < 64)
        {
            __sig_int   = __sig >> __shiftDist;
            __sig_extra = __sig << (-__shiftDist & 63);
        }
        else
        {
            __sig_int   = 0;
            __sig_extra = (__shiftDist == 64) ? __sig : (uint64_t)(__sig != 0);
        }
        return __nv_internal_fp64emu_round_to_i64<_Rm>(__sign, __sig_int, __sig_extra);
    }

    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API  uint64_t __nv_internal_fp64emu_fpbits64_unpacked_to_ull (fpbits64_unpacked_t __x) noexcept
    {
        const bool    __sign = (__x.sign != 0);
        const int32_t __exp  = (int32_t)__x.exponent;
        if (__exp == __FP64EMU_CVT_NAN_EXP) return 0x8000000000000000ULL;                     // NaN -> indefinite
        if (__exp == __FP64EMU_CVT_INF_EXP) return __sign ? 0ULL : 0xFFFFFFFFFFFFFFFFULL;
        uint64_t __sig = __x.mantissa >> EXTRA_BITS;
        int32_t  __shiftDist = 0x433 - __exp;
        uint64_t __sig_int, __sig_extra;
        if (__shiftDist <= 0)
        {
            if (__shiftDist < -11) return __sign ? 0ULL : 0xFFFFFFFFFFFFFFFFULL;
            __sig_int   = __sig << (-__shiftDist);
            __sig_extra = 0;
        }
        else if (__shiftDist < 64)
        {
            __sig_int   = __sig >> __shiftDist;
            __sig_extra = __sig << (-__shiftDist & 63);
        }
        else
        {
            __sig_int   = 0;
            __sig_extra = (__shiftDist == 64) ? __sig : (uint64_t)(__sig != 0);
        }
        return __nv_internal_fp64emu_round_to_ui64<_Rm>(__sign, __sig_int, __sig_extra);
    }

    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API  float __nv_internal_fp64emu_fpbits64_unpacked_to_float (fpbits64_unpacked_t __x) noexcept
    {
        fpbits64_t __x_packed = __nv_internal_fp64emu_pack<_Rm>(__x);
        return __nv_internal_fp64emu_fpbits64_to_float(__x_packed);
    }

    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API fpbits64_unpacked_t __nv_internal_fp64emu_float_to_fpbits64_unpacked  (float __x) noexcept     
    { 
        fpbits64_t __x_packed = __nv_internal_fp64emu_float_to_fpbits64(__x);
        return __nv_internal_fp64emu_unpack(__x_packed);
    }

    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API fpbits64_unpacked_t __nv_internal_fp64emu_int_to_fpbits64_unpacked  (int32_t __x) noexcept     
    { 
        fpbits64_t __x_packed = __nv_internal_fp64emu_int_to_fpbits64(__x);
        return __nv_internal_fp64emu_unpack(__x_packed);
    }

    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API fpbits64_unpacked_t __nv_internal_fp64emu_uint_to_fpbits64_unpacked  (uint32_t __x) noexcept     
    { 
        fpbits64_t __x_packed = __nv_internal_fp64emu_uint_to_fpbits64(__x);
        return __nv_internal_fp64emu_unpack(__x_packed);
    }   

    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API fpbits64_unpacked_t __nv_internal_fp64emu_ull_to_fpbits64_unpacked  (uint64_t __x) noexcept     
    { 
        fpbits64_t __x_packed = __nv_internal_fp64emu_ull_to_fpbits64(__x);
        return __nv_internal_fp64emu_unpack(__x_packed);
    }

    template<fpemu::rounding _Rm   = fpemu::rounding::def,
             fp64emu_accuracy   _Acc = fp64emu_accuracy::def>
    _CCCL_TRIVIAL_API fpbits64_unpacked_t __nv_internal_fp64emu_ll_to_fpbits64_unpacked  (int64_t __x) noexcept     
    { 
        fpbits64_t __x_packed = __nv_internal_fp64emu_ll_to_fpbits64(__x);
        return __nv_internal_fp64emu_unpack(__x_packed);
    }

#endif // __FPEMU_UNPACKED__ == 1

} // namespace impl

// ============================================================================
// Builtin declarations/implementations for conversion operations
// ============================================================================
#if defined(__FPEMU_INLINE__)
#if (__FPEMU_PACKED_VIA_UNPACKED__ == 1)
// Packed-via-unpacked (testing): route the packed conversion builtins through the
// unpacked cores. fp->int unpack(x) then the rounding-aware unpacked core; fp->fp
// goes through the universal unpack/pack; int/fp->fp builds the unpacked value and
// packs (rn -- the integer/widening conversions are exact or already rounded).
// The pure bit-reinterpret casts (fpbits64<->ull) are NOT rerouted: they must
// preserve the exact bit pattern and have no unpacked-core equivalent.
__FPEMU_BUILTIN_DECL__ double     __nv_fp64emu_to_double (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ float      __nv_fp64emu_to_float  (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_float (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rn (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rz (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::rz> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_ru (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::ru> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rd (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::rd> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rn (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rz (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::rz> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_ru (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::ru> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rd (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::rd> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rn (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rz (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::rz> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_ru (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::ru> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rd (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::rd> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rn (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rz (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::rz> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_ru (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::ru> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rd (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::rd> (impl::__nv_internal_fp64emu_unpack (__x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_double (double __x) noexcept   { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked (__x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_float  (float __x) noexcept    { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_float_to_fpbits64_unpacked (__x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_int    (int32_t __x) noexcept  { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_int_to_fpbits64_unpacked (__x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_uint   (uint32_t __x) noexcept { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_uint_to_fpbits64_unpacked (__x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ll     (int64_t __x) noexcept  { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_ll_to_fpbits64_unpacked (__x)); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ull    (uint64_t __x) noexcept { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn> (impl::__nv_internal_fp64emu_ull_to_fpbits64_unpacked (__x)); }
#else
__FPEMU_BUILTIN_DECL__ double     __nv_fp64emu_to_double (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_double (__x); }
__FPEMU_BUILTIN_DECL__ float      __nv_fp64emu_to_float  (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_float (__x); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rn (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_int<fpemu::rounding::rn> (__x); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rz (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_int<fpemu::rounding::rz> (__x); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_ru (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_int<fpemu::rounding::ru> (__x); }
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rd (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_int<fpemu::rounding::rd> (__x); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rn (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_uint<fpemu::rounding::rn> (__x); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rz (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_uint<fpemu::rounding::rz> (__x); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_ru (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_uint<fpemu::rounding::ru> (__x); }
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rd (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_uint<fpemu::rounding::rd> (__x); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rn (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_ll<fpemu::rounding::rn> (__x); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rz (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_ll<fpemu::rounding::rz> (__x); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_ru (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_ll<fpemu::rounding::ru> (__x); }
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rd (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_ll<fpemu::rounding::rd> (__x); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rn (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_ull<fpemu::rounding::rn> (__x); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rz (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_ull<fpemu::rounding::rz> (__x); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_ru (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_ull<fpemu::rounding::ru> (__x); }
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rd (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_to_ull<fpemu::rounding::rd> (__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_double (double __x) noexcept   { return impl::__nv_internal_fp64emu_double_to_fpbits64 (__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_float  (float __x) noexcept    { return impl::__nv_internal_fp64emu_float_to_fpbits64 (__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_int    (int32_t __x) noexcept  { return impl::__nv_internal_fp64emu_int_to_fpbits64 (__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_uint   (uint32_t __x) noexcept { return impl::__nv_internal_fp64emu_uint_to_fpbits64 (__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ll     (int64_t __x) noexcept  { return impl::__nv_internal_fp64emu_ll_to_fpbits64 (__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ull    (uint64_t __x) noexcept { return impl::__nv_internal_fp64emu_ull_to_fpbits64 (__x); }
#endif // __FPEMU_PACKED_VIA_UNPACKED__
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_fpbits64_cast_ull  (fpbits64_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_cast_ull (__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ull_cast_fpbits64  (uint64_t __x) noexcept   { return impl::__nv_internal_fp64emu_ull_cast_fpbits64 (__x); }
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpack  (fpbits64_t __a) noexcept          { return impl::__nv_internal_fp64emu_unpack(__a); }
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rn (fpbits64_unpacked_t __a) noexcept { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rn>(__a); }
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rz (fpbits64_unpacked_t __a) noexcept { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rz>(__a); }
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_ru (fpbits64_unpacked_t __a) noexcept { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::ru>(__a); }
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rd (fpbits64_unpacked_t __a) noexcept { return impl::__nv_internal_fp64emu_pack<fpemu::rounding::rd>(__a); }
__FPEMU_BUILTIN_DECL__ int32_t  __nv_fp64emu_unpacked_to_int            (fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_int<fpemu::rounding::rz>(__x); }
__FPEMU_BUILTIN_DECL__ uint32_t __nv_fp64emu_unpacked_to_uint           (fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_uint<fpemu::rounding::rz>(__x); }
__FPEMU_BUILTIN_DECL__ int64_t  __nv_fp64emu_unpacked_to_ll             (fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ll<fpemu::rounding::rz>(__x); }
__FPEMU_BUILTIN_DECL__ uint64_t __nv_fp64emu_unpacked_to_ull            (fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_ull<fpemu::rounding::rz>(__x); }
__FPEMU_BUILTIN_DECL__ float    __nv_fp64emu_unpacked_to_float          (fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_float(__x); }
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_to_double         (fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double(__x); }
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_high_to_double(fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double<fpemu::rounding::rn, fp64emu_accuracy::high>(__x); }
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_mid_to_double     (fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double<fpemu::rounding::rn, fp64emu_accuracy::mid>(__x); }
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_low_to_double    (fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_to_double<fpemu::rounding::rn, fp64emu_accuracy::low>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_int             (int32_t __x) noexcept  { return impl::__nv_internal_fp64emu_int_to_fpbits64_unpacked(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_uint            (uint32_t __x) noexcept { return impl::__nv_internal_fp64emu_uint_to_fpbits64_unpacked(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ll              (int64_t __x) noexcept  { return impl::__nv_internal_fp64emu_ll_to_fpbits64_unpacked(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ull             (uint64_t __x) noexcept { return impl::__nv_internal_fp64emu_ull_to_fpbits64_unpacked(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_float           (float __x) noexcept    { return impl::__nv_internal_fp64emu_float_to_fpbits64_unpacked(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_double          (double __x) noexcept   { return impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_from_double (double __x) noexcept   { return impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked<fpemu::rounding::rn, fp64emu_accuracy::high>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_from_double      (double __x) noexcept   { return impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked<fpemu::rounding::rn, fp64emu_accuracy::mid>(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_from_double     (double __x) noexcept   { return impl::__nv_internal_fp64emu_double_to_fpbits64_unpacked<fpemu::rounding::rn, fp64emu_accuracy::low>(__x); }
__FPEMU_BUILTIN_DECL__ uint64_t            __nv_fp64emu_unpacked_fpbits64_cast_ull           (fpbits64_unpacked_t __x) noexcept { return impl::__nv_internal_fp64emu_fpbits64_unpacked_cast_ull(__x); }
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_ull_cast_fpbits64           (uint64_t __x) noexcept { return impl::__nv_internal_fp64emu_ull_cast_fpbits64_unpacked(__x); }
#endif
#else
__FPEMU_BUILTIN_DECL__ double     __nv_fp64emu_to_double (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_double (double x) noexcept ;
__FPEMU_BUILTIN_DECL__ float      __nv_fp64emu_to_float  (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_float  (float x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_int    (int32_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_uint   (uint32_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ll     (int64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_from_ull    (uint64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_fpbits64_cast_ull  (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t __nv_fp64emu_ull_cast_fpbits64  (uint64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rn (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rz (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_ru (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ int32_t    __nv_fp64emu_to_int_rd (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rn (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rz (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_ru (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint32_t   __nv_fp64emu_to_uint_rd (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rn (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rz (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_ru (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ int64_t    __nv_fp64emu_to_ll_rd (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rn (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rz (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_ru (fpbits64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint64_t   __nv_fp64emu_to_ull_rd (fpbits64_t x) noexcept ;
#if __FPEMU_UNPACKED__ == 1
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpack  (fpbits64_t a) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rn (fpbits64_unpacked_t a) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rz (fpbits64_unpacked_t a) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_ru (fpbits64_unpacked_t a) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_t          __nv_fp64emu_pack_rd (fpbits64_unpacked_t a) noexcept ;
__FPEMU_BUILTIN_DECL__ int32_t  __nv_fp64emu_unpacked_to_int            (fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint32_t __nv_fp64emu_unpacked_to_uint           (fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ int64_t  __nv_fp64emu_unpacked_to_ll             (fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint64_t __nv_fp64emu_unpacked_to_ull            (fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ float    __nv_fp64emu_unpacked_to_float          (fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_to_double         (fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_high_to_double(fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_mid_to_double     (fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ double   __nv_fp64emu_unpacked_low_to_double    (fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_int             (int32_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_uint            (uint32_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ll              (int64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_ull             (uint64_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_float           (float x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_from_double          (double x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_high_from_double (double x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_mid_from_double      (double x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_low_from_double     (double x) noexcept ;
__FPEMU_BUILTIN_DECL__ uint64_t            __nv_fp64emu_unpacked_fpbits64_cast_ull           (fpbits64_unpacked_t x) noexcept ;
__FPEMU_BUILTIN_DECL__ fpbits64_unpacked_t __nv_fp64emu_unpacked_ull_cast_fpbits64           (uint64_t x) noexcept ;
#endif
#endif // __FPEMU_INLINE__

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_IMPL_CVT_HPP__

#if defined(__FPEMU_API_CLASSES_DEFINED__) && !defined(__FPEMU_CVT_API_MERGED__)
#define __FPEMU_CVT_API_MERGED__
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


// ============================================================================
// API (merged from fp64emu_cvt_api.hpp)
// ============================================================================

    // Type conversion to fp64emu_t with other method
    template<fp64emu_accuracy _AccSrc> 
    template<fp64emu_accuracy _AccDst> 
        _CCCL_API inline fp64emu_t<_AccSrc>::operator fp64emu_t<_AccDst>() const noexcept 
        { 
            return fp64emu_t<_AccDst>(fpbits64_construct, bits); 
        }

#if __FPEMU_UNPACKED__ == 1
    // Type conversion from fp64emu_t to fp64emu_unpacked_t
    template<fp64emu_accuracy _AccSrc> 
    template<fp64emu_accuracy _AccDst> 
        _CCCL_API inline fp64emu_t<_AccSrc>::operator fp64emu_unpacked_t<_AccDst>() const noexcept 
        { 
            fpbits64_unpacked_t __bits_unpacked = __nv_fp64emu_unpack(bits);
            return fp64emu_unpacked_t<_AccDst>(fpbits64_construct, __bits_unpacked); 
        }
#endif // __FPEMU_UNPACKED__ == 1

    /*
    // Type conversions from other types to fp64emu_t 
    */
    // from double
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::fp64emu_t(double __d) noexcept { 
        bits = __nv_fp64emu_from_double (__d); }
    // from float
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::fp64emu_t(float __d) noexcept { 
        bits = __nv_fp64emu_from_float (__d); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline fp64emu_t<_Acc> __float2double  (float __x) noexcept { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_from_float (__x)); }
    // from int32_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::fp64emu_t(int32_t __d) noexcept { 
        bits = __nv_fp64emu_from_int (__d); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc> __int2double  (int32_t __x) noexcept { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_from_int (__x)); }
    // from uint32_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::fp64emu_t(uint32_t __d) noexcept { 
        bits = __nv_fp64emu_from_uint (__d); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc> __uint2double (uint32_t __x) noexcept { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_from_uint (__x)); }
    // from int64_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::fp64emu_t(int64_t __d) noexcept { 
        bits = __nv_fp64emu_from_ll (__d); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc> __ll2double (int64_t __x) noexcept { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_from_ll (__x)); }
    // from uint64_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::fp64emu_t(uint64_t __d) noexcept { 
        bits = __nv_fp64emu_from_ull (__d); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc> __ull2double(uint64_t __x) noexcept { 
        return fp64emu_t<_Acc>(fpbits64_construct, __nv_fp64emu_from_ull (__x)); }

    /*
    // Type conversions from fp64emu_t to other types
    */
    // to double
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::operator double() const noexcept { 
        return __nv_fp64emu_to_double (bits); }
    // to float
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::operator float()  const noexcept { 
        return __nv_fp64emu_to_float (bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline float  __double2float (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_float (__x.bits); }
    // to int32_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::operator int32_t()  const noexcept { 
        return __nv_fp64emu_to_int_rz (bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline int32_t __double2int_rn (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_int_rn (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline int32_t __double2int_rz (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_int_rz (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline int32_t __double2int_ru (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_int_ru (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline int32_t __double2int_rd (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_int_rd (__x.bits); }
    // to uint32_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::operator uint32_t() const noexcept { 
        return __nv_fp64emu_to_uint_rz (bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline uint32_t __double2uint_rn (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_uint_rn (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline uint32_t __double2uint_rz (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_uint_rz (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline uint32_t __double2uint_ru (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_uint_ru (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline uint32_t __double2uint_rd (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_uint_rd (__x.bits); }
    // to int64_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::operator int64_t()  const noexcept { 
        return __nv_fp64emu_to_ll_rz (bits); }    
    template<fp64emu_accuracy _Acc>  _CCCL_API inline int64_t __double2ll_rn (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_ll_rn (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline int64_t __double2ll_rz (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_ll_rz (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline int64_t __double2ll_ru (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_ll_ru (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline int64_t __double2ll_rd (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_ll_rd (__x.bits); }    
    // to uint64_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_t<_Acc>::operator uint64_t() const noexcept { 
        return __nv_fp64emu_to_ull_rz (bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline uint64_t __double2ull_rn (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_ull_rn (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline uint64_t __double2ull_rz (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_ull_rz (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline uint64_t __double2ull_ru (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_ull_ru (__x.bits); }
    template<fp64emu_accuracy _Acc>  _CCCL_API inline uint64_t __double2ull_rd (fp64emu_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_to_ull_rd (__x.bits); }

#if __FPEMU_UNPACKED__ == 1

    // Type conversion from fp64emu_unpacked_t with other method
    template<fp64emu_accuracy _AccSrc> 
    template<fp64emu_accuracy _AccDst> 
        _CCCL_API inline fp64emu_unpacked_t<_AccSrc>::operator fp64emu_unpacked_t<_AccDst>() const noexcept 
        { 
            return fp64emu_unpacked_t<_AccDst>(fpbits64_construct, bits); 
        }

    // Type conversion from fp64emu_unpacked_t to fp64emu_t
    template<fp64emu_accuracy _AccSrc> 
    template<fp64emu_accuracy _AccDst> 
        _CCCL_API inline fp64emu_unpacked_t<_AccSrc>::operator fp64emu_t<_AccDst>() const noexcept 
        { 
            fpbits64_t __bits_packed = __nv_fp64emu_pack_rn(bits);
            return fp64emu_t<_AccDst>(fpbits64_construct, __bits_packed); 
        }

    /*
    // Type conversions from other types to fp64emu_unpacked_t 
    */
    // from double 
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::fp64emu_unpacked_t(double __d) noexcept 
    { 
    if      constexpr (_Acc == fp64emu_accuracy::high) { bits = __nv_fp64emu_unpacked_high_from_double(__d); }
        else if constexpr (_Acc == fp64emu_accuracy::mid)      { bits = __nv_fp64emu_unpacked_mid_from_double(__d); }
        else                                      { bits = __nv_fp64emu_unpacked_from_double(__d); }
    }
    // from float 
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::fp64emu_unpacked_t(float __d) noexcept { 
        bits = __nv_fp64emu_unpacked_from_float (__d);  }
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>  __float2double (float __x) noexcept { 
        return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_from_float (__x)); }
    // from int32_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::fp64emu_unpacked_t(int32_t __d) noexcept { 
        bits = __nv_fp64emu_unpacked_from_int (__d); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>  __int2double (int32_t __x) noexcept { 
        return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_from_int (__x)); }
    // from uint32_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::fp64emu_unpacked_t(uint32_t __d) noexcept { 
        bits = __nv_fp64emu_unpacked_from_uint (__d); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>  __uint2double (uint32_t __x) noexcept { 
        return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_from_uint (__x)); }
    // from int64_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::fp64emu_unpacked_t(int64_t __d) noexcept { 
        bits = __nv_fp64emu_unpacked_from_ll (__d); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>  __ll2double (int64_t __x) noexcept  { 
        return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_from_ll (__x)); }
    // from uint64_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::fp64emu_unpacked_t(uint64_t __d) noexcept { 
        bits = __nv_fp64emu_unpacked_from_ull (__d); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>  __ull2double (uint64_t __x) noexcept { 
        return fp64emu_unpacked_t<_Acc>(fpbits64_construct, __nv_fp64emu_unpacked_from_ull (__x)); }

    /*
    // Conversion operators from fp64emu_unpacked_t to other types
    */
    // to double
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::operator double() const noexcept 
    { 
if      constexpr (_Acc == fp64emu_accuracy::high) { return __nv_fp64emu_unpacked_high_to_double(bits); }
       else if constexpr (_Acc == fp64emu_accuracy::mid)      { return __nv_fp64emu_unpacked_mid_to_double(bits); }
       else                                      { return __nv_fp64emu_unpacked_to_double(bits); }
    }
    // to float
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::operator float()  const noexcept { 
        return __nv_fp64emu_unpacked_to_float (bits); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline float    __double2float   (fp64emu_unpacked_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_unpacked_to_float (__x.bits); }
    // to int32_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::operator int32_t()  const noexcept { 
        return __nv_fp64emu_unpacked_to_int (bits); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline int32_t  __double2int_rz  (fp64emu_unpacked_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_unpacked_to_int (__x.bits); }
    // to uint32_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::operator uint32_t() const noexcept { 
        return __nv_fp64emu_unpacked_to_uint (bits); }
    template<fp64emu_accuracy _Acc> _CCCL_API inline uint32_t __double2uint_rz (fp64emu_unpacked_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_unpacked_to_uint (__x.bits); }
    // to int64_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::operator int64_t()  const noexcept { 
        return __nv_fp64emu_unpacked_to_ll (bits); } 
    template<fp64emu_accuracy _Acc> _CCCL_API inline int64_t  __double2ll_rz   (fp64emu_unpacked_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_unpacked_to_ll (__x.bits); }
    // to uint64_t
    template<fp64emu_accuracy _Acc> _CCCL_API inline fp64emu_unpacked_t<_Acc>::operator uint64_t() const noexcept { 
        return __nv_fp64emu_unpacked_to_ull (bits); } 
    template<fp64emu_accuracy _Acc> _CCCL_API inline uint64_t __double2ull_rz  (fp64emu_unpacked_t<_Acc> __x) noexcept { 
        return __nv_fp64emu_unpacked_to_ull (__x.bits); }
    template<typename _To, fp64emu_accuracy _M2>
        _CCCL_API inline _To bit_cast(const fp64emu_unpacked_t<_M2>& __from) noexcept
        {
            // Pack the unpacked value to get IEEE-754 representation
            fpbits64_t __packed = __nv_fp64emu_pack_rn(__from.bits);
            return fpemu::bit_cast<_To>(__packed);
        }
#endif // __FPEMU_UNPACKED__ == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>
#endif // __FPEMU_CVT_API_MERGED__
