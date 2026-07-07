//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_H
#define _CUDA___FP_FPEMU_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/**
 * @file fpemu.h
 * @brief Main header file for the FPEMU floating point scalar emulation library
 *
 * This is the main header file that provides access to the complete FPEMU library.
 * It includes all the necessary headers for:
 *
 * - Core definitions, macros and enumerations (fpemu_common.h)
 * - Class templates (fp64emu_t, fp64emu_unpacked_t)
 * - Public API functions (operators, builtins, conversions)
 * - Implementation files for specific scalar operations:
 *   - Comparison operations (fpemu_impl_cmp.h)
 *   - Type conversions (fpemu_impl_cvt.h)
 *   - Fused multiply-add (fpemu_impl_fma.h)
 *   - Addition (fpemu_impl_add.h)
 *   - Subtraction (fpemu_impl_sub.h)
 *   - Multiplication (fpemu_impl_mul.h)
 *   - Division (fpemu_impl_div.h)
 *   - Square root (fpemu_impl_sqrt.h)
 *   - Other operations (fpemu_impl_others.h)
 *
 * The library provides IEEE-754 compliant emulated scalar floating point operations
 * with configurable rounding modes and computation methods.
 *
 * Accuracy levels (template parameter 'fp64emu_accuracy'):
 *   - fp64emu_accuracy::high — correctly rounded, full IEEE-754 range including
 *                        infinities, NaNs, and subnormals
 *   - fp64emu_accuracy::mid  — up to 1-2 least significant mantissa bits of error,
 *                        limited INF, NaN and subnormal support
 *   - fp64emu_accuracy::low  — up to half of the mantissa bits may be lost,
 *                        limited INF, NaN and subnormal support
 *   - fp64emu_accuracy::def  — default selector; equals high (IEEE-correct)
 *
 * The API supports both host and device code through appropriate decorators and
 * can utilize different computational backends based on template parameters.
 */
 
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/cstdint>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/__fp/fpemu_common.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

    /**
    * @brief Tag type for explicit construction of fpbits64_t values
    *
    * This struct serves as a tag type to disambiguate constructors that take
    * raw bit values. It prevents implicit conversions from raw integers to
    * floating-point values and ensures that bit-level construction is explicit.
    *
    * Usage:
    *   fpbits64_t value = fpbits64_construct_t{}, raw_bits;
    *   // or by the constexpr instance:
    *   fpbits64_t value = fpbits64_construct, raw_bits;
    */
    struct fpbits64_construct_t { explicit fpbits64_construct_t() = default; };

    // Global constant instance of fpbits64_construct_t for convenient usage
    // (host/device accessible)
    _CCCL_GLOBAL_CONSTANT fpbits64_construct_t fpbits64_construct{};

    // Forward declaration of unpacked floating-point class
    template <fp64emu_accuracy _Met> class fp64emu_unpacked_t;

    /**
    * @brief Primary emulated double-precision floating-point class template
    *
    * The fp64emu_t class template represents a double-precision (64-bit)
    * floating-point number, emulated according to IEEE-754 semantics but with 
    * configurable accuracy level.
    *
    * @tparam met Accuracy level (fp64emu_accuracy::high, mid, low; def == high)
    *              - high: Correctly rounded with full IEEE-754 range
    *              - mid: 1-2 LSB error with normal range
    *              - low: Low accuracy with normal range
    *
    * This class provides:
    *   - Storage of the value as fpbits64_t (raw IEEE-754 format)
    *   - Construction from and conversion to standard C++ types (int, float, double)
    *   - Arithmetic operators and mathematical functions
    *   - Fine-grained control over rounding and accuracy level
    *   - Portable host/device compatibility (CUDA/HIP/etc)
    * 
    * Usage:
    *   fp64emu_t<fp64emu_accuracy::high> x{1.5};
    *   fp64emu_t<> y = x + 2.0;
    *   double z = static_cast<double>(y);
    */
    template <fp64emu_accuracy _Met = fp64emu_accuracy::def> 
    class fp64emu_t 
    {
     public:

        // Internal representation of the floating-point value
        // fpbits64_t is defined in fpemu_common.h
        fpbits64_t bits;
        
        /*
        // Constructors and assignment operators
        */
        // Basic constructors
        _CCCL_API inline fp64emu_t() noexcept : bits{0u} {}
        _CCCL_API inline fp64emu_t(fpbits64_construct_t, const fpbits64_t& __f) noexcept : bits(__f) {}
        /*
        // Defaulted copy constructor (trivially copyable)
        // Note: NVCC implicitly makes defaulted special members __host__ __device__
        */
        fp64emu_t(const fp64emu_t& __other) = default;

        /*
        // Copy constructor from volatile fp64emu_t
        // Template so it is NOT a copy constructor per the C++ standard.
        // The volatile overloads are wrapped in dummy templates
        // so that the C++ standard does not consider them copy constructors/assignment
        // operators (a template is never a copy constructor or copy assignment operator),
        // preserving trivial copyability while retaining volatile access support.
        */
        template<typename _Dummy = void>
        _CCCL_API inline fp64emu_t(const volatile fp64emu_t& __other) noexcept : bits(__other.bits) {}

        // Defaulted copy assignment operator (trivially copyable)
        fp64emu_t& operator=(const fp64emu_t& __other) = default;

        /*
        // Assignment operator to volatile fp64emu_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
        */
        template<typename _Dummy = void>
        _CCCL_API inline void operator=(const fp64emu_t& __other) volatile noexcept { bits = __other.bits; }

        /*
        // Assignment operator from volatile fp64emu_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        */
        template<typename _Dummy = void>
        _CCCL_API inline fp64emu_t& operator=(const volatile fp64emu_t& __other) noexcept { bits = __other.bits; return *this; }

        /*
        // Conversion operators
        */
        // ==== Conversions from other types to fp64emu_t:
        // Implicit conversions from floating-point types
        _CCCL_API inline fp64emu_t(float __f) noexcept ;
        _CCCL_API inline fp64emu_t(double __d) noexcept ;
        // Construction from any standard integer type (int / long / long long + unsigned).
        // 32-bit and narrower are lossless in double and stay implicit; 64-bit may lose
        // precision and are explicit (as the prior fixed-width API required). Dispatch is
        // by width and signedness to the accuracy-correct integer builtins (via the private
        // out-of-line helpers below), so every integer type is handled portably.
        // bool / character types are excluded by __cccl_is_integer_v.
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND(sizeof(_Tp) <= sizeof(int32_t)))
        _CCCL_API inline fp64emu_t(_Tp __i) noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>) { __set_from_int (static_cast<int32_t>(__i)); }
            else                                                        { __set_from_uint(static_cast<uint32_t>(__i)); }
        }
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND(sizeof(_Tp) > sizeof(int32_t)))
        _CCCL_API explicit inline fp64emu_t(_Tp __i) noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>) { __set_from_ll (static_cast<int64_t>(__i)); }
            else                                                        { __set_from_ull(static_cast<uint64_t>(__i)); }
        }
        // Type conversion to fp64emu_t with other accuracy and range
        template<fp64emu_accuracy _Acc = _Met> _CCCL_API inline operator fp64emu_t<_Acc>() const noexcept ;
        // Type conversion from fp64emu_t to fp64emu_unpacked_t (explicit to avoid overload ambiguity)
        template<fp64emu_accuracy _Acc = _Met> _CCCL_API explicit inline operator fp64emu_unpacked_t<_Acc>() const noexcept ;

        // ==== Conversion from fp64emu_t to other types:
        // Implicit conversion to double
        _CCCL_API inline operator double() const noexcept ;
        // Explicit conversions to other types
        _CCCL_API explicit inline operator float()    const noexcept ;
        // Explicit conversion to any standard integer type (int / long / long long + unsigned).
        // Dispatches by width and signedness to the accuracy-correct integer builtins (via the
        // private out-of-line helpers below); excludes bool / character types.
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
        _CCCL_API inline explicit operator _Tp() const noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>)
            {
                if constexpr (sizeof(_Tp) <= sizeof(int32_t)) { return static_cast<_Tp>(__to_int()); }
                else                                          { return static_cast<_Tp>(__to_ll());  }
            }
            else
            {
                if constexpr (sizeof(_Tp) <= sizeof(uint32_t)) { return static_cast<_Tp>(__to_uint()); }
                else                                           { return static_cast<_Tp>(__to_ull());  }
            }
        }

     private:
        // Accuracy-correct integer <-> value helpers (defined out-of-line where the fpemu
        // builtins are visible). Kept non-template so the definitions stay out-of-line.
        _CCCL_API inline void     __set_from_int (int32_t  __i) noexcept ;
        _CCCL_API inline void     __set_from_uint(uint32_t __i) noexcept ;
        _CCCL_API inline void     __set_from_ll  (int64_t  __i) noexcept ;
        _CCCL_API inline void     __set_from_ull (uint64_t __i) noexcept ;
        _CCCL_API inline int32_t  __to_int () const noexcept ;
        _CCCL_API inline uint32_t __to_uint() const noexcept ;
        _CCCL_API inline int64_t  __to_ll  () const noexcept ;
        _CCCL_API inline uint64_t __to_ull () const noexcept ;
     public:

        /*
        //  CUDA builtins functions for conversions
        */
        // double to float
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline float  __double2float (fp64emu_t<_Acc> __x) noexcept ;
        // double to integer
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rn (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rz (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_ru (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rd (fp64emu_t<_Acc> __x) noexcept ;
        // double to unsigned integer
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rn (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rz (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_ru (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rd (fp64emu_t<_Acc> __x) noexcept ;
        // double to signed integer
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rn (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rz (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_ru (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rd (fp64emu_t<_Acc> __x) noexcept ;
        // double to unsigned integer
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rn (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rz (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_ru (fp64emu_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rd (fp64emu_t<_Acc> __x) noexcept ;
        // other types to double
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __int2double   (int32_t __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __uint2double  (uint32_t __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __ll2double    (int64_t __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __ull2double   (uint64_t __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __float2double (float __x) noexcept ;
    
        /*
        // Arithmetic operations:
        */
        // === mul ===
        // (*)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_t<_Acc> operator*(const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) noexcept ;
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t operator*(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) * fp64emu_t(__y); }
        // dmul_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dmul_rn(const _T1& __x, const _T2& __y) noexcept { return __dmul_rn(fp64emu_t(__x), fp64emu_t(__y)); }
        // dmul_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dmul_rz(const _T1& __x, const _T2& __y) noexcept { return __dmul_rz(fp64emu_t(__x), fp64emu_t(__y)); }
        // dmul_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dmul_ru(const _T1& __x, const _T2& __y) noexcept { return __dmul_ru(fp64emu_t(__x), fp64emu_t(__y)); }
        // dmul_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dmul_rd(const _T1& __x, const _T2& __y) noexcept { return __dmul_rd(fp64emu_t(__x), fp64emu_t(__y)); }
        
        // === div ===
        // (/)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_t<_Acc> operator/(const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) noexcept ;
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t operator/(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) / fp64emu_t(__y); }
        // ddiv_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __ddiv_rn(const _T1& __x, const _T2& __y) noexcept { return __ddiv_rn(fp64emu_t(__x), fp64emu_t(__y)); }
        // ddiv_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __ddiv_rz(const _T1& __x, const _T2& __y) noexcept { return __ddiv_rz(fp64emu_t(__x), fp64emu_t(__y)); }
        // ddiv_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __ddiv_ru(const _T1& __x, const _T2& __y) noexcept { return __ddiv_ru(fp64emu_t(__x), fp64emu_t(__y)); }
        // ddiv_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __ddiv_rd(const _T1& __x, const _T2& __y) noexcept { return __ddiv_rd(fp64emu_t(__x), fp64emu_t(__y)); }

        // === add ===
        // (+)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_t<_Acc> operator+(const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) noexcept ;
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t operator+(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) + fp64emu_t(__y); }
        // dadd_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dadd_rn(const _T1& __x, const _T2& __y) noexcept {  return __dadd_rn(fp64emu_t(__x), fp64emu_t(__y)); }
        // dadd_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dadd_rz(const _T1& __x, const _T2& __y) noexcept {  return __dadd_rz(fp64emu_t(__x), fp64emu_t(__y)); }
        // dadd_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dadd_ru(const _T1& __x, const _T2& __y) noexcept { return __dadd_ru(fp64emu_t(__x), fp64emu_t(__y)); }
        // dadd_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dadd_rd(const _T1& __x, const _T2& __y) noexcept { return __dadd_rd(fp64emu_t(__x), fp64emu_t(__y)); }

        // === sub ===
        // (-)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_t<_Acc> operator-(const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y) noexcept ;
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t operator-(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) - fp64emu_t(__y); }
        // dsub_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dsub_rn(const _T1& __x, const _T2& __y) noexcept { return __dsub_rn(fp64emu_t(__x), fp64emu_t(__y)); }
        // dsub_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dsub_rz(const _T1& __x, const _T2& __y) noexcept { return __dsub_rz(fp64emu_t(__x), fp64emu_t(__y)); }
        // dsub_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dsub_ru(const _T1& __x, const _T2& __y) noexcept { return __dsub_ru(fp64emu_t(__x), fp64emu_t(__y)); }
        // dsub_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_t __dsub_rd(const _T1& __x, const _T2& __y) noexcept { return __dsub_rd(fp64emu_t(__x), fp64emu_t(__y)); }

        // === sqrt ===
        // sqrt
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fp64emu_t sqrt(const _T1& __x) noexcept { return sqrt(fp64emu_t(__x)); }        
        // dsqrt_rn
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fp64emu_t __dsqrt_rn(const _T1& __x) noexcept { return __dsqrt_rn(fp64emu_t(__x)); }

        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fp64emu_t __dsqrt_rz(const _T1& __x) noexcept { return __dsqrt_rz(fp64emu_t(__x)); }
        // dsqrt_ru
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fp64emu_t __dsqrt_ru(const _T1& __x) noexcept { return __dsqrt_ru(fp64emu_t(__x)); }
        // dsqrt_rd
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fp64emu_t __dsqrt_rd(const _T1& __x) noexcept { return __dsqrt_rd(fp64emu_t(__x)); }

        // === fma ===
        // fma
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t> || ::cuda::std::is_same_v<_T3,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_t fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return fma(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dfma_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t> || ::cuda::std::is_same_v<_T3,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_t __fma_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_rn(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dfma_rz
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t> || ::cuda::std::is_same_v<_T3,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_t __fma_rz(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_rz(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dfma_ru
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t> || ::cuda::std::is_same_v<_T3,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_t __fma_ru(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_ru(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dfma_rd
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t> || ::cuda::std::is_same_v<_T3,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_t __fma_rd(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_rd(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }

        // === mad ===
        // mad
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t> || ::cuda::std::is_same_v<_T3,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_t mad(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return mad(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dmad_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t> || ::cuda::std::is_same_v<_T3,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_t __mad_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __mad_rn(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }

        // === dot ===
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3, typename _T4)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t> || ::cuda::std::is_same_v<_T3,fp64emu_t> || ::cuda::std::is_same_v<_T4,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3> || ::cuda::std::is_arithmetic_v<_T4>)))
            _CCCL_API friend  fp64emu_t dot(const _T1& __x1, const _T2& __y1, const _T3& __x2, const _T4& __y2) noexcept { return dot(fp64emu_t(__x1), fp64emu_t(__y1), fp64emu_t(__x2), fp64emu_t(__y2)); }

         // === cmul ===
         _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3, typename _T4)
         _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t> || ::cuda::std::is_same_v<_T3,fp64emu_t> || ::cuda::std::is_same_v<_T4,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3> || ::cuda::std::is_arithmetic_v<_T4>)))
             _CCCL_API friend void cmul(const _T1& __x_re, const _T2& __x_im, const _T3& __y_re, const _T4& __y_im, fp64emu_t& __r_re, fp64emu_t& __r_im) noexcept { cmul(fp64emu_t(__x_re), fp64emu_t(__x_im), fp64emu_t(__y_re), fp64emu_t(__y_im), __r_re, __r_im); }

        // Prefix increment/decrement
        _CCCL_API fp64emu_t& operator++() noexcept { this = this + fp64emu_t(1.0); return *this; }
        _CCCL_API fp64emu_t& operator--() noexcept { this = this - fp64emu_t(1.0); return *this; }
        // Postfix increment/decrement
        _CCCL_API fp64emu_t  operator++(int) noexcept { fp64emu_t __temp(*this); this = this + fp64emu_t(1.0); return __temp; }
        _CCCL_API fp64emu_t  operator--(int) noexcept { fp64emu_t __temp(*this); this = this - fp64emu_t(1.0); return __temp; }
        // Compound assignment operators
        _CCCL_API fp64emu_t& operator+=(const fp64emu_t& __other) noexcept { *this = *this + __other; return *this; }
        _CCCL_API fp64emu_t& operator-=(const fp64emu_t& __other) noexcept { *this = *this - __other; return *this; }
        _CCCL_API fp64emu_t& operator*=(const fp64emu_t& __other) noexcept { *this = *this * __other; return *this; }
        _CCCL_API fp64emu_t& operator/=(const fp64emu_t& __other) noexcept { *this = *this / __other; return *this; }
        // Unary negation operator (implementation in fpemu_impl_others.h)
        _CCCL_API fp64emu_t  operator-() const noexcept ;

        /*
        // Comparison operators:
        */       
        // equality (==)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator==(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) == fp64emu_t(__y); }
        // inequality (!=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator!=(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) != fp64emu_t(__y); }
        // less than (<)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator<(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) < fp64emu_t(__y); }
        // greater than (>)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator>(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) > fp64emu_t(__y); }
        // less than or equal to (<=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator<=(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) <= fp64emu_t(__y); }
        // greater than or equal to (>=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_t> || ::cuda::std::is_same_v<_T2,fp64emu_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator>=(const _T1& __x, const _T2& __y) noexcept { return fp64emu_t(__x) >= fp64emu_t(__y); }
    }; // class fp64emu_t 


    template <fp64emu_accuracy _Met = fp64emu_accuracy::def> 
    class fp64emu_unpacked_t 
    {
     public:

        // Internal representation of the unpacked floating-point value
        // fpbits64_unpacked_t is defined in fpemu_common.h
        fpbits64_unpacked_t bits;
        
        /*
        // Constructors and assignment operators
        */
        // Basic constructors
        _CCCL_API inline fp64emu_unpacked_t() noexcept : bits{0u, 0, 0} {}
        _CCCL_API inline fp64emu_unpacked_t(fpbits64_construct_t, const fpbits64_unpacked_t& __f) noexcept : bits(__f) {}
        /*
        // Defaulted copy constructor (trivially copyable)
        // Note: NVCC implicitly makes defaulted special members __host__ __device__
        */
        fp64emu_unpacked_t(const fp64emu_unpacked_t& __other) = default;

        /*
        // Copy constructor from volatile fp64emu_unpacked_t
        // Template so it is NOT a copy constructor per the C++ standard.
        // The volatile overloads are wrapped in dummy templates
        // so that the C++ standard does not consider them copy constructors/assignment
        // operators (a template is never a copy constructor or copy assignment operator),
        // preserving trivial copyability while retaining volatile access support.
        */
        template<typename _Dummy = void>
        _CCCL_API inline fp64emu_unpacked_t(const volatile fp64emu_unpacked_t& __other) noexcept
        { 
            bits.sign = __other.bits.sign; 
            bits.exponent = __other.bits.exponent; 
            bits.mantissa = __other.bits.mantissa; 
        }

        // Defaulted copy assignment operator (trivially copyable)
        fp64emu_unpacked_t& operator=(const fp64emu_unpacked_t& __other) = default;

        /*
        // Assignment operator to volatile fp64emu_unpacked_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
        */
        template<typename _Dummy = void>
        _CCCL_API inline void operator=(const fp64emu_unpacked_t& __other) volatile noexcept
        { 
            bits.sign = __other.bits.sign; 
            bits.exponent = __other.bits.exponent; 
            bits.mantissa = __other.bits.mantissa; 
        }

        /*
        // Assignment operator from volatile fp64emu_unpacked_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        */
        template<typename _Dummy = void>
        _CCCL_API inline fp64emu_unpacked_t& operator=(const volatile fp64emu_unpacked_t& __other) noexcept
        { 
            bits.sign = __other.bits.sign; 
            bits.exponent = __other.bits.exponent; 
            bits.mantissa = __other.bits.mantissa; 
            return *this; 
        }
        /*
        // Conversion operators
        */
        // ==== Conversions from other types to fp64emu_unpacked_t:
#if defined __CUDACC__
        // Implicit conversions from floating-point types 
        _CCCL_API  inline fp64emu_unpacked_t(float f) noexcept ;
        _CCCL_API  inline fp64emu_unpacked_t(double d) noexcept ;        
#  define _CCCL_FPEMU_UNP_NARROW_EXPLICIT
#else
        // Explicit conversions from floating-point types (to avoid ambiguity with packed type)
        _CCCL_API explicit inline fp64emu_unpacked_t(float __f) noexcept ;
        _CCCL_API explicit inline fp64emu_unpacked_t(double __d) noexcept ;        
#  define _CCCL_FPEMU_UNP_NARROW_EXPLICIT explicit
#endif
        // Construction from any standard integer type (int / long / long long + unsigned).
        // 32-bit and narrower are lossless in double; 64-bit values may lose precision. The
        // narrow-integer explicitness follows the surrounding float ctors (implicit on device,
        // explicit on host) to avoid ambiguity with the packed type; 64-bit is always explicit.
        // Dispatch is by width and signedness to the accuracy-correct integer builtins (via the
        // private out-of-line helpers below); bool / character types are excluded.
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND(sizeof(_Tp) <= sizeof(int32_t)))
        _CCCL_API _CCCL_FPEMU_UNP_NARROW_EXPLICIT inline fp64emu_unpacked_t(_Tp __i) noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>) { __set_from_int (static_cast<int32_t>(__i)); }
            else                                                        { __set_from_uint(static_cast<uint32_t>(__i)); }
        }
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND(sizeof(_Tp) > sizeof(int32_t)))
        _CCCL_API explicit inline fp64emu_unpacked_t(_Tp __i) noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>) { __set_from_ll (static_cast<int64_t>(__i)); }
            else                                                        { __set_from_ull(static_cast<uint64_t>(__i)); }
        }
#undef _CCCL_FPEMU_UNP_NARROW_EXPLICIT
        // Type conversion to fp64emu_unpacked_t with other accuracy and range
        template<fp64emu_accuracy _Acc = _Met> _CCCL_API inline operator fp64emu_unpacked_t<_Acc>() const noexcept ;
        // Type conversion from fp64emu_unpacked_t to fp64emu_t (explicit to avoid overload ambiguity)
        template<fp64emu_accuracy _Acc = _Met> _CCCL_API explicit inline operator fp64emu_t<_Acc>() const noexcept ;

        // ==== Conversion from fp64emu_unpacked_t to other types:
        // Implicit conversion to double
        _CCCL_API inline operator double() const noexcept ;
        // Explicit conversions to other types
        _CCCL_API explicit inline operator float()    const noexcept ;
        // Explicit conversion to any standard integer type (int / long / long long + unsigned).
        // Dispatches by width and signedness to the accuracy-correct integer builtins (via the
        // private out-of-line helpers below); excludes bool / character types.
        _CCCL_TEMPLATE(class _Tp)
        _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
        _CCCL_API inline explicit operator _Tp() const noexcept
        {
            if constexpr (::cuda::std::__cccl_is_signed_integer_v<_Tp>)
            {
                if constexpr (sizeof(_Tp) <= sizeof(int32_t)) { return static_cast<_Tp>(__to_int()); }
                else                                          { return static_cast<_Tp>(__to_ll());  }
            }
            else
            {
                if constexpr (sizeof(_Tp) <= sizeof(uint32_t)) { return static_cast<_Tp>(__to_uint()); }
                else                                           { return static_cast<_Tp>(__to_ull());  }
            }
        }

     private:
        // Accuracy-correct integer <-> value helpers (defined out-of-line where the fpemu
        // builtins are visible). Kept non-template so the definitions stay out-of-line.
        _CCCL_API inline void     __set_from_int (int32_t  __i) noexcept ;
        _CCCL_API inline void     __set_from_uint(uint32_t __i) noexcept ;
        _CCCL_API inline void     __set_from_ll  (int64_t  __i) noexcept ;
        _CCCL_API inline void     __set_from_ull (uint64_t __i) noexcept ;
        _CCCL_API inline int32_t  __to_int () const noexcept ;
        _CCCL_API inline uint32_t __to_uint() const noexcept ;
        _CCCL_API inline int64_t  __to_ll  () const noexcept ;
        _CCCL_API inline uint64_t __to_ull () const noexcept ;
     public:

        /*
        //  CUDA builtins functions for conversions
        */
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline float __double2float(fp64emu_unpacked_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rz(fp64emu_unpacked_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rz(fp64emu_unpacked_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rz(fp64emu_unpacked_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rz(fp64emu_unpacked_t<_Acc> __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __float2double (float __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __int2double   (int32_t __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __uint2double  (uint32_t __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __ll2double    (int64_t __x) noexcept ;
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __ull2double   (uint64_t __x) noexcept ;

        /*
        // Arithmetic operations:
        */
        // === mul ===
        // (*)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_unpacked_t<_Acc> operator*(const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y) noexcept ;
        // (/)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_unpacked_t<_Acc> operator/(const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y) noexcept ;
        // (+)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_unpacked_t<_Acc> operator+(const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y) noexcept ;
        // (-)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_unpacked_t<_Acc> operator-(const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y) noexcept ;
        

        // == mul ==
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_unpacked_t operator*(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) * fp64emu_unpacked_t(__y); }
        // dmul_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_unpacked_t __dmul_rn(const _T1& __x, const _T2& __y) noexcept { return __dmul_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y)); }

        // === div ===
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_unpacked_t operator/(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) / fp64emu_unpacked_t(__y); }
        // ddiv_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_unpacked_t __ddiv_rn(const _T1& __x, const _T2& __y) noexcept { return __ddiv_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y)); }

        // === add ===
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_unpacked_t operator+(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) + fp64emu_unpacked_t(__y); }
        // dadd_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_unpacked_t __dadd_rn(const _T1& __x, const _T2& __y) noexcept {  return __dadd_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y)); }

        // === sub ===
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_unpacked_t operator-(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) - fp64emu_unpacked_t(__y); }
        // dsub_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend  fp64emu_unpacked_t __dsub_rn(const _T1& __x, const _T2& __y) noexcept { return __dsub_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y)); }

        // === sqrt ===
        // sqrt
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fp64emu_unpacked_t sqrt(const _T1& __x) noexcept { return sqrt(fp64emu_unpacked_t(__x)); }        
        // dsqrt_rn
        _CCCL_TEMPLATE(typename _T1)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1>)))
            _CCCL_API friend  fp64emu_unpacked_t __dsqrt_rn(const _T1& __x) noexcept { return __dsqrt_rn(fp64emu_unpacked_t(__x)); }

        // === fma ===
        // fma
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T3,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_unpacked_t fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return fma(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y), fp64emu_unpacked_t(__z)); }
        // dfma_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T3,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_unpacked_t __fma_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __fma_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y), fp64emu_unpacked_t(__z)); }

        // === mad ===
        // mad
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T3,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_unpacked_t mad(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return mad(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y), fp64emu_unpacked_t(__z)); }
        // dmad_rn
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T3,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3>)))
            _CCCL_API friend  fp64emu_unpacked_t __mad_rn(const _T1& __x, const _T2& __y, const _T3& __z) noexcept { return __mad_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y), fp64emu_unpacked_t(__z)); }

        // === dot ===
        _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3, typename _T4)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T3,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T4,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3> || ::cuda::std::is_arithmetic_v<_T4>)))
            _CCCL_API friend  fp64emu_unpacked_t dot(const _T1& __x1, const _T2& __y1, const _T3& __x2, const _T4& __y2) noexcept { return dot(fp64emu_unpacked_t(__x1), fp64emu_unpacked_t(__y1), fp64emu_unpacked_t(__x2), fp64emu_unpacked_t(__y2)); }

         // === cmul ===
         _CCCL_TEMPLATE(typename _T1, typename _T2, typename _T3, typename _T4)
         _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T3,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T4,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2> || ::cuda::std::is_arithmetic_v<_T3> || ::cuda::std::is_arithmetic_v<_T4>)))
             _CCCL_API friend void cmul(const _T1& __x_re, const _T2& __x_im, const _T3& __y_re, const _T4& __y_im, fp64emu_unpacked_t& __r_re, fp64emu_unpacked_t& __r_im) noexcept { cmul(fp64emu_unpacked_t(__x_re), fp64emu_unpacked_t(__x_im), fp64emu_unpacked_t(__y_re), fp64emu_unpacked_t(__y_im), __r_re, __r_im); }

        // Prefix increment/decrement
        _CCCL_API fp64emu_unpacked_t& operator++() noexcept { this = this + fp64emu_unpacked_t(1.0); return *this; }
        _CCCL_API fp64emu_unpacked_t& operator--() noexcept { this = this - fp64emu_unpacked_t(1.0); return *this; }
        // Postfix increment/decrement
        _CCCL_API fp64emu_unpacked_t  operator++(int) noexcept { fp64emu_unpacked_t __temp(*this); this = this + fp64emu_unpacked_t(1.0); return __temp; }
        _CCCL_API fp64emu_unpacked_t  operator--(int) noexcept { fp64emu_unpacked_t __temp(*this); this = this - fp64emu_unpacked_t(1.0); return __temp; }
        // Compound assignment operators
        _CCCL_API fp64emu_unpacked_t& operator+=(const fp64emu_unpacked_t& __other) noexcept { *this = *this + __other; return *this; }
        _CCCL_API fp64emu_unpacked_t& operator-=(const fp64emu_unpacked_t& __other) noexcept { *this = *this - __other; return *this; }
        _CCCL_API fp64emu_unpacked_t& operator*=(const fp64emu_unpacked_t& __other) noexcept { *this = *this * __other; return *this; }
        _CCCL_API fp64emu_unpacked_t& operator/=(const fp64emu_unpacked_t& __other) noexcept { *this = *this / __other; return *this; }
        // Unary negation operator (implementation in fpemu_impl_others.h)
        _CCCL_API fp64emu_unpacked_t  operator-() const noexcept ;

        /*
        // Comparison operators:
        */       
        // equality (==)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator==(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) == fp64emu_unpacked_t(__y); }
        // inequality (!=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator!=(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) != fp64emu_unpacked_t(__y); }
        // less than (<)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator<(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) < fp64emu_unpacked_t(__y); }
        // greater than (>)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator>(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) > fp64emu_unpacked_t(__y); }
        // less than or equal to (<=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator<=(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) <= fp64emu_unpacked_t(__y); }
        // greater than or equal to (>=)
        _CCCL_TEMPLATE(typename _T1, typename _T2)
        _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1,fp64emu_unpacked_t> || ::cuda::std::is_same_v<_T2,fp64emu_unpacked_t>) && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>)))
            _CCCL_API friend bool
            operator>=(const _T1& __x, const _T2& __y) noexcept { return fp64emu_unpacked_t(__x) >= fp64emu_unpacked_t(__y); }

        // C++20-style bit_cast for unpacked floating-point types
        template<typename _To, fp64emu_accuracy _Acc> 
        _CCCL_API friend inline _To bit_cast(const fp64emu_unpacked_t<_Acc>& __from) noexcept ;

    }; // class fp64emu_unpacked_t 

    /*
    // Aliases for the emulated floating-point types
    */
    using fp64emu               = fp64emu_t<fp64emu_accuracy::def>;
    using fp64emu_low           = fp64emu_t<fp64emu_accuracy::low>;
    using fp64emu_mid           = fp64emu_t<fp64emu_accuracy::mid>;
    using fp64emu_high          = fp64emu_t<fp64emu_accuracy::high>;

    using fp64emu_unpacked      = fp64emu_unpacked_t<fp64emu_accuracy::def>;
    using fp64emu_unpacked_low  = fp64emu_unpacked_t<fp64emu_accuracy::low>;
    using fp64emu_unpacked_mid  = fp64emu_unpacked_t<fp64emu_accuracy::mid>;
    using fp64emu_unpacked_high = fp64emu_unpacked_t<fp64emu_accuracy::high>;

// Define this macro so that the API sections in _impl.hpp files are activated.
// The _impl.hpp files are structured with implementation code under their own
// include guard, and API code (operators, class methods) under this guard.
// This ensures API code is only compiled after class definitions are complete.
#define _CCCL_FPEMU_API_CLASSES_DEFINED

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#include <cuda/__fp/fpemu_impl_cmp.h>
#include <cuda/__fp/fpemu_impl_cvt.h>
#include <cuda/__fp/fpemu_impl_fma.h>
#include <cuda/__fp/fpemu_impl_add.h>
#include <cuda/__fp/fpemu_impl_sub.h>
#include <cuda/__fp/fpemu_impl_mul.h>
#include <cuda/__fp/fpemu_impl_div.h>
#include <cuda/__fp/fpemu_impl_sqrt.h>
#include <cuda/__fp/fpemu_impl_others.h>

#endif // _CUDA___FP_FPEMU_H
