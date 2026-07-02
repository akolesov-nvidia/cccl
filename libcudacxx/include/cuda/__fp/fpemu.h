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
 
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
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

#if __FPEMU_UNPACKED__ == 1
    // Forward declaration of unpacked floating-point class
    template <fp64emu_accuracy _Met> class fp64emu_unpacked_t;
#endif

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
        _CCCL_API inline fp64emu_t() : bits{0u} {}
        _CCCL_API inline fp64emu_t(fpbits64_construct_t, const fpbits64_t& __f) : bits(__f) {}
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
        _CCCL_API inline fp64emu_t(const volatile fp64emu_t& __other) : bits(__other.bits) {}

        // Defaulted copy assignment operator (trivially copyable)
        fp64emu_t& operator=(const fp64emu_t& __other) = default;

        /*
        // Assignment operator to volatile fp64emu_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
        */
        template<typename _Dummy = void>
        _CCCL_API inline void operator=(const fp64emu_t& __other) volatile { bits = __other.bits; }

        /*
        // Assignment operator from volatile fp64emu_t
        // Template so it is NOT a copy assignment operator per the C++ standard
        */
        template<typename _Dummy = void>
        _CCCL_API inline fp64emu_t& operator=(const volatile fp64emu_t& __other) { bits = __other.bits; return *this; }

        /*
        // Conversion operators
        */
        // ==== Conversions from other types to fp64emu_t:
        // Implicit conversions from floating-point types
        _CCCL_API inline fp64emu_t(float __f);
        _CCCL_API inline fp64emu_t(double __d);
        // Implicit conversions from integer types
        _CCCL_API inline fp64emu_t(int32_t __i);
        _CCCL_API inline fp64emu_t(uint32_t __i);

        // Explicit conversions from 64-bit integers 
        // required due to ambiguity with other constructors
        _CCCL_API explicit inline fp64emu_t(int64_t __i);
        _CCCL_API explicit inline fp64emu_t(uint64_t __i);
        // Explicit conversion from long long int types when their range is wider than int64_t
        _CCCL_API explicit inline fp64emu_t(long long unsigned int __i) { *this = fp64emu_t((uint64_t)__i); }
        _CCCL_API explicit inline fp64emu_t(long long  int __i)         { *this = fp64emu_t((int64_t)__i);  }
        // Type conversion to fp64emu_t with other accuracy and range
        template<fp64emu_accuracy _Acc = _Met> _CCCL_API inline operator fp64emu_t<_Acc>() const;
#if __FPEMU_UNPACKED__ == 1
        // Type conversion from fp64emu_t to fp64emu_unpacked_t (explicit to avoid overload ambiguity)
        template<fp64emu_accuracy _Acc = _Met> _CCCL_API explicit inline operator fp64emu_unpacked_t<_Acc>() const;
#endif

        // ==== Conversion from fp64emu_t to other types:
        // Implicit conversion to double
        _CCCL_API inline operator double() const;
        // Explicit conversions to other types
        _CCCL_API explicit inline operator float()    const;
        _CCCL_API explicit inline operator int32_t()  const;
        _CCCL_API explicit inline operator uint32_t() const;
        _CCCL_API explicit inline operator int64_t()  const;
        _CCCL_API explicit inline operator uint64_t() const;
        // Explicit conversion to long long int types when their range is wider than int64_t
        _CCCL_API explicit inline operator long long unsigned int() const { return (uint64_t)(*this); }
        _CCCL_API explicit inline operator long long int() const          { return (int64_t)(*this); }

        /*
        //  CUDA builtins functions for conversions
        */
        // double to float
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline float  __double2float (fp64emu_t<_Acc> __x);
        // double to integer
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rn (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rz (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_ru (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rd (fp64emu_t<_Acc> __x);
        // double to unsigned integer
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rn (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rz (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_ru (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rd (fp64emu_t<_Acc> __x);
        // double to signed integer
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rn (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rz (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_ru (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rd (fp64emu_t<_Acc> __x);
        // double to unsigned integer
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rn (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rz (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_ru (fp64emu_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rd (fp64emu_t<_Acc> __x);
        // other types to double
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __int2double   (int32_t __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __uint2double  (uint32_t __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __ll2double    (int64_t __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __ull2double   (uint64_t __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_t<_Acc> __float2double (float __x);
    
        /*
        // Arithmetic operations:
        */
        // === mul ===
        // (*)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_t<_Acc> operator*(const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y);
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type> 
            _CCCL_API friend  fp64emu_t operator*(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) * fp64emu_t(__y); }
        // dmul_rn
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dmul_rn(const _T1& __x, const _T2& __y) { return __dmul_rn(fp64emu_t(__x), fp64emu_t(__y)); }
        // dmul_rz
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dmul_rz(const _T1& __x, const _T2& __y) { return __dmul_rz(fp64emu_t(__x), fp64emu_t(__y)); }
        // dmul_ru
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dmul_ru(const _T1& __x, const _T2& __y) { return __dmul_ru(fp64emu_t(__x), fp64emu_t(__y)); }
        // dmul_rd
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dmul_rd(const _T1& __x, const _T2& __y) { return __dmul_rd(fp64emu_t(__x), fp64emu_t(__y)); }
        
        // === div ===
        // (/)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_t<_Acc> operator/(const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y);
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type> 
            _CCCL_API friend  fp64emu_t operator/(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) / fp64emu_t(__y); }
        // ddiv_rn
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __ddiv_rn(const _T1& __x, const _T2& __y) { return __ddiv_rn(fp64emu_t(__x), fp64emu_t(__y)); }
        // ddiv_rz
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __ddiv_rz(const _T1& __x, const _T2& __y) { return __ddiv_rz(fp64emu_t(__x), fp64emu_t(__y)); }
        // ddiv_ru
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __ddiv_ru(const _T1& __x, const _T2& __y) { return __ddiv_ru(fp64emu_t(__x), fp64emu_t(__y)); }
        // ddiv_rd
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __ddiv_rd(const _T1& __x, const _T2& __y) { return __ddiv_rd(fp64emu_t(__x), fp64emu_t(__y)); }

        // === add ===
        // (+)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_t<_Acc> operator+(const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y);
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type> 
            _CCCL_API friend  fp64emu_t operator+(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) + fp64emu_t(__y); }
        // dadd_rn
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dadd_rn(const _T1& __x, const _T2& __y) {  return __dadd_rn(fp64emu_t(__x), fp64emu_t(__y)); }
        // dadd_rz
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dadd_rz(const _T1& __x, const _T2& __y) {  return __dadd_rz(fp64emu_t(__x), fp64emu_t(__y)); }
        // dadd_ru
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dadd_ru(const _T1& __x, const _T2& __y) { return __dadd_ru(fp64emu_t(__x), fp64emu_t(__y)); }
        // dadd_rd
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dadd_rd(const _T1& __x, const _T2& __y) { return __dadd_rd(fp64emu_t(__x), fp64emu_t(__y)); }

        // === sub ===
        // (-)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_t<_Acc> operator-(const fp64emu_t<_Acc>& __x, const fp64emu_t<_Acc>& __y);
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type> 
            _CCCL_API friend  fp64emu_t operator-(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) - fp64emu_t(__y); }
        // dsub_rn
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dsub_rn(const _T1& __x, const _T2& __y) { return __dsub_rn(fp64emu_t(__x), fp64emu_t(__y)); }
        // dsub_rz
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dsub_rz(const _T1& __x, const _T2& __y) { return __dsub_rz(fp64emu_t(__x), fp64emu_t(__y)); }
        // dsub_ru
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dsub_ru(const _T1& __x, const _T2& __y) { return __dsub_ru(fp64emu_t(__x), fp64emu_t(__y)); }
        // dsub_rd
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_t __dsub_rd(const _T1& __x, const _T2& __y) { return __dsub_rd(fp64emu_t(__x), fp64emu_t(__y)); }

        // === sqrt ===
        // sqrt
        template<typename _T1, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value))>::type>
            _CCCL_API friend  fp64emu_t sqrt(const _T1& __x) { return sqrt(fp64emu_t(__x)); }        
        // dsqrt_rn
        template<typename _T1, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value))>::type>
            _CCCL_API friend  fp64emu_t __dsqrt_rn(const _T1& __x) { return __dsqrt_rn(fp64emu_t(__x)); }

        template<typename _T1, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value))>::type>
            _CCCL_API friend  fp64emu_t __dsqrt_rz(const _T1& __x) { return __dsqrt_rz(fp64emu_t(__x)); }
        // dsqrt_ru
        template<typename _T1, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value))>::type>
            _CCCL_API friend  fp64emu_t __dsqrt_ru(const _T1& __x) { return __dsqrt_ru(fp64emu_t(__x)); }
        // dsqrt_rd
        template<typename _T1, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value))>::type>
            _CCCL_API friend  fp64emu_t __dsqrt_rd(const _T1& __x) { return __dsqrt_rd(fp64emu_t(__x)); }

        // === fma ===
        // fma
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value || std::is_same<_T3,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_t fma(const _T1& __x, const _T2& __y, const _T3& __z) { return fma(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dfma_rn
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value || std::is_same<_T3,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_t __fma_rn(const _T1& __x, const _T2& __y, const _T3& __z) { return __fma_rn(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dfma_rz
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value || std::is_same<_T3,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_t __fma_rz(const _T1& __x, const _T2& __y, const _T3& __z) { return __fma_rz(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dfma_ru
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value || std::is_same<_T3,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_t __fma_ru(const _T1& __x, const _T2& __y, const _T3& __z) { return __fma_ru(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dfma_rd
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value || std::is_same<_T3,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_t __fma_rd(const _T1& __x, const _T2& __y, const _T3& __z) { return __fma_rd(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }

        // === mad ===
        // mad
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value || std::is_same<_T3,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_t mad(const _T1& __x, const _T2& __y, const _T3& __z) { return mad(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }
        // dmad_rn
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value || std::is_same<_T3,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_t __mad_rn(const _T1& __x, const _T2& __y, const _T3& __z) { return __mad_rn(fp64emu_t(__x), fp64emu_t(__y), fp64emu_t(__z)); }

        // === dot ===
        template<typename _T1, typename _T2, typename _T3, typename _T4, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value || std::is_same<_T3,fp64emu_t>::value || std::is_same<_T4,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value || std::is_arithmetic<_T4>::value))>::type>
            _CCCL_API friend  fp64emu_t dot(const _T1& __x1, const _T2& __y1, const _T3& __x2, const _T4& __y2) { return dot(fp64emu_t(__x1), fp64emu_t(__y1), fp64emu_t(__x2), fp64emu_t(__y2)); }

         // === cmul ===
         template<typename _T1, typename _T2, typename _T3, typename _T4, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value || std::is_same<_T3,fp64emu_t>::value || std::is_same<_T4,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value || std::is_arithmetic<_T4>::value))>::type>
             _CCCL_API friend void cmul(const _T1& __x_re, const _T2& __x_im, const _T3& __y_re, const _T4& __y_im, fp64emu_t& __r_re, fp64emu_t& __r_im) { cmul(fp64emu_t(__x_re), fp64emu_t(__x_im), fp64emu_t(__y_re), fp64emu_t(__y_im), __r_re, __r_im); }

        // Prefix increment/decrement
        _CCCL_API fp64emu_t& operator++() { this = this + fp64emu_t(1.0); return *this; }
        _CCCL_API fp64emu_t& operator--() { this = this - fp64emu_t(1.0); return *this; }
        // Postfix increment/decrement
        _CCCL_API fp64emu_t  operator++(int) { fp64emu_t __temp(*this); this = this + fp64emu_t(1.0); return __temp; }
        _CCCL_API fp64emu_t  operator--(int) { fp64emu_t __temp(*this); this = this - fp64emu_t(1.0); return __temp; }
        // Compound assignment operators
        _CCCL_API fp64emu_t& operator+=(const fp64emu_t& __other) { *this = *this + __other; return *this; }
        _CCCL_API fp64emu_t& operator-=(const fp64emu_t& __other) { *this = *this - __other; return *this; }
        _CCCL_API fp64emu_t& operator*=(const fp64emu_t& __other) { *this = *this * __other; return *this; }
        _CCCL_API fp64emu_t& operator/=(const fp64emu_t& __other) { *this = *this / __other; return *this; }
        // Unary negation operator (implementation in fpemu_impl_others.h)
        _CCCL_API fp64emu_t  operator-() const;

        /*
        // Comparison operators:
        */       
        // equality (==)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator==(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) == fp64emu_t(__y); }
        // inequality (!=)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator!=(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) != fp64emu_t(__y); }
        // less than (<)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator<(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) < fp64emu_t(__y); }
        // greater than (>)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator>(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) > fp64emu_t(__y); }
        // less than or equal to (<=)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator<=(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) <= fp64emu_t(__y); }
        // greater than or equal to (>=)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_t>::value || std::is_same<_T2,fp64emu_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator>=(const _T1& __x, const _T2& __y) { return fp64emu_t(__x) >= fp64emu_t(__y); }
    }; // class fp64emu_t 

#if __FPEMU_UNPACKED__ == 1

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
        _CCCL_API inline fp64emu_unpacked_t() : bits{0u, 0, 0} {}
        _CCCL_API inline fp64emu_unpacked_t(fpbits64_construct_t, const fpbits64_unpacked_t& __f) : bits(__f) {}
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
        _CCCL_API inline fp64emu_unpacked_t(const volatile fp64emu_unpacked_t& __other)
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
        _CCCL_API inline void operator=(const fp64emu_unpacked_t& __other) volatile
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
        _CCCL_API inline fp64emu_unpacked_t& operator=(const volatile fp64emu_unpacked_t& __other)
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
        _CCCL_API  inline fp64emu_unpacked_t(float f);
        _CCCL_API  inline fp64emu_unpacked_t(double d);        
        // Explicit conversions from integer types
        _CCCL_API  inline fp64emu_unpacked_t(int32_t i);
        _CCCL_API  inline fp64emu_unpacked_t(uint32_t i);
#else
        // Explicit conversions from floating-point types (to avoid ambiguity with packed type)
        _CCCL_API explicit inline fp64emu_unpacked_t(float __f);
        _CCCL_API explicit inline fp64emu_unpacked_t(double __d);        
        // Explicit conversions from integer types (to avoid ambiguity with packed type)
        _CCCL_API explicit inline fp64emu_unpacked_t(int32_t __i);
        _CCCL_API explicit inline fp64emu_unpacked_t(uint32_t __i);
#endif
        // Explicit conversions from 64-bit integers 
        // required due to ambiguity with other constructors
        _CCCL_API explicit inline fp64emu_unpacked_t(int64_t __i);
        _CCCL_API explicit inline fp64emu_unpacked_t(uint64_t __i);

        // Explicit conversion from long long int types when their range is wider than int64_t
        _CCCL_API explicit inline fp64emu_unpacked_t(long long unsigned int __i) { *this = fp64emu_unpacked_t((uint64_t)__i); }
        _CCCL_API explicit inline fp64emu_unpacked_t(long long  int __i)         { *this = fp64emu_unpacked_t((int64_t)__i);  }
        // Type conversion to fp64emu_unpacked_t with other accuracy and range
        template<fp64emu_accuracy _Acc = _Met> _CCCL_API inline operator fp64emu_unpacked_t<_Acc>() const;
        // Type conversion from fp64emu_unpacked_t to fp64emu_t (explicit to avoid overload ambiguity)
        template<fp64emu_accuracy _Acc = _Met> _CCCL_API explicit inline operator fp64emu_t<_Acc>() const;

        // ==== Conversion from fp64emu_unpacked_t to other types:
        // Implicit conversion to double
        _CCCL_API inline operator double() const;
        // Explicit conversions to other types
        _CCCL_API explicit inline operator float()    const;
        _CCCL_API explicit inline operator int32_t()  const;
        _CCCL_API explicit inline operator uint32_t() const;
        _CCCL_API explicit inline operator int64_t()  const;
        _CCCL_API explicit inline operator uint64_t() const;
        // Explicit conversion to long long int types when their range is wider than int64_t
        _CCCL_API explicit inline operator long long unsigned int() const { return (uint64_t)(*this); }
        _CCCL_API explicit inline operator long long int() const          { return (int64_t)(*this); }

        /*
        //  CUDA builtins functions for conversions
        */
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline float __double2float(fp64emu_unpacked_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int32_t __double2int_rz(fp64emu_unpacked_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint32_t __double2uint_rz(fp64emu_unpacked_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline int64_t __double2ll_rz(fp64emu_unpacked_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline uint64_t __double2ull_rz(fp64emu_unpacked_t<_Acc> __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __float2double (float __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __int2double   (int32_t __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __uint2double  (uint32_t __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __ll2double    (int64_t __x);
        template<fp64emu_accuracy _Acc> _CCCL_API friend inline fp64emu_unpacked_t<_Acc> __ull2double   (uint64_t __x);

        /*
        // Arithmetic operations:
        */
        // === mul ===
        // (*)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_unpacked_t<_Acc> operator*(const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y);
        // (/)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_unpacked_t<_Acc> operator/(const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y);
        // (+)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_unpacked_t<_Acc> operator+(const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y);
        // (-)
        template<fp64emu_accuracy _Acc> _CCCL_API friend fp64emu_unpacked_t<_Acc> operator-(const fp64emu_unpacked_t<_Acc>& __x, const fp64emu_unpacked_t<_Acc>& __y);
        

        // == mul ==
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type> 
            _CCCL_API friend  fp64emu_unpacked_t operator*(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) * fp64emu_unpacked_t(__y); }
        // dmul_rn
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t __dmul_rn(const _T1& __x, const _T2& __y) { return __dmul_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y)); }

        // === div ===
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type> 
            _CCCL_API friend  fp64emu_unpacked_t operator/(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) / fp64emu_unpacked_t(__y); }
        // ddiv_rn
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t __ddiv_rn(const _T1& __x, const _T2& __y) { return __ddiv_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y)); }

        // === add ===
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type> 
            _CCCL_API friend  fp64emu_unpacked_t operator+(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) + fp64emu_unpacked_t(__y); }
        // dadd_rn
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t __dadd_rn(const _T1& __x, const _T2& __y) {  return __dadd_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y)); }

        // === sub ===
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type> 
            _CCCL_API friend  fp64emu_unpacked_t operator-(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) - fp64emu_unpacked_t(__y); }
        // dsub_rn
        template<typename _T1, typename _T2, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t __dsub_rn(const _T1& __x, const _T2& __y) { return __dsub_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y)); }

        // === sqrt ===
        // sqrt
        template<typename _T1, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t sqrt(const _T1& __x) { return sqrt(fp64emu_unpacked_t(__x)); }        
        // dsqrt_rn
        template<typename _T1, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t __dsqrt_rn(const _T1& __x) { return __dsqrt_rn(fp64emu_unpacked_t(__x)); }

        // === fma ===
        // fma
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value || std::is_same<_T3,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t fma(const _T1& __x, const _T2& __y, const _T3& __z) { return fma(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y), fp64emu_unpacked_t(__z)); }
        // dfma_rn
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value || std::is_same<_T3,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t __fma_rn(const _T1& __x, const _T2& __y, const _T3& __z) { return __fma_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y), fp64emu_unpacked_t(__z)); }

        // === mad ===
        // mad
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value || std::is_same<_T3,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t mad(const _T1& __x, const _T2& __y, const _T3& __z) { return mad(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y), fp64emu_unpacked_t(__z)); }
        // dmad_rn
        template<typename _T1, typename _T2, typename _T3, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value || std::is_same<_T3,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t __mad_rn(const _T1& __x, const _T2& __y, const _T3& __z) { return __mad_rn(fp64emu_unpacked_t(__x), fp64emu_unpacked_t(__y), fp64emu_unpacked_t(__z)); }

        // === dot ===
        template<typename _T1, typename _T2, typename _T3, typename _T4, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value || std::is_same<_T3,fp64emu_unpacked_t>::value || std::is_same<_T4,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value || std::is_arithmetic<_T4>::value))>::type>
            _CCCL_API friend  fp64emu_unpacked_t dot(const _T1& __x1, const _T2& __y1, const _T3& __x2, const _T4& __y2) { return dot(fp64emu_unpacked_t(__x1), fp64emu_unpacked_t(__y1), fp64emu_unpacked_t(__x2), fp64emu_unpacked_t(__y2)); }

         // === cmul ===
         template<typename _T1, typename _T2, typename _T3, typename _T4, typename = typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value || std::is_same<_T3,fp64emu_unpacked_t>::value || std::is_same<_T4,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value || std::is_arithmetic<_T3>::value || std::is_arithmetic<_T4>::value))>::type>
             _CCCL_API friend void cmul(const _T1& __x_re, const _T2& __x_im, const _T3& __y_re, const _T4& __y_im, fp64emu_unpacked_t& __r_re, fp64emu_unpacked_t& __r_im) { cmul(fp64emu_unpacked_t(__x_re), fp64emu_unpacked_t(__x_im), fp64emu_unpacked_t(__y_re), fp64emu_unpacked_t(__y_im), __r_re, __r_im); }

        // Prefix increment/decrement
        _CCCL_API fp64emu_unpacked_t& operator++() { this = this + fp64emu_unpacked_t(1.0); return *this; }
        _CCCL_API fp64emu_unpacked_t& operator--() { this = this - fp64emu_unpacked_t(1.0); return *this; }
        // Postfix increment/decrement
        _CCCL_API fp64emu_unpacked_t  operator++(int) { fp64emu_unpacked_t __temp(*this); this = this + fp64emu_unpacked_t(1.0); return __temp; }
        _CCCL_API fp64emu_unpacked_t  operator--(int) { fp64emu_unpacked_t __temp(*this); this = this - fp64emu_unpacked_t(1.0); return __temp; }
        // Compound assignment operators
        _CCCL_API fp64emu_unpacked_t& operator+=(const fp64emu_unpacked_t& __other) { *this = *this + __other; return *this; }
        _CCCL_API fp64emu_unpacked_t& operator-=(const fp64emu_unpacked_t& __other) { *this = *this - __other; return *this; }
        _CCCL_API fp64emu_unpacked_t& operator*=(const fp64emu_unpacked_t& __other) { *this = *this * __other; return *this; }
        _CCCL_API fp64emu_unpacked_t& operator/=(const fp64emu_unpacked_t& __other) { *this = *this / __other; return *this; }
        // Unary negation operator (implementation in fpemu_impl_others.h)
        _CCCL_API fp64emu_unpacked_t  operator-() const;

        /*
        // Comparison operators:
        */       
        // equality (==)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator==(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) == fp64emu_unpacked_t(__y); }
        // inequality (!=)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator!=(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) != fp64emu_unpacked_t(__y); }
        // less than (<)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator<(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) < fp64emu_unpacked_t(__y); }
        // greater than (>)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator>(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) > fp64emu_unpacked_t(__y); }
        // less than or equal to (<=)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator<=(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) <= fp64emu_unpacked_t(__y); }
        // greater than or equal to (>=)
        template<typename _T1, typename _T2>
            _CCCL_API friend typename std::enable_if<((std::is_same<_T1,fp64emu_unpacked_t>::value || std::is_same<_T2,fp64emu_unpacked_t>::value) && (std::is_arithmetic<_T1>::value || std::is_arithmetic<_T2>::value)), bool>::type
            operator>=(const _T1& __x, const _T2& __y) { return fp64emu_unpacked_t(__x) >= fp64emu_unpacked_t(__y); }

        // C++20-style bit_cast for unpacked floating-point types
        template<typename _To, fp64emu_accuracy _Acc> 
        _CCCL_API friend inline _To bit_cast(const fp64emu_unpacked_t<_Acc>& __from);

    }; // class fp64emu_unpacked_t 
#endif // __FPEMU_UNPACKED__ == 1

    /*
    // Aliases for the emulated floating-point types
    */
    using fp64emu      = fp64emu_t<fp64emu_accuracy::def>;
    using fp64emu_low  = fp64emu_t<fp64emu_accuracy::low>;
    using fp64emu_mid  = fp64emu_t<fp64emu_accuracy::mid>;
    using fp64emu_high = fp64emu_t<fp64emu_accuracy::high>;
#if __FPEMU_UNPACKED__ == 1
    using fp64emu_unpacked      = fp64emu_unpacked_t<fp64emu_accuracy::def>;
    using fp64emu_unpacked_low  = fp64emu_unpacked_t<fp64emu_accuracy::low>;
    using fp64emu_unpacked_mid  = fp64emu_unpacked_t<fp64emu_accuracy::mid>;
    using fp64emu_unpacked_high = fp64emu_unpacked_t<fp64emu_accuracy::high>;
#endif

// Define this macro so that the API sections in _impl.hpp files are activated.
// The _impl.hpp files are structured with implementation code under their own
// include guard, and API code (operators, class methods) under this guard.
// This ensures API code is only compiled after class definitions are complete.
#define __FPEMU_API_CLASSES_DEFINED__

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
