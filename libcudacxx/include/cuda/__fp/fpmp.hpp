
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_H
#define _CUDA___FP_FPMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/*
    fpmp.hpp - Multi-Precision Floating-Point Types and Core Operations (double-float / double-double)
    ======================================================================================================
    This header defines the primary public types and core operations for multi-component floating-point
    arithmetic using pairs of IEEE-754 floating-point values. It supports both "double-float" (fp32mp2)
    and "double-double" (fp64mp2) representations and can be used from both CPU and GPU (CUDA) code.
    
    Linkage note:
    - In header mode (default, FPMP_INLINE=1), built-in entry points are defined as inline/static.
    - In library mode (FPMP_LIB=1), built-in entry points are provided by a
      separately compiled object/library (see `src/fpmp_lib.cpp`), while the C++ API remains header-based.

    Supported Formats:
    -------------------------------------------------------------------------
    - Double-Float (fp32mp2): Pairs of single-precision floats (2×32-bit), providing up to ~46 bits of effective mantissa
      (algorithm/method dependent; between IEEE-754 float and double).
      Ideal for consumer GPUs where FP64 performance is limited (often 1/32 of FP32 throughput).
    - Double-Double (fp64mp2): Pairs of double-precision floats (2×64-bit), providing up to ~104 bits of effective mantissa
      (algorithm/method dependent; between IEEE-754 double and binary128).
      Enabled via FPMP_FP64MP2_ENABLE=1 macro.

    Key Features:
    -------------------------------------------------------------------------
    - Data Types
        - fp32mp2 : Double-float type using two single-precision floats (hi, lo), representing a number as the unevaluated sum hi + lo.
        - fpmp2_t<FpType, fpmp2_accuracy> : C++ class template providing operator overloading and type safety.
          * FpType=float for double-float, FpType=double for double-double.

    - Operations Supported
        * Conversion: Safe conversion from/to double, float, int32_t, uint32_t, int64_t, uint64_t.
        * Normalization: Ensures the two-float representation maintains strict ordering and non-overlapping components.
        * Basic Arithmetic: Addition, subtraction, multiplication, negation, and division for double-float.
        * Accuracy-Explicit Arithmetic: Free functions add<m>, sub<m>, mul<m>, div<m>, fma<m>, mad<m>
          that override the arithmetic method for a single operation without changing the type.
        * Advanced: Square root, reciprocal square root, fused multiply-add (FMA), multiply-add (MAD), exponential function (exp).
        * Utility: Renormalization, error-aware summation and multiplication with or without FMA.
        * Comparison: Supports all common relational operators (==, !=, <, <=, >, >=).
        * Bit-casting: Supports reinterpretation of the value as a 64-bit integer (IEEE-754 format).
        * Atomic Operations: Supports atomic addition and subtraction on multi-precision floating point numbers (CUDA only).
        * Warp Shuffle: Overloads of CUDA's __shfl_sync family for fpmp2 pairs (CUDA only, header-only,
          declared in fpmp_math.hpp; not exposed via fpmp_lib).
        * GPU & Host Compatibility: All operations and members are decorated for both device and host use.

    - Implementation Aspects
        * Header-based C++ API with optional library-provided built-in symbols (see linkage note above).
        * Multiple accuracy levels: mid (default, Dekker-based), low (minimal renormalization), and high (Thall-based).
        * Template-based design allows compile-time selection of arithmetic precision/speed trade-offs.
        * Error-free transformations using Dekker's 2Sum and 2Mult algorithms.

    - Usage Scenarios
        * Double-float (fp32mp2): Useful in numerical algorithms requiring more accuracy than float, without the full cost of FP64.
          Particularly suitable for GPUs with limited FP64 performance (consumer GPUs).
        * Double-double (fp64mp2): For applications requiring quad-like accuracy (extended precision beyond IEEE double).
          Ideal for scientific computing, high-precision simulations, and financial calculations. 
          Suitable when expensive FP128 operations are not required and GPU .
        * Both formats are suitable for high-performance computing, GPGPU kernels requiring extended precision.
        * Applications needing reproducible results across different hardware platforms.

    Example Usage:
    -------------------------------------------------------------------------
        #include "fpmp.hpp"
        #include "fpmp_math.hpp"
        
        // Basic arithmetic with double-float precision
        fp32mp2 a = 1.23456789123456789;       // double-float precision from double value
        fp32mp2 b = 9.87654321987654321;       // double-float precision from double value
        auto sum = a + b;                        // High-precision addition
        auto product = a * b;                    // High-precision multiplication
        auto result  = fma(a, b, sum);           // Fused multiply-add: a*b + sum
        auto root = sqrt(a);                     // High-precision square root
        auto exponential = exp(a);               // High-precision exponential
        double d = static_cast<double>(result);  // Convert to double
        float f  = static_cast<float>(result);   // Convert to float (high part only)
        uint64_t bits = bit_cast<uint64_t>(result); // Bit-cast to 64-bit integer (IEEE-754 format)

        // Accuracy-explicit operations: override arithmetic accuracy for a single operation
        using namespace fpmp;
        fp32mp2_low x = ..., y = ...;
        auto diff = sub<fpmp2_accuracy::high>(x, y); // Accurate subtraction, result stays fp32mp2_low

    Naming Convention:
    -------------------------------------------------------------------------
    - Built-in functions (C-style API):
        * __nv_fpmp2_add, __nv_fpmp2_sub, __nv_fpmp2_mul, __nv_fpmp2_div : Basic arithmetic operations
        * __nv_fpmp2_acc, __nv_fpmp2_low_acc, __nv_fpmp2_high_acc : Optimized single-component accumulate
        * __nv_fpmp2_low_add, __nv_fpmp2_high_add : Method-specific variants
        * __nv_fpmp2_fma, __nv_fpmp2_mad : Fused multiply-add and multiply-add operations
        * __nv_fpmp2_sqrt, __nv_fpmp2_rsqrt : Square root and reciprocal square root
        * __nv_fpmp2_exp : Exponential function
        * __nv_fpmp2_from_double, __nv_fpmp2_to_double : Type conversions
        * __nv_fpmp2_cmp_eq, __nv_fpmp2_cmp_lt, etc. : Comparison operations
        * __nv_fpmp2_atomicAdd, __nv_fpmp2_atomicSub : Atomic operations (CUDA only, slower than hardware atomics)
        * __shfl_sync, __shfl_xor_sync, __shfl_down_sync, __shfl_up_sync :
          Warp shuffle overloads for fpmp2 pairs (CUDA only, header-only via fpmp_math.hpp).
    
    - C++ class template:
        * fpmp2_t<FpType, fpmp2_accuracy> : Template class with operator overloading
        * fp32mp2, fp32mp2_low, fp32mp2_high : Double-float type aliases
        * fp64mp2, fp64mp2_low, fp64mp2_high : Double-double type aliases

    - Accuracy-explicit free functions:
        * add<fpmp2_accuracy::m>(x, y) : Addition with explicit accuracy override
        * sub<fpmp2_accuracy::m>(x, y) : Subtraction with explicit accuracy override
        * mul<fpmp2_accuracy::m>(x, y) : Multiplication with explicit accuracy override
        * div<fpmp2_accuracy::m>(x, y) : Division with explicit accuracy override
        * fma<fpmp2_accuracy::m>(x, y, z) : Fused multiply-add with explicit accuracy override
        * mad<fpmp2_accuracy::m>(x, y, z) : Multiply-add with explicit accuracy override
        where m is one of: def, low, mid, high
        Each operation also has a mixed-type overload: any operand may be
        a built-in arithmetic scalar as long as at least one operand is
        fpmp2_t. The scalar is converted to the fpmp2 side's type.
        Example:  ffloat r = sub<fpmp2_accuracy::high>(a, 1.0f);

    Reference Papers:
    -------------------------------------------------------------------------
    [1] Dekker, T. (1971). A floating-point technique for extending the available precision. Numerische Mathematik, 18, 224–242.
    [2] Karp, A. H., & Markstein, P. (1997). High Precision Division and Square Root. ACM Transactions on Mathematical Software, 23(4), 561–589.
    [3] Thall, Andrew. Extended-Precision Floating-Point Numbers for GPU Computation. (http://andrewthall.org/papers/df64_qf128.pdf)
    [4] Nagai et al. (2008). Fast Quadruple Precision Arithmetic Library on Parallel Computer SR11000/J2. ICCS '08.

    Configuration Macros:
    -------------------------------------------------------------------------
    - FPMP_EXPLICIT_CASTS: When 1 (default), lossy/narrowing conversions INTO fpmp2_t (e.g., double
      to fp32mp2, fp64mp2 to fp32mp2, and integer (int32/uint32/int64/uint64) to fpmp2_t) require explicit casts, matching
      CCCL's strict-cast conventions. The widening conversion OUT to double (operator double()) is
      always implicit and is not affected by this macro. Set to 0 to restore the fully-implicit model
      (all conversions implicit) for easier migration of existing code from standard types.
    - FPMP_FP64MP2_ENABLE: When 1 (default), enables fp64mp2 (double-double) support. Set to 0
      to speed up builds and reduce code size when only fp32mp2 is needed.
    - FPMP_FP128_ENABLE: Automatically detected from compiler version and CUDA capabilities.
      Can be explicitly set to 0 to disable 128-bit float support (older compilers, compatibility).
    - FPMP_FP128_MATH_FALLBACK: When 1, fp64mp2 math functions use quad-precision (__fpmp_fp128)
      for higher accuracy. Requires libquadmath linkage, slower compilation, larger code.
      When 0 (default), falls back to double precision—faster builds, smaller code, but
      reduced accuracy for transcendental functions.
    - FPMP_OPTIMIZED_DOUBLE_TO_FPMP: When 1, double-to-fpmp2 conversion uses integer bit
      manipulation instead of FP64 casts. Avoids the slow FP64 pipeline on GPUs with limited
      double-precision throughput (1:64 ratio). When 0 (default), uses standard casts.
    - FPMP_OPTIMIZED_FPMP_TO_DOUBLE: When 1, fpmp2-to-double conversion reconstructs the
      double bit pattern using integer arithmetic (no FP64 ops). More complex than the forward
      direction (full software double-add). When 0 (default), uses (double)hi + (double)lo.

    Important Notes:
    -------------------------------------------------------------------------
    - The error-free transformations (e.g., 2Sum, 2Mult) are essential for guaranteeing that hi/lo have the non-overlapping property.
    - It is possible to mix and match routines according to desired accuracy, speed, and hardware FMA support.
    - The library requires C++11 minimum (for alignas, constexpr) and works best with C++17 or later (for if constexpr).
    - For CUDA code, the library requires CUDA Toolkit 11.0 or later.
    - All operations are fully inlined for optimal performance when not using library mode.
*/


#include <cuda/__fp/fpmp_common.hpp>
#include <cuda/__fp/fpmp_impl.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{


/*********************************************************************
 * Multi-precision 32-bit floating-point emulation type (double-float)
 *********************************************************************/
/**
    * @brief Multi-precision 32-bit floating-point emulation type (double-float)
    * 
    * The `fpmp2_t` class provides a flexible, efficient, and accurate software-emulated 
    * floating-point type supporting extended precision beyond standard IEEE 754 single-precision (float). 
    * It is designed for GPU and CPU usage with double-float precision (hi, lo components),
    * with templates to control the underlying accuracy level (via the `fpmp2_accuracy` parameter).
    * 
    * ## Template Parameters
    * - `fpmp2_accuracy`: The arithmetic accuracy level. Usually one of:
    *     - `fpmp2_accuracy::low` (favor speed, possibly at minor cost in accuracy),
    *     - `fpmp2_accuracy::def`, `fpmp2_accuracy::mid`, `fpmp2_accuracy::high`
    * 
    * ## Features and Operations
    * - **Arithmetic**: Supports standard operators (+, -, *, /), fused multiply-add (fma), square roots (rsqrt, sqrt).
    * - **Renormalization**: Supports renormalization of the result of the arithmetic operations (useful for fast mode).
    * - **Construction**: Can be constructed from float, double, int32_t, uint32_t, int64_t, uint64_t.
    * - **Conversion**: Provides explicit and implicit conversion to standard C++ scalar types.
    * - **Comparison**: Supports all common relational operators (==, !=, <, <=, >, >=)
    * - **Bit-casting**: Supports reinterpretation of the value as a 64-bit integer (IEEE-754 format)
    * - **GPU & Host Compatibility**: All operations and members are decorated for both device and host use.
    * 
    * ## Internal Representation
    * The class internally stores its value in `val`, a structure `fp32m2_t`
    * encoding the high and low float components following the multi-precision scheme.
    * 
    * **Example Usage**:
    * @code
    * fpmp2_t<> a = 1.0f;
    * fpmp2_t<> b = 2.0f;
    * auto c = a + b; // High-precision addition
    * float f = static_cast<float>(c); // Convert back to float
    * uint64_t bits = bit_cast<uint64_t>(c); // Bit-cast to 64-bit integer (IEEE-754 format)
    * @endcode
    * 
    * ## Motivation
    * This class is intended for scenarios requiring higher precision than float offers,
    * for example in scientific computing, GPU linear algebra, or when porting algorithms requiring 
    * quad/double-scalar emulation to platforms where native double/quad is slow or unavailable.
    * 
    * ## Thread Safety
    * - Each instance manages its own state and is safe for concurrent use in different threads.
    * 
    * ## Limitations
    * - Denormals, NaN, and Inf handling may differ from IEEE 754 strict standards, depending on accuracy level.
    * - Performance depends on template parameters and underlying hardware.
    */

// fpmp2_t class template
// met: arithmetic accuracy level
//     - fpmp2_accuracy::mid (default): Dekker-based split and error accumulation technique
//     - fpmp2_accuracy::high: Thall-based and other techniques
//     - fpmp2_accuracy::low: fast arithmetic operation without re-normalizations
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
class alignas(2 * alignof(FpType)) fpmp2_t 
{
    public:

    /*
    // Accessor functions for hi and lo fields.
    // constexpr so the cross-method converting constructor below (and any
    // other context that needs (hi, lo) at compile time) can stay constexpr.
    */
    constexpr __FPMP_API_DECL__ FpType hi() const { return mp2_hi; }
    constexpr __FPMP_API_DECL__ FpType lo() const { return mp2_lo; }

    /*
    // Basic constructors
    */
    // Default constructor
    fpmp2_t() = default;


    // Constructor from hi and lo floats (direct initialization).
    // constexpr so constant `fpmp2_t` arrays can live in constexpr
    // context
    constexpr __FPMP_API_DECL__ fpmp2_t(FpType hi, FpType lo) : mp2_hi(hi), mp2_lo(lo) {}

    /*
    // Defaulted copy constructor (trivially copyable)
    // Note: NVCC implicitly makes defaulted special members __host__ __device__
    */
    fpmp2_t(const fpmp2_t& other) = default;

    /*
    // Copy constructor from volatile fpmp2_t
    // Template so it is NOT a copy constructor per the C++ standard.
    // The volatile overloads are wrapped in dummy templates
    // so that the C++ standard does not consider them copy constructors/assignment
    // operators (a template is never a copy constructor or copy assignment operator),
    // preserving trivial copyability while retaining volatile access support.
    */
    template<typename Dummy = void>
    __FPMP_API_DECL__ fpmp2_t(const volatile fpmp2_t& other)  
    { 
        mp2_hi = other.mp2_hi; 
        mp2_lo = other.mp2_lo;  
    }

    // Defaulted copy assignment operator (trivially copyable)
    constexpr fpmp2_t& operator=(const fpmp2_t& other) = default;

    /*
    // Assignment operator to volatile fpmp2_t
    // Template so it is NOT a copy assignment operator per the C++ standard
    // Returns void to avoid C++20 -Wvolatile (deprecated volatile return)
    */
    template<typename Dummy = void>
    __FPMP_API_DECL__ void operator=(const fpmp2_t& other) volatile 
    { 
        mp2_hi = other.mp2_hi; 
        mp2_lo = other.mp2_lo; 
    }

    /*
    // Assignment operator from volatile fpmp2_t
    // Template so it is NOT a copy assignment operator per the C++ standard
    */
    template<typename Dummy = void>
    __FPMP_API_DECL__ fpmp2_t& operator=(const volatile fpmp2_t& other) 
    { 
        mp2_hi = other.mp2_hi; 
        mp2_lo = other.mp2_lo; 
        return *this; 
    }

    /*
    // Cross-method converting constructor (same FpType, different method tag).
    //
    // Different fpmp2_t specializations with the same FpType share an
    // identical (hi, lo) representation; only the method tag (which selects
    // the algorithm used by downstream arithmetic) differs. Without this
    // overload, a direct-init like
    //     fp32mp2_low c(b);   // b is fp32mp2_high
    // would silently route through fpmp2_t(double), i.e. operator
    // double() + ctor(double), an expensive round trip — particularly bad on
    // GPUs with limited FP64 throughput. With this overload, the same
    // direct-init becomes a plain (hi, lo) copy.
    //
    // Marked `explicit` on purpose: copy-initialization
    //     fp32mp2_low d = b;   // ill-formed
    // and copy-assignment
    //     a = b;                  // ill-formed
    // continue to fail to compile (a single implicit conversion sequence
    // can't chain two user-defined conversions). To opt in, write
    //     fp32mp2_low c(b);              // direct-init
    //     a = fp32mp2_low(b);            // explicit conversion + assign
    //     a = static_cast<fp32mp2_low>(b);
    //
    // SFINAE excludes met2 == met to avoid clashing with the defaulted
    // copy constructor.
    */
    template<fpmp2_accuracy met2,
             typename = typename std::enable_if<met2 != met>::type>
    constexpr __FPMP_API_DECL__ explicit fpmp2_t(const fpmp2_t<FpType, met2>& other)
        : mp2_hi(other.hi()), mp2_lo(other.lo())
    {
    }

#if (FPMP_FP64MP2_ENABLE == 1)
    /*
    // Cross-precision converting constructor + assignment (upconvert): fp32mp2 -> fp64mp2.
    //
    // Enabled only when FpType == double. The conversion is exact and lossless:
    //   - Each float component casts losslessly to double (float values are a
    //     subset of double values), so (d_hi, d_lo) = ((double)hi, (double)lo)
    //     is an exact representation of the original mathematical value.
    //   - The pair is then renormalized with fast_two_sum so the result is a
    //     valid fp64mp2 (i.e. |out.lo| <= ulp_double(out.hi)/2).
    //
    // Why renormalize instead of just collapsing to (d_hi + d_lo, 0.0):
    //   In a renormalized fp32mp2 only |lo| <= ulp_float(hi)/2 = 2^-24*|hi|
    //   is guaranteed, but lo may be far smaller — e.g. (1.0f, 2^-100f) is a
    //   valid pair representing 1 + 2^-100. That value is NOT representable
    //   in a single double (2^-100 falls below the 53-bit precision of 1.0),
    //   but it IS representable as the fp64mp2 pair (1.0, 2^-100) because
    //   fp64mp2's renormalization bound is 2^-53*|hi|. fast_two_sum captures
    //   exactly this residual, so no precision is ever lost.
    //
    // Implicit on purpose: mirrors the IEEE-754 float -> double widening
    // (no precision loss). Accepts any source `met2`; the destination
    // method tag is preserved.
    */
    template<typename U = FpType, fpmp2_accuracy met2,
             typename = typename std::enable_if<std::is_same<U, double>::value>::type>
    __FPMP_API_DECL__ fpmp2_t(const fpmp2_t<float, met2>& src)
    {
        const double d_hi_in = static_cast<double>(src.hi());
        const double d_lo_in = static_cast<double>(src.lo());
        // Renormalized fp32mp2 has |hi| >= |lo|, so fast_two_sum is safe.
        mp2_hi = fpmp::fast_two_sum(d_hi_in, d_lo_in, &mp2_lo);
    }

    template<typename U = FpType, fpmp2_accuracy met2,
             typename = typename std::enable_if<std::is_same<U, double>::value>::type>
    __FPMP_API_DECL__ fpmp2_t& operator=(const fpmp2_t<float, met2>& src)
    {
        const double d_hi_in = static_cast<double>(src.hi());
        const double d_lo_in = static_cast<double>(src.lo());
        mp2_hi = fpmp::fast_two_sum(d_hi_in, d_lo_in, &mp2_lo);
        return *this;
    }

    /*
    // Cross-precision converting constructor + assignment (downconvert): fp64mp2 -> fp32mp2.
    //
    // Enabled only when FpType == float. The conversion is lossy (double's
    // 53 bits do not fit in float's 24), so it is performed via:
    //   1. Split src.hi() into a (float, float) pair: (a_hi, a_lo).
    //   2. Split src.lo() into a (float, float) pair: (b_hi, b_lo).
    //   3. Sum the two fp32mp2 pairs with __nv_fpmp2_add<float> to obtain a
    //      renormalized fp32mp2 result.
    // This typically preserves ~48 bits of effective precision (the fp32mp2
    // limit), losing only ~5 bits relative to the fp64mp2 input.
    //
    // Marked __FPMP_EXPLICIT__ (matches the existing double -> fp32mp2
    // narrowing constructor) so callers must opt in via static_cast or
    // direct-init, mirroring the IEEE-754 double -> float narrowing.
    // The companion assignment operator is provided for symmetry; both
    // perform the same precision-preserving 2-pair add.
    */
    template<typename U = FpType, fpmp2_accuracy met2,
             typename = typename std::enable_if<std::is_same<U, float>::value>::type>
    __FPMP_API_DECL__ __FPMP_EXPLICIT__ fpmp2_t(const fpmp2_t<double, met2>& src)
    {
        float a_hi, a_lo, b_hi, b_lo;
        __nv_fpmp2_from_double<float>(src.hi(), &a_hi, &a_lo);
        __nv_fpmp2_from_double<float>(src.lo(), &b_hi, &b_lo);
        __nv_fpmp2_add<float>(a_hi, a_lo, b_hi, b_lo, &mp2_hi, &mp2_lo);
    }

    template<typename U = FpType, fpmp2_accuracy met2,
             typename = typename std::enable_if<std::is_same<U, float>::value>::type>
    __FPMP_API_DECL__ fpmp2_t& operator=(const fpmp2_t<double, met2>& src)
    {
        float a_hi, a_lo, b_hi, b_lo;
        __nv_fpmp2_from_double<float>(src.hi(), &a_hi, &a_lo);
        __nv_fpmp2_from_double<float>(src.lo(), &b_hi, &b_lo);
        __nv_fpmp2_add<float>(a_hi, a_lo, b_hi, b_lo, &mp2_hi, &mp2_lo);
        return *this;
    }
#endif // FPMP_FP64MP2_ENABLE == 1

    /*
    // Conversion operators
    */
    // ==== Conversions from other types to fpmp2_t:
    // Implicit conversion from a single FpType (lo == 0).
    // constexpr so float/double constants flow into constexpr coefficient
    // tables without forcing callers to materialise the (hi, lo) pair.
    constexpr __FPMP_API_DECL__ fpmp2_t(FpType f)
        : mp2_hi(f), mp2_lo((FpType)0)
    {
    }

    /*
    // Constructor from double (only for FpType == float)
    // C++20: compile-time uses simple float casts, runtime delegates to __nv_fpmp2_from_double
    // C++17: always uses simple float casts (member initializer list, constexpr-safe)
    // When FpType is double, use the regular FpType constructor instead
    */
    template<typename U = FpType, typename = typename std::enable_if<std::is_same<U, float>::value>::type>
#if __cplusplus >= 202002L
    constexpr __FPMP_API_DECL__ __FPMP_EXPLICIT__ fpmp2_t(double d)
    {
        if (__FPMP_IS_CONSTEVAL__()) {
            mp2_hi = (FpType)d;
            mp2_lo = (FpType)(d - (double)(FpType)d);
        } else {
            __nv_fpmp2_from_double(d, &mp2_hi, &mp2_lo);
        }
    }
#else
    constexpr __FPMP_API_DECL__ __FPMP_EXPLICIT__ fpmp2_t(double d)
        : mp2_hi((FpType)d), mp2_lo((FpType)(d - (double)(FpType)d))
    {
    }
#endif

    #if (FPMP_FP64MP2_ENABLE == 1)
        //  __fpmp_fp128  operations (only for FpType == double)
        // available only for CUDA architectures >= 1000 or when FPMP_FP128_ENABLE is defined
        #if FPMP_FP128_ENABLE == 1
            // Constructor from __fpmp_fp128 (only for FpType == double)
            // C++20: compile-time uses simple double casts, runtime delegates to __nv_fpmp2_from_quad
            // C++17: always uses simple double casts (member initializer list, same as original)
            template<typename U = FpType, typename = typename std::enable_if<std::is_same<U, double>::value>::type>
            constexpr __FPMP_API_DECL__ __FPMP_EXPLICIT__ fpmp2_t(__fpmp_fp128 d)
        #if __cplusplus >= 202002L
            {
                if (__FPMP_IS_CONSTEVAL__()) {
                    mp2_hi = (FpType)d;
                    mp2_lo = (FpType)(d - (__fpmp_fp128)(FpType)d);
                } else {
                    __nv_fpmp2_from_quad(d, &mp2_hi, &mp2_lo);
                }
            }
        #else
                : mp2_hi((FpType)d), mp2_lo((FpType)(d - (__fpmp_fp128)(FpType)d))
            {
            }
        #endif
            // Explicit conversion to __fpmp_fp128
            __FPMP_API_DECL__ explicit operator __fpmp_fp128() const { 
                return __nv_fpmp2_to_quad(mp2_hi, mp2_lo);
            }
        #endif // FPMP_FP128_ENABLE == 1
    #endif // FPMP_FP64MP2_ENABLE == 1

    // Constructor from int32_t
    __FPMP_API_DECL__ __FPMP_EXPLICIT__ fpmp2_t(int32_t i) { __nv_fpmp2_from_int(i, &mp2_hi, &mp2_lo);}

    // Constructor from uint32_t
    __FPMP_API_DECL__ __FPMP_EXPLICIT__ fpmp2_t(uint32_t i) { __nv_fpmp2_from_uint(i, &mp2_hi, &mp2_lo);}

    // Constructor from int64_t
    __FPMP_API_DECL__ __FPMP_EXPLICIT__ fpmp2_t(int64_t i) { __nv_fpmp2_from_ll(i, &mp2_hi, &mp2_lo);}
    
    // Constructor from uint64_t
    __FPMP_API_DECL__ __FPMP_EXPLICIT__ fpmp2_t(uint64_t i) { __nv_fpmp2_from_ull(i, &mp2_hi, &mp2_lo);}

    // ==== Conversion from fpmp2_t to other types:
    // Conversion to double is ALWAYS implicit (never gated by FPMP_EXPLICIT_CASTS).
    // It is a value-preserving widening conversion (the analog of the implicit
    // IEEE-754 float -> double), so it stays implicit for ergonomics. This does NOT
    // cause hidden FP64 in fpmp<->fpmp conversions or fpmp arithmetic: cross-method
    // conversions use a direct (hi,lo) copy, and mixed fpmp/scalar operators promote
    // the scalar up to fpmp. It only takes effect when an fpmp value is fed into a
    // double-typed sink.
    __FPMP_API_DECL__ operator double() const          { return __nv_fpmp2_to_double(mp2_hi, mp2_lo);}
    __FPMP_API_DECL__ operator double() const volatile { return __nv_fpmp2_to_double(mp2_hi, mp2_lo);}

    // Explicit conversions to other types
    // Conversion to float
    __FPMP_API_DECL__ explicit operator float() const          { return __nv_fpmp2_to_float(mp2_hi, mp2_lo);}
    __FPMP_API_DECL__ explicit operator float() const volatile { return __nv_fpmp2_to_float(mp2_hi, mp2_lo);}
    
    // Conversion to int32_t
    __FPMP_API_DECL__ explicit operator int32_t() const          { return __nv_fpmp2_to_int(mp2_hi, mp2_lo);}
    __FPMP_API_DECL__ explicit operator int32_t() const volatile { return __nv_fpmp2_to_int(mp2_hi, mp2_lo);}
    
    // Conversion to uint32_t
    __FPMP_API_DECL__ explicit operator uint32_t() const          { return __nv_fpmp2_to_uint(mp2_hi, mp2_lo);}
    __FPMP_API_DECL__ explicit operator uint32_t() const volatile { return __nv_fpmp2_to_uint(mp2_hi, mp2_lo);}
    
    // Conversion to int64_t
    __FPMP_API_DECL__ explicit operator int64_t() const          { return __nv_fpmp2_to_ll(mp2_hi, mp2_lo);}
    __FPMP_API_DECL__ explicit operator int64_t() const volatile { return __nv_fpmp2_to_ll(mp2_hi, mp2_lo);}
    
    // Conversion to uint64_t
    __FPMP_API_DECL__ explicit operator uint64_t() const          { return __nv_fpmp2_to_ull(mp2_hi, mp2_lo);}
    __FPMP_API_DECL__ explicit operator uint64_t() const volatile { return __nv_fpmp2_to_ull(mp2_hi, mp2_lo);}
    
    // (renormalize)
    __FPMP_API_DECL__ friend fpmp2_t renormalize(const fpmp2_t& x) 
    { 
        fpmp2_t res; 
        __nv_fpmp2_renormalize(x.mp2_hi, x.mp2_lo, &res.mp2_hi, &res.mp2_lo);
        return res; 
    }
    
    /*
    // Arithmetic operations:
    */
    // (+)
    __FPMP_API_DECL__ friend fpmp2_t operator+(const fpmp2_t& x, const fpmp2_t& y) 
    { 
        fpmp2_t res; 
        if constexpr (met == fpmp2_accuracy::low)          { __nv_fpmp2_low_add  (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        else if constexpr (met == fpmp2_accuracy::high)    { __nv_fpmp2_high_add (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        else                                               { __nv_fpmp2_add      (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); }
        return res; 
    }

    // (-)
    __FPMP_API_DECL__ friend fpmp2_t operator-(const fpmp2_t& x, const fpmp2_t& y) 
    { 
        fpmp2_t res;
        if constexpr (met == fpmp2_accuracy::low)          { __nv_fpmp2_low_sub  (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        else if constexpr (met == fpmp2_accuracy::high)    { __nv_fpmp2_high_sub (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        else                                               { __nv_fpmp2_sub      (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); }
        return res; 
    }

    // (*)
    __FPMP_API_DECL__ friend fpmp2_t operator*(const fpmp2_t& x, const fpmp2_t& y) 
    { 
        fpmp2_t res; 
        if constexpr (met == fpmp2_accuracy::low)          { __nv_fpmp2_low_mul  (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        #if __FPMP_USE_ACCURATE_MUL__ == 1
        else if constexpr (met == fpmp2_accuracy::high)    { __nv_fpmp2_high_mul (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); }
        #endif
        else                                               { __nv_fpmp2_mul      (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); }
        return res; 
    }

    // (/)
    __FPMP_API_DECL__ friend fpmp2_t operator/(const fpmp2_t& x, const fpmp2_t& y) 
    { 
        fpmp2_t res;
        if constexpr (met == fpmp2_accuracy::low)          { __nv_fpmp2_low_div  (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        #if __FPMP_USE_ACCURATE_DIV__ == 1
        else if constexpr (met == fpmp2_accuracy::high)    { __nv_fpmp2_high_div (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); }
        #endif
        else                                               { __nv_fpmp2_div      (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        return res; 
    }

    // (sqrt)
    __FPMP_API_DECL__ friend fpmp2_t sqrt(const fpmp2_t& x) 
    { 
        fpmp2_t res; 
        __nv_fpmp2_sqrt(x.mp2_hi, x.mp2_lo, &res.mp2_hi, &res.mp2_lo); 
        return res; 
    }

    // (rsqrt)
    __FPMP_API_DECL__ friend fpmp2_t rsqrt(const fpmp2_t& x) 
    { 
        fpmp2_t res; 
        __nv_fpmp2_rsqrt(x.mp2_hi, x.mp2_lo, &res.mp2_hi, &res.mp2_lo);
        return res; 
    }
    
    // (fma)
    __FPMP_API_DECL__ friend fpmp2_t fma(const fpmp2_t& x, const fpmp2_t& y, const fpmp2_t& z) 
    { 
        fpmp2_t res; 
        if constexpr (met == fpmp2_accuracy::low)          { __nv_fpmp2_low_fma (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, z.mp2_hi, z.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        else if constexpr (met == fpmp2_accuracy::high)    { __nv_fpmp2_high_fma(x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, z.mp2_hi, z.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        else                                               { __nv_fpmp2_fma     (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, z.mp2_hi, z.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        return res; 
    }

    // (mad)
    __FPMP_API_DECL__ friend fpmp2_t mad(const fpmp2_t& x, const fpmp2_t& y, const fpmp2_t& z) 
    { 
        fpmp2_t res; 
        if constexpr (met == fpmp2_accuracy::low)          { __nv_fpmp2_low_mad  (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, z.mp2_hi, z.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        else if constexpr (met == fpmp2_accuracy::high)    { __nv_fpmp2_high_mad (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, z.mp2_hi, z.mp2_lo, &res.mp2_hi, &res.mp2_lo); } 
        else                                               { __nv_fpmp2_mad      (x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo, z.mp2_hi, z.mp2_lo, &res.mp2_hi, &res.mp2_lo); }
        return res; 
    }

    /*
    // Optimized compound assignment for single-component operands (accumulate)
    // Uses specialized __nv_fpmp2_acc functions which are more efficient than
    // full mp2+mp2 addition (saves ~6 operations by avoiding low-part 2Sum).
    */
    __FPMP_API_DECL__ fpmp2_t& operator+=(const FpType c) { 
        if constexpr (met == fpmp2_accuracy::low)          { __nv_fpmp2_low_acc  (c, &mp2_hi, &mp2_lo); }
        else if constexpr (met == fpmp2_accuracy::high)    { __nv_fpmp2_high_acc (c, &mp2_hi, &mp2_lo); }
        else                                               { __nv_fpmp2_acc      (c, &mp2_hi, &mp2_lo); }
        return *this; 
    }
    __FPMP_API_DECL__ fpmp2_t& operator-=(const FpType c) { 
        if constexpr (met == fpmp2_accuracy::low)          { __nv_fpmp2_low_acc  (-c, &mp2_hi, &mp2_lo); }
        else if constexpr (met == fpmp2_accuracy::high)    { __nv_fpmp2_high_acc (-c, &mp2_hi, &mp2_lo); }
        else                                               { __nv_fpmp2_acc      (-c, &mp2_hi, &mp2_lo); }
        return *this; 
    }

    // (neg)
    __FPMP_API_DECL__ fpmp2_t  operator-() const 
    { 
        fpmp2_t res;
        __nv_fpmp2_neg(mp2_hi, mp2_lo, &res.mp2_hi, &res.mp2_lo); 
        return res; 
    }

    /*
    // Comparison operators:
    */ 
    // equality (==)
    __FPMP_API_DECL__ friend bool operator==(const fpmp2_t& x, const fpmp2_t& y) { 
        return __nv_fpmp2_cmp_eq(x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo); }
    // inequality (!=)
    __FPMP_API_DECL__ friend bool operator!=(const fpmp2_t& x, const fpmp2_t& y) { 
        return __nv_fpmp2_cmp_ne(x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo); }
    // less than (<)
    __FPMP_API_DECL__ friend bool operator<(const fpmp2_t& x, const fpmp2_t& y) { 
        return __nv_fpmp2_cmp_lt(x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo); }
    // greater than (>)
    __FPMP_API_DECL__ friend bool operator>(const fpmp2_t& x, const fpmp2_t& y) { 
        return __nv_fpmp2_cmp_gt(x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo); }
    // less than or equal to (<=)
    __FPMP_API_DECL__ friend bool operator<=(const fpmp2_t& x, const fpmp2_t& y) { 
        return __nv_fpmp2_cmp_le(x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo); }
    // greater than or equal to (>=)
    __FPMP_API_DECL__ friend bool operator>=(const fpmp2_t& x, const fpmp2_t& y) { 
        return __nv_fpmp2_cmp_ge(x.mp2_hi, x.mp2_lo, y.mp2_hi, y.mp2_lo); }

    /*
    // C++20-style bit_cast for unpacked floating-point types
    // Bit-cast to 64-bit integer (IEEE-754 format)
    */
    template<typename To> 
    __FPMP_API_DECL__ friend To bit_cast(const fpmp2_t& from) 
    { 
        return static_cast<To>(__nv_fpmp2_bit_cast(from.mp2_hi, from.mp2_lo)); 
    }

    // Prefix increment/decrement
    __FPMP_API_DECL__ fpmp2_t& operator++() { *this = *this + fpmp2_t(1.0f); return *this; }
    __FPMP_API_DECL__ fpmp2_t& operator--() { *this = *this - fpmp2_t(1.0f); return *this; }
    // Postfix increment/decrement
    __FPMP_API_DECL__ fpmp2_t  operator++(int) { fpmp2_t temp(*this); *this = *this + fpmp2_t(1.0f); return temp; }
    __FPMP_API_DECL__ fpmp2_t  operator--(int) { fpmp2_t temp(*this); *this = *this - fpmp2_t(1.0f); return temp; }
    // Compound assignment operators (multi-precision operand)
    __FPMP_API_DECL__ fpmp2_t& operator+=(const fpmp2_t& other) { *this = *this + other; return *this; }
    __FPMP_API_DECL__ fpmp2_t& operator-=(const fpmp2_t& other) { *this = *this - other; return *this; }
    __FPMP_API_DECL__ fpmp2_t& operator*=(const fpmp2_t& other) { *this = *this * other; return *this; }
    __FPMP_API_DECL__ fpmp2_t& operator/=(const fpmp2_t& other) { *this = *this / other; return *this; }
    
    /*
    // Mixed types arithmetic operations
    // Support for mixed arithmetic and emulation types
    */
    // === mul ===
    template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                                           (std::is_arithmetic<T1>::value          || std::is_arithmetic<T2>::value))>::type> 
        __FPMP_API_DECL__ friend  fpmp2_t operator*(const T1& x, const T2& y) { 
            return fpmp2_t(x) * fpmp2_t(y); }
    // === div ===
    template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                                           (std::is_arithmetic<T1>::value          || std::is_arithmetic<T2>::value))>::type> 
        __FPMP_API_DECL__ friend  fpmp2_t operator/(const T1& x, const T2& y) { 
            return fpmp2_t(x) / fpmp2_t(y); }
    // === add ===
    template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                                           (std::is_arithmetic<T1>::value          || std::is_arithmetic<T2>::value))>::type> 
        __FPMP_API_DECL__ friend  fpmp2_t operator+(const T1& x, const T2& y) { 
            return fpmp2_t(x) + fpmp2_t(y); }
    // === sub ===
    template<typename T1, typename T2, typename = typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                                           (std::is_arithmetic<T1>::value          || std::is_arithmetic<T2>::value))>::type> 
        __FPMP_API_DECL__ friend  fpmp2_t operator-(const T1& x, const T2& y) { 
            return fpmp2_t(x) - fpmp2_t(y); }
    // === fma ===
    template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value || std::is_same<T3,fpmp2_t>::value) && 
                                                                                        (std::is_arithmetic<T1>::value          || std::is_arithmetic<T2>::value              || std::is_arithmetic<T3>::value))>::type>
        __FPMP_API_DECL__ friend fpmp2_t fma(const T1& x, const T2& y, const T3& z) { 
            return fma(fpmp2_t(x), fpmp2_t(y), fpmp2_t(z)); }
    // === mad ===
    template<typename T1, typename T2, typename T3, typename = typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value || std::is_same<T3,fpmp2_t>::value) && 
                                                                                        (std::is_arithmetic<T1>::value          || std::is_arithmetic<T2>::value              || std::is_arithmetic<T3>::value))>::type>
        __FPMP_API_DECL__ friend fpmp2_t mad(const T1& x, const T2& y, const T3& z) { 
            return mad(fpmp2_t(x), fpmp2_t(y), fpmp2_t(z)); }

    // equality (==)
    template<typename T1, typename T2>
        __FPMP_API_DECL__ friend typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                              (std::is_arithmetic<T1>::value         || std::is_arithmetic<T2>::value)), bool>::type
        operator==(const T1& x, const T2& y) { 
            return fpmp2_t(x) == fpmp2_t(y); }
    // inequality (!=)
    template<typename T1, typename T2>
        __FPMP_API_DECL__ friend typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                              (std::is_arithmetic<T1>::value         || std::is_arithmetic<T2>::value)), bool>::type
        operator!=(const T1& x, const T2& y) { 
            return fpmp2_t(x) != fpmp2_t(y); }
    // less than (<)
    template<typename T1, typename T2>
        __FPMP_API_DECL__ friend typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                              (std::is_arithmetic<T1>::value         || std::is_arithmetic<T2>::value)), bool>::type
        operator<(const T1& x, const T2& y) { 
            return fpmp2_t(x) < fpmp2_t(y); }
    // greater than (>)
    template<typename T1, typename T2>
        __FPMP_API_DECL__ friend typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                              (std::is_arithmetic<T1>::value         || std::is_arithmetic<T2>::value)), bool>::type
        operator>(const T1& x, const T2& y) { 
            return fpmp2_t(x) > fpmp2_t(y); }
    // less than or equal to (<=)
    template<typename T1, typename T2>
        __FPMP_API_DECL__ friend typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                              (std::is_arithmetic<T1>::value         || std::is_arithmetic<T2>::value)), bool>::type
        operator<=(const T1& x, const T2& y) { 
            return fpmp2_t(x) <= fpmp2_t(y); }
    // greater than or equal to (>=)
    template<typename T1, typename T2>
        __FPMP_API_DECL__ friend typename std::enable_if<((std::is_same<T1,fpmp2_t>::value || std::is_same<T2,fpmp2_t>::value) && 
                                                              (std::is_arithmetic<T1>::value         || std::is_arithmetic<T2>::value)), bool>::type
        operator>=(const T1& x, const T2& y) { 
            return fpmp2_t(x) >= fpmp2_t(y); }

    private:
    /*
    // Internal storage - two floats (hi, lo) representing double-float precision
    */
    FpType mp2_hi;
    FpType mp2_lo;
}; // class fpmp2_t 

/*********************************************************************
 * Accuracy-explicit arithmetic free functions
 *
 * Allow overriding the arithmetic method for a single operation
 * without changing the type, e.g.:
 *   using ffloat = fp32mp2_low;
 *   ffloat x = sub<fpmp2_accuracy::high>(a, b);  // accurate sub, result stays ffloat
 *
 * This avoids instantiating a second fpmp2_t specialization and
 * the associated register pressure on GPU.
 *
 * Two forms are provided per operation:
 *   - Strict form: both/all operands are fpmp2_t<FpType, met>. Used
 *     when types are already matched.
 *   - Mixed form (one template, symmetric): accepts any combination where
 *     at least one operand is fpmp2_t and at least one is a built-in
 *     arithmetic type. The arithmetic side is converted to the fpmp2 side
 *     via its (implicit/explicit) constructor, then dispatched to the
 *     strict form. Examples:
 *       ffloat x = sub<fpmp2_accuracy::high>(a, 1.0f);  // ffloat - float
 *       ffloat y = sub<fpmp2_accuracy::high>(1.0f, a);  // float  - ffloat
 *
 * The predicate "at least one is fpmp2_t AND at least one is
 * arithmetic" is the same disjoint-categories trick used by the operator
 * overloads above, so one symmetric template suffices to cover both
 * argument orders without ambiguity against the strict form.
 *********************************************************************/

// Trait: detect any specialization of fpmp2_t<FpType, met>.
// Kept in namespace fpmp to avoid polluting the global namespace.
namespace fpmp {
    template<typename T> struct is_fpmp2 : std::false_type {};
    template<typename FpType, fpmp2_accuracy met>
    struct is_fpmp2<fpmp2_t<FpType, met>> : std::true_type {};
}

template<fpmp2_accuracy m, typename FpType, fpmp2_accuracy met>
__FPMP_API_DECL__ fpmp2_t<FpType, met> add (const fpmp2_t<FpType, met>& x,
                                                 const fpmp2_t<FpType, met>& y) 
{
    FpType rhi, rlo;
    if constexpr (m == fpmp2_accuracy::low)          { __nv_fpmp2_low_add  (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
    else if constexpr (m == fpmp2_accuracy::high)    { __nv_fpmp2_high_add (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
    else                                             { __nv_fpmp2_add      (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
    return fpmp2_t<FpType, met>(rhi, rlo);
}

template<fpmp2_accuracy m, typename T1, typename T2,
         typename = typename std::enable_if<((fpmp::is_fpmp2<T1>::value     || fpmp::is_fpmp2<T2>::value) &&
                                             (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
__FPMP_API_DECL__ auto add(const T1& x, const T2& y)
{
    using mp2 = typename std::conditional<fpmp::is_fpmp2<T1>::value, T1, T2>::type;
    return add<m>(mp2(x), mp2(y));
}

template<fpmp2_accuracy m, typename FpType, fpmp2_accuracy met>
__FPMP_API_DECL__ fpmp2_t<FpType, met> sub (const fpmp2_t<FpType, met>& x,
                                                 const fpmp2_t<FpType, met>& y) 
{
    FpType rhi, rlo;
    if constexpr (m == fpmp2_accuracy::low)          { __nv_fpmp2_low_sub  (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
    else if constexpr (m == fpmp2_accuracy::high)    { __nv_fpmp2_high_sub (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
    else                                             { __nv_fpmp2_sub      (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
    return fpmp2_t<FpType, met>(rhi, rlo);
}

template<fpmp2_accuracy m, typename T1, typename T2,
         typename = typename std::enable_if<((fpmp::is_fpmp2<T1>::value     || fpmp::is_fpmp2<T2>::value) &&
                                             (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
__FPMP_API_DECL__ auto sub(const T1& x, const T2& y)
{
    using mp2 = typename std::conditional<fpmp::is_fpmp2<T1>::value, T1, T2>::type;
    return sub<m>(mp2(x), mp2(y));
}

template<fpmp2_accuracy m, typename FpType, fpmp2_accuracy met>
__FPMP_API_DECL__ fpmp2_t<FpType, met> mul (const fpmp2_t<FpType, met>& x,
                                                 const fpmp2_t<FpType, met>& y) 
{
    FpType rhi, rlo;
    if constexpr (m == fpmp2_accuracy::low)          { __nv_fpmp2_low_mul  (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
#if __FPMP_USE_ACCURATE_MUL__ == 1
    else if constexpr (m == fpmp2_accuracy::high)    { __nv_fpmp2_high_mul (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
#endif
    else                                             { __nv_fpmp2_mul      (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
    return fpmp2_t<FpType, met>(rhi, rlo);
}

template<fpmp2_accuracy m, typename T1, typename T2,
         typename = typename std::enable_if<((fpmp::is_fpmp2<T1>::value     || fpmp::is_fpmp2<T2>::value) &&
                                             (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
__FPMP_API_DECL__ auto mul(const T1& x, const T2& y)
{
    using mp2 = typename std::conditional<fpmp::is_fpmp2<T1>::value, T1, T2>::type;
    return mul<m>(mp2(x), mp2(y));
}

template<fpmp2_accuracy m, typename FpType, fpmp2_accuracy met>
__FPMP_API_DECL__ fpmp2_t<FpType, met> div (const fpmp2_t<FpType, met>& x,
                                                 const fpmp2_t<FpType, met>& y) 
{
    FpType rhi, rlo;
    if constexpr (m == fpmp2_accuracy::low)          { __nv_fpmp2_low_div  (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
#if __FPMP_USE_ACCURATE_DIV__ == 1
    else if constexpr (m == fpmp2_accuracy::high)    { __nv_fpmp2_high_div (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
#endif
    else                                             { __nv_fpmp2_div      (x.hi(), x.lo(), y.hi(), y.lo(), &rhi, &rlo); }
    return fpmp2_t<FpType, met>(rhi, rlo);
}

template<fpmp2_accuracy m, typename T1, typename T2,
         typename = typename std::enable_if<((fpmp::is_fpmp2<T1>::value     || fpmp::is_fpmp2<T2>::value) &&
                                             (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value))>::type>
__FPMP_API_DECL__ auto div(const T1& x, const T2& y)
{
    using mp2 = typename std::conditional<fpmp::is_fpmp2<T1>::value, T1, T2>::type;
    return div<m>(mp2(x), mp2(y));
}

template<fpmp2_accuracy m, typename FpType, fpmp2_accuracy met>
__FPMP_API_DECL__ fpmp2_t<FpType, met> fma (const fpmp2_t<FpType, met>& x,
                                                 const fpmp2_t<FpType, met>& y,
                                                 const fpmp2_t<FpType, met>& z) 
{
    FpType rhi, rlo;
    if constexpr (m == fpmp2_accuracy::low)          { __nv_fpmp2_low_fma  (x.hi(), x.lo(), y.hi(), y.lo(), z.hi(), z.lo(), &rhi, &rlo); }
    else if constexpr (m == fpmp2_accuracy::high)    { __nv_fpmp2_high_fma (x.hi(), x.lo(), y.hi(), y.lo(), z.hi(), z.lo(), &rhi, &rlo); }
    else                                             { __nv_fpmp2_fma      (x.hi(), x.lo(), y.hi(), y.lo(), z.hi(), z.lo(), &rhi, &rlo); }
    return fpmp2_t<FpType, met>(rhi, rlo);
}

template<fpmp2_accuracy m, typename T1, typename T2, typename T3,
         typename = typename std::enable_if<((fpmp::is_fpmp2<T1>::value     || fpmp::is_fpmp2<T2>::value     || fpmp::is_fpmp2<T3>::value) &&
                                             (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
__FPMP_API_DECL__ auto fma(const T1& x, const T2& y, const T3& z)
{
    using mp2 = typename std::conditional<fpmp::is_fpmp2<T1>::value, T1,
                typename std::conditional<fpmp::is_fpmp2<T2>::value, T2, T3>::type>::type;
    return fma<m>(mp2(x), mp2(y), mp2(z));
}

template<fpmp2_accuracy m, typename FpType, fpmp2_accuracy met>
__FPMP_API_DECL__ fpmp2_t<FpType, met> mad (const fpmp2_t<FpType, met>& x,
                                                 const fpmp2_t<FpType, met>& y,
                                                 const fpmp2_t<FpType, met>& z) 
{
    FpType rhi, rlo;
    if constexpr (m == fpmp2_accuracy::low)          { __nv_fpmp2_low_mad  (x.hi(), x.lo(), y.hi(), y.lo(), z.hi(), z.lo(), &rhi, &rlo); }
    else if constexpr (m == fpmp2_accuracy::high)    { __nv_fpmp2_high_mad (x.hi(), x.lo(), y.hi(), y.lo(), z.hi(), z.lo(), &rhi, &rlo); }
    else                                             { __nv_fpmp2_mad      (x.hi(), x.lo(), y.hi(), y.lo(), z.hi(), z.lo(), &rhi, &rlo); }
    return fpmp2_t<FpType, met>(rhi, rlo);
}

template<fpmp2_accuracy m, typename T1, typename T2, typename T3,
         typename = typename std::enable_if<((fpmp::is_fpmp2<T1>::value     || fpmp::is_fpmp2<T2>::value     || fpmp::is_fpmp2<T3>::value) &&
                                             (std::is_arithmetic<T1>::value || std::is_arithmetic<T2>::value || std::is_arithmetic<T3>::value))>::type>
__FPMP_API_DECL__ auto mad(const T1& x, const T2& y, const T3& z)
{
    using mp2 = typename std::conditional<fpmp::is_fpmp2<T1>::value, T1,
                typename std::conditional<fpmp::is_fpmp2<T2>::value, T2, T3>::type>::type;
    return mad<m>(mp2(x), mp2(y), mp2(z));
}


#if defined(__CUDACC__)
/*
* ============================================================================
* Warp Shuffle Helpers (CUDA-only, header-only)
* ============================================================================
* Overloads of CUDA's modern __shfl_sync family for the fpmp2_t pair.
* Each shuffle operates independently on the (hi, lo) components: the same
* mask, lane/delta, and width are used for both halves so the two parts of
* a multi-precision value always travel together to the same destination lane.
*
* Mirrors CUDA's API exactly:
*   __shfl_sync     (mask, var, srcLane,  width = warpSize)
*   __shfl_xor_sync (mask, var, laneMask, width = warpSize)
*   __shfl_down_sync(mask, var, delta,    width = warpSize)
*   __shfl_up_sync  (mask, var, delta,    width = warpSize)
*
* Defined as overloads in the global namespace (matching CUDA's convention)
* so the same call site works for built-in scalars and fpmp2_t.  The
* recursive `::__shfl_sync(mask, var.hi(), ...)` resolves to CUDA's
* float/double overload (our template only matches fpmp2_t arguments).
*
* Defined only for CUDA compilation (NVCC); the warp shuffle primitives have
* no host counterpart, so host-only translation units never see them.
*
* These are thread-cooperation primitives (not math), so they live in the core
* header and are available via <cuda/fpmp> without pulling in <cuda/fpmp_math>.
* ============================================================================
*/

template <typename FpType, fpmp2_accuracy met>
__FPMP_API_DEVICE_DECL__ fpmp2_t<FpType, met>
__shfl_sync(unsigned mask, const fpmp2_t<FpType, met>& var,
            int srcLane, int width = warpSize)
{
    return fpmp2_t<FpType, met>(
        ::__shfl_sync(mask, var.hi(), srcLane, width),
        ::__shfl_sync(mask, var.lo(), srcLane, width)
    );
}

template <typename FpType, fpmp2_accuracy met>
__FPMP_API_DEVICE_DECL__ fpmp2_t<FpType, met>
__shfl_xor_sync(unsigned mask, const fpmp2_t<FpType, met>& var,
                int laneMask, int width = warpSize)
{
    return fpmp2_t<FpType, met>(
        ::__shfl_xor_sync(mask, var.hi(), laneMask, width),
        ::__shfl_xor_sync(mask, var.lo(), laneMask, width)
    );
}

template <typename FpType, fpmp2_accuracy met>
__FPMP_API_DEVICE_DECL__ fpmp2_t<FpType, met>
__shfl_down_sync(unsigned mask, const fpmp2_t<FpType, met>& var,
                 unsigned int delta, int width = warpSize)
{
    return fpmp2_t<FpType, met>(
        ::__shfl_down_sync(mask, var.hi(), delta, width),
        ::__shfl_down_sync(mask, var.lo(), delta, width)
    );
}

template <typename FpType, fpmp2_accuracy met>
__FPMP_API_DEVICE_DECL__ fpmp2_t<FpType, met>
__shfl_up_sync(unsigned mask, const fpmp2_t<FpType, met>& var,
               unsigned int delta, int width = warpSize)
{
    return fpmp2_t<FpType, met>(
        ::__shfl_up_sync(mask, var.hi(), delta, width),
        ::__shfl_up_sync(mask, var.lo(), delta, width)
    );
}

#endif // __CUDACC__

/*********************************************************************
 * Aliases for the most common use cases
 *********************************************************************/
using fp32mp2      = fpmp2_t<float, fpmp2_accuracy::def>;
using fp32mp2_low  = fpmp2_t<float, fpmp2_accuracy::low>;
using fp32mp2_mid  = fpmp2_t<float, fpmp2_accuracy::mid>;
using fp32mp2_high = fpmp2_t<float, fpmp2_accuracy::high>;

#if FPMP_FP64MP2_ENABLE == 1
    using fp64mp2      = fpmp2_t<double, fpmp2_accuracy::def>;
    using fp64mp2_low  = fpmp2_t<double, fpmp2_accuracy::low>;
    using fp64mp2_mid  = fpmp2_t<double, fpmp2_accuracy::mid>;
    using fp64mp2_high = fpmp2_t<double, fpmp2_accuracy::high>;
#endif // FPMP_FP64MP2_ENABLE == 1

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_H

