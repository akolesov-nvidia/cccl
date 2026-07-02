//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_COMMON_H
#define _CUDA___FP_FPMP_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
// Pulling in cuda::std::bit_cast above surfaces the cuda::std namespace; make the
// type-trait and <cmath> sets complete so the library's unqualified std:: names
// inside namespace cuda::experimental keep resolving (to cuda::std equivalents).
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

/*
    fpmp_common.h - Common Definitions and Internal Utilities for Multi-Precision Floating-Point Library
    ======================================================================================================
    This header provides common macros, type definitions, and internal utility functions used throughout
    the multi-precision floating-point arithmetic library. It serves as the foundation for both host and 
    device code compilation.
    
    Key Components:
    -------------------------------------------------------------------------
    - Compiler Detection and Attributes:
        * __FPMP_API_DECL__, __FPMP_API_DEVICE_DECL__ : API function declarations
        * __FPMP_INTERNAL_DECL__, __FPMP_INTERNAL_DEVICE_DECL__ : Internal function declarations
        * __FPMP_BUILTIN_DECL__, __FPMP_BUILTIN_DEVICE_DECL__ : Built-in function declarations
        * Cross-platform __forceinline__ support (CUDA, GCC, Clang, MSVC)
    
    - Configuration Macros:
        * FPMP_EXPLICIT_CASTS : Control implicit/explicit conversion behavior
        * FPMP_FP64MP2_ENABLE : Enable/disable double precision support
        * FPMP_OPTIMIZED_DOUBLE_TO_FPMP : Use integer bit manipulation for double -> fpmp2 conversion
        * FPMP_OPTIMIZED_FPMP_TO_DOUBLE : Use integer bit manipulation for fpmp2 -> double conversion
        * FPMP_FP128_MATH_FALLBACK : Fallback fp128 math functions to system implementation
        * __FPMP_LARGE_TRIG_FP64_FALLBACK__ : Fallback fp32mp2 large-arg trig to system fp64 sin/cos
        * FPMP_LIB (0/1), FPMP_INLINE (0/1) : Compilation mode control
        * __FPMP_BUILD_LIB__, __FPMP_USE_LIB__ : Internal library build/usage mode control
    
    - Arithmetic Accuracy Enumeration:
        * fpmp2_accuracy::def : Default (== mid) Dekker-based arithmetic
        * fpmp2_accuracy::low : Fast arithmetic without full renormalization
        * fpmp2_accuracy::mid : Dekker-based arithmetic (default level)
        * fpmp2_accuracy::high : Thall-based accurate arithmetic
    
    - Internal Utility Functions:
        * internal_bit_cast<To, From> : Type-safe bit casting (C++20 std::bit_cast polyfill)
        * internal_fabs, internal_isnan : Single-precision/double-precision scalar helpers (host + device)
          (intentionally prefixed to avoid colliding with ::fabs / ::isnan from <cmath> when
           a TU uses `using namespace fpmp;`)
        * add_rn, add_rz, sub_rn, mul_rn, fma_rn : Rounding mode specific operations
        * rcp_rn, rsqrt_rn : Reciprocal and reciprocal square root operations
        * fast_exp2, fast_log2 : Fast SFU-based base-2 exp/log (CUDA: ex2/lg2.approx; host: exp2f/log2f)
        * fp2int_rz, int2fp_rz, etc. : Floating point to integer conversions
        * two_mult_fma, fast_two_sum, two_sum : Error-free transformation algorithms
        * from_double : Double to (hi, lo) conversion utility
    
    Compatibility:
    -------------------------------------------------------------------------
    - Requires C++11 minimum (for alignas, constexpr)
    - Works best with C++17 or later (for if constexpr)
    - Supports both CUDA (nvcc) and standard C++ compilers (GCC, Clang, MSVC)
    - Provides fallback implementations for host-only compilation
*/

#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <iostream>
#include <string>
#include <cuda/std/cmath>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

/*
// Require C++11 minimum for alignas support
*/
#if !defined(__cplusplus) || __cplusplus < 201103L
    #error "This header requires C++11 or later (for alignas, constexpr, etc.)"
#endif

/*
// Check for if constexpr support (C++17)
*/
#if __cplusplus < 201703L
    #warning "This header works best with C++17 or later for if constexpr support"
#endif

/*
// External configuration macros
*/

/*
// FPMP_LIB: Compilation mode control.
//   1 = link against precompiled library (maps to __FPMP_USE_LIB__)
//   0 = header-only inline mode (default)
// FPMP_INLINE is the inverse alias: FPMP_INLINE=1 is equivalent to FPMP_LIB=0.
*/
#ifndef FPMP_LIB
    #ifdef FPMP_INLINE
        #if FPMP_INLINE == 1
            #define FPMP_LIB 0
        #else
            #define FPMP_LIB 1
        #endif
    #else
        #define FPMP_LIB 0
    #endif
#endif
#ifndef FPMP_INLINE
    #if FPMP_LIB == 1
        #define FPMP_INLINE 0
    #else
        #define FPMP_INLINE 1
    #endif
#endif
#if FPMP_LIB == 1 && !defined(__FPMP_USE_LIB__)
    #define __FPMP_USE_LIB__
#endif

// FPMP_EXPLICIT_CASTS controls whether lossy/narrowing conversions INTO fpmp2_t
// are explicit. It gates only the constructors:
//   - double      -> fp32mp2   (narrowing)
//   - fp64mp2     -> fp32mp2   (narrowing)
//   - __float128  -> fp64mp2   (narrowing)
//   - int32_t / uint32_t -> fpmp2_t
//   - int64_t / uint64_t -> fpmp2_t
// The conversion OUT to double (operator double()) is always implicit and is NOT
// affected by this macro (it is a value-preserving widening conversion).
//
// Default is 1 (lossy casts explicit), matching CCCL's strict-cast conventions.
// Existing users who rely on the fully-implicit model can restore it with
// -DFPMP_EXPLICIT_CASTS=0 (makes the constructors above implicit again).
#ifndef FPMP_EXPLICIT_CASTS
    #define FPMP_EXPLICIT_CASTS 1
#endif

/*
// FPMP_OPTIMIZED_DOUBLE_TO_FPMP: Use integer bit manipulation for double -> fpmp2 conversion.
// When 1: the conversion uses integer shifts/masks to split the double mantissa
//   into two float components without FP64 arithmetic. This avoids the slow FP64 pipeline
//   on GPUs with limited double-precision throughput (e.g., consumer GPUs with 1:64 ratio).
// When 0 (default): uses the standard cast-based approach: hi = (float)x; lo = (float)(x - (double)hi).
//
// NOTE: The optimized path increases register usage due to additional integer operations.
//   In large kernels with high register pressure, this may cause register spills to local
//   memory, negating the performance benefit. Profile your specific kernel to verify.
*/
#ifndef FPMP_OPTIMIZED_DOUBLE_TO_FPMP
    #define FPMP_OPTIMIZED_DOUBLE_TO_FPMP 0
#endif

/*
// FPMP_OPTIMIZED_FPMP_TO_DOUBLE: Use integer bit manipulation for fpmp2 -> double conversion.
// When 1: the conversion reconstructs the double bit pattern from the two float
//   components using integer shifts/masks and a software double-add, without any FP64
//   arithmetic.  This avoids the slow FP64 pipeline on GPUs with limited double-precision
//   throughput (e.g., consumer GPUs with 1:64 ratio).
// When 0 (default): uses the standard cast-based approach: (double)hi + (double)lo
//   (2x F2D + 1x DADD = 3 FP64 operations).
//
// NOTE: The optimized path increases register usage due to additional integer operations.
//   In large kernels with high register pressure, this may cause register spills to local
//   memory, negating the performance benefit. Profile your specific kernel to verify.
*/
#ifndef FPMP_OPTIMIZED_FPMP_TO_DOUBLE
    #define FPMP_OPTIMIZED_FPMP_TO_DOUBLE 0
#endif

/*
// Define if double precision based types (double-double) are enabled
*/
#ifndef FPMP_FP64MP2_ENABLE
    #define FPMP_FP64MP2_ENABLE 1
#endif

/*
// libquadmath availability detection
// -----------------------------------
// libquadmath ships only with GCC's runtime on x86 (32/64-bit) and provides
// <quadmath.h>, the *q math suite (expq, sinq, sqrtq, ...), and the
// __float128 type. It is NOT available on:
//   - Non-x86 architectures (ARM, ARM64, RISC-V, PowerPC, ...)
//   - MSVC (no GCC runtime)
//   - Most Windows toolchains (MinGW configurations vary)
//
// Override at compile time with -DFPMP_HOST_SUPPORTS_LIBQUADMATH=1 (force enable) or
// -DFPMP_HOST_SUPPORTS_LIBQUADMATH=0 (force disable) when the auto-detection is wrong
// for your environment.
*/
#ifndef FPMP_HOST_SUPPORTS_LIBQUADMATH
    #if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && !defined(_MSC_VER) && !defined(_WIN32)
        #define FPMP_HOST_SUPPORTS_LIBQUADMATH 1
    #else
        #define FPMP_HOST_SUPPORTS_LIBQUADMATH 0
    #endif
#endif

/*
// Detection of systems where 'long double' is 128-bit IEEE 754 quadruple precision
// ---------------------------------------------------------------------------------
// On these platforms, 'long double' has the same binary format as __float128,
// enabling the use of standard <cmath> functions (expl, sinl, sqrtl, logl, etc.)
// instead of libquadmath (*q functions like expq, sinq, sqrtq, logq).
//
// True on:
//   - aarch64/ARM64: IEEE quad is the standard long double
//   - s390x (IBM Z): IEEE quad long double
//   - PowerPC with -mabi=ieeelongdouble (GCC defines __LONG_DOUBLE_IEEE128__)
//
// NOT true on:
//   - x86/x86_64: long double is 80-bit x87 extended precision (different format)
//   - PowerPC default: long double is IBM double-double format (different format)
//   - Windows: long double equals double (64-bit)
*/
#ifndef FPMP_HOST_SUPPORTS_LDOUBLE128
    #if defined(__aarch64__) || defined(_M_ARM64) || defined(__s390x__) || defined(__LONG_DOUBLE_IEEE128__)
        #define FPMP_HOST_SUPPORTS_LDOUBLE128 1
    #else
        #define FPMP_HOST_SUPPORTS_LDOUBLE128 0
    #endif
#endif

/*
// Automatic detection of FP128 support based on platform capabilities
// --------------------------------------------------------------------
// FP128 is enabled when the platform provides 128-bit IEEE 754 quadruple
// precision arithmetic, either via:
//   - __float128 + libquadmath (x86 Linux/Unix with GCC)
//   - 128-bit long double (aarch64, s390x, PowerPC with IEEE long double)
//   - CUDA device intrinsics (sm_100+ / Blackwell)
//
// Disabled on:
//   - Windows (MSVC has no __float128; MinGW varies)
//   - x86 without libquadmath
//   - Older CUDA architectures (< sm_100)
*/
#ifndef FPMP_FP128_ENABLE
    #if defined(__CUDACC__)
        #if (FPMP_FP64MP2_ENABLE == 1) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
            #define FPMP_FP128_ENABLE 1
        #else
            #define FPMP_FP128_ENABLE 0
        #endif
    #else
        #if (FPMP_HOST_SUPPORTS_LIBQUADMATH == 1) || (FPMP_HOST_SUPPORTS_LDOUBLE128 == 1)
            #define FPMP_FP128_ENABLE 1
        #else
            #define FPMP_FP128_ENABLE 0
        #endif
    #endif
#endif

/*
// fp128 math functions fallback to system implementation enabling
*/
#ifndef FPMP_FP128_MATH_FALLBACK
    #if (FPMP_FP128_ENABLE == 1)
        #define FPMP_FP128_MATH_FALLBACK 1
    #else
        #define FPMP_FP128_MATH_FALLBACK 0
    #endif
#endif

/*
// Internal 128-bit floating-point type definition
// ------------------------------------------------
// __fpmp_fp128 is the library's internal quad-precision type, mapped to the
// platform-specific 128-bit IEEE 754 floating-point type:
//
//   - x86 Linux/Unix with libquadmath: __float128 (GCC extension)
//   - ARM64, s390x, PowerPC with IEEE long double: long double
//
// Only defined when FPMP_FP128_ENABLE == 1.
*/
#if (FPMP_FP128_ENABLE == 1)
    #if (defined(__CUDA_ARCH__)) || (FPMP_HOST_SUPPORTS_LIBQUADMATH == 1)
        typedef __float128 __fpmp_fp128;
    #elif (FPMP_HOST_SUPPORTS_LDOUBLE128 == 1)
        typedef long double __fpmp_fp128;
    #else
        #error "FPMP_FP128_ENABLE=1 but no 128-bit float type available"
    #endif
#endif

/*
// Internal macro definitions
*/

/*
// Custom ABI for builtins in static library
*/
#if ((defined __CUDA_LIBDEVICE__) || (defined __FPMP_BUILD_LIB__) || (defined __FPMP_USE_LIB__)) && \
     (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13))
  #ifndef __FPMP_ABI_PRESERVE_N_DATA__
    #define __FPMP_ABI_PRESERVE_N_DATA__    -1
  #endif
  #ifndef __FPMP_ABI_PRESERVE_N_CONTROL__
    #define __FPMP_ABI_PRESERVE_N_CONTROL__ -1
  #endif
  #if (__FPMP_ABI_PRESERVE_N_DATA__ != -1) && (__FPMP_ABI_PRESERVE_N_CONTROL__ != -1)
    #define __FPMP_ABI_STR1__(x) #x
    #define __FPMP_ABI_STR__(x) __FPMP_ABI_STR1__(x)
    #define __FPMP_ABI_PRAGMA_TEXT__ nv_abi preserve_n_data(__FPMP_ABI_PRESERVE_N_DATA__) preserve_n_control(__FPMP_ABI_PRESERVE_N_CONTROL__)
    #define __FPMP_ABI__ _Pragma(__FPMP_ABI_STR__(__FPMP_ABI_PRAGMA_TEXT__))
  #else
    #define __FPMP_ABI__
  #endif
#else
  #define __FPMP_ABI__
#endif

/*
// Internal defines for API, builtin, and internal declarations
// For CUDA, use __host__ __device__ and static __forceinline__
// For host, use __forceinline__ with compiler-specific attributes
*/
#if defined __CUDACC__

    #define __FPMP_API_DECL__                  __forceinline__ __host__ __device__ 
    #define __FPMP_API_DEVICE_DECL__           __forceinline__ __device__ 
    #define __FPMP_INTERNAL_DECL__             static __forceinline__ __host__ __device__
    #define __FPMP_INTERNAL_DEVICE_DECL__      static __forceinline__  __device__


    #if (defined __FPMP_BUILD_LIB__) || (defined __FPMP_USE_LIB__)
      #define __FPMP_BUILTIN_DECL__            __FPMP_ABI__ extern "C"  __host__ __device__
      #define __FPMP_BUILTIN_DEVICE_DECL__     __FPMP_ABI__ extern "C"  __device__
    #else
      #define __FPMP_BUILTIN_DECL__            static __forceinline__ __host__ __device__
      #define __FPMP_BUILTIN_DEVICE_DECL__     static __forceinline__ __device__
    #endif

#else // !defined __CUDACC__

    // Define __forceinline__ for non-CUDA compilers
    #ifndef __forceinline__
        #if defined(__GNUC__) || defined(__clang__)
            #define __forceinline__            inline __attribute__((always_inline))
        #elif defined(_MSC_VER)
            #define __forceinline__            __forceinline
        #else
            #define __forceinline__            inline
        #endif
    #endif

    #define __FPMP_API_DECL__                  __forceinline__
    #define __FPMP_API_DEVICE_DECL__           __forceinline__
    #define __FPMP_INTERNAL_DECL__             static __forceinline__
    #define __FPMP_INTERNAL_DEVICE_DECL__      static __forceinline__


    #if (defined __FPMP_BUILD_LIB__) || (defined __FPMP_USE_LIB__)
      #define __FPMP_BUILTIN_DECL__            extern "C"
      #define __FPMP_BUILTIN_DEVICE_DECL__     extern "C"
    #else
      #define __FPMP_BUILTIN_DECL__            static __forceinline__
      #define __FPMP_BUILTIN_DEVICE_DECL__     static __forceinline__
    #endif

#endif // !defined __CUDACC__

/*
// Optional function qualifiers for portable API annotation.
// __FPMP_CONSTEXPR__ can be used only on functions whose bodies are valid
// constant-evaluation code across all supported toolchains.
*/
#ifndef __FPMP_CONSTEXPR__
    #define __FPMP_CONSTEXPR__ constexpr
#endif

#ifndef __FPMP_NOEXCEPT__
    #define __FPMP_NOEXCEPT__ noexcept
#endif

/*
// fp32mp2 large-argument trig: fallback to system fp64 sin/cos (1) or use dedicated Payne-Hanek reduction (0)
*/
#ifndef __FPMP_LARGE_TRIG_FP64_FALLBACK__
    #define __FPMP_LARGE_TRIG_FP64_FALLBACK__ 0
#endif

/*
// Internal explicit cast macro
// When FPMP_EXPLICIT_CASTS is 1, the explicit cast is used
// When FPMP_EXPLICIT_CASTS is 0, the explicit cast is not used
// The default is to not use explicit cast
*/
#if  FPMP_EXPLICIT_CASTS == 1
    #define __FPMP_EXPLICIT__ explicit
#else
    #define __FPMP_EXPLICIT__ 
#endif

/*
// C++20 is_constant_evaluated() compatibility.
// NVCC, GCC, and Clang provide __builtin_is_constant_evaluated() which works
// in __host__ __device__ context without warnings.  std::is_constant_evaluated()
// is a __host__-only constexpr function under NVCC, triggering warning #20015-D.
// Fall back to std:: only for compilers that lack the built-in (e.g., MSVC).
*/
#if defined(__CUDACC__) || defined(__GNUC__) || defined(__clang__)
    #define __FPMP_IS_CONSTEVAL__() __builtin_is_constant_evaluated()
#else
    #define __FPMP_IS_CONSTEVAL__() std::is_constant_evaluated()
#endif

/*
// Internal bit cast utility
// This utility is used to bit cast a value from one type to another
// Provides C++20 bit_cast functionality when it's absent
*/
#undef __FPMP_HAS_BIT_CAST__
#ifdef __has_builtin
    # define __FPMP_HAS_BIT_CAST__ __has_builtin(__builtin_bit_cast)
#else
    # define __FPMP_HAS_BIT_CAST__ 0
#endif

/*
// by default route fpmp's internal bit-casts through CCCL's
// cuda::std::bit_cast. __FPMP_BIT_CAST__ is the single switch point -- define it
// before including the fpmp headers for a fast re-map back to the in-house
// polyfill, e.g.:
//   #define __FPMP_BIT_CAST__(To, v) \
//       ::cuda::experimental::fpmp::__fpmp_builtin_bit_cast<To>(v)
*/
#ifndef __FPMP_BIT_CAST__
    #define __FPMP_BIT_CAST__(To, v) ::cuda::std::bit_cast<To>(v)
#endif

/*
// Internal macro for inline assembly support 
// for reciprocal and reciprocal square root operations when available (CUDA)
// When __FPMP_USE_INLINE_ASM_RSQRT__ is 1, the inline assembly is used
// When __FPMP_USE_INLINE_ASM_RSQRT__ is 0, the inline assembly is not used
// When __FPMP_USE_INLINE_ASM_RCP__ is 1, the inline assembly is used
// When __FPMP_USE_INLINE_ASM_RCP__ is 0, the inline assembly is not used
// The default is to use inline assembly
// This is the fastest option, but may cause accuracy loss in subtle domains 
// close to denormals or large numbers.
*/
#ifndef __FPMP_USE_INLINE_ASM_RSQRT__
    #define __FPMP_USE_INLINE_ASM_RSQRT__ 1
#endif
#ifndef __FPMP_USE_INLINE_ASM_RCP__
    #define __FPMP_USE_INLINE_ASM_RCP__   1
#endif
/*
// __FPMP_USE_INLINE_ASM_EX2_LG2__ controls the implementation of the single-precision
// fast exp2 / log2 helpers used by the fp32mp2 transcendental kernels (cbrt, ...).
// When 1 (default on CUDA): emit ex2.approx.ftz.f32 / lg2.approx.ftz.f32 inline asm.
// When 0: fall back to the __exp2f / __log2f device intrinsics.
*/
#ifndef __FPMP_USE_INLINE_ASM_EX2_LG2__
    #define __FPMP_USE_INLINE_ASM_EX2_LG2__ 1
#endif
// Internal: map user-facing macros to internal names
#ifndef __FPMP_USE_OPT_FROM_DOUBLE__
    #define __FPMP_USE_OPT_FROM_DOUBLE__ FPMP_OPTIMIZED_DOUBLE_TO_FPMP
#endif
#ifndef __FPMP_USE_OPT_TO_DOUBLE__
    #define __FPMP_USE_OPT_TO_DOUBLE__   FPMP_OPTIMIZED_FPMP_TO_DOUBLE
#endif

/*
// Internal macro for accurate multiplication & division support
// When __FPMP_USE_ACCURATE_MUL__ is 1, the accurate multiplication is used
// When __FPMP_USE_ACCURATE_MUL__ is 0, the accurate multiplication is not used
// When __FPMP_USE_ACCURATE_DIV__ is 1, the accurate division is used
// When __FPMP_USE_ACCURATE_DIV__ is 0, the accurate division is not used
// The default is to not use accurate multiplication & division.
// These implementations scale values to improve accuray on denormals
// but may cause about 1.5x slowdown.
*/  
#ifndef __FPMP_USE_ACCURATE_MUL__
    #define __FPMP_USE_ACCURATE_MUL__ 0
#endif
#ifndef __FPMP_USE_ACCURATE_DIV__
    #define __FPMP_USE_ACCURATE_DIV__ 0
#endif

/*********************************************************************
 * Internal utilities
 *********************************************************************/
/*
// Accuracy level for fpmp arithmetic (public; defined directly in
// cuda::experimental, no internal fpmp:: namespace). Named fpmp2_accuracy, so
// callers write e.g. fpmp2_t<float, fpmp2_accuracy::high>.
// mid is the Dekker-based split and error accumulation technique
// high is the Thall-based split and error accumulation technique
// low is the fast arithmetic operation without re-normalizations
// def is the default selector; equals mid.
*/
enum struct fpmp2_accuracy
{
    unset = -1,
    low   =  1,
    mid   =  2,
    high  =  3,
    def   =  2,
};

 namespace fpmp
 {
    /*
    // In-house bit cast polyfill, kept available as the __FPMP_BIT_CAST__
    // fallback target. Provides C++20 bit_cast functionality when it's absent.
    */
    template<typename _To, typename _From>
    __FPMP_INTERNAL_DECL__ _To __fpmp_builtin_bit_cast(_From __v)
    {
        // Static checks to ensure bit_cast requirements
        static_assert(sizeof(_To) == sizeof(_From),              "bit_cast requires source and destination types to have the same size");
        static_assert(std::is_trivially_copyable<_From>::value, "bit_cast requires From to be trivially copyable");
        static_assert(std::is_trivially_copyable<_To>::value,   "bit_cast requires To to be trivially copyable");
        
    #if __FPMP_HAS_BIT_CAST__
        // Prefer compiler builtin if available (C++20 or compiler extension)
        return __builtin_bit_cast(_To, __v);
    #else
        // Fallback using reinterpret_cast for performance
        // Note: This technically violates strict aliasing rules but is widely supported
        // and works correctly in practice on all target platforms
        return *reinterpret_cast<To*>(&__v);
    #endif
    } // __fpmp_builtin_bit_cast

    /*
    // Internal bit cast utility used throughout the library. Delegates to the
    // __FPMP_BIT_CAST__ switch macro (cuda::std::bit_cast by default).
    */
    template<typename _To, typename _From>
    __FPMP_INTERNAL_DECL__ _To internal_bit_cast(_From __v)
    {
        return __FPMP_BIT_CAST__(_To, __v);
    } // internal_bit_cast

    /*
    // Internal basic arith operations 
    // dispatched to the appropriate built-in for host and device
    // if not available, use the appropriate fallback
    // the fallback is the appropriate arithmetic operation
    */
    #ifdef __CUDA_ARCH__    
        __FPMP_INTERNAL_DECL__ float    internal_fabs(float __x)   {return fabsf(__x);}
        __FPMP_INTERNAL_DECL__ bool     internal_isnan(float __x)  {return ::isnan(__x);}
        __FPMP_INTERNAL_DECL__ float    add_rn(float __x, float __y) {return __fadd_rn(__x, __y);}
        __FPMP_INTERNAL_DECL__ float    add_rz(float __x, float __y) {return __fadd_rz(__x, __y);}
        __FPMP_INTERNAL_DECL__ float    sub_rn(float __x, float __y) {return __fsub_rn(__x, __y);}
        __FPMP_INTERNAL_DECL__ float    mul_rn(float __x, float __y) {return __fmul_rn(__x, __y);}
        __FPMP_INTERNAL_DECL__ float    fma_rn(float __x, float __y, float __z) {return __fmaf_ieee_rn(__x, __y, __z);}
        #if __FPMP_USE_INLINE_ASM_RCP__ == 1
        __FPMP_INTERNAL_DECL__ float    rcp_rn(float __x)  { float __r; asm ("rcp.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x)); return __r; }
        #else
        __FPMP_INTERNAL_DECL__ float    rcp_rn(float __x)  {return __frcp_rn(__x);}
        #endif
        #if __FPMP_USE_INLINE_ASM_RSQRT__ == 1
        __FPMP_INTERNAL_DECL__ float    rsqrt_rn(float __x) { float __r; asm ("rsqrt.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x)); return __r; }
        #else
        __FPMP_INTERNAL_DECL__ float    rsqrt_rn(float __x) {return __frsqrt_rn(__x);}
        #endif
        // Fast single-precision base-2 exp / log mapped to the FP32 SFU
        // approximation units (ex2.approx / lg2.approx). These are not
        // correctly rounded; they are used as initial estimates for
        // higher-precision Newton/Halley refinement.
        #if __FPMP_USE_INLINE_ASM_EX2_LG2__ == 1
        __FPMP_INTERNAL_DECL__ float    fast_exp2(float __x) { float __r; asm ("ex2.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x)); return __r; }
        __FPMP_INTERNAL_DECL__ float    fast_log2(float __x) { float __r; asm ("lg2.approx.ftz.f32 %0,%1;" : "=f"(__r) : "f"(__x)); return __r; }
        #else
        __FPMP_INTERNAL_DECL__ float    fast_exp2(float __x) {return __exp2f(__x);}
        __FPMP_INTERNAL_DECL__ float    fast_log2(float __x) {return __log2f(__x);}
        #endif
        __FPMP_INTERNAL_DECL__ int32_t  fp2int_rz(float __x)  {return __float2int_rz(__x);}
        __FPMP_INTERNAL_DECL__ int32_t  fp2int_rn(float __x)  {return __float2int_rn(__x);}
        __FPMP_INTERNAL_DECL__ uint32_t fp2uint_rz(float __x) {return __float2uint_rz(__x);}
        __FPMP_INTERNAL_DECL__ int64_t  fp2ll_rz(float __x)   {return __float2ll_rz(__x);}
        __FPMP_INTERNAL_DECL__ uint64_t fp2ull_rz(float __x)  {return __float2ull_rz(__x);}

        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType int2fp_rn(int32_t __x)   {return static_cast<_FpType>(__int2float_rn(__x));}
        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType int2fp_rz(int32_t __x)   {return static_cast<_FpType>(__int2float_rz(__x));}
        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType uint2fp_rz(uint32_t __x) {return static_cast<_FpType>(__uint2float_rz(__x));}
        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType ll2fp_rz(int64_t __x)    {return static_cast<_FpType>(__ll2float_rz(__x));}
        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType ull2fp_rz(uint64_t __x)  {return static_cast<_FpType>(__ull2float_rz(__x));}
    #else // !__CUDA_ARCH__
        __FPMP_INTERNAL_DECL__ float    internal_fabs(float __x)   {return fabsf(__x);}
        __FPMP_INTERNAL_DECL__ bool     internal_isnan(float __x)  {return std::isnan(__x);}
        __FPMP_INTERNAL_DECL__ float    add_rn(float __x, float __y) {return __x + __y;}
        __FPMP_INTERNAL_DECL__ float    add_rz(float __x, float __y)             
        {
            float __sum = __x + __y;
            if (__sum == 0.0f) return __sum; 
            float __error = fmaf(-1.0f, __sum, __x) + __y;
            if (__error == 0.0f) return __sum;          
            if ((__sum > 0.0f && __error < 0.0f) || 
                (__sum < 0.0f && __error > 0.0f)) 
            {
                // Rounded away from zero - need to adjust mantissa toward zero
                uint32_t __bits = internal_bit_cast<uint32_t>(__sum);
                // Decrement mantissa (moves toward zero for both positive and negative)
                __bits--;
                __sum = internal_bit_cast<float>(__bits);
            }
            return __sum;
        }
        __FPMP_INTERNAL_DECL__ float    sub_rn(float __x, float __y) {return __x - __y;}
        __FPMP_INTERNAL_DECL__ float    mul_rn(float __x, float __y) {return __x * __y;}
        __FPMP_INTERNAL_DECL__ float    fma_rn(float __x, float __y, float __z) {return fmaf(__x, __y, __z);}
        __FPMP_INTERNAL_DECL__ float    rcp_rn(float __x)   {return 1.0f / __x;}
        __FPMP_INTERNAL_DECL__ float    rsqrt_rn(float __x) {return 1.0f / sqrtf(__x);}
        // Host fallback for the fast SFU-style exp2 / log2; uses the libm
        // single-precision routines.  Same use case as the device path:
        // a low-cost initial estimate for Newton/Halley refinement.
        __FPMP_INTERNAL_DECL__ float    fast_exp2(float __x) {return ::exp2f(__x);}
        __FPMP_INTERNAL_DECL__ float    fast_log2(float __x) {return ::log2f(__x);}
        __FPMP_INTERNAL_DECL__ int32_t  fp2int_rz(float __x)  {return static_cast<int32_t>(__x);}
        __FPMP_INTERNAL_DECL__ int32_t  fp2int_rn(float __x)  {return static_cast<int32_t>(roundf(__x));}
        __FPMP_INTERNAL_DECL__ uint32_t fp2uint_rz(float __x) {return static_cast<uint32_t>(__x);}
        __FPMP_INTERNAL_DECL__ int64_t  fp2ll_rz(float __x)   {return static_cast<int64_t>(__x);}
        __FPMP_INTERNAL_DECL__ uint64_t fp2ull_rz(float __x)  {return static_cast<uint64_t>(__x);}

        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType int2fp_rn(int32_t __x) 
        {
            return static_cast<_FpType>(roundf(__x));
        }
        
        /*
        // Round-toward-zero (truncation) versions for integer constructors
        // For CPU: implement round-to-zero by checking if round-to-nearest went away from zero,
        // then use nextafter to get the next representable float toward zero
        // Using double for exact comparison (double has 53 bits, enough for int32_t's 32 bits)
        // Template versions for both float and double
        */
        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType int2fp_rz(int32_t __x) 
        {
            _FpType __f = static_cast<_FpType>(__x);
            double __exact = static_cast<double>(__x);
            if ((__x > 0 && __f > __exact) || (__x < 0 && __f < __exact)) { 
                __f = std::is_same<_FpType, float>::value ? nextafterf(__f, 0.0f) : nextafter(__f, 0.0);
            }
            return __f;
        }
        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType uint2fp_rz(uint32_t __x) 
        {
            _FpType __f = static_cast<_FpType>(__x);
            double __exact = static_cast<double>(__x);
            if (__f > __exact) { 
                __f = std::is_same<_FpType, float>::value ? nextafterf(__f, 0.0f) : nextafter(__f, 0.0);
            }
            return __f;
        }
        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType ll2fp_rz(int64_t __x) 
        {
            _FpType __f = static_cast<_FpType>(__x);
            double __exact = static_cast<double>(__x);
            if ((__x > 0 && __f > __exact) || (__x < 0 && __f < __exact)) { 
                __f = std::is_same<_FpType, float>::value ? nextafterf(__f, 0.0f) : nextafter(__f, 0.0);
            }
            return __f;
        }
        template<typename _FpType = float> __FPMP_INTERNAL_DECL__ _FpType ull2fp_rz(uint64_t __x) 
        {
            _FpType __f = static_cast<_FpType>(__x);
            double __exact = static_cast<double>(__x);
            if (__f > __exact) { 
                __f = std::is_same<_FpType, float>::value ? nextafterf(__f, 0.0f) : nextafter(__f, 0.0);
            }
            return __f;
        }
    #endif // __CUDA_ARCH__

#if FPMP_FP64MP2_ENABLE == 1
    #ifdef __CUDA_ARCH__
        __FPMP_INTERNAL_DECL__ double   internal_fabs(double __x)    {return ::fabs(__x);}
        __FPMP_INTERNAL_DECL__ bool     internal_isnan(double __x)   {return ::isnan(__x);}
        __FPMP_INTERNAL_DECL__ double   add_rn(double __x, double __y) {return __dadd_rn(__x, __y);}
        __FPMP_INTERNAL_DECL__ double   add_rz(double __x, double __y) {return __dadd_rz(__x, __y);}
        __FPMP_INTERNAL_DECL__ double   sub_rn(double __x, double __y) {return __dsub_rn(__x, __y);}
        __FPMP_INTERNAL_DECL__ double   mul_rn(double __x, double __y) {return __dmul_rn(__x, __y);}
        __FPMP_INTERNAL_DECL__ double   fma_rn(double __x, double __y, double __z) {return __fma_rn(__x, __y, __z);}
        __FPMP_INTERNAL_DECL__ double   rcp_rn(double __x)     {return __drcp_rn(__x);}
        __FPMP_INTERNAL_DECL__ double   rsqrt_rn(double __x)   {return rsqrt(__x);}
        __FPMP_INTERNAL_DECL__ int32_t  fp2int_rz(double __x)  {return __double2int_rz(__x);}
        __FPMP_INTERNAL_DECL__ int32_t  fp2int_rn(double __x)  {return __double2int_rn(__x);}
        __FPMP_INTERNAL_DECL__ uint32_t fp2uint_rz(double __x) {return __double2uint_rz(__x);}
        __FPMP_INTERNAL_DECL__ int64_t  fp2ll_rz(double __x)   {return __double2ll_rz(__x);}
        __FPMP_INTERNAL_DECL__ uint64_t fp2ull_rz(double __x)  {return __double2ull_rz(__x);}
        // int32_t and uint32_t always fit exactly in double (52-bit mantissa vs 32-bit values)
        template<> __FPMP_API_DECL__ double int2fp_rn<double>(int32_t __x)   {return __int2double_rn(__x);}
        template<> __FPMP_API_DECL__ double int2fp_rz<double>(int32_t __x)   {return static_cast<double>(__x);}
        template<> __FPMP_API_DECL__ double uint2fp_rz<double>(uint32_t __x) {return static_cast<double>(__x);}
        // int64_t and uint64_t: use CUDA intrinsics for round-toward-zero
        template<> __FPMP_API_DECL__ double ll2fp_rz<double>(int64_t __x)    {return __ll2double_rz(__x);}
        template<> __FPMP_API_DECL__ double ull2fp_rz<double>(uint64_t __x)  {return __ull2double_rz(__x);}
    #else // !__CUDA_ARCH__
        __FPMP_INTERNAL_DECL__ double   internal_fabs(double __x)    {return ::fabs(__x);}
        __FPMP_INTERNAL_DECL__ bool     internal_isnan(double __x)   {return std::isnan(__x);}
        __FPMP_INTERNAL_DECL__ double   add_rn(double __x, double __y) {return __x + __y;}
        __FPMP_INTERNAL_DECL__ double   add_rz(double __x, double __y)             
        {
            double __sum = __x + __y;
            if (__sum == 0.0) return __sum; 
            double __error = fma(-1.0, __sum, __x) + __y;
            if (__error == 0.0) return __sum;          
            if ((__sum > 0.0 && __error < 0.0) || 
                (__sum < 0.0 && __error > 0.0)) 
            {
                // Rounded away from zero - need to adjust mantissa toward zero
                uint64_t __bits = internal_bit_cast<uint64_t>(__sum);
                // Decrement mantissa (moves toward zero for both positive and negative)
                __bits--;
                __sum = internal_bit_cast<double>(__bits);
            }
            return __sum;
        }
        __FPMP_INTERNAL_DECL__ double   sub_rn(double __x, double __y) {return __x - __y;}
        __FPMP_INTERNAL_DECL__ double   mul_rn(double __x, double __y) {return __x * __y;}
        __FPMP_INTERNAL_DECL__ double   fma_rn(double __x, double __y, double __z) {return fma(__x, __y, __z);}
        __FPMP_INTERNAL_DECL__ double   rcp_rn(double __x)     {return 1.0 / __x;}
        __FPMP_INTERNAL_DECL__ double   rsqrt_rn(double __x)   {return 1.0 / sqrt(__x);}
        __FPMP_INTERNAL_DECL__ int32_t  fp2int_rz(double __x)  {return static_cast<int32_t>(__x);}
        __FPMP_INTERNAL_DECL__ int32_t  fp2int_rn(double __x)  {return static_cast<int32_t>(round(__x));}
        __FPMP_INTERNAL_DECL__ uint32_t fp2uint_rz(double __x) {return static_cast<uint32_t>(__x);}
        __FPMP_INTERNAL_DECL__ int64_t  fp2ll_rz(double __x)   {return static_cast<int64_t>(__x);}
        __FPMP_INTERNAL_DECL__ uint64_t fp2ull_rz(double __x)  {return static_cast<uint64_t>(__x);}
        /*
        // Round-toward-zero (truncation) versions for integer-to-double constructors
        // For double, we need to use long double for exact comparison where possible
        // Template specializations for double type
        */
        template<> __FPMP_API_DECL__ double int2fp_rn<double>(int32_t __x)   { return round(__x); }
        template<> __FPMP_API_DECL__ double int2fp_rz<double>(int32_t __x)   { return static_cast<double>(__x); }
        template<> __FPMP_API_DECL__ double uint2fp_rz<double>(uint32_t __x) { return static_cast<double>(__x); }
        template<> __FPMP_API_DECL__ double ll2fp_rz<double>(int64_t __x) 
        {
            // int64_t may not fit exactly in double
            double __d = static_cast<double>(__x);
            long double __exact = static_cast<long double>(__x);
            if ((__x > 0 && __d > __exact) || (__x < 0 && __d < __exact)) {  __d = nextafter(__d, 0.0); }
            return __d;
        }
        template<> __FPMP_API_DECL__ double ull2fp_rz<double>(uint64_t __x) 
        {
            // uint64_t may not fit exactly in double
            double __d = static_cast<double>(__x);
            long double __exact = static_cast<long double>(__x);
            if (__d > __exact) { __d = nextafter(__d, 0.0); }
            return __d;
        }
    #endif // __CUDA_ARCH__
#endif // FPMP_FP64MP2_ENABLE == 1

    /*
    // Scalar rounding helpers (host + device)
    // Intentionally named under fpmp:: namespace and used by fpmp_math.h
    // dedicated fp32mp2 rounding implementations.
    */
    template<typename _FpType = float>
    __FPMP_INTERNAL_DECL__ _FpType internal_trunc (const _FpType __x)
    {
        if constexpr (std::is_same<_FpType, float>::value)
        {
            const _FpType __abs_x = fpmp::internal_fabs(__x);
            if (__abs_x >= _FpType(0x1.0p23f)) { return __x; }
        #if defined(__CUDA_ARCH__)
            const int32_t __xi = __float2int_rz(__x);
            return __int2float_rz(__xi);
        #else
            const int32_t __xi = fpmp::fp2int_rz(__x);
            return fpmp::int2fp_rz<_FpType>(__xi);
        #endif
        }
        else
        {
            return static_cast<_FpType>(::trunc(static_cast<double>(__x)));
        }
    }

    template<typename _FpType = float>
    __FPMP_INTERNAL_DECL__ _FpType internal_floor (const _FpType __x)
    {
        if constexpr (std::is_same<_FpType, float>::value)
        {
            const _FpType __abs_x = fpmp::internal_fabs(__x);
            if (__abs_x >= _FpType(0x1.0p23f)) { return __x; }
        #if defined(__CUDA_ARCH__)
            const int32_t __xi = __float2int_rd(__x);
            return __int2float_rn(__xi);
        #else
            return floorf(__x);
        #endif
        }
        else
        {
            return static_cast<_FpType>(::floor(static_cast<double>(__x)));
        }
    }

    template<typename _FpType = float>
    __FPMP_INTERNAL_DECL__ _FpType internal_ceil (const _FpType __x)
    {
        if constexpr (std::is_same<_FpType, float>::value)
        {
            const _FpType __abs_x = fpmp::internal_fabs(__x);
            if (__abs_x >= _FpType(0x1.0p23f)) { return __x; }
        #if defined(__CUDA_ARCH__)
            const int32_t __xi = __float2int_ru(__x);
            return __int2float_rn(__xi);
        #else
            return ceilf(__x);
        #endif
        }
        else
        {
            return static_cast<_FpType>(::ceil(static_cast<double>(__x)));
        }
    }

    /*
    // Internal operations for 2-precision arithmetic
    */
    // Multiply 2 floats exactly, assuming no over/underflow.
    template<typename _FpType = float>
    __FPMP_INTERNAL_DECL__ _FpType two_mult_fma (const _FpType __x, 
                                                const _FpType __y,
                                                _FpType* const __res_lo)
    {
        _FpType __res_hi = mul_rn(__x, __y);
        *__res_lo       = fma_rn(__x, __y, -__res_hi);
        return __res_hi;
    }

    // Add 2 floats, returning the answer exactly in 'hi' and 'lo' parts.
    // Assumes the exponent of 'x' is >= exponent of 'y'.
    // (Usually we just check if |x| >= |y|).
    // If this is not known use the function below.
    template<typename _FpType = float>
    __FPMP_INTERNAL_DECL__ _FpType fast_two_sum (const _FpType __x, 
                                                const _FpType __y, 
                                                _FpType* const __res_lo)
    {
        _FpType __res_hi = add_rn(__x, __y);
        _FpType __diff   = sub_rn(__res_hi, __x);
        *__res_lo       = sub_rn(__y, __diff);
        return __res_hi;
    }

    // Add 2 floats, returning the answer exactly in 'hi' and 'lo' parts.
    // This makes no assumptions on the magnitudes of |x| and |y|.
    template<typename _FpType = float>
    __FPMP_INTERNAL_DECL__ _FpType two_sum (const _FpType __x, 
                                           const _FpType __y,
                                           _FpType* const __res_lo)
    {
        _FpType __res_hi  = add_rn(__x, __y);
        _FpType __a_prime = sub_rn(__res_hi, __y);
        _FpType __b_prime = sub_rn(__res_hi, __a_prime);
        _FpType __delta_a = sub_rn(__x, __a_prime);
        _FpType __delta_b = sub_rn(__y, __b_prime);
        *__res_lo        = add_rn(__delta_a, __delta_b);
        return __res_hi;
    }

    // double -> (hi, lo) conversions (plain versions)
    // only for the C++ class below to be optimized in compile-time
    __FPMP_CONSTEXPR__ __FPMP_API_DECL__ void from_double (const double __x, 
                                                           float*       __res_hi, 
                                                           float*       __res_lo) __FPMP_NOEXCEPT__
    {
        *__res_hi = (float)__x;
        *__res_lo = (float)(__x - (double)(float)__x);
    }

 } // namespace fpmp

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_COMMON_H