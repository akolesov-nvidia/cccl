//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPEMU_COMMON_H
#define _CUDA___FP_FPEMU_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/**
 * @file fpemu_common.h
 * @brief Common definitions, macros, and enumerations for the FPEMU library
 *
 * This header provides the core definitions, macros, and enumerations used throughout 
 * the FPEMU library. It defines:
 *
 * - Macros for controlling inline vs library compilation
 * - Host/device decorators for CUDA compilation
 * - Function declaration macros for different contexts
 * - Rounding modes (nearest, zero, up, down)
 * - Accuracy levels (high, mid, low)
 *
 * The macros handle:
 * - Inline vs library compilation modes
 * - Host vs device code compilation for CUDA
 * - Function visibility and linkage
 *
 * This provides a unified way to handle compilation across different platforms
 * and compilation modes while maintaining consistent behavior.
 */

#if !defined(__CUDA_LIBDEVICE__)
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#endif

/*********************************************************************
 * Compilation mode macros
 *********************************************************************/

// CCCL_FPEMU_LIB: Compilation mode control.
//   1 = link against precompiled library (maps to _CCCL_FPEMU_USE_LIB)
//   0 = header-only inline mode (default)
// CCCL_FPEMU_INLINE is the inverse alias: CCCL_FPEMU_INLINE=1 is equivalent to CCCL_FPEMU_LIB=0.
#ifndef CCCL_FPEMU_LIB
    #ifdef CCCL_FPEMU_INLINE
        #if CCCL_FPEMU_INLINE == 1
            #define CCCL_FPEMU_LIB 0
        #else
            #define CCCL_FPEMU_LIB 1
        #endif
    #else
        #define CCCL_FPEMU_LIB 0
    #endif
#endif
#ifndef CCCL_FPEMU_INLINE
    #if CCCL_FPEMU_LIB == 1
        #define CCCL_FPEMU_INLINE 0
    #else
        #define CCCL_FPEMU_INLINE 1
    #endif
#endif
#if CCCL_FPEMU_LIB == 1 && !defined(_CCCL_FPEMU_USE_LIB)
    #define _CCCL_FPEMU_USE_LIB
#endif

// If neither inline nor lib is defined internally, default to inline
#if !defined _CCCL_FPEMU_INLINE && !defined _CCCL_FPEMU_BUILD_LIB && !defined _CCCL_FPEMU_USE_LIB
    #define _CCCL_FPEMU_INLINE
#endif

// If neither host nor device is defined, default to device
#if defined __CUDACC__
    #undef  _CCCL_FPEMU_HOST
    #undef  _CCCL_FPEMU_DEVICE
    #define _CCCL_FPEMU_DEVICE
#else
    #undef  _CCCL_FPEMU_DEVICE
    #undef  _CCCL_FPEMU_HOST
    #define _CCCL_FPEMU_HOST
#endif

/*
// Custom ABI for builtins in static library
*/
#if ((defined __CUDA_LIBDEVICE__) || (defined _CCCL_FPEMU_BUILD_LIB) || (defined _CCCL_FPEMU_USE_LIB)) && \
     (defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13))
  #ifndef _CCCL_FPEMU_ABI_PRESERVE_N_DATA
    #define _CCCL_FPEMU_ABI_PRESERVE_N_DATA    -1
  #endif
  #ifndef _CCCL_FPEMU_ABI_PRESERVE_N_CONTROL
    #define _CCCL_FPEMU_ABI_PRESERVE_N_CONTROL -1
  #endif
  #if (_CCCL_FPEMU_ABI_PRESERVE_N_DATA != -1) && (_CCCL_FPEMU_ABI_PRESERVE_N_CONTROL != -1)
    #define _CCCL_FPEMU_ABI_STR1(x) #x
    #define _CCCL_FPEMU_ABI_STR(x) _CCCL_FPEMU_ABI_STR1(x)
    #define _CCCL_FPEMU_ABI_PRAGMA_TEXT nv_abi preserve_n_data(_CCCL_FPEMU_ABI_PRESERVE_N_DATA) preserve_n_control(_CCCL_FPEMU_ABI_PRESERVE_N_CONTROL)
    #define _CCCL_FPEMU_ABI _Pragma(_CCCL_FPEMU_ABI_STR(_CCCL_FPEMU_ABI_PRAGMA_TEXT))
  #else
    #define _CCCL_FPEMU_ABI
  #endif
#else
  #define _CCCL_FPEMU_ABI
#endif

/*********************************************************************
 * Declaration macros
 *********************************************************************/

// Header (inline) builds decorate functions at the call site with CCCL
// visibility macros directly:
//   _CCCL_API         — public entry points (host/device, hidden from ABI)
//   _CCCL_TRIVIAL_API — force-inlined internal/impl helpers (hot paths)
//   _CCCL_DEVICE_API  — device-only overloads
// The only decorator that still needs a dedicated macro is the extern-"C"
// ABI symbol used when building or linking the standalone libcufp library.
#if defined __CUDA_LIBDEVICE__
    #define _CCCL_FPEMU_BUILTIN_DECL   _CCCL_FPEMU_ABI extern "C" _CCCL_HOST_DEVICE
#elif defined _CCCL_FPEMU_INLINE
    #define _CCCL_FPEMU_BUILTIN_DECL   _CCCL_TRIVIAL_API
#else // _CCCL_FPEMU_BUILD_LIB or _CCCL_FPEMU_USE_LIB
    #define _CCCL_FPEMU_BUILTIN_DECL   _CCCL_FPEMU_ABI extern "C" _CCCL_HOST_DEVICE
#endif

/*********************************************************************
 * Default configuration values
 *********************************************************************/

// Define the default values for the enums
#ifndef _CCCL_FPEMU_DEFAULT_ROUNDING
    #define _CCCL_FPEMU_DEFAULT_ROUNDING rn
#endif

// Unpacked API presence. Default ON so the packed (fpbits64_t) legacy API and
// the unpacked (fpbits64_unpacked_t) API co-exist in the same package: this
// gates the *_unpacked cores, the universal pack/unpack, and the *_unpacked
// builtins/operators. Set to 0 only to strip the unpacked API entirely.

// Route the PACKED API through the unpack -> *_unpacked core -> pack pipeline
// instead of the legacy fused kernels. Set by the Makefile's PACKED_VIA_UNPACKED=y.
// This is a TESTING knob (it lets the packed test harness exercise the unpacked
// cores); default OFF keeps the packed API on its legacy implementations.
#ifndef _CCCL_FPEMU_PACKED_VIA_UNPACKED
    #define _CCCL_FPEMU_PACKED_VIA_UNPACKED 0
#endif

/*********************************************************************
 * Enumerations
 *********************************************************************/

/**
* @brief Bit representation of a double-precision floating point number
*
* This struct provides a simple wrapper around a 64-bit integer value that represents
* the IEEE-754 binary encoding of a double-precision floating point number.
* The value field contains the raw bits including sign, exponent and mantissa.
*/
typedef uint64_t fpbits64_t;

/**
* @brief Unpacked representation of a double-precision floating point number
* 
* This struct represents a double-precision floating point number in an unpacked format.
* It contains the sign, exponent, and mantissa components.
*/
typedef struct 
{
    uint32_t sign;
    uint32_t exponent;
    uint64_t mantissa;
} fpbits64_unpacked_t;

/**
* @brief Accuracy level for floating-point emulation (public).
*
* Defined directly in cuda::experimental (no internal fpemu:: namespace) and named
* fp64emu_accuracy, so callers write e.g. fp64emu_t<fp64emu_accuracy::high>.
* - high: Correctly rounded with full IEEE-754 range (infinities, NaNs, subnormals)
* - mid:  High accuracy (1-2 ULP) with normal range
* - low:  Low accuracy (up to half mantissa) with normal range
* - def:  Default selector; equals high so the default is IEEE-correct.
*/
enum struct fp64emu_accuracy
{
    unset = -1,
    low   =  1,
    mid   =  2,
    high  =  3,
    def   =  3,
};

namespace fpemu
{
    /**
    * @brief Rounding modes for floating point operations
    *
    * Enumeration of supported rounding modes:
    * - rn: Round to nearest (ties to even) - default IEEE-754 rounding
    * - rz: Round toward zero (truncation)
    * - ru: Round toward positive infinity 
    * - rd: Round toward negative infinity
    */
    enum struct rounding
    {
        unset = -1,
        rn    =  0,
        rz    =  1,
        ru    =  2,
        rd    =  3,
        def   = _CCCL_FPEMU_DEFAULT_ROUNDING
    };

} // namespace fpemu

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPEMU_COMMON_H
