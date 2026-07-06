//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FP64_TOOL_H
#define _CUDA___FP_FP64_TOOL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/is_integer.h>
// Pulling in cuda::std::bit_cast above surfaces the cuda::std namespace; make the
// type-trait set complete so this header's unqualified std:: traits keep
// resolving (to cuda::std equivalents).
#include <cuda/std/type_traits>

/**
 * @file fp64_tool.h
 * @brief FP64 Precision Emulation Tool - A drop-in double replacement with configurable precision
 *
 * This header-only library provides a `fp64_tool_t` class that wraps native double-precision
 * floating-point operations while allowing compile-time precision reduction. It's designed
 * for:
 *
 *   - **Algorithm sensitivity analysis**: Test how algorithms behave with reduced precision
 *   - **Mixed-precision research**: Emulate lower-precision formats (float, bfloat16, etc.)
 *   - **CUDA/CPU compatibility**: Works identically on both host and device code
 *   - **Drop-in replacement**: Use `fp64_tool` where you would use `double`
 *
 * ## Quick Start
 *
 * ```cpp
 * #define FP64_TOOL_MANTISSA_BITS 23  // Emulate float precision (52 bits -> 23 bits)
 * #include "fp64_tool.h"
 *
 * fp64_tool a = 1.5, b = 2.5;
 * fp64_tool result = a + b;  // Precision callbacks applied automatically
 * double native = result;      // Convert back to native double
 * ```
 *
 * ## Configuration Macros (define BEFORE including this header)
 *
 * | Macro                        | Default   | Description                                    |
 * |------------------------------|-----------|------------------------------------------------|
 * | `FP64_TOOL_MANTISSA_BITS`    | 52        | Number of mantissa bits to preserve (1-52)     |
 * | `FP64_TOOL_EXPONENT_BITS`    | 11        | Number of exponent bits to preserve (1-11)     |
 * | `FP64_TOOL_IEEE_ROUNDING`    | default   | Use IEEE 754 round-to-nearest-even             |
 * | `FP64_TOOL_ROUND_TO_NEAREST` | -         | Simple round-to-nearest                        |
 * | `FP64_TOOL_TRUNCATION`       | -         | Simple truncation (floor toward zero)          |
 * | `FP64_TOOL_NO_UNDERFLOW`     | -         | Preserve underflowing values (no flush to zero)|
 * | `FP64_TOOL_NO_OVERFLOW`      | -         | Preserve overflowing values (no clamp to INF)  |
 * | `FP64_TOOL_DISABLE`          | undefined | Disable precision emulation                    |
 * | `FP64_TOOL_RUNTIME_SIZE`     | undefined | Enable runtime precision control (see below)   |
 *
 * ## Underflow/Overflow Control
 *
 * When exponent bits are reduced (e.g., from 11 to 8 for FP32 emulation), values outside
 * the new dynamic range will normally be clamped:
 *   - **Overflow**: Values too large for reduced exponent → Infinity (±INF)
 *   - **Underflow**: Values too small for reduced exponent → Zero (±0)
 *
 * Defining these macros changes this behavior:
 *   - `FP64_TOOL_NO_OVERFLOW`: Keep original FP64 value when it would overflow
 *   - `FP64_TOOL_NO_UNDERFLOW`: Keep original FP64 value when it would underflow
 *
 * This is useful for algorithms that need extended dynamic range while still
 * emulating reduced mantissa precision.
 *
 * ## Common Precision Configurations
 *
 * | Format   | Mantissa | Exponent | Configuration                                          |
 * |----------|----------|----------|--------------------------------------------------------|
 * | FP64     | 52       | 11       | Default (no callbacks applied)                         |
 * | FP32     | 23       | 8        | `FP64_TOOL_MANTISSA_BITS=23, FP64_TOOL_EXPONENT_BITS=8`|
 * | BF16     | 7        | 8        | `FP64_TOOL_MANTISSA_BITS=7, FP64_TOOL_EXPONENT_BITS=8` |
 * | FP16     | 10       | 5        | `FP64_TOOL_MANTISSA_BITS=10, FP64_TOOL_EXPONENT_BITS=5`|
 * | TF32     | 10       | 8        | `FP64_TOOL_MANTISSA_BITS=10, FP64_TOOL_EXPONENT_BITS=8`|
 *
 * ## How It Works
 *
 * Each arithmetic operation follows this pattern:
 * 1. Apply callback to input operands (reduce precision)
 * 2. Perform native FP64 operation
 * 3. Apply callback to result (reduce precision)
 *
 * This models how lower-precision hardware would handle the computation while
 * maintaining full FP64 representation for intermediate storage.
 *
 * ## Runtime Precision Control
 *
 * By default, mantissa and exponent sizes are compile-time constants. Defining
 * `FP64_TOOL_RUNTIME_SIZE` before including this header enables changing them
 * at runtime without recompilation. The `FP64_TOOL_MANTISSA_BITS` and
 * `FP64_TOOL_EXPONENT_BITS` macros then set the initial values only.
 *
 * ```cpp
 * #define FP64_TOOL_RUNTIME_SIZE
 * #define FP64_TOOL_MANTISSA_BITS 52   // initial mantissa (full precision)
 * #define FP64_TOOL_EXPONENT_BITS 11   // initial exponent (full range)
 * #include "fp64_tool.h"
 * ```
 *
 * ### Setter Functions
 *
 * | Function                                  | Target | Description                        |
 * |-------------------------------------------|--------|------------------------------------|
 * | `fp64_tool_set_host_mantissa_size(int)`    | CPU    | Set mantissa bits on host (1-52)   |
 * | `fp64_tool_set_host_exponent_size(int)`    | CPU    | Set exponent bits on host (1-11)   |
 * | `fp64_tool_set_device_mantissa_size(int)`  | GPU    | Set mantissa bits on device (1-52) |
 * | `fp64_tool_set_device_exponent_size(int)`  | GPU    | Set exponent bits on device (1-11) |
 *
 * Host and device sizes are independent — changing one does not affect the other.
 * The device setters use `cudaMemcpyToSymbol` when called from host code; a
 * `cudaDeviceSynchronize()` before the next kernel launch ensures the new value
 * is visible on the GPU. On the device side, only thread 0 of block 0 writes
 * the global variable to avoid race conditions.
 *
 * ### CPU Example
 *
 * ```cpp
 * // Full precision initially
 * fp64_tool a = 1.0, b = 1e-15;
 * double full = (double)(a + b);       // preserves small term
 *
 * // Switch to float-like precision at runtime
 * fp64_tool_set_host_mantissa_size(23);
 * fp64_tool c = 1.0, d = 1e-15;
 * double reduced = (double)(c + d);    // small term lost → 1.0
 * ```
 *
 * @note There is a small performance cost compared to compile-time mode because
 *       the bit counts are read from memory rather than compiled as constants.
 *
 * @note Thread Safety: All operations are thread-safe (no shared mutable state)
 *
 * @copyright NVIDIA Corporation
 * @license Apache 2.0
 */

#include <cuda/std/cstdint>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>

#if defined(__CUDACC__) && !defined(__CUDA_ARCH__)
    // Include CUDA runtime for host-side functions like cudaMemcpyToSymbol
    #include <cuda_runtime.h>
#endif

#include <cuda/std/__cccl/prologue.h>

//=============================================================================
// SECTION 1: Configuration and Feature Flags
//=============================================================================

/**
 * @brief Number of mantissa bits to preserve (1-52)
 * 
 * FP64 has 52 explicit mantissa bits. Setting this lower truncates/rounds
 * the mantissa to simulate reduced precision. Common values:
 *   - 52: Full FP64 precision (no reduction)
 *   - 23: FP32/float precision
 *   - 10: FP16/TF32 precision
 *   - 7:  BF16 precision
 */
#ifndef FP64_TOOL_MANTISSA_BITS
  #define FP64_TOOL_MANTISSA_BITS 52
#endif

/**
 * @brief Number of exponent bits to preserve (1-11)
 * 
 * FP64 has 11 exponent bits (bias 1023). Setting this lower reduces the
 * dynamic range, potentially causing overflow to infinity or underflow to zero.
 *   - 11: Full FP64 range (no reduction)
 *   - 8:  FP32/BF16/TF32 range
 *   - 5:  FP16 range
 */
#ifndef FP64_TOOL_EXPONENT_BITS
  #define FP64_TOOL_EXPONENT_BITS 11
#endif

#if defined(FP64_TOOL_RUNTIME_SIZE)
  #define __FP64_TOOL_RUNTIME_SIZE__ 1
#else
  #define __FP64_TOOL_RUNTIME_SIZE__ 0
#endif

/* Internal flags: auto-detect if reduction is needed */
#if __FP64_TOOL_RUNTIME_SIZE__ || FP64_TOOL_EXPONENT_BITS < 11
  #define __FP64_TOOL_REDUCE_EXPONENT__
  #ifndef __FP64_TOOL_ENABLE__
    #define __FP64_TOOL_ENABLE__
  #endif
#endif
#if __FP64_TOOL_RUNTIME_SIZE__ || FP64_TOOL_MANTISSA_BITS < 52
  #define __FP64_TOOL_REDUCE_MANTISSA__
  #ifndef __FP64_TOOL_ENABLE__
    #define __FP64_TOOL_ENABLE__
  #endif
#endif

// Master switch for precision emulation disabling
#if defined FP64_TOOL_DISABLE
  #undef __FP64_TOOL_ENABLE__
#endif

#if defined FP64_TOOL_RUNTIME_SIZE
    #define __FP64_TOOL_CONST_QUALIFIER const
#else
    #define __FP64_TOOL_CONST_QUALIFIER constexpr
#endif

//=============================================================================
// SECTION 2: Platform Abstraction (CUDA/Host Compatibility)
//=============================================================================

// Function decorators come from CCCL directly (see <cuda/std/__cccl/...>):
//   _CCCL_API inline   — public host/device entry points (hidden from ABI)
//   _CCCL_TRIVIAL_API  — force-inlined internal helpers (hot paths)
//   _CCCL_HOST_DEVICE / _CCCL_HOST — plain execution-space qualifiers, used on
//                        the static setters that must keep internal linkage.

//=============================================================================
// SECTION 3: Type Definitions and Utilities
//=============================================================================

/**
 * @brief 64-bit unsigned integer type for bit manipulation of doubles
 * 
 * This type is used to access the raw IEEE 754 bit representation of
 * double-precision floating-point values.
 */
namespace cuda::experimental
{

using fpbits64_t = uint64_t;

/**
 * @brief Tag type for raw bit construction
 * 
 * Used to disambiguate constructors that take raw bit patterns from those
 * that take numeric values. Example:
 *   fp64_tool_t(fpbits64_raw, 0x3FF0000000000000ULL)  // 1.0 from bits
 *   fp64_tool_t(1.0)                                   // 1.0 from value
 */
struct fpbits64_raw_t { explicit fpbits64_raw_t() = default; };

/** @brief Global instance of the raw bit construction tag */
inline constexpr fpbits64_raw_t fpbits64_raw{};

/**
 * @brief Check for __builtin_bit_cast availability (C++20 or compiler extension)
 */
#undef __FP64_TOOL_HAS_BUILTIN_BIT_CAST__
#ifdef __has_builtin
    #define __FP64_TOOL_HAS_BUILTIN_BIT_CAST__ __has_builtin(__builtin_bit_cast)
#else
    #define __FP64_TOOL_HAS_BUILTIN_BIT_CAST__ 0
#endif

/**
 * @brief Type-safe bit reinterpretation utility (internal)
 * 
 * Reinterprets the bit pattern of one type as another type of the same size.
 * Uses __builtin_bit_cast when available (optimal), falls back to memcpy.
 * 
 * @tparam To   Target type
 * @tparam From Source type
 * @param src   Value to reinterpret
 * @return      The same bit pattern interpreted as type To
 * 
 * @note Uses compiler builtins for optimal codegen (compiles to zero instructions)
 * @note Named with __ prefix to avoid conflict with std::bit_cast (C++20)
 */
template<typename _To, typename _From>
_CCCL_TRIVIAL_API _To __fp64_tool_bit_cast(const _From& __src) noexcept {
    static_assert(sizeof(_To) == sizeof(_From), "Size mismatch in __fp64_tool_bit_cast");
#if __FP64_TOOL_HAS_BUILTIN_BIT_CAST__ && !defined(__CUDA_ARCH__)
    // Prefer compiler builtin if available (C++20 or compiler extension)
    return __builtin_bit_cast(_To, __src);
#else
    // Fallback using memcpy (works on CUDA and older compilers)
    _To __dst;
    #if defined(__CUDA_ARCH__)
        memcpy(&__dst, &__src, sizeof(_To));
    #else
        __builtin_memcpy(&__dst, &__src, sizeof(_To));
    #endif
    return __dst;
#endif
}

/*
 * @brief Internal macro wrapper for bit_cast.
 *
 * CCCL integration: routes through CCCL's cuda::std::bit_cast by default.
 * __FP64_TOOL_BIT_CAST__ is the single switch point -- define it before
 * including this header for a fast re-map back to the in-house polyfill:
 *   #define __FP64_TOOL_BIT_CAST__(To, src) __fp64_tool_bit_cast<To>(src)
 */
#ifndef __FP64_TOOL_BIT_CAST__
#  define __FP64_TOOL_BIT_CAST__(To, src) ::cuda::std::bit_cast<To>(src)
#endif

#if defined(FP64_TOOL_RUNTIME_SIZE)

// Global device variables (shared across all threads) - must be non-static for CUDA
// On device: __device__ variables are in global memory, shared across all threads
// On host: static variables for normal C++ behavior
#if defined(__CUDACC__)
[[maybe_unused]] __device__ static int __fp64_tool_device_mantissa_bits = FP64_TOOL_MANTISSA_BITS;
[[maybe_unused]] __device__ static int __fp64_tool_device_exponent_bits = FP64_TOOL_EXPONENT_BITS;
#endif

[[maybe_unused]] static int __fp64_tool_host_mantissa_bits = FP64_TOOL_MANTISSA_BITS;
[[maybe_unused]] static int __fp64_tool_host_exponent_bits = FP64_TOOL_EXPONENT_BITS;

// Device-side setter (can be called from __device__ or __global__ functions)
// Only thread 0 in block 0 sets the value to avoid race conditions
#if defined(__CUDACC__)
[[maybe_unused]] __device__ static void __fp64_tool_set_device_mantissa_size(int __new_size) { 
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __fp64_tool_device_mantissa_bits = __new_size; 
    }
}
[[maybe_unused]] __device__ static void __fp64_tool_set_device_exponent_size(int __new_size) { 
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __fp64_tool_device_exponent_bits = __new_size; 
    }
}
#endif

// Device setter - works on both host and device
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
[[maybe_unused]] static _CCCL_HOST_DEVICE void fp64_tool_set_device_mantissa_size(int __new_size) noexcept { 
#if defined(__CUDA_ARCH__)
    // On device: call device function
    __fp64_tool_set_device_mantissa_size(__new_size);
#elif defined(__CUDACC__)
    // On host (CUDA compilation): use cudaMemcpyToSymbol
    cudaMemcpyToSymbol(__fp64_tool_device_mantissa_bits, &__new_size, sizeof(int));
#endif
}

[[maybe_unused]] static _CCCL_HOST_DEVICE void fp64_tool_set_device_exponent_size(int __new_size) noexcept { 
#if defined(__CUDA_ARCH__)
    // On device: call device function
    __fp64_tool_set_device_exponent_size(__new_size);
#elif defined(__CUDACC__)
    // On host (CUDA compilation): use cudaMemcpyToSymbol
    cudaMemcpyToSymbol(__fp64_tool_device_exponent_bits, &__new_size, sizeof(int));
#endif
}
#endif

// Host setter - works on host only
#if !defined(__CUDACC__)
[[maybe_unused]] static _CCCL_HOST void fp64_tool_set_host_mantissa_size(int __new_size) noexcept { 
    // On host (non-CUDA): direct assignment
    __fp64_tool_host_mantissa_bits = __new_size;
}

[[maybe_unused]] static _CCCL_HOST void fp64_tool_set_host_exponent_size(int __new_size) noexcept { 
    // On host (non-CUDA): direct assignment
    __fp64_tool_host_exponent_bits = __new_size;    
}
#endif

#endif

//=============================================================================
// SECTION 4: Precision Callback Implementation
//=============================================================================

/**
 * @brief Precision reduction callback function
 * 
 * This function modifies the bit representation of a double to simulate
 * reduced precision. It's called before and after each arithmetic operation.
 * 
 * The reduction happens in two phases:
 * 1. **Exponent reduction** (if FP64_TOOL_EXPONENT_BITS < 11):
 *    - Values outside the reduced exponent range become infinity or zero
 *    - Preserves the sign bit
 * 
 * 2. **Mantissa reduction** (if FP64_TOOL_MANTISSA_BITS < 52):
 *    - Excess mantissa bits are removed via truncation or rounding
 *    - Three rounding modes available: truncation, round-to-nearest, IEEE 754
 * 
 * @param v  Reference to the bit pattern to modify (modified in place)
 * 
 * @note This function is only compiled when __FP64_TOOL_ENABLE__ is defined
 * @note Thread-safe: no shared state is modified
 */
#if defined __FP64_TOOL_ENABLE__
_CCCL_TRIVIAL_API void __fp64_tool_callback(fpbits64_t& __v) noexcept 
{
    (void)__v;  /* Suppress unused parameter warning when no reduction is configured */
    //-------------------------------------------------------------------------
    // Phase 1: Exponent Range Reduction
    //-------------------------------------------------------------------------
#if defined(__FP64_TOOL_REDUCE_EXPONENT__)    
    /* Only check overflow/underflow if at least one clamping mode is enabled */
    #if !defined(FP64_TOOL_NO_OVERFLOW) || !defined(FP64_TOOL_NO_UNDERFLOW)
    {
        /* Get the exponent bits for the device or host */
        #if __FP64_TOOL_RUNTIME_SIZE__
            #if defined(__CUDA_ARCH__)
        __FP64_TOOL_CONST_QUALIFIER int __exponent_bits = __fp64_tool_device_exponent_bits;
            #else
        __FP64_TOOL_CONST_QUALIFIER int __exponent_bits = __fp64_tool_host_exponent_bits;
            #endif
        #else
        __FP64_TOOL_CONST_QUALIFIER int __exponent_bits = FP64_TOOL_EXPONENT_BITS;
        #endif

        /* IEEE 754 double-precision bit layout:
         * [63]    - Sign bit
         * [62:52] - 11-bit exponent (bias 1023)
         * [51:0]  - 52-bit mantissa (implicit leading 1)
         */
        constexpr uint64_t __EXP_MASK = 0x7FFULL << 52;    // Bits 52-62
        constexpr int64_t __ORIGINAL_BIAS = 1023;          // FP64 exponent bias
        __FP64_TOOL_CONST_QUALIFIER int64_t __NEW_BIAS = (1LL << (__exponent_bits - 1)) - 1;
        __FP64_TOOL_CONST_QUALIFIER int64_t __max_encoded = (1LL << __exponent_bits) - 2;

        uint64_t __bits = __v;
        uint64_t __exp_bits = (__bits & __EXP_MASK) >> 52;
        int64_t __unbiased_exp = (int64_t)__exp_bits - __ORIGINAL_BIAS;
        int64_t __new_exp_bits = __unbiased_exp + __NEW_BIAS;

        /* Check for overflow/underflow in reduced exponent range */
        #if !defined(FP64_TOOL_NO_OVERFLOW)
        if (__new_exp_bits > __max_encoded) 
        {
            /* Overflow: clamp to FP64 infinity (preserve sign) */
            constexpr uint64_t __SIGN_MASK = 1ULL << 63;
            constexpr uint64_t __FP64_INF_EXP = 0x7FFULL << 52;  /* FP64 infinity exponent */
            __v = (__bits & __SIGN_MASK) | __FP64_INF_EXP;
            return;  /* INF doesn't need mantissa reduction */
        }
        #endif
        
        #if !defined(FP64_TOOL_NO_UNDERFLOW)
        if (__new_exp_bits < 1) 
        {
            /* Underflow: flush to signed zero */
            constexpr uint64_t __SIGN_MASK = 1ULL << 63;
            __v = __bits & __SIGN_MASK;
            return;  /* Zero doesn't need mantissa reduction */
        }
        #endif
        /* Normal range: fall through to mantissa reduction */
    }
    #endif /* !NO_OVERFLOW || !NO_UNDERFLOW */
#endif /* __FP64_TOOL_REDUCE_EXPONENT__ */

    //-------------------------------------------------------------------------
    // Phase 2: Mantissa Precision Reduction
    //-------------------------------------------------------------------------
#if defined(__FP64_TOOL_REDUCE_MANTISSA__)
    /* Get the mantissa bits for the device or host */
    #if __FP64_TOOL_RUNTIME_SIZE__
        #if defined(__CUDA_ARCH__)
    __FP64_TOOL_CONST_QUALIFIER int __mantissa_bits = __fp64_tool_device_mantissa_bits;
        #else
    __FP64_TOOL_CONST_QUALIFIER int __mantissa_bits = __fp64_tool_host_mantissa_bits;
        #endif
    #else
    __FP64_TOOL_CONST_QUALIFIER int __mantissa_bits = FP64_TOOL_MANTISSA_BITS;
    #endif

    /* Calculate how many low bits to discard */
    const int __reduce_mantissa_bits = 52 - __mantissa_bits;
    
    #if defined(FP64_TOOL_TRUNCATION)
        //---------------------------------------------------------------------
        // Mode A: Simple Truncation (round toward zero)
        //---------------------------------------------------------------------
        /* Just shift out the low bits and shift back (zeros the low bits) */
        __v >>= __reduce_mantissa_bits;
        __v <<= __reduce_mantissa_bits;
        
    #elif defined(FP64_TOOL_ROUND_TO_NEAREST)
        //---------------------------------------------------------------------
        // Mode B: Round to Nearest (biased - always rounds 0.5 up)
        //---------------------------------------------------------------------
        /* Add half the quantization step before truncating */
        /* Incorrect for Infs and NaNs */
        __v += 1ULL << (__reduce_mantissa_bits - 1);
        __v >>= __reduce_mantissa_bits;
        __v <<= __reduce_mantissa_bits;
        
    #else /* FP64_TOOL_IEEE_ROUNDING (default) */
        //---------------------------------------------------------------------
        // Mode C: IEEE 754 Round-to-Nearest-Even (banker's rounding)
        //---------------------------------------------------------------------
        /* This is the default rounding mode in IEEE 754 and produces
         * statistically unbiased results for random data.
         * 
         * Rules:
         * - If discarded bits > 0.5: round up
         * - If discarded bits < 0.5: round down (truncate)
         * - If discarded bits == 0.5: round to nearest even
         */
        uint64_t __exponent = (__v >> 52) & 0x7FF;
        if (__exponent != 0x7FF) {  /* Skip NaN and Infinity */
            /* __half_mask: bit at position (bits_to_remove - 1), represents 0.5 */
            uint64_t __half_mask = 1ULL << (__reduce_mantissa_bits - 1);
            /* __upper_mask: the two MSBs of the bits being removed */
            uint64_t __upper_mask = __half_mask * 3;
            uint64_t __two_bits = __v & __upper_mask;
            
            if (__two_bits & __half_mask) {
                /* Discarded value >= 0.5, need to decide between up/down */
                /* If exactly 0.5, round to even; otherwise round up */
                __v += (__two_bits == __half_mask) ? (__half_mask - 1) : __half_mask;
            }
            __v >>= __reduce_mantissa_bits;
            __v <<= __reduce_mantissa_bits;
        }
    #endif /* rounding mode selection */
#endif /* __FP64_TOOL_REDUCE_MANTISSA__ */

} /* __fp64_tool_callback */

/** @brief Macro to invoke the precision callback */
#define __FP64_TOOL_CALLBACK__(v) __fp64_tool_callback(v)

#else /* !__FP64_TOOL_ENABLE__ */

/** @brief No-op when precision emulation is disabled */
#define __FP64_TOOL_CALLBACK__(v)

#endif /* __FP64_TOOL_ENABLE__ */

//=============================================================================
// SECTION 5: Main Class Definition
//=============================================================================

/**
 * @brief Emulated double-precision floating-point type with precision callbacks
 * 
 * This class provides a drop-in replacement for `double` that applies precision
 * reduction callbacks to all arithmetic operations. It stores values using the
 * standard IEEE 754 double-precision format but can simulate lower precisions.
 * 
 * ## Features
 * - Implicit conversion from all numeric types
 * - Full operator overloading (+, -, *, /, comparisons)
 * - CUDA host/device compatibility
 * - Zero overhead when callbacks are disabled
 * 
 * ## Memory Layout
 * - Size: 8 bytes (same as double)
 * - Alignment: 8 bytes
 * - Stores raw IEEE 754 bit pattern
 * 
 * @note The class is trivially copyable and can be used in CUDA kernels
 */
class fp64_tool_t
{
public:
    /** @brief Raw IEEE 754 bit representation of the value */
    fpbits64_t bits;

    //=========================================================================
    // Constructors
    //=========================================================================

    /** @brief Default constructor: initializes to zero */
    _CCCL_API inline fp64_tool_t() noexcept : bits{0u} {}
    
    /** 
     * @brief Raw bit constructor (use fpbits64_raw tag)
     * @param raw The raw IEEE 754 bit pattern
     * 
     * Example: fp64_tool_t(fpbits64_raw, 0x3FF0000000000000ULL) creates 1.0
     */
    _CCCL_API inline fp64_tool_t(fpbits64_raw_t, fpbits64_t __raw) noexcept : bits(__raw) {}
    
    /** @brief Copy constructor */
    _CCCL_API inline fp64_tool_t(const fp64_tool_t& __o) noexcept : bits(__o.bits) {}
    
    /** @brief Copy constructor from volatile (for atomic operations) */
    _CCCL_API inline fp64_tool_t(const volatile fp64_tool_t& __o) noexcept : bits(__o.bits) {}

    /** @brief Construct from double (implicit conversion) */
    _CCCL_API inline fp64_tool_t(double __d) noexcept : bits(__FP64_TOOL_BIT_CAST__(fpbits64_t, __d)) {}
    
    /** @brief Construct from float (implicit conversion with promotion) */
    _CCCL_API inline fp64_tool_t(float __f) noexcept : bits(__FP64_TOOL_BIT_CAST__(fpbits64_t, static_cast<double>(__f))) {}

    /** @brief Construct from any standard integer type (int / long / long long + unsigned).
     *  Routes through double, so every width/signedness is handled uniformly and
     *  portably (LP64 and LLP64). Excludes bool / character types by design. */
    _CCCL_TEMPLATE(class _Tp)
    _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
    _CCCL_API inline fp64_tool_t(_Tp __i) noexcept : bits(__FP64_TOOL_BIT_CAST__(fpbits64_t, static_cast<double>(__i))) {}

    //=========================================================================
    // Assignment Operators
    //=========================================================================

    /** @brief Copy assignment */
    _CCCL_API inline fp64_tool_t& operator=(const fp64_tool_t& __o) noexcept { 
        bits = __o.bits; 
        return *this; 
    }
    
    /** @brief Volatile copy assignment (for atomic operations) */
    _CCCL_API inline volatile fp64_tool_t& operator=(const fp64_tool_t& __o) volatile noexcept { 
        bits = __o.bits; 
        return *this; 
    }

    //=========================================================================
    // Type Conversion Operators
    //=========================================================================

    /** @brief Convert to double (implicit, preserves full precision) */
    _CCCL_API inline operator double() const noexcept { 
        return __FP64_TOOL_BIT_CAST__(double, bits); 
    }
    
    /** @brief Convert to float (explicit, may lose precision) */
    _CCCL_API inline explicit operator float() const noexcept { 
        return static_cast<float>(__FP64_TOOL_BIT_CAST__(double, bits)); 
    }
    
    /** @brief Convert to any standard integer type (explicit, truncates toward zero).
     *  Covers int / long / long long + unsigned uniformly; excludes bool / char. */
    _CCCL_TEMPLATE(class _Tp)
    _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
    _CCCL_API inline explicit operator _Tp() const noexcept { 
        return static_cast<_Tp>(__FP64_TOOL_BIT_CAST__(double, bits)); 
    }

    //=========================================================================
    // Arithmetic Operators (with precision callbacks)
    //=========================================================================

    /**
     * @brief Addition with precision callbacks
     * 
     * Operation flow:
     * 1. Apply callback to both operands
     * 2. Perform native FP64 addition
     * 3. Apply callback to result
     */
    _CCCL_API inline fp64_tool_t operator+(const fp64_tool_t& __y) const noexcept {
        fpbits64_t __a = bits, __b = __y.bits;
        __FP64_TOOL_CALLBACK__(__a); 
        __FP64_TOOL_CALLBACK__(__b);
        #if defined(__CUDA_ARCH__)
            fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __dadd_rn(__FP64_TOOL_BIT_CAST__(double, __a), __FP64_TOOL_BIT_CAST__(double, __b)));
        #else
            fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __FP64_TOOL_BIT_CAST__(double, __a) + __FP64_TOOL_BIT_CAST__(double, __b));
        #endif
        __FP64_TOOL_CALLBACK__(__r);
        return fp64_tool_t(fpbits64_raw, __r);
    }

    /** @brief Subtraction with precision callbacks */
    _CCCL_API inline fp64_tool_t operator-(const fp64_tool_t& __y) const noexcept {
        fpbits64_t __a = bits, __b = __y.bits;
        __FP64_TOOL_CALLBACK__(__a); 
        __FP64_TOOL_CALLBACK__(__b);
        #if defined(__CUDA_ARCH__)
            fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __dsub_rn(__FP64_TOOL_BIT_CAST__(double, __a), __FP64_TOOL_BIT_CAST__(double, __b)));
        #else
            fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __FP64_TOOL_BIT_CAST__(double, __a) - __FP64_TOOL_BIT_CAST__(double, __b));
        #endif
        __FP64_TOOL_CALLBACK__(__r);
        return fp64_tool_t(fpbits64_raw, __r);
    }

    /** @brief Multiplication with precision callbacks */
    _CCCL_API inline fp64_tool_t operator*(const fp64_tool_t& __y) const noexcept {
        fpbits64_t __a = bits, __b = __y.bits;
        __FP64_TOOL_CALLBACK__(__a); 
        __FP64_TOOL_CALLBACK__(__b);
        #if defined(__CUDA_ARCH__)
            fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __dmul_rn(__FP64_TOOL_BIT_CAST__(double, __a), __FP64_TOOL_BIT_CAST__(double, __b)));
        #else
            fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __FP64_TOOL_BIT_CAST__(double, __a) * __FP64_TOOL_BIT_CAST__(double, __b));
        #endif
        __FP64_TOOL_CALLBACK__(__r);
        return fp64_tool_t(fpbits64_raw, __r);
    }

    /** @brief Division with precision callbacks */
    _CCCL_API inline fp64_tool_t operator/(const fp64_tool_t& __y) const noexcept {
        fpbits64_t __a = bits, __b = __y.bits;
        __FP64_TOOL_CALLBACK__(__a); 
        __FP64_TOOL_CALLBACK__(__b);
        #if defined(__CUDA_ARCH__)
            fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __ddiv_rn(__FP64_TOOL_BIT_CAST__(double, __a), __FP64_TOOL_BIT_CAST__(double, __b)));
        #else
            fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __FP64_TOOL_BIT_CAST__(double, __a) / __FP64_TOOL_BIT_CAST__(double, __b));
        #endif
        __FP64_TOOL_CALLBACK__(__r);
        return fp64_tool_t(fpbits64_raw, __r);
    }

    /** 
     * @brief Unary negation (sign flip)
     * @note No precision callback - just flips the sign bit
     */
    _CCCL_API inline fp64_tool_t operator-() const noexcept {
        return fp64_tool_t(fpbits64_raw, bits ^ (1ULL << 63));
    }

    //=========================================================================
    // Compound Assignment Operators
    //=========================================================================

    /** @brief Add and assign */
    _CCCL_API inline fp64_tool_t& operator+=(const fp64_tool_t& __o) noexcept { 
        *this = *this + __o; 
        return *this; 
    }
    
    /** @brief Subtract and assign */
    _CCCL_API inline fp64_tool_t& operator-=(const fp64_tool_t& __o) noexcept { 
        *this = *this - __o; 
        return *this; 
    }
    
    /** @brief Multiply and assign */
    _CCCL_API inline fp64_tool_t& operator*=(const fp64_tool_t& __o) noexcept { 
        *this = *this * __o; 
        return *this; 
    }
    
    /** @brief Divide and assign */
    _CCCL_API inline fp64_tool_t& operator/=(const fp64_tool_t& __o) noexcept { 
        *this = *this / __o; 
        return *this; 
    }

    //=========================================================================
    // Increment/Decrement Operators
    //=========================================================================

    /** @brief Pre-increment */
    _CCCL_API inline fp64_tool_t& operator++() noexcept { 
        *this = *this + fp64_tool_t(1.0); 
        return *this; 
    }
    
    /** @brief Pre-decrement */
    _CCCL_API inline fp64_tool_t& operator--() noexcept { 
        *this = *this - fp64_tool_t(1.0); 
        return *this; 
    }
    
    /** @brief Post-increment */
    _CCCL_API inline fp64_tool_t operator++(int) noexcept { 
        auto __t = *this; 
        ++(*this); 
        return __t; 
    }
    
    /** @brief Post-decrement */
    _CCCL_API inline fp64_tool_t operator--(int) noexcept { 
        auto __t = *this; 
        --(*this); 
        return __t; 
    }

    //=========================================================================
    // Comparison Operators
    //=========================================================================

    /** @brief Equality comparison */
    _CCCL_API inline bool operator==(const fp64_tool_t& __y) const noexcept { 
        return __FP64_TOOL_BIT_CAST__(double, bits) == __FP64_TOOL_BIT_CAST__(double, __y.bits); 
    }
    
    /** @brief Inequality comparison */
    _CCCL_API inline bool operator!=(const fp64_tool_t& __y) const noexcept { 
        return __FP64_TOOL_BIT_CAST__(double, bits) != __FP64_TOOL_BIT_CAST__(double, __y.bits); 
    }
    
    /** @brief Less than comparison */
    _CCCL_API inline bool operator<(const fp64_tool_t& __y) const noexcept { 
        return __FP64_TOOL_BIT_CAST__(double, bits) < __FP64_TOOL_BIT_CAST__(double, __y.bits); 
    }
    
    /** @brief Greater than comparison */
    _CCCL_API inline bool operator>(const fp64_tool_t& __y) const noexcept { 
        return __FP64_TOOL_BIT_CAST__(double, bits) > __FP64_TOOL_BIT_CAST__(double, __y.bits); 
    }
    
    /** @brief Less than or equal comparison */
    _CCCL_API inline bool operator<=(const fp64_tool_t& __y) const noexcept { 
        return __FP64_TOOL_BIT_CAST__(double, bits) <= __FP64_TOOL_BIT_CAST__(double, __y.bits); 
    }
    
    /** @brief Greater than or equal comparison */
    _CCCL_API inline bool operator>=(const fp64_tool_t& __y) const noexcept { 
        return __FP64_TOOL_BIT_CAST__(double, bits) >= __FP64_TOOL_BIT_CAST__(double, __y.bits); 
    }
};

//=============================================================================
// SECTION 6: Math Functions
//=============================================================================

/**
 * @brief Square root with precision callbacks
 * 
 * @param x Input value
 * @return Square root of x with precision callbacks applied
 * 
 * @note Uses __dsqrt_rn intrinsic on CUDA, ::sqrt on host
 */
_CCCL_API inline fp64_tool_t sqrt(const fp64_tool_t& __x) noexcept {
    fpbits64_t __a = __x.bits;
    __FP64_TOOL_CALLBACK__(__a);
    #if defined(__CUDA_ARCH__)
        fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __dsqrt_rn(__FP64_TOOL_BIT_CAST__(double, __a)));
    #else
        fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, ::sqrt(__FP64_TOOL_BIT_CAST__(double, __a)));
    #endif
    __FP64_TOOL_CALLBACK__(__r);
    return fp64_tool_t(fpbits64_raw, __r);
}

/**
 * @brief Fused multiply-add with precision callbacks
 * 
 * Computes (x * y) + z with a single rounding operation.
 * 
 * @param x First multiplicand
 * @param y Second multiplicand
 * @param z Addend
 * @return (x * y) + z with precision callbacks applied to all operands and result
 * 
 * @note Uses __fma_rn intrinsic on CUDA, ::fma on host
 */
_CCCL_API inline fp64_tool_t fma(const fp64_tool_t& __x, const fp64_tool_t& __y, const fp64_tool_t& __z) noexcept {
    fpbits64_t __a = __x.bits, __b = __y.bits, __c = __z.bits;
    __FP64_TOOL_CALLBACK__(__a); 
    __FP64_TOOL_CALLBACK__(__b); 
    __FP64_TOOL_CALLBACK__(__c);
    #if defined(__CUDA_ARCH__)
        fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, __fma_rn(__FP64_TOOL_BIT_CAST__(double, __a), __FP64_TOOL_BIT_CAST__(double, __b), __FP64_TOOL_BIT_CAST__(double, __c)));
    #else
        fpbits64_t __r = __FP64_TOOL_BIT_CAST__(fpbits64_t, ::fma(__FP64_TOOL_BIT_CAST__(double, __a), __FP64_TOOL_BIT_CAST__(double, __b), __FP64_TOOL_BIT_CAST__(double, __c)));
    #endif
    __FP64_TOOL_CALLBACK__(__r);
    return fp64_tool_t(fpbits64_raw, __r);
}

//=============================================================================
// SECTION 7: Mixed-Type Operator Overloads
//=============================================================================

/**
 * @name Mixed-Type Arithmetic Operators
 * @brief Operators for combining fp64_tool with native arithmetic types
 * 
 * These templates enable natural expressions like:
 *   fp64_tool x = 1.5;
 *   auto y = x + 2.0;   // fp64_tool + double
 *   auto z = 3 * x;     // int * fp64_tool
 */
///@{

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline fp64_tool_t operator+(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x + fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline fp64_tool_t operator+(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) + __y; 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline fp64_tool_t operator-(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x - fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline fp64_tool_t operator-(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) - __y; 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline fp64_tool_t operator*(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x * fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline fp64_tool_t operator*(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) * __y; 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline fp64_tool_t operator/(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x / fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline fp64_tool_t operator/(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) / __y; 
}

///@}

/**
 * @name Mixed-Type Comparison Operators
 * @brief Comparison operators for fp64_tool with native arithmetic types
 */
///@{

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator==(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x == fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator==(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) == __y; 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator!=(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x != fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator!=(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) != __y; 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator<(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x < fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator<(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) < __y; 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator>(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x > fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator>(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) > __y; 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator<=(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x <= fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator<=(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) <= __y; 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator>=(const fp64_tool_t& __x, _Tp __y) noexcept { 
    return __x >= fp64_tool_t(static_cast<double>(__y)); 
}

template<typename _Tp, typename = typename std::enable_if<std::is_arithmetic<_Tp>::value>::type>
_CCCL_API inline bool operator>=(_Tp __x, const fp64_tool_t& __y) noexcept { 
    return fp64_tool_t(static_cast<double>(__x)) >= __y; 
}

///@}

//=============================================================================
// SECTION 8: Type Aliases
//=============================================================================

/**
 * @brief Convenient type alias for fp64_tool_t
 * 
 * Use this as a drop-in replacement for `double` in your code:
 * @code
 * using Real = fp64_tool;  // or double for production
 * Real x = 1.5, y = 2.5;
 * Real result = x + y;
 * @endcode
 */
using fp64_tool = fp64_tool_t;

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FP64_TOOL_H
