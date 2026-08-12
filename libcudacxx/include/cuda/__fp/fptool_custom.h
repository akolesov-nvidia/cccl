//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPTOOL_CUSTOM_H
#define _CUDA___FP_FPTOOL_CUSTOM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integer.h>

//! @file fptool_custom.h
//! @brief fp_custom - a drop-in `double` replacement with configurable precision
//!
//! This header-only library provides a `fp_custom` class that wraps native double-precision
//! floating-point operations while allowing compile-time precision reduction. It's designed
//! for:
//!
//!   - **Algorithm sensitivity analysis**: Test how algorithms behave with reduced precision
//!   - **Mixed-precision research**: Emulate lower-precision formats (float, bfloat16, etc.)
//!   - **CUDA/CPU compatibility**: Works identically on both host and device code
//!   - **Drop-in replacement**: Use `fp_custom` where you would use `double`
//!
//! ## Quick Start
//!
//! ```cpp
//! #define CCCL_FPTOOL_CUSTOM_MANTISSA_BITS 23  // Emulate float precision (52 bits -> 23 bits)
//! #include <cuda/fptool>
//!
//! fp_custom a = 1.5, b = 2.5;
//! fp_custom result = a + b;  // Precision callbacks applied automatically
//! double native = result;      // Convert back to native double
//! ```
//!
//! ## Configuration Macros (define BEFORE including this header)
//!
//! | Macro                              | Default   | Description                                  |
//! |------------------------------------|-----------|----------------------------------------------|
//! | `CCCL_FPTOOL_CUSTOM_MANTISSA_BITS` | 52        | Number of mantissa bits to preserve (1-52)   |
//! | `CCCL_FPTOOL_CUSTOM_EXPONENT_BITS` | 11        | Number of exponent bits to preserve (1-11)   |
//! | `CCCL_FPTOOL_CUSTOM_DISABLE`       | undefined | Disable precision emulation                  |
//! | `CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE`  | undefined | Enable runtime precision control (see below) |
//!
//! Mantissa reduction always uses IEEE 754 round-to-nearest-even.
//!
//! ## Underflow/Overflow Behavior
//!
//! When exponent bits are reduced (e.g., from 11 to 8 for FP32 emulation), values outside
//! the new dynamic range are clamped:
//!   - **Overflow**: Values too large for reduced exponent → Infinity (±INF)
//!   - **Underflow**: Values too small for reduced exponent → Zero (±0)
//!
//! In both cases the sign is preserved, and the clamped result skips mantissa reduction.
//! NaN and infinity are never clamped.
//!
//! ## Common Precision Configurations
//!
//! | Format | Mantissa | Exponent | Configuration                                                             |
//! |--------|----------|----------|---------------------------------------------------------------------------|
//! | FP64   | 52       | 11       | Default (no callbacks applied)                                            |
//! | FP32   | 23       | 8        | `CCCL_FPTOOL_CUSTOM_MANTISSA_BITS=23, CCCL_FPTOOL_CUSTOM_EXPONENT_BITS=8` |
//! | BF16   | 7        | 8        | `CCCL_FPTOOL_CUSTOM_MANTISSA_BITS=7, CCCL_FPTOOL_CUSTOM_EXPONENT_BITS=8`  |
//! | FP16   | 10       | 5        | `CCCL_FPTOOL_CUSTOM_MANTISSA_BITS=10, CCCL_FPTOOL_CUSTOM_EXPONENT_BITS=5` |
//! | TF32   | 10       | 8        | `CCCL_FPTOOL_CUSTOM_MANTISSA_BITS=10, CCCL_FPTOOL_CUSTOM_EXPONENT_BITS=8` |
//!
//! ## How It Works
//!
//! Each arithmetic operation follows this pattern:
//! 1. Apply callback to input operands (reduce precision)
//! 2. Perform native FP64 operation
//! 3. Apply callback to result (reduce precision)
//!
//! This models how lower-precision hardware would handle the computation while
//! maintaining full FP64 representation for intermediate storage.
//!
//! ## Runtime Precision Control
//!
//! By default, mantissa and exponent sizes are compile-time constants. Defining
//! `CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE` before including this header enables changing them
//! at runtime without recompilation. The `CCCL_FPTOOL_CUSTOM_MANTISSA_BITS` and
//! `CCCL_FPTOOL_CUSTOM_EXPONENT_BITS` macros then set the initial values only.
//!
//! ```cpp
//! #define CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE
//! #define CCCL_FPTOOL_CUSTOM_MANTISSA_BITS 52   // initial mantissa (full precision)
//! #define CCCL_FPTOOL_CUSTOM_EXPONENT_BITS 11   // initial exponent (full range)
//! #include <cuda/fptool>
//! ```
//!
//! ### Setter Functions
//!
//! | Function                                  | Target | Description                        |
//! |-------------------------------------------|--------|------------------------------------|
//! | `fp_custom_set_host_mantissa_size(int)`   | CPU    | Set mantissa bits on host (1-52)   |
//! | `fp_custom_set_host_exponent_size(int)`   | CPU    | Set exponent bits on host (1-11)   |
//! | `fp_custom_set_device_mantissa_size(int)` | GPU    | Set mantissa bits on device (1-52) |
//! | `fp_custom_set_device_exponent_size(int)` | GPU    | Set exponent bits on device (1-11) |
//!
//! Host and device sizes are independent — changing one does not affect the other.
//! The device setters use `cudaMemcpyToSymbol` when called from host code; a
//! `cudaDeviceSynchronize()` before the next kernel launch ensures the new value
//! is visible on the GPU. On the device side, only thread 0 of block 0 writes
//! the global variable to avoid race conditions.
//!
//! ### CPU Example
//!
//! ```cpp
//! // Full precision initially
//! fp_custom a = 1.0, b = 1e-15;
//! double full = (double)(a + b);       // preserves small term
//!
//! // Switch to float-like precision at runtime
//! fp_custom_set_host_mantissa_size(23);
//! fp_custom c = 1.0, d = 1e-15;
//! double reduced = (double)(c + d);    // small term lost → 1.0
//! ```
//!
//! @note There is a small performance cost compared to compile-time mode because
//!       the bit counts are read from memory rather than compiled as constants.
//!
//! @note Thread Safety: All operations are thread-safe (no shared mutable state)
//!
//! @copyright NVIDIA Corporation
//! @license Apache 2.0

#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>

#if _CCCL_CUDA_COMPILATION() && _CCCL_HOST_COMPILATION()
// Include CUDA runtime for host-side functions like cudaMemcpyToSymbol
#  include <cuda_runtime.h>
#endif

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

//=============================================================================
// SECTION 1: Configuration and Feature Flags
//=============================================================================

//! @brief Number of mantissa bits to preserve (1-52)
//!
//! FP64 has 52 explicit mantissa bits. Setting this lower rounds the mantissa
//! to simulate reduced precision. Common values:
//!   - 52: Full FP64 precision (no reduction)
//!   - 23: FP32/float precision
//!   - 10: FP16/TF32 precision
//!   - 7:  BF16 precision
#ifndef CCCL_FPTOOL_CUSTOM_MANTISSA_BITS
#  define CCCL_FPTOOL_CUSTOM_MANTISSA_BITS 52
#endif

//! @brief Number of exponent bits to preserve (1-11)
//!
//! FP64 has 11 exponent bits (bias 1023). Setting this lower reduces the
//! dynamic range, potentially causing overflow to infinity or underflow to zero.
//!   - 11: Full FP64 range (no reduction)
//!   - 8:  FP32/BF16/TF32 range
//!   - 5:  FP16 range
#ifndef CCCL_FPTOOL_CUSTOM_EXPONENT_BITS
#  define CCCL_FPTOOL_CUSTOM_EXPONENT_BITS 11
#endif

#if defined(CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE)
#  define _CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE 1
#else
#  define _CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE 0
#endif

/* Internal flags: auto-detect if reduction is needed */
#if _CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE || CCCL_FPTOOL_CUSTOM_EXPONENT_BITS < 11
#  define _CCCL_FPTOOL_CUSTOM_REDUCE_EXPONENT
#  ifndef _CCCL_FPTOOL_CUSTOM_ENABLE
#    define _CCCL_FPTOOL_CUSTOM_ENABLE
#  endif
#endif
#if _CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE || CCCL_FPTOOL_CUSTOM_MANTISSA_BITS < 52
#  define _CCCL_FPTOOL_CUSTOM_REDUCE_MANTISSA
#  ifndef _CCCL_FPTOOL_CUSTOM_ENABLE
#    define _CCCL_FPTOOL_CUSTOM_ENABLE
#  endif
#endif

// Master switch for precision emulation disabling
#if defined CCCL_FPTOOL_CUSTOM_DISABLE
#  undef _CCCL_FPTOOL_CUSTOM_ENABLE
#endif

#if defined CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE
#  define _CCCL_FPTOOL_CUSTOM_CONST_QUALIFIER const
#else
#  define _CCCL_FPTOOL_CUSTOM_CONST_QUALIFIER constexpr
#endif

//=============================================================================
// SECTION 2: Platform Abstraction (CUDA/Host Compatibility)
//=============================================================================

// Function decorators come from CCCL directly (see <cuda/std/__cccl/...>):
//   _CCCL_HOST_DEVICE_API inline   — public host/device entry points (hidden from ABI)
//   _CCCL_TRIVIAL_HOST_DEVICE_API  — force-inlined internal helpers (hot paths)
//   _CCCL_HOST_DEVICE / _CCCL_HOST — plain execution-space qualifiers, used on
//                        the static setters that must keep internal linkage.

//=============================================================================
// SECTION 3: Type Definitions and Utilities
//=============================================================================

//! @brief 64-bit unsigned integer type for bit manipulation of doubles
//!
//! This type is used to access the raw IEEE 754 bit representation of
//! double-precision floating-point values.
namespace cuda::experimental
{
using fpbits64 = uint64_t;

//! @brief Tag type for raw bit construction
//!
//! Used to disambiguate constructors that take raw bit patterns from those
//! that take numeric values. Example:
//!   fp_custom(fpbits64_raw, 0x3FF0000000000000ULL)  // 1.0 from bits
//!   fp_custom(1.0)                                   // 1.0 from value
struct fpbits64_raw_tag
{
  explicit fpbits64_raw_tag() = default;
};

//! @brief Global instance of the raw bit construction tag
inline constexpr fpbits64_raw_tag fpbits64_raw{};

#if defined(CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE)

// Global device variables (shared across all threads) - must be non-static for CUDA
// On device: __device__ variables are in global memory, shared across all threads
// On host: static variables for normal C++ behavior
#  if _CCCL_CUDA_COMPILATION()
[[maybe_unused]] __device__ static int __fp_custom_device_mantissa_bits = CCCL_FPTOOL_CUSTOM_MANTISSA_BITS;
[[maybe_unused]] __device__ static int __fp_custom_device_exponent_bits = CCCL_FPTOOL_CUSTOM_EXPONENT_BITS;
#  endif

[[maybe_unused]] static int __fp_custom_host_mantissa_bits = CCCL_FPTOOL_CUSTOM_MANTISSA_BITS;
[[maybe_unused]] static int __fp_custom_host_exponent_bits = CCCL_FPTOOL_CUSTOM_EXPONENT_BITS;

// Readers for the active copy: device code sees the __device__ variables, host
// code the static ones.
[[maybe_unused]] static _CCCL_HOST_DEVICE int __fp_custom_mantissa_bits() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return __fp_custom_device_mantissa_bits;), (return __fp_custom_host_mantissa_bits;))
}

[[maybe_unused]] static _CCCL_HOST_DEVICE int __fp_custom_exponent_bits() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return __fp_custom_device_exponent_bits;), (return __fp_custom_host_exponent_bits;))
}

// Device-side setter (can be called from __device__ or __global__ functions)
// Only thread 0 in block 0 sets the value to avoid race conditions
#  if _CCCL_CUDA_COMPILATION()
[[maybe_unused]] __device__ static void __fp_custom_set_device_mantissa_size(int __new_size)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    __fp_custom_device_mantissa_bits = __new_size;
  }
}
[[maybe_unused]] __device__ static void __fp_custom_set_device_exponent_size(int __new_size)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    __fp_custom_device_exponent_bits = __new_size;
  }
}
#  endif

// Device setter - works on both host and device
#  if _CCCL_CUDA_COMPILATION()
[[maybe_unused]] static _CCCL_HOST_DEVICE void fp_custom_set_device_mantissa_size(int __new_size) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__fp_custom_set_device_mantissa_size(__new_size);),
                    (cudaMemcpyToSymbol(__fp_custom_device_mantissa_bits, &__new_size, sizeof(int));))
}

[[maybe_unused]] static _CCCL_HOST_DEVICE void fp_custom_set_device_exponent_size(int __new_size) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__fp_custom_set_device_exponent_size(__new_size);),
                    (cudaMemcpyToSymbol(__fp_custom_device_exponent_bits, &__new_size, sizeof(int));))
}
#  endif

// Host setter - works on host only
#  if !_CCCL_CUDA_COMPILATION()
[[maybe_unused]] static _CCCL_HOST void fp_custom_set_host_mantissa_size(int __new_size) noexcept
{
  // On host (non-CUDA): direct assignment
  __fp_custom_host_mantissa_bits = __new_size;
}

[[maybe_unused]] static _CCCL_HOST void fp_custom_set_host_exponent_size(int __new_size) noexcept
{
  // On host (non-CUDA): direct assignment
  __fp_custom_host_exponent_bits = __new_size;
}
#  endif

#endif

//=============================================================================
// SECTION 4: Precision Callback Implementation
//=============================================================================

//! @brief Precision reduction callback function
//!
//! This function modifies the bit representation of a double to simulate
//! reduced precision. It's called before and after each arithmetic operation.
//!
//! The reduction happens in two phases:
//! 1. **Exponent reduction** (if CCCL_FPTOOL_CUSTOM_EXPONENT_BITS < 11):
//!    - Values outside the reduced exponent range become infinity or zero
//!    - Preserves the sign bit
//!    - NaN and infinity pass through unchanged
//!
//! 2. **Mantissa reduction** (if CCCL_FPTOOL_CUSTOM_MANTISSA_BITS < 52):
//!    - Excess mantissa bits are removed using IEEE 754 round-to-nearest-even
//!    - NaN and infinity are left untouched
//!
//! @param v  Reference to the bit pattern to modify (modified in place)
//!
//! @note This function is only compiled when _CCCL_FPTOOL_CUSTOM_ENABLE is defined
//! @note Thread-safe: no shared state is modified
#if defined _CCCL_FPTOOL_CUSTOM_ENABLE
_CCCL_TRIVIAL_HOST_DEVICE_API void __fp_custom_callback(fpbits64& __v) noexcept
{
  //-------------------------------------------------------------------------
  // Phase 1: Exponent Range Reduction
  //-------------------------------------------------------------------------
#  if defined(_CCCL_FPTOOL_CUSTOM_REDUCE_EXPONENT)
  {
/* Get the exponent bits for the device or host */
#    if _CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE
    _CCCL_FPTOOL_CUSTOM_CONST_QUALIFIER int __exponent_bits = __fp_custom_exponent_bits();
#    else
    _CCCL_FPTOOL_CUSTOM_CONST_QUALIFIER int __exponent_bits = CCCL_FPTOOL_CUSTOM_EXPONENT_BITS;
#    endif

    /* IEEE 754 double-precision bit layout:
     * [63]    - Sign bit
     * [62:52] - 11-bit exponent (bias 1023)
     * [51:0]  - 52-bit mantissa (implicit leading 1)
     */
    constexpr uint64_t __exp_mask                             = 0x7FFULL << 52; // Bits 52-62
    constexpr int64_t __original_bias                         = 1023; // FP64 exponent bias
    _CCCL_FPTOOL_CUSTOM_CONST_QUALIFIER int64_t __new_bias    = (1LL << (__exponent_bits - 1)) - 1;
    _CCCL_FPTOOL_CUSTOM_CONST_QUALIFIER int64_t __max_encoded = (1LL << __exponent_bits) - 2;

    uint64_t __bits     = __v;
    uint64_t __exp_bits = (__bits & __exp_mask) >> 52;

    /* Infinity and NaN carry an all-ones exponent, which must not be mistaken
     * for a large finite exponent and clamped: that would turn NaN into INF.
     */
    if (__exp_bits == 0x7FF)
    {
      return;
    }

    int64_t __unbiased_exp = (int64_t) __exp_bits - __original_bias;
    int64_t __new_exp_bits = __unbiased_exp + __new_bias;

    /* Check for overflow/underflow in reduced exponent range */
    if (__new_exp_bits > __max_encoded)
    {
      /* Overflow: clamp to FP64 infinity (preserve sign) */
      constexpr uint64_t __sign_mask    = 1ULL << 63;
      constexpr uint64_t __fp64_inf_exp = 0x7FFULL << 52; /* FP64 infinity exponent */
      __v                               = (__bits & __sign_mask) | __fp64_inf_exp;
      return; /* INF doesn't need mantissa reduction */
    }

    if (__new_exp_bits < 1)
    {
      /* Underflow: flush to signed zero */
      constexpr uint64_t __sign_mask = 1ULL << 63;
      __v                            = __bits & __sign_mask;
      return; /* Zero doesn't need mantissa reduction */
    }
    /* Normal range: fall through to mantissa reduction */
  }
#  endif /* _CCCL_FPTOOL_CUSTOM_REDUCE_EXPONENT */

  //-------------------------------------------------------------------------
  // Phase 2: Mantissa Precision Reduction
  //-------------------------------------------------------------------------
#  if defined(_CCCL_FPTOOL_CUSTOM_REDUCE_MANTISSA)
/* Get the mantissa bits for the device or host */
#    if _CCCL_FPTOOL_CUSTOM_RUNTIME_SIZE
  _CCCL_FPTOOL_CUSTOM_CONST_QUALIFIER int __mantissa_bits = __fp_custom_mantissa_bits();
#    else
  _CCCL_FPTOOL_CUSTOM_CONST_QUALIFIER int __mantissa_bits = CCCL_FPTOOL_CUSTOM_MANTISSA_BITS;
#    endif

  /* Calculate how many low bits to discard. In runtime mode this can be zero
   * (full precision requested), and rounding must then be skipped entirely:
   * the masks below would shift by -1. In compile-time mode the count is a
   * constant of at least 1, so the guard folds away.
   */
  const int __reduce_mantissa_bits = 52 - __mantissa_bits;

  //---------------------------------------------------------------------
  // IEEE 754 Round-to-Nearest-Even (banker's rounding)
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
  if (__reduce_mantissa_bits > 0 && __exponent != 0x7FF)
  { /* Skip NaN and Infinity */
    /* __half_mask: bit at position (bits_to_remove - 1), represents 0.5 */
    uint64_t __half_mask = 1ULL << (__reduce_mantissa_bits - 1);
    /* __upper_mask: the two MSBs of the bits being removed */
    uint64_t __upper_mask = __half_mask * 3;
    uint64_t __two_bits   = __v & __upper_mask;

    if (__two_bits & __half_mask)
    {
      /* Discarded value >= 0.5, need to decide between up/down */
      /* If exactly 0.5, round to even; otherwise round up */
      __v += (__two_bits == __half_mask) ? (__half_mask - 1) : __half_mask;
    }
    __v >>= __reduce_mantissa_bits;
    __v <<= __reduce_mantissa_bits;
  }
#  endif /* _CCCL_FPTOOL_CUSTOM_REDUCE_MANTISSA */

} /* __fp_custom_callback */

//! @brief Macro to invoke the precision callback
#  define _CCCL_FPTOOL_CUSTOM_CALLBACK(v) __fp_custom_callback(v)

#else /* !_CCCL_FPTOOL_CUSTOM_ENABLE */

//! @brief No-op when precision emulation is disabled
#  define _CCCL_FPTOOL_CUSTOM_CALLBACK(v)

#endif /* _CCCL_FPTOOL_CUSTOM_ENABLE */

//=============================================================================
// SECTION 5: Main Class Definition
//=============================================================================

//! @brief Emulated double-precision floating-point type with precision callbacks
//!
//! This class provides a drop-in replacement for `double` that applies precision
//! reduction callbacks to all arithmetic operations. It stores values using the
//! standard IEEE 754 double-precision format but can simulate lower precisions.
//!
//! ## Features
//! - Implicit conversion from all numeric types
//! - Full operator overloading (+, -, *, /, comparisons)
//! - CUDA host/device compatibility
//! - Zero overhead when callbacks are disabled
//!
//! ## Memory Layout
//! - Size: 8 bytes (same as double)
//! - Alignment: 8 bytes
//! - Stores raw IEEE 754 bit pattern
//!
//! ## Usage
//! @code
//! using Real = cuda::experimental::fp_custom;  // or double for production
//! Real x = 1.5, y = 2.5;
//! Real result = x + y;
//! @endcode
//!
//! @note The class is trivially copyable and can be used in CUDA kernels
class fp_custom
{
public:
  //=========================================================================
  // Constructors
  //=========================================================================

  //! @brief Default constructor: initializes to zero
  _CCCL_HOST_DEVICE_API constexpr fp_custom() noexcept
      : __bits_{0u}
  {}

  //! @brief Raw bit constructor (use fpbits64_raw tag)
  //! @param raw The raw IEEE 754 bit pattern
  //!
  //! Example: fp_custom(fpbits64_raw, 0x3FF0000000000000ULL) creates 1.0
  _CCCL_HOST_DEVICE_API constexpr fp_custom(fpbits64_raw_tag, fpbits64 __raw) noexcept
      : __bits_{__raw}
  {}

  //! @brief Copy constructor
  _CCCL_HOST_DEVICE_API fp_custom(const fp_custom& __o) noexcept
      : __bits_{__o.__bits_}
  {}

  //! @brief Copy constructor from volatile (for atomic operations)
  _CCCL_HOST_DEVICE_API fp_custom(const volatile fp_custom& __o) noexcept
      : __bits_{__o.__bits_}
  {}

  //! @brief Construct from double (implicit conversion)
  _CCCL_HOST_DEVICE_API fp_custom(double __d) noexcept
      : __bits_{::cuda::std::bit_cast<fpbits64>(__d)}
  {}

  //! @brief Construct from float (implicit conversion with promotion)
  _CCCL_HOST_DEVICE_API fp_custom(float __f) noexcept
      : __bits_{::cuda::std::bit_cast<fpbits64>(static_cast<double>(__f))}
  {}

  //! @brief Construct from any standard integer type (int / long / long long + unsigned).
  //!  Routes through double, so every width/signedness is handled uniformly and
  //! portably (LP64 and LLP64). Excludes bool / character types by design.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API fp_custom(_Tp __i) noexcept
      : __bits_{::cuda::std::bit_cast<fpbits64>(static_cast<double>(__i))}
  {}

  //=========================================================================
  // Assignment Operators
  //=========================================================================

  //! @brief Copy assignment
  _CCCL_HOST_DEVICE_API fp_custom& operator=(const fp_custom& __o) noexcept
  {
    __bits_ = __o.__bits_;
    return *this;
  }

  //! @brief Volatile copy assignment (for atomic operations)
  _CCCL_HOST_DEVICE_API volatile fp_custom& operator=(const fp_custom& __o) volatile noexcept
  {
    __bits_ = __o.__bits_;
    return *this;
  }

  //=========================================================================
  // Type Conversion Operators
  //=========================================================================

  //! @brief Convert to double (implicit, preserves full precision)
  _CCCL_HOST_DEVICE_API operator double() const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_);
  }

  //! @brief Convert to float (explicit, may lose precision)
  _CCCL_HOST_DEVICE_API explicit operator float() const noexcept
  {
    return static_cast<float>(::cuda::std::bit_cast<double>(__bits_));
  }

  //! @brief Convert to any standard integer type (explicit, truncates toward zero).
  //! Covers int / long / long long + unsigned uniformly; excludes bool / char.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API explicit operator _Tp() const noexcept
  {
    return static_cast<_Tp>(::cuda::std::bit_cast<double>(__bits_));
  }

  //=========================================================================
  // Arithmetic Operators (with precision callbacks)
  //=========================================================================
  //
  // The CUDA intrinsics are called as ::__dadd_rn etc. because this class lives in
  // cuda::experimental, where <cuda/fpemu> declares same-named overloads for its own
  // types: unqualified lookup would stop there and never reach the global scope.

  //! @brief Addition with precision callbacks
  //!
  //! Operation flow:
  //! 1. Apply callback to both operands
  //! 2. Perform native FP64 addition
  //! 3. Apply callback to result
  _CCCL_HOST_DEVICE_API fp_custom operator+(const fp_custom& __y) const noexcept
  {
    fpbits64 __a = __bits_, __b = __y.__bits_;
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__a);
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__b);
    fpbits64 __r{};
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (__r = ::cuda::std::bit_cast<fpbits64>(
         ::__dadd_rn(::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b)));),
      (__r = ::cuda::std::bit_cast<fpbits64>(::cuda::std::bit_cast<double>(__a) + ::cuda::std::bit_cast<double>(__b));))
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__r);
    return fp_custom(fpbits64_raw, __r);
  }

  //! @brief Subtraction with precision callbacks
  _CCCL_HOST_DEVICE_API fp_custom operator-(const fp_custom& __y) const noexcept
  {
    fpbits64 __a = __bits_, __b = __y.__bits_;
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__a);
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__b);
    fpbits64 __r{};
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (__r = ::cuda::std::bit_cast<fpbits64>(
         ::__dsub_rn(::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b)));),
      (__r = ::cuda::std::bit_cast<fpbits64>(::cuda::std::bit_cast<double>(__a) - ::cuda::std::bit_cast<double>(__b));))
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__r);
    return fp_custom(fpbits64_raw, __r);
  }

  //! @brief Multiplication with precision callbacks
  _CCCL_HOST_DEVICE_API fp_custom operator*(const fp_custom& __y) const noexcept
  {
    fpbits64 __a = __bits_, __b = __y.__bits_;
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__a);
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__b);
    fpbits64 __r{};
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (__r = ::cuda::std::bit_cast<fpbits64>(
         ::__dmul_rn(::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b)));),
      (__r = ::cuda::std::bit_cast<fpbits64>(::cuda::std::bit_cast<double>(__a) * ::cuda::std::bit_cast<double>(__b));))
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__r);
    return fp_custom(fpbits64_raw, __r);
  }

  //! @brief Division with precision callbacks
  _CCCL_HOST_DEVICE_API fp_custom operator/(const fp_custom& __y) const noexcept
  {
    fpbits64 __a = __bits_, __b = __y.__bits_;
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__a);
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__b);
    fpbits64 __r{};
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE,
      (__r = ::cuda::std::bit_cast<fpbits64>(
         ::__ddiv_rn(::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b)));),
      (__r = ::cuda::std::bit_cast<fpbits64>(::cuda::std::bit_cast<double>(__a) / ::cuda::std::bit_cast<double>(__b));))
    _CCCL_FPTOOL_CUSTOM_CALLBACK(__r);
    return fp_custom(fpbits64_raw, __r);
  }

  //! @brief Unary negation (sign flip)
  //! @note No precision callback - just flips the sign bit
  _CCCL_HOST_DEVICE_API fp_custom operator-() const noexcept
  {
    return fp_custom(fpbits64_raw, __bits_ ^ (1ULL << 63));
  }

  //=========================================================================
  // Compound Assignment Operators
  //=========================================================================

  //! @brief Add and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator+=(const fp_custom& __o) noexcept
  {
    *this = *this + __o;
    return *this;
  }

  //! @brief Subtract and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator-=(const fp_custom& __o) noexcept
  {
    *this = *this - __o;
    return *this;
  }

  //! @brief Multiply and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator*=(const fp_custom& __o) noexcept
  {
    *this = *this * __o;
    return *this;
  }

  //! @brief Divide and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator/=(const fp_custom& __o) noexcept
  {
    *this = *this / __o;
    return *this;
  }

  //=========================================================================
  // Increment/Decrement Operators
  //=========================================================================

  //! @brief Pre-increment
  _CCCL_HOST_DEVICE_API fp_custom& operator++() noexcept
  {
    *this = *this + fp_custom(1.0);
    return *this;
  }

  //! @brief Pre-decrement
  _CCCL_HOST_DEVICE_API fp_custom& operator--() noexcept
  {
    *this = *this - fp_custom(1.0);
    return *this;
  }

  //! @brief Post-increment
  _CCCL_HOST_DEVICE_API fp_custom operator++(int) noexcept
  {
    auto __t = *this;
    ++(*this);
    return __t;
  }

  //! @brief Post-decrement
  _CCCL_HOST_DEVICE_API fp_custom operator--(int) noexcept
  {
    auto __t = *this;
    --(*this);
    return __t;
  }

  //=========================================================================
  // Comparison Operators
  //=========================================================================

  //! @brief Equality comparison
  _CCCL_HOST_DEVICE_API bool operator==(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) == ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Inequality comparison
  _CCCL_HOST_DEVICE_API bool operator!=(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) != ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Less than comparison
  _CCCL_HOST_DEVICE_API bool operator<(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) < ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Greater than comparison
  _CCCL_HOST_DEVICE_API bool operator>(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) > ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Less than or equal comparison
  _CCCL_HOST_DEVICE_API bool operator<=(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) <= ::cuda::std::bit_cast<double>(__y.__bits_);
  }

  //! @brief Greater than or equal comparison
  _CCCL_HOST_DEVICE_API bool operator>=(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<double>(__bits_) >= ::cuda::std::bit_cast<double>(__y.__bits_);
  }

private:
  //! @brief Raw IEEE 754 bit representation of the value
  fpbits64 __bits_;
};

//=============================================================================
// SECTION 6: Math Functions
//=============================================================================

//! @brief Square root with precision callbacks
//!
//! @param x Input value
//! @return Square root of x with precision callbacks applied
//!
//! @note Uses __dsqrt_rn intrinsic on CUDA, ::sqrt on host
_CCCL_HOST_DEVICE_API inline fp_custom sqrt(const fp_custom& __x) noexcept
{
  fpbits64 __a = ::cuda::std::bit_cast<fpbits64>(static_cast<double>(__x));
  _CCCL_FPTOOL_CUSTOM_CALLBACK(__a);
  fpbits64 __r{};
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (__r = ::cuda::std::bit_cast<fpbits64>(::__dsqrt_rn(::cuda::std::bit_cast<double>(__a)));),
                    (__r = ::cuda::std::bit_cast<fpbits64>(::sqrt(::cuda::std::bit_cast<double>(__a)));))
  _CCCL_FPTOOL_CUSTOM_CALLBACK(__r);
  return fp_custom(fpbits64_raw, __r);
}

//! @brief Fused multiply-add with precision callbacks
//!
//! Computes (x * y) + z with a single rounding operation.
//!
//! @param x First multiplicand
//! @param y Second multiplicand
//! @param z Addend
//! @return (x * y) + z with precision callbacks applied to all operands and result
//!
//! @note Uses __fma_rn intrinsic on CUDA, ::fma on host
_CCCL_HOST_DEVICE_API inline fp_custom fma(const fp_custom& __x, const fp_custom& __y, const fp_custom& __z) noexcept
{
  fpbits64 __a = ::cuda::std::bit_cast<fpbits64>(static_cast<double>(__x));
  fpbits64 __b = ::cuda::std::bit_cast<fpbits64>(static_cast<double>(__y));
  fpbits64 __c = ::cuda::std::bit_cast<fpbits64>(static_cast<double>(__z));
  _CCCL_FPTOOL_CUSTOM_CALLBACK(__a);
  _CCCL_FPTOOL_CUSTOM_CALLBACK(__b);
  _CCCL_FPTOOL_CUSTOM_CALLBACK(__c);
  fpbits64 __r{};
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (__r = ::cuda::std::bit_cast<fpbits64>(::__fma_rn(
       ::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b), ::cuda::std::bit_cast<double>(__c)));),
    (__r = ::cuda::std::bit_cast<fpbits64>(::fma(
       ::cuda::std::bit_cast<double>(__a), ::cuda::std::bit_cast<double>(__b), ::cuda::std::bit_cast<double>(__c)));))
  _CCCL_FPTOOL_CUSTOM_CALLBACK(__r);
  return fp_custom(fpbits64_raw, __r);
}

//=============================================================================
// SECTION 7: Mixed-Type Operator Overloads
//=============================================================================

//! @name Mixed-Type Arithmetic Operators
//! @brief Operators for combining fp_custom with native arithmetic types
//!
//! These templates enable natural expressions like:
//!   fp_custom x = 1.5;
//!   auto y = x + 2.0;   // fp_custom + double
//!   auto z = 3 * x;     // int * fp_custom
//! @{

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline fp_custom operator+(const fp_custom& __x, _Tp __y) noexcept
{
  return __x + fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline fp_custom operator+(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) + __y;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline fp_custom operator-(const fp_custom& __x, _Tp __y) noexcept
{
  return __x - fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline fp_custom operator-(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) - __y;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline fp_custom operator*(const fp_custom& __x, _Tp __y) noexcept
{
  return __x * fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline fp_custom operator*(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) * __y;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline fp_custom operator/(const fp_custom& __x, _Tp __y) noexcept
{
  return __x / fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline fp_custom operator/(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) / __y;
}

//! @}

//! @name Mixed-Type Comparison Operators
//! @brief Comparison operators for fp_custom with native arithmetic types
//! @{

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator==(const fp_custom& __x, _Tp __y) noexcept
{
  return __x == fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator==(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) == __y;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator!=(const fp_custom& __x, _Tp __y) noexcept
{
  return __x != fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator!=(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) != __y;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator<(const fp_custom& __x, _Tp __y) noexcept
{
  return __x < fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator<(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) < __y;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator>(const fp_custom& __x, _Tp __y) noexcept
{
  return __x > fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator>(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) > __y;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator<=(const fp_custom& __x, _Tp __y) noexcept
{
  return __x <= fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator<=(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) <= __y;
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator>=(const fp_custom& __x, _Tp __y) noexcept
{
  return __x >= fp_custom(static_cast<double>(__y));
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
_CCCL_HOST_DEVICE_API inline bool operator>=(_Tp __x, const fp_custom& __y) noexcept
{
  return fp_custom(static_cast<double>(__x)) >= __y;
}

//! @}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPTOOL_CUSTOM_H
