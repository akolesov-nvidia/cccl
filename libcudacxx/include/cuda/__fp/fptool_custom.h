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

//! @file fptool_custom.h
//! @brief fp_custom - a drop-in floating-point replacement with configurable precision
//!
//! This header-only library provides an `fp_custom` class template that wraps native
//! floating-point operations while reducing the exponent and mantissa to a chosen size.
//! It's designed for:
//!
//!   - **Algorithm sensitivity analysis**: Test how algorithms behave with reduced precision
//!   - **Mixed-precision research**: Emulate lower-precision formats (float, bfloat16, etc.)
//!   - **CUDA/CPU compatibility**: Works identically on both host and device code
//!   - **Drop-in replacement**: Use `fp64_custom<>` where you would use `double`, and
//!     `fp32_custom<>` where you would use `float`
//!
//! ## Quick Start
//!
//! ```cpp
//! #include <cuda/fptool>
//!
//! using namespace cuda::experimental;
//!
//! // Step 1: swap `double` for `fp64_custom<>`; behavior is unchanged.
//! fp64_custom<> a = 1.5, b = 2.5;
//! double native = a + b;
//!
//! // Step 2: ask for a reduced format, here float-like (8 exponent, 23 mantissa bits).
//! // A format narrower than the source takes a value explicitly, see Conversions below.
//! fp64_custom<8, 23> c{1.5}, d{2.5};
//!
//! // The same over `float`, which stores 4 bytes and computes in binary32.
//! fp32_custom<> e = 1.5f;         // native binary32, no reduction
//! fp32_custom<8, 10> f{1.5f};     // TF32-like
//!
//! // And over binary128, where the platform has a 128-bit type, which is what emulating
//! // binary64 itself takes.
//! fp128_custom<11, 52> g{1.5};    // FP64, with headroom above it
//! ```
//!
//! ## Template Parameters
//!
//! | Parameter  | Meaning                                                              |
//! |------------|----------------------------------------------------------------------|
//! | `_FpType`  | Base type holding the value: `double`, `float` or binary128          |
//! | `_ExpSize` | Exponent bits to preserve, or `fp_custom_dynamic_size` for runtime   |
//! | `_MantSize`| Mantissa bits to preserve, or `fp_custom_dynamic_size` for runtime   |
//!
//! Each size runs from its minimum up to the base type's native size: 2 to 11 exponent and
//! 0 to 52 mantissa bits over `double`, 2 to 8 and 0 to 23 over `float`, 2 to 15 and 0 to 112
//! over binary128. The `fp64_custom`, `fp32_custom` and `fp128_custom` aliases fix the base
//! type and default both sizes to its native ones.
//!
//! Sizes equal to the native ones disable the corresponding reduction entirely: the
//! emulation code is discarded at compile time, leaving native base-type arithmetic.
//! Mantissa reduction always uses IEEE 754 round-to-nearest-even.
//!
//! ## Common Precision Configurations
//!
//! | Format | Exponent | Mantissa | over `double`        | over `float`         |
//! |--------|----------|----------|----------------------|----------------------|
//! | FP64   | 11       | 52       | `fp64_custom<>`      | -                    |
//! | FP32   | 8        | 23       | `fp64_custom<8, 23>` | `fp32_custom<>`      |
//! | BF16   | 8        | 7        | `fp64_custom<8, 7>`  | `fp32_custom<8, 7>`  |
//! | FP16   | 5        | 10       | `fp64_custom<5, 10>` | `fp32_custom<5, 10>` |
//! | TF32   | 8        | 10       | `fp64_custom<8, 10>` | `fp32_custom<8, 10>` |
//! | PO2    | native   | 0        | `fp64_custom<11, 0>` | `fp32_custom<8, 0>`  |
//!
//! Zero mantissa bits leave only the implicit leading 1, so a format like `fp64_custom<11, 0>`
//! rounds every value to the nearest power of two. The exponent, in contrast, starts at two
//! bits: an n-bit exponent field spends its all-ones pattern on infinity and NaN, so it
//! covers 2^n - 2 binades and a single bit would leave none.
//!
//! A format that several base types can hold may be emulated over any of them. The narrower
//! base is the cheaper one - `fp32_custom` stores 4 bytes and computes at binary32 throughput
//! - while a wider one carries the native operation's result further from the reduced format,
//! so the two can differ where that intermediate rounding is what decides a tie. A base type
//! wider than the format is also the only way to see the format's own rounding in isolation,
//! which is what `fp128_custom<11, 52>` is for: over `double` the same request reduces
//! nothing, the native operation having already rounded to binary64.
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
//! ## How It Works
//!
//! Each arithmetic operation follows this pattern:
//! 1. Apply the precision reduction to input operands
//! 2. Perform the native operation on the base type
//! 3. Apply the precision reduction to the result
//!
//! This models how lower-precision hardware would handle the computation while
//! maintaining the base type's representation for intermediate storage.
//!
//! ## Conversions
//!
//! A conversion into `fp_custom` is implicit where the requested format is at least as wide
//! as the source, and explicit where it is narrower - the rank rule CCCL applies to its
//! floating-point types, with integers counting as `double`:
//!
//! | Format                | from `double`, integers | from `float` | from binary128 |
//! |-----------------------|-------------------------|--------------|----------------|
//! | `fp64_custom<>`       | implicit                | implicit     | explicit       |
//! | `fp64_custom<8, 23>`  | explicit                | implicit     | explicit       |
//! | `fp64_custom<5, 10>`  | explicit                | explicit     | explicit       |
//! | `fp32_custom<>`       | explicit                | implicit     | explicit       |
//! | `fp32_custom<5, 10>`  | explicit                | explicit     | explicit       |
//! | `fp128_custom<>`      | implicit                | implicit     | implicit       |
//! | `fp128_custom<11, 52>`| implicit                | implicit     | explicit       |
//! | dynamic sizes         | explicit                | explicit     | explicit       |
//!
//! So `fp64_custom<>` is a `double` in every respect and `fp32_custom<>` a `float`, while a
//! reduced format marks values entering it: `fp64_custom<8, 23> x{d}`. Note that the base
//! type is not what decides this, the requested format is: a `double` is cast into
//! `fp32_custom<>` because binary32 is the narrower format, not because it is the base type.
//! Mixed arithmetic is unaffected, `x + 2.0` and `x < 2.0` taking a scalar operand whatever
//! the sizes are.
//!
//! What an explicit constructor reports is the format the value is entering, not a loss in
//! the constructor: the value is stored in the base type unreduced, and the sizes are
//! applied by the first arithmetic operation. `fp64_custom<8, 23>{1e300}` therefore reads
//! back as `1e300` and turns into infinity as soon as it is used. The base type itself is the
//! one boundary that does bind at construction, a `double` entering an `fp32_custom` being
//! rounded to binary32 there while the requested format still waits.
//!
//! Adopting the type across a codebase written against `double` means respelling every
//! initialization the table above makes explicit. Defining `CCCL_FP_CUSTOM_EXPLICIT_CASTS`
//! to 0 makes that column implicit instead, so those call sites compile unchanged; see the
//! macro in this header for what the setting gives up.
//!
//! Coming out, `operator double()` is always implicit, and `operator float()` is implicit
//! exactly where `float` holds the requested format, which over a `float` base is always.
//! Note what the explicit `operator float()` does and does not do: it decides which conversion
//! function a `float` target picks, but it cannot stop the target being reached, because the
//! implicit `operator double()` followed by the standard `double` to `float` conversion is a
//! valid path from any format. `float f = x;` compiles for every instantiation, and rounds, as
//! it would from a `double`.
//!
//! Over a base type no wider than binary64 the `double` on the way out is exact. Over
//! binary128 it rounds, and stays implicit anyway, so that a native format stays a drop-in for
//! its base type. `static_cast<__fp_custom_fp128>(x)` is the exact way out, offered by every
//! base type where the platform has the type.
//!
//! ## Runtime Precision Control
//!
//! Passing `fp_custom_dynamic_size` instead of a size takes that field's size from a
//! global variable that can be changed at runtime, without recompiling:
//!
//! ```cpp
//! using Real = fp64_custom<fp_custom_dynamic_size, fp_custom_dynamic_size>;
//!
//! Real a = 1.0, b = 1e-15;
//! double full = a + b;                        // starts at full FP64 precision
//!
//! fp_custom_set_host_mantissa_size(23);       // switch to float-like precision
//! double reduced = a + b;                     // small term is now lost
//! ```
//!
//! The sizes start at the base type's native values, so a program that never calls a setter
//! behaves exactly like that base type.
//!
//! ### Size Accessors
//!
//! | Function                                               | Called from | Description               |
//! |--------------------------------------------------------|-------------|---------------------------|
//! | `fp_custom_set_host_mantissa_size(int)`                | host        | Set host mantissa         |
//! | `fp_custom_set_host_exponent_size(int)`                | host        | Set host exponent         |
//! | `fp_custom_get_host_mantissa_size()`                   | host        | Read host mantissa        |
//! | `fp_custom_get_host_exponent_size()`                   | host        | Read host exponent        |
//! | `fp_custom_set_device_mantissa_size(int, stream_ref)`  | host        | Set device mantissa       |
//! | `fp_custom_set_device_exponent_size(int, stream_ref)`  | host        | Set device exponent       |
//! | `fp_custom_get_device_mantissa_size(stream_ref)`       | host        | Read device mantissa      |
//! | `fp_custom_get_device_exponent_size(stream_ref)`       | host        | Read device exponent      |
//! | `fp_custom_set_device_mantissa_size(int)`              | device      | Set device mantissa       |
//! | `fp_custom_set_device_exponent_size(int)`              | device      | Set device exponent       |
//! | `fp_custom_get_device_mantissa_size()`                 | device      | Read device mantissa      |
//! | `fp_custom_get_device_exponent_size()`                 | device      | Read device exponent      |
//!
//! Each takes the base type as a template argument defaulting to `double`, so
//! `fp_custom_set_host_mantissa_size(23)` sizes the `fp64_custom` instantiations and
//! `fp_custom_set_host_mantissa_size<float>(10)` the `fp32_custom` ones. A size runs from
//! its minimum to that base type's native one, 2 exponent bits upward and 0 mantissa bits
//! upward.
//!
//! Host and device sizes are independent — changing one does not affect the other — and so
//! are the sizes of the different base types, each holding its own pair.
//!
//! A device size is per-device state, so the host accessors say which device to touch and
//! when, through one `cuda::stream_ref`: the write is enqueued on that stream, and every
//! kernel that runs after it there sees the new size. Reading waits on the stream, since
//! the value has to come back to the host. The accessors throw `cuda::cuda_error` if the
//! copy fails, rather than leaving a status to be checked.
//!
//! From device code the accessors touch the variable directly. Only thread 0 of block 0
//! writes it, and nothing propagates the value to the rest of the grid, so a write from a
//! kernel belongs in a single-block setup kernel.
//!
//! The sizes live in one program-wide copy each, shared by all translation units. On the
//! device that sharing needs relocatable device code (`-rdc=true`); in whole-program mode
//! each translation unit necessarily gets its own device copy.
//!
//! Under NVRTC only the device-code accessors exist, since a JIT compilation has no host
//! side to set the sizes from.
//!
//! @note There is a small performance cost compared to fixed sizes because the sizes are
//! read from memory instead of being folded into the code.
//! @note Thread Safety: All operations are thread-safe (no shared mutable state) unless a
//! setter runs concurrently with arithmetic.

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__cccl/preprocessor.h> // _CCCL_PP_FOR_EACH, to fold __CUDA_ARCH_LIST__
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cfloat> // LDBL_*, to recognize a binary128 long double
#include <cuda/std/cmath>
#include <cuda/std/cstdint>

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)
// The host half of the runtime size control: a stream-ordered copy to the device globals
#  include <cuda/__memory/get_device_address.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/stream_ref.h>
#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

#include <nv/target>

// CCCL_FP_CUSTOM_EXPLICIT_CASTS controls whether conversions INTO a reduced fp_custom are
// explicit. It gates only the constructors, and only where the rank rule makes them
// explicit to begin with:
//   - double, integers -> a format narrower than double
//   - float            -> a format narrower than float
//   - binary128        -> a format narrower than binary128
//   - any of them      -> dynamic sizes, whose format is not known at compile time
// A format that holds the source exactly takes it implicitly either way. The conversions OUT
// are not affected here at all: operator double() is implicit whatever this is set to, and the
// explicitness of operator float() and of the binary128 one is fixed at their declarations.
//
// Default is 1 (narrowing casts explicit), matching CCCL's strict-cast conventions and the
// CCCL_FPMP_EXPLICIT_CASTS default.
//
// Set to 0 when adopting fp_custom across an existing codebase, which is what the type is
// for: swap a `double` typedef for a reduced format, recompile, and see how the algorithm
// behaves. Under the default every `T x = 1.0;` and `T x = 0;` in that codebase has to be
// respelled before it compiles; with 0 they compile unchanged.
//
// What setting 0 gives up is narrow: an fp_custom constructor does not reduce, storing the
// value in the base type and leaving the sizes to the first arithmetic operation, so an
// implicit conversion drops nothing at the conversion itself. What it drops is the annotation
// that the value is entering an emulated format, and with it the warning that a value outside
// the reduced range - 1e300 for fp64_custom<8, 23>, say - turns into infinity once used.
//
// Note what this does not reach: overload resolution on the way out. A reduced format
// converts implicitly to both float and double, so a call overloaded on those two is
// ambiguous for it at either setting, and wants an fp_custom overload of its own.
#ifndef CCCL_FP_CUSTOM_EXPLICIT_CASTS
#  define CCCL_FP_CUSTOM_EXPLICIT_CASTS 1
#endif
#if CCCL_FP_CUSTOM_EXPLICIT_CASTS == 1
#  define _CCCL_FP_CUSTOM_EXPLICIT explicit
#else
#  define _CCCL_FP_CUSTOM_EXPLICIT
#endif

// === the binary128 base type ===
// Availability of __float128. Deliberately broader than _CCCL_HAS_FLOAT128(), which also
// requires the q/Q literal suffixes that fptool never writes.
#ifndef _CCCL_FP_CUSTOM_HAS_FLOAT128_TYPE
#  if _CCCL_HAS_FLOAT128()
#    define _CCCL_FP_CUSTOM_HAS_FLOAT128_TYPE 1
#  elif (defined(__SIZEOF_FLOAT128__) || defined(__FLOAT128__) || defined(__CUDACC_RTC_FLOAT128__)) \
    && !_CCCL_HOST_ARCH(ARM64)
#    define _CCCL_FP_CUSTOM_HAS_FLOAT128_TYPE 1
#  else
#    define _CCCL_FP_CUSTOM_HAS_FLOAT128_TYPE 0
#  endif
#endif

// Platforms whose `long double` is binary128.
#ifndef _CCCL_FP_CUSTOM_HAS_LDOUBLE128
#  if LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && LDBL_MANT_DIG == 113
#    define _CCCL_FP_CUSTOM_HAS_LDOUBLE128 1
#  else
#    define _CCCL_FP_CUSTOM_HAS_LDOUBLE128 0
#  endif
#endif

// Whether __fp_custom_fp128 exists, needing a 128-bit integer for the bit pattern too. Read
// from the host toolchain only, never __CUDA_ARCH__: both passes must agree on the member set.
#ifndef _CCCL_FP_CUSTOM_FP128_ENABLE
#  if !_CCCL_HAS_INT128()
#    define _CCCL_FP_CUSTOM_FP128_ENABLE 0
#  elif _CCCL_COMPILER(NVRTC)
#    if (_CCCL_FP_CUSTOM_HAS_FLOAT128_TYPE == 1) && (_CCCL_PTX_ARCH() >= 1000)
#      define _CCCL_FP_CUSTOM_FP128_ENABLE 1
#    else
#      define _CCCL_FP_CUSTOM_FP128_ENABLE 0
#    endif
#  elif _CCCL_CUDA_COMPILATION() && !_CCCL_CUDA_COMPILER(NVCC)
#    define _CCCL_FP_CUSTOM_FP128_ENABLE 0
#  elif (_CCCL_FP_CUSTOM_HAS_FLOAT128_TYPE == 1) || (_CCCL_FP_CUSTOM_HAS_LDOUBLE128 == 1)
#    define _CCCL_FP_CUSTOM_FP128_ENABLE 1
#  else
#    define _CCCL_FP_CUSTOM_FP128_ENABLE 0
#  endif
#endif

// Whether binary128 arithmetic is device-callable: all-or-nothing for the compilation, so every
// targeted architecture must be sm_100 or later. Only __float128 stays 128-bit on the device.
#ifndef _CCCL_FP_CUSTOM_FP128_DEVICE_OPS
#  if (_CCCL_FP_CUSTOM_FP128_ENABLE == 0) || !_CCCL_CUDA_COMPILATION() \
    || (_CCCL_FP_CUSTOM_HAS_FLOAT128_TYPE == 0)
#    define _CCCL_FP_CUSTOM_FP128_DEVICE_OPS 0
#  elif _CCCL_COMPILER(NVRTC)
#    define _CCCL_FP_CUSTOM_FP128_DEVICE_OPS 1
#  elif defined(__CUDA_ARCH_LIST__)
// Folds the list into one conjunction: 800,1000 becomes 1 &&(800 >= 1000) &&(1000 >= 1000).
#    define _CCCL_FP_CUSTOM_FP128_ARCH_IS_SM100(_Arch) &&((_Arch) >= 1000)
#    if 1 _CCCL_PP_FOR_EACH(_CCCL_FP_CUSTOM_FP128_ARCH_IS_SM100, __CUDA_ARCH_LIST__)
#      define _CCCL_FP_CUSTOM_FP128_DEVICE_OPS 1
#    else
#      define _CCCL_FP_CUSTOM_FP128_DEVICE_OPS 0
#    endif
#    undef _CCCL_FP_CUSTOM_FP128_ARCH_IS_SM100
#  else
#    define _CCCL_FP_CUSTOM_FP128_DEVICE_OPS 0
#  endif
#endif

// Execution space of the entry points naming binary128. Declared in both passes either way, so
// the class stays identical and a device call without device ops fails at its call site.
#if (_CCCL_FP_CUSTOM_FP128_DEVICE_OPS == 1) || !_CCCL_CUDA_COMPILATION()
#  define _CCCL_FP_CUSTOM_FP128_API         _CCCL_HOST_DEVICE_API
#  define _CCCL_FP_CUSTOM_FP128_TRIVIAL_API _CCCL_TRIVIAL_HOST_DEVICE_API
#else
#  define _CCCL_FP_CUSTOM_FP128_API         _CCCL_HOST_API
#  define _CCCL_FP_CUSTOM_FP128_TRIVIAL_API _CCCL_TRIVIAL_HOST_API
#endif

// Whether sqrt and fma are available. They come from builtins that lower to a host backend, so
// they are host-only, and where no host toolchain stands behind the pass, absent.
#ifndef _CCCL_FP_CUSTOM_FP128_MATH
#  if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1) && !_CCCL_COMPILER(NVRTC)
#    define _CCCL_FP_CUSTOM_FP128_MATH 1
#  else
#    define _CCCL_FP_CUSTOM_FP128_MATH 0
#  endif
#endif

// The f128 builtins name __float128; the *l ones name long double.
#if (_CCCL_FP_CUSTOM_FP128_MATH == 1)
#  if (_CCCL_FP_CUSTOM_HAS_FLOAT128_TYPE == 1)
#    define _CCCL_FP_CUSTOM_FP128_SQRT(__x)          __builtin_sqrtf128(__x)
#    define _CCCL_FP_CUSTOM_FP128_FMA(__x, __y, __z) __builtin_fmaf128(__x, __y, __z)
#  else
#    define _CCCL_FP_CUSTOM_FP128_SQRT(__x)          __builtin_sqrtl(__x)
#    define _CCCL_FP_CUSTOM_FP128_FMA(__x, __y, __z) __builtin_fmal(__x, __y, __z)
#  endif
#endif // _CCCL_FP_CUSTOM_FP128_MATH == 1

#include <cuda/std/__cccl/prologue.h>

// === supported base types and field sizes ===
namespace cuda::experimental
{
//! @brief Size value selecting runtime control of a field
//!
//! Passed as `_ExpSize` or `_MantSize`, it takes that field's size from a global
//! variable instead of the template argument. See the setters and getters below.
inline constexpr uint16_t fp_custom_dynamic_size = static_cast<uint16_t>(-1);

#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1)
//! @brief The binary128 base type: `__float128`, or `long double` where that is binary128
//!
//! Forcing _CCCL_FP_CUSTOM_FP128_ENABLE=1 without both halves the base type needs is a hard
//! error rather than an alias that silently degrades.
#  if !_CCCL_HAS_INT128()
#    error "_CCCL_FP_CUSTOM_FP128_ENABLE=1 but this platform has no 128-bit integer to hold the bit pattern in"
#  elif (_CCCL_FP_CUSTOM_HAS_FLOAT128_TYPE == 1)
using __fp_custom_fp128 = __float128;
#  elif (_CCCL_FP_CUSTOM_HAS_LDOUBLE128 == 1)
using __fp_custom_fp128 = long double;
#  else
#    error "_CCCL_FP_CUSTOM_FP128_ENABLE=1 but this platform provides no 128-bit floating-point type"
#  endif
static_assert(sizeof(__fp_custom_fp128) == 16, "__fp_custom_fp128 must be a 128-bit floating-point type");
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1

//! @brief Base types fp_custom can hold values in
//!
//! binary64 and binary32, spelled `double` and `float`, and binary128 where the platform has
//! a 128-bit type to spell it with. _Float64 is accepted as a bit-identical alias for double;
//! there is no _Float32 counterpart because CCCL does not detect that type.
template <typename _Tp>
inline constexpr bool __fp_custom_is_supported_fp_v =
  ::cuda::std::is_same_v<_Tp, double> || ::cuda::std::is_same_v<_Tp, float>
#if _CCCL_HAS_FLOAT64()
  || ::cuda::std::is_same_v<_Tp, _Float64>
#endif // _CCCL_HAS_FLOAT64()
#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1)
  || ::cuda::std::is_same_v<_Tp, __fp_custom_fp128>
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1
  ;

//! @brief Native layout of a base type: everything the emulation needs to know about the
//! format _FpType comes with
//!
//! One specialization per supported base type, each spelling out the whole set below. Native
//! as opposed to requested - a requested size is measured against these and clamped to them
//! by the setters, and the distance between the two is what a reduction closes.
//!
//! Members:
//! - `__fp_type`      the type the arithmetic runs in. Differs from _FpType where a format is
//!                    spelled by more than one C++ type: one of them carries the arithmetic
//!                    and the rest are aliased onto its specialization, so that nothing
//!                    downstream sees the difference.
//! - `__bits_type`    the unsigned integer of the same width, which the reduction works on
//! - `__exp_size`     width of the exponent field
//! - `__mant_size`    width of the stored mantissa, the implicit leading bit excluded
//! - `__exp_bias`     the exponent bias, which IEEE 754 puts at half the all-ones pattern
//! - `__exp_all_ones` the all-ones exponent field, right-aligned: what infinity and NaN carry
//!                    and nothing finite does, in the form an extracted exponent is compared
//!                    against
//! - `__exp_mask`     the exponent field in place, which over a zero mantissa is infinity's
//!                    encoding
//! - `__sign_mask`    the sign bit, at the top of the word
//!
//! The primary template reports zeros so that an unsupported _FpType produces the single
//! static_assert below rather than an incomplete-type error; everything it names is a
//! placeholder that keeps the class well-formed until that assert fires.
template <typename _FpType>
struct __fp_custom_native_layout
{
  using __fp_type   = double;
  using __bits_type = uint64_t;

  static constexpr uint16_t __exp_size  = 0;
  static constexpr uint16_t __mant_size = 0;
  static constexpr int64_t __exp_bias   = 0;

  static constexpr __bits_type __exp_all_ones = 0;
  static constexpr __bits_type __exp_mask     = 0;
  static constexpr __bits_type __sign_mask    = 0;
};

//! @brief binary64: 1 sign bit, 11 exponent bits, 52 mantissa bits
template <>
struct __fp_custom_native_layout<double>
{
  using __fp_type   = double;
  using __bits_type = uint64_t;

  static constexpr uint16_t __exp_size  = 11;
  static constexpr uint16_t __mant_size = 52;
  static constexpr int64_t __exp_bias   = 1023;

  static constexpr __bits_type __exp_all_ones = 0x7ffULL;
  static constexpr __bits_type __exp_mask     = 0x7ff0000000000000ULL;
  static constexpr __bits_type __sign_mask    = 0x8000000000000000ULL;
};

//! @brief binary32: 1 sign bit, 8 exponent bits, 23 mantissa bits
template <>
struct __fp_custom_native_layout<float>
{
  using __fp_type   = float;
  using __bits_type = uint32_t;

  static constexpr uint16_t __exp_size  = 8;
  static constexpr uint16_t __mant_size = 23;
  static constexpr int64_t __exp_bias   = 127;

  static constexpr __bits_type __exp_all_ones = 0xffU;
  static constexpr __bits_type __exp_mask     = 0x7f800000U;
  static constexpr __bits_type __sign_mask    = 0x80000000U;
};

#if _CCCL_HAS_FLOAT64()
template <>
struct __fp_custom_native_layout<_Float64> : __fp_custom_native_layout<double>
{};
#endif // _CCCL_HAS_FLOAT64()

#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1)
//! @brief binary128: 1 sign bit, 15 exponent bits, 112 mantissa bits
//!
//! The in-place masks are shifted rather than written out, C++ having no 128-bit literal.
template <>
struct __fp_custom_native_layout<__fp_custom_fp128>
{
  using __fp_type   = __fp_custom_fp128;
  using __bits_type = __uint128_t;

  static constexpr uint16_t __exp_size  = 15;
  static constexpr uint16_t __mant_size = 112;
  static constexpr int64_t __exp_bias   = 16383;

  static constexpr __bits_type __exp_all_ones = 0x7fffU;
  static constexpr __bits_type __exp_mask     = __bits_type{0x7fffU} << 112;
  static constexpr __bits_type __sign_mask    = __bits_type{1} << 127;
};
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1

template <typename _FpType>
using __fp_custom_fp_t = typename __fp_custom_native_layout<_FpType>::__fp_type;

template <typename _FpType>
using __fp_custom_bits_t = typename __fp_custom_native_layout<_FpType>::__bits_type;

// Largest value a field size can take: the size itself where it is fixed, and the base
// type's native size where it is dynamic, the setters accepting nothing wider.
template <typename _FpType, uint16_t _ExpSize>
inline constexpr uint16_t __fp_custom_max_exp_size_v =
  _ExpSize == fp_custom_dynamic_size ? __fp_custom_native_layout<_FpType>::__exp_size : _ExpSize;

template <typename _FpType, uint16_t _MantSize>
inline constexpr uint16_t __fp_custom_max_mant_size_v =
  _MantSize == fp_custom_dynamic_size ? __fp_custom_native_layout<_FpType>::__mant_size : _MantSize;

// Whether float holds every value the requested format can take. binary32 has 8 exponent
// and 23 mantissa bits, and the reduction clamps whatever leaves the reduced exponent
// range to zero or infinity, so a format inside those two bounds reaches float without
// rounding. A runtime size is bounded only by its base type's native size, which is why
// _FpType enters: a base no wider than binary32 stays inside the bounds whatever its sizes
// turn out to be, while a wider one cannot promise that. _FpType also makes the value
// dependent where the conversion operators use it as a constraint.
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
inline constexpr bool __fp_custom_fits_in_float_v =
  __fp_custom_is_supported_fp_v<_FpType> && __fp_custom_max_exp_size_v<_FpType, _ExpSize> <= 8
  && __fp_custom_max_mant_size_v<_FpType, _MantSize> <= 23;

// The opposite direction: whether the requested format holds every value of a source
// format, which is what decides whether a conversion into fp_custom is implicit. CCCL
// ranks a floating-point format by its field sizes and calls a conversion implicit where
// the target's rank is at least the source's (cuda::std::__fp_is_implicit_conversion_v);
// read on two independent fields, that is "neither field narrower". A dynamic size answers
// false, as it has to: the size is not known here, so nothing can be proven about it.
// _FpType makes the value dependent where the constructors use it as a constraint.
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize, uint16_t _SrcExpSize, uint16_t _SrcMantSize>
inline constexpr bool __fp_custom_holds_v =
  __fp_custom_is_supported_fp_v<_FpType> && _ExpSize != fp_custom_dynamic_size && _MantSize != fp_custom_dynamic_size
  && _ExpSize >= _SrcExpSize && _MantSize >= _SrcMantSize;

// The sources a constructor takes a value from: binary64, which has 11 exponent and 52
// mantissa bits, binary32, which has 8 and 23, and binary128, which has 15 and 112. An
// integer counts as a binary64 source, the rank CCCL gives it
// (cuda::std::__fp_conv_rank_order_int_ext_v).
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
inline constexpr bool __fp_custom_holds_double_v = __fp_custom_holds_v<_FpType, _ExpSize, _MantSize, 11, 52>;

template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
inline constexpr bool __fp_custom_holds_float_v = __fp_custom_holds_v<_FpType, _ExpSize, _MantSize, 8, 23>;

#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1)
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
inline constexpr bool __fp_custom_holds_fp128_v = __fp_custom_holds_v<_FpType, _ExpSize, _MantSize, 15, 112>;
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1

// Sources with a constructor pair of their own, which the double constructor must refuse.
// Copy-initialization drops explicit constructors outright, so such a source would fall through
// to the double one and be taken implicitly, losing the required cast and, for binary128, bits.
template <class _Tp>
inline constexpr bool __fp_custom_has_own_ctor_v =
  ::cuda::std::is_same_v<_Tp, float>
#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1)
  || ::cuda::std::is_same_v<_Tp, __fp_custom_fp128>
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1
  ;

template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
class fp_custom;

// Detects an fp_custom specialization. Declared here rather than with the fma trait machinery
// below because the double constructor's constraint needs it.
template <class _Tp>
inline constexpr bool __is_fp_custom_v = false;
template <class _FpType, uint16_t _ExpSize, uint16_t _MantSize>
inline constexpr bool __is_fp_custom_v<fp_custom<_FpType, _ExpSize, _MantSize>> = true;

// What is left for the double constructor: anything reaching a double that is not spoken for
// elsewhere - binary64 however it is spelled, long double, which the class does not rank, and
// any type with a conversion to double of its own. Integers are excluded as having their own
// pair; another fp_custom outright, since sizes and base types never mix and a deduced
// parameter would collapse the two user-defined conversions that rule relies on into one.
template <class _Tp>
inline constexpr bool __fp_custom_is_double_src_v =
  ::cuda::std::is_convertible_v<_Tp, double> && !::cuda::std::is_integral_v<_Tp>
  && !__fp_custom_has_own_ctor_v<_Tp> && !__is_fp_custom_v<_Tp>;

// === runtime sizes ===
// Sizes used by instantiations that pass fp_custom_dynamic_size, initialised to the
// unreduced sizes of the base type so that a program which never calls a setter keeps
// native behavior.
//
// They have to be variable templates: a variable template has vague linkage, so all
// translation units share one copy, while a plain `inline _CCCL_DEVICE` variable is rejected
// by nvcc outside relocatable device code.
//
// They are the only mutable namespace-scope variables in this library; everything else
// at namespace scope is an immutable _CCCL_GLOBAL_CONSTANT or a constexpr trait.
//
// NVRTC compiles device code only, so the host half of the state and of the accessors
// does not exist there; the device half below is what a JIT-compiled kernel uses.
#if !_CCCL_COMPILER(NVRTC)
template <typename _FpType>
int __fp_custom_host_mantissa_size = __fp_custom_native_layout<_FpType>::__mant_size;

template <typename _FpType>
int __fp_custom_host_exponent_size = __fp_custom_native_layout<_FpType>::__exp_size;
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
template <typename _FpType>
_CCCL_DEVICE int __fp_custom_device_mantissa_size = __fp_custom_native_layout<_FpType>::__mant_size;

template <typename _FpType>
_CCCL_DEVICE int __fp_custom_device_exponent_size = __fp_custom_native_layout<_FpType>::__exp_size;
#endif // _CCCL_CUDA_COMPILATION()

#if !_CCCL_COMPILER(NVRTC)

// === host sizes ===
// Every accessor below, host and device alike, takes the base type as a template argument
// defaulting to double, and reaches that type's own pair of sizes. Each base type holds an
// independent pair, so sizing one leaves every other one alone.

//! @brief Set the mantissa size used by host code, from 0 to the base type's mantissa size
template <typename _FpType = double>
_CCCL_HOST_API inline void fp_custom_set_host_mantissa_size(int __new_size) noexcept
{
  _CCCL_ASSERT(__new_size >= 0 && __new_size <= __fp_custom_native_layout<_FpType>::__mant_size,
               "fp_custom mantissa size out of range");
  __fp_custom_host_mantissa_size<_FpType> = __new_size;
}

//! @brief Set the exponent size used by host code, from 2 to the base type's exponent size
template <typename _FpType = double>
_CCCL_HOST_API inline void fp_custom_set_host_exponent_size(int __new_size) noexcept
{
  _CCCL_ASSERT(__new_size >= 2 && __new_size <= __fp_custom_native_layout<_FpType>::__exp_size,
               "fp_custom exponent size out of range");
  __fp_custom_host_exponent_size<_FpType> = __new_size;
}

//! @brief Read the mantissa size used by host code
template <typename _FpType = double>
[[nodiscard]] _CCCL_HOST_API inline int fp_custom_get_host_mantissa_size() noexcept
{
  return __fp_custom_host_mantissa_size<_FpType>;
}

//! @brief Read the exponent size used by host code
template <typename _FpType = double>
[[nodiscard]] _CCCL_HOST_API inline int fp_custom_get_host_exponent_size() noexcept
{
  return __fp_custom_host_exponent_size<_FpType>;
}

#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()

// === device sizes, from device code ===
// These touch the globals directly and are the form NVRTC has, a JIT compilation having
// no host side. Host code goes through the stream-ordered overloads further down.

//! @brief Set the mantissa size used by device code, from a kernel
//!
//! The size runs from 0 to the base type's mantissa size.
//!
//! Only thread 0 of block 0 writes, so that a whole grid calling this does not race, but
//! nothing makes the new size visible to the rest of the grid: this is for a single-block
//! setup kernel, or for a JIT-compiled program that has no host side to set it from.
template <typename _FpType = double>
_CCCL_DEVICE_API inline void fp_custom_set_device_mantissa_size(int __new_size) noexcept
{
  _CCCL_ASSERT(__new_size >= 0 && __new_size <= __fp_custom_native_layout<_FpType>::__mant_size,
               "fp_custom mantissa size out of range");
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    __fp_custom_device_mantissa_size<_FpType> = __new_size;
  }
}

//! @brief Set the exponent size used by device code, from a kernel
//!
//! The size runs from 2 to the base type's exponent size.
//! @copydetails fp_custom_set_device_mantissa_size
template <typename _FpType = double>
_CCCL_DEVICE_API inline void fp_custom_set_device_exponent_size(int __new_size) noexcept
{
  _CCCL_ASSERT(__new_size >= 2 && __new_size <= __fp_custom_native_layout<_FpType>::__exp_size,
               "fp_custom exponent size out of range");
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    __fp_custom_device_exponent_size<_FpType> = __new_size;
  }
}

//! @brief Read the mantissa size used by device code, from a kernel
template <typename _FpType = double>
[[nodiscard]] _CCCL_DEVICE_API inline int fp_custom_get_device_mantissa_size() noexcept
{
  return __fp_custom_device_mantissa_size<_FpType>;
}

//! @brief Read the exponent size used by device code, from a kernel
template <typename _FpType = double>
[[nodiscard]] _CCCL_DEVICE_API inline int fp_custom_get_device_exponent_size() noexcept
{
  return __fp_custom_device_exponent_size<_FpType>;
}

#endif // _CCCL_CUDA_COMPILATION()

#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

// === device sizes, from host code ===
// A size is per-device state, so where to write it is part of the call: the device comes
// from the stream, which also orders the write against the kernels that read it. The
// value being copied is an ordinary host object, which a pageable host-to-device copy
// consumes before returning, so it needs to outlive the call no longer than that.

//! @brief Set the mantissa size used by device code on the stream's device
//!
//! The size runs from 0 to the base type's mantissa size. The copy is enqueued on
//! `__stream`, so every kernel that runs after it there sees the new size and no
//! synchronization is needed. A program using several devices sets the size once per device.
//!
//! @param __new_size Mantissa bits to keep
//! @param __stream Stream to order the write against, and whose device to write on
//! @throws cuda::cuda_error if the copy cannot be enqueued
template <typename _FpType = double>
_CCCL_HOST_API void fp_custom_set_device_mantissa_size(int __new_size, ::cuda::stream_ref __stream)
{
  _CCCL_ASSERT(__new_size >= 0 && __new_size <= __fp_custom_native_layout<_FpType>::__mant_size,
               "fp_custom mantissa size out of range");
  int* __size_ptr = ::cuda::get_device_address(__fp_custom_device_mantissa_size<_FpType>, __stream.device());
  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync,
    "failed to set the fp_custom device mantissa size",
    __size_ptr,
    &__new_size,
    sizeof(int),
    ::cudaMemcpyHostToDevice,
    __stream.get());
}

//! @brief Set the exponent size used by device code on the stream's device
//!
//! The size runs from 2 to the base type's exponent size.
//! @copydetails fp_custom_set_device_mantissa_size
template <typename _FpType = double>
_CCCL_HOST_API void fp_custom_set_device_exponent_size(int __new_size, ::cuda::stream_ref __stream)
{
  _CCCL_ASSERT(__new_size >= 2 && __new_size <= __fp_custom_native_layout<_FpType>::__exp_size,
               "fp_custom exponent size out of range");
  int* __size_ptr = ::cuda::get_device_address(__fp_custom_device_exponent_size<_FpType>, __stream.device());
  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync,
    "failed to set the fp_custom device exponent size",
    __size_ptr,
    &__new_size,
    sizeof(int),
    ::cudaMemcpyHostToDevice,
    __stream.get());
}

//! @brief Read the mantissa size used by device code on the stream's device
//!
//! Waits on `__stream`, so a size set on it is included and the value read is the one the
//! next kernel there would use.
//!
//! @param __stream Stream to order the read against, and whose device to read from
//! @return The mantissa size in effect on that device
//! @throws cuda::cuda_error if the copy fails
template <typename _FpType = double>
[[nodiscard]] _CCCL_HOST_API int fp_custom_get_device_mantissa_size(::cuda::stream_ref __stream)
{
  int __size            = 0;
  const int* __size_ptr = ::cuda::get_device_address(__fp_custom_device_mantissa_size<_FpType>, __stream.device());
  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync,
    "failed to read the fp_custom device mantissa size",
    &__size,
    __size_ptr,
    sizeof(int),
    ::cudaMemcpyDeviceToHost,
    __stream.get());
  __stream.sync();
  return __size;
}

//! @brief Read the exponent size used by device code on the stream's device
//! @copydetails fp_custom_get_device_mantissa_size
template <typename _FpType = double>
[[nodiscard]] _CCCL_HOST_API int fp_custom_get_device_exponent_size(::cuda::stream_ref __stream)
{
  int __size            = 0;
  const int* __size_ptr = ::cuda::get_device_address(__fp_custom_device_exponent_size<_FpType>, __stream.device());
  _CCCL_TRY_CUDA_API(
    ::cudaMemcpyAsync,
    "failed to read the fp_custom device exponent size",
    &__size,
    __size_ptr,
    sizeof(int),
    ::cudaMemcpyDeviceToHost,
    __stream.get());
  __stream.sync();
  return __size;
}

#endif // _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)

//! @brief Mantissa size in effect for a given _MantSize argument
//!
//! Folds to a constant unless the size is runtime-controlled.
template <typename _FpType, uint16_t _MantSize>
[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API int __fp_custom_mantissa_size() noexcept
{
  if constexpr (_MantSize == fp_custom_dynamic_size)
  {
    // A host-only compilation drops the device branch in the preprocessor, before the
    // device global it names has to exist.
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return __fp_custom_device_mantissa_size<_FpType>;),
                      (return __fp_custom_host_mantissa_size<_FpType>;))
  }
  else
  {
    return static_cast<int>(_MantSize);
  }
}

//! @brief Exponent size in effect for a given _ExpSize argument
template <typename _FpType, uint16_t _ExpSize>
[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API int __fp_custom_exponent_size() noexcept
{
  if constexpr (_ExpSize == fp_custom_dynamic_size)
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                      (return __fp_custom_device_exponent_size<_FpType>;),
                      (return __fp_custom_host_exponent_size<_FpType>;))
  }
  else
  {
    return static_cast<int>(_ExpSize);
  }
}

// === precision reduction ===
//! @brief Precision reduction applied to operands and results
//!
//! This function modifies the bit representation of a base-type value to simulate
//! reduced precision. It's called before and after each arithmetic operation.
//!
//! The reduction happens in two phases:
//! 1. **Exponent reduction** (if _ExpSize is below the native size, or dynamic):
//!    - Values outside the reduced exponent range become infinity or zero
//!    - Preserves the sign bit
//!    - NaN and infinity pass through unchanged
//!
//! 2. **Mantissa reduction** (if _MantSize is below the native size, or dynamic):
//!    - Excess mantissa bits are removed using IEEE 754 round-to-nearest-even
//!    - NaN and infinity are left untouched
//!
//! With native sizes on both axes the whole body is discarded, so an instantiation asking for
//! its base type's own format compiles down to plain base-type arithmetic.
//!
//! @param __v  Reference to the bit pattern to modify (modified in place)
//!
//! @note Thread-safe: no shared state is modified
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
_CCCL_TRIVIAL_HOST_DEVICE_API void __fp_custom_reduce(__fp_custom_bits_t<_FpType>& __v) noexcept
{
  using _Bits   = __fp_custom_bits_t<_FpType>;
  using _Layout = __fp_custom_native_layout<_FpType>;

  constexpr uint16_t __native_exp_size  = _Layout::__exp_size;
  constexpr uint16_t __native_mant_size = _Layout::__mant_size;

  // === phase 1: exponent range reduction ===
  if constexpr (_ExpSize == fp_custom_dynamic_size || _ExpSize < __native_exp_size)
  {
    const int __exp_size = __fp_custom_exponent_size<_FpType, _ExpSize>();

    const int64_t __new_bias    = (1LL << (__exp_size - 1)) - 1;
    const int64_t __max_encoded = (1LL << __exp_size) - 2;

    const _Bits __bits     = __v;
    const _Bits __exp_bits = static_cast<_Bits>((__bits & _Layout::__exp_mask) >> __native_mant_size);

    /* Infinity and NaN carry an all-ones exponent, which must not be mistaken
     * for a large finite exponent and clamped: that would turn NaN into INF.
     */
    if (__exp_bits == _Layout::__exp_all_ones)
    {
      return;
    }

    const int64_t __unbiased_exp = static_cast<int64_t>(__exp_bits) - _Layout::__exp_bias;
    const int64_t __new_exp_bits = __unbiased_exp + __new_bias;

    /* Check for overflow/underflow in reduced exponent range */
    if (__new_exp_bits > __max_encoded)
    {
      /* Overflow: clamp to the base type's infinity (preserve sign), which an all-ones
       * exponent over a zero mantissa is, so __exp_mask is that pattern.
       */
      __v = static_cast<_Bits>((__bits & _Layout::__sign_mask) | _Layout::__exp_mask);
      return; /* INF doesn't need mantissa reduction */
    }

    if (__new_exp_bits < 1)
    {
      /* Underflow: flush to signed zero */
      __v = static_cast<_Bits>(__bits & _Layout::__sign_mask);
      return; /* Zero doesn't need mantissa reduction */
    }
    /* Normal range: fall through to mantissa reduction */
  }

  // === phase 2: mantissa precision reduction ===
  if constexpr (_MantSize == fp_custom_dynamic_size || _MantSize < __native_mant_size)
  {
    const int __mant_size = __fp_custom_mantissa_size<_FpType, _MantSize>();

    /* Number of low bits to discard. A runtime size can ask for full precision,
     * and rounding must then be skipped entirely: the masks below would shift
     * by -1. With a fixed size the count is a constant of at least 1, so the
     * guard folds away.
     */
    const int __dropped_mant_size = static_cast<int>(__native_mant_size) - __mant_size;

    // === IEEE 754 round-to-nearest-even (banker's rounding) ===
    /* This is the default rounding mode in IEEE 754 and produces
     * statistically unbiased results for random data.
     *
     * Rules:
     * - If discarded bits > 0.5: round up
     * - If discarded bits < 0.5: round down (truncate)
     * - If discarded bits == 0.5: round to nearest even
     */
    const _Bits __exponent = static_cast<_Bits>((__v >> __native_mant_size) & _Layout::__exp_all_ones);
    if (__dropped_mant_size > 0 && __exponent != _Layout::__exp_all_ones)
    { /* Skip NaN and Infinity */
      /* __half_mask: bit at position (bits_to_remove - 1), represents 0.5 */
      const _Bits __half_mask = static_cast<_Bits>(_Bits{1} << (__dropped_mant_size - 1));
      /* __upper_mask: the two MSBs of the bits being removed */
      const _Bits __upper_mask = static_cast<_Bits>(__half_mask * 3);
      const _Bits __two_bits   = static_cast<_Bits>(__v & __upper_mask);

      if (__two_bits & __half_mask)
      {
        /* Discarded value >= 0.5, need to decide between up/down */
        /* If exactly 0.5, round to even; otherwise round up */
        __v = static_cast<_Bits>(__v + ((__two_bits == __half_mask) ? (__half_mask - 1) : __half_mask));
      }
      __v = static_cast<_Bits>(__v >> __dropped_mant_size);
      __v = static_cast<_Bits>(__v << __dropped_mant_size);
    }
  }
}

// === base-type arithmetic ===
// The round-to-nearest-even operations the class performs between the two reductions, one
// overload per base type so that the operators can name them without asking which type they
// hold. The device side uses the intrinsics rather than the built-in operators so that a
// multiply and an add are never contracted into an FMA.

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API float __fp_custom_add(float __x, float __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fadd_rn(__x, __y);), (return __x + __y;))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API double __fp_custom_add(double __x, double __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__dadd_rn(__x, __y);), (return __x + __y;))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API float __fp_custom_sub(float __x, float __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fsub_rn(__x, __y);), (return __x - __y;))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API double __fp_custom_sub(double __x, double __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__dsub_rn(__x, __y);), (return __x - __y;))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API float __fp_custom_mul(float __x, float __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fmul_rn(__x, __y);), (return __x * __y;))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API double __fp_custom_mul(double __x, double __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__dmul_rn(__x, __y);), (return __x * __y;))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API float __fp_custom_div(float __x, float __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fdiv_rn(__x, __y);), (return __x / __y;))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API double __fp_custom_div(double __x, double __y) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__ddiv_rn(__x, __y);), (return __x / __y;))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API float __fp_custom_sqrt(float __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fsqrt_rn(__x);), (return ::cuda::std::sqrt(__x);))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API double __fp_custom_sqrt(double __x) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__dsqrt_rn(__x);), (return ::cuda::std::sqrt(__x);))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API float __fp_custom_fma(float __x, float __y, float __z) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fmaf_rn(__x, __y, __z);), (return ::cuda::std::fma(__x, __y, __z);))
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API double __fp_custom_fma(double __x, double __y, double __z) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return ::__fma_rn(__x, __y, __z);), (return ::cuda::std::fma(__x, __y, __z);))
}

#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1)
// binary128 has no rounding-mode intrinsic on either side, so these four are the built-in
// operators. One operation apiece leaves a compiler nothing to contract.

[[nodiscard]] _CCCL_FP_CUSTOM_FP128_TRIVIAL_API __fp_custom_fp128
__fp_custom_add(__fp_custom_fp128 __x, __fp_custom_fp128 __y) noexcept
{
  return __x + __y;
}

[[nodiscard]] _CCCL_FP_CUSTOM_FP128_TRIVIAL_API __fp_custom_fp128
__fp_custom_sub(__fp_custom_fp128 __x, __fp_custom_fp128 __y) noexcept
{
  return __x - __y;
}

[[nodiscard]] _CCCL_FP_CUSTOM_FP128_TRIVIAL_API __fp_custom_fp128
__fp_custom_mul(__fp_custom_fp128 __x, __fp_custom_fp128 __y) noexcept
{
  return __x * __y;
}

[[nodiscard]] _CCCL_FP_CUSTOM_FP128_TRIVIAL_API __fp_custom_fp128
__fp_custom_div(__fp_custom_fp128 __x, __fp_custom_fp128 __y) noexcept
{
  return __x / __y;
}

#  if (_CCCL_FP_CUSTOM_FP128_MATH == 1)
// Host-only whatever _CCCL_FP_CUSTOM_FP128_DEVICE_OPS says: the builtins lower to a host
// backend, and the device has no entry point to reach instead.

[[nodiscard]] _CCCL_TRIVIAL_HOST_API __fp_custom_fp128 __fp_custom_sqrt(__fp_custom_fp128 __x) noexcept
{
  return _CCCL_FP_CUSTOM_FP128_SQRT(__x);
}

[[nodiscard]] _CCCL_TRIVIAL_HOST_API __fp_custom_fp128
__fp_custom_fma(__fp_custom_fp128 __x, __fp_custom_fp128 __y, __fp_custom_fp128 __z) noexcept
{
  return _CCCL_FP_CUSTOM_FP128_FMA(__x, __y, __z);
}
#  endif // _CCCL_FP_CUSTOM_FP128_MATH == 1
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1

// === main class definition ===
//! @brief Floating-point type with a configurable exponent and mantissa size
//!
//! This class template provides a drop-in replacement for its base type that reduces the
//! precision of every arithmetic operation to the requested field sizes. It stores values in
//! the base type's standard IEEE 754 format but can simulate lower precisions.
//!
//! ## Features
//! - Implicit conversion from all numeric types
//! - Full operator overloading (+, -, *, /, comparisons)
//! - CUDA host/device compatibility
//! - Zero overhead at native sizes, where the emulation is compiled out
//!
//! ## Memory Layout
//! - Size and alignment: the base type's, the requested sizes costing nothing
//! - Stores raw IEEE 754 bit pattern
//!
//! ## Usage
//! ```cpp
//! using Real = cuda::experimental::fp64_custom<>; // or double for production
//! Real x = 1.5, y = 2.5;
//! Real result = x + y;
//! ```
//!
//! @tparam _FpType Base type the value is held and computed in, one of the supported base
//! types; see the static_assert below for the current set
//! @tparam _ExpSize Exponent bits to preserve, or `fp_custom_dynamic_size`
//! @tparam _MantSize Mantissa bits to preserve, or `fp_custom_dynamic_size`
//!
//! @note The class is trivially copyable and can be used in CUDA kernels
//! @note Instantiations with different sizes or base types are distinct types and do not
//! mix in arithmetic; convert through a base type to combine them.
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
class fp_custom
{
  static_assert(__fp_custom_is_supported_fp_v<_FpType>,
                "cuda::experimental::fp_custom supports _FpType == double (or the bit-identical _Float64), "
                "_FpType == float, and _FpType == __fp_custom_fp128 where the platform has a binary128 type, "
                "possible future extension to other base types");
  // An n-bit exponent field reserves the all-ones pattern for infinity and NaN, so it
  // covers 2^n - 2 binades: at least two bits are needed for a single usable one.
  static_assert(!__fp_custom_is_supported_fp_v<_FpType> || _ExpSize == fp_custom_dynamic_size
                  || (_ExpSize >= 2 && _ExpSize <= __fp_custom_native_layout<_FpType>::__exp_size),
                "cuda::experimental::fp_custom exponent size must be between 2 and the exponent size of the base "
                "type, or fp_custom_dynamic_size");
  // Zero mantissa bits leave only the implicit leading 1, which is a valid request: it
  // rounds every value to the nearest power of two.
  static_assert(!__fp_custom_is_supported_fp_v<_FpType> || _MantSize == fp_custom_dynamic_size
                  || _MantSize <= __fp_custom_native_layout<_FpType>::__mant_size,
                "cuda::experimental::fp_custom mantissa size must not exceed the mantissa size of the base type, "
                "unless it is fp_custom_dynamic_size");

  // The type the value is computed in, and the unsigned integer it is stored as, both taken
  // from the base type's native layout; see there for why the first can differ from _FpType.
  using __value_type   = __fp_custom_fp_t<_FpType>;
  using __storage_type = __fp_custom_bits_t<_FpType>;

public:
  // === constructors ===
  //! @brief Default constructor: initializes to zero
  _CCCL_HOST_DEVICE_API constexpr fp_custom() noexcept
      : __bits_{0u}
  {}

  //! @brief Copy constructor, defaulted so the type stays trivially copyable
  //! @note NVCC implicitly makes defaulted special members __host__ __device__
  _CCCL_HIDE_FROM_ABI fp_custom(const fp_custom&) = default;

  // Volatile support: the constructor and assignment operators below cover storage
  // only, i.e. load, store and a bit-preserving round-trip, which is what the legacy
  // pattern of keeping shared-memory scalars in volatile variables needs.
  //
  // A volatile object cannot be an operand of arithmetic or comparison: those take
  // const fp_custom&, and a volatile lvalue never binds to it, not even through the
  // constructor below, because reference-related types are required to bind directly.
  // Copy into a non-volatile local, compute there, store the result back. bit_cast is
  // the exception, deducing volatile fp_custom and performing a volatile load.
  //
  // Each volatile overload is wrapped in a dummy template so that the C++ standard
  // does not consider it a copy constructor or copy assignment operator (a template
  // never is), which preserves trivial copyability.

  //! @brief Copy constructor from volatile
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API fp_custom(const volatile fp_custom& __other) noexcept
      : __bits_{__other.__bits_}
  {}

  /*
  // A conversion is implicit where the requested format is at least as wide as the type on
  // the other side, and explicit where it is narrower, in both directions: a format that
  // matches a source exactly takes it implicitly and behaves like it in every respect, while
  // one narrower than a source marks every crossing of its boundary. Which base type holds
  // the value does not enter into it, only the format asked for. The conversions out do the
  // same, see operator float() below.
  //
  // What an explicit constructor here reports is the format the value is entering, not a loss
  // in the constructor itself: the value is stored in the base type unreduced, and the sizes
  // are applied by the first arithmetic operation. So a request for 8 and 23 bits keeps 1e300
  // until it is used, and yields infinity once it is - which is the surprise the cast marks.
  // The base type is the one boundary that does bind here, a source wider than it being
  // rounded to it on the way in.
  //
  // A dynamic size is explicit for every source, since which format the value enters is not
  // known at compile time.
  //
  // CCCL_FP_CUSTOM_EXPLICIT_CASTS = 0 drops the specifier from the narrowing side, for
  // adopting the type across a codebase written against double. The pairs stay as they are;
  // only the keyword goes, leaving both sides implicit. See the macro for what that gives up.
  //
  // Written as constrained pairs because the specifier cannot depend on the class parameters
  // before C++20. The condition runs through _Up to stay dependent, as the conversion
  // operators do.
  */

  //! @brief Construct from double, implicit where the format holds every double
  //!
  //! Deduced and constrained rather than named double: see __fp_custom_is_double_src_v for
  //! which sources have to be kept off this constructor, and why.
  _CCCL_TEMPLATE(class _Tp, typename _Up = _FpType)
  _CCCL_REQUIRES(__fp_custom_is_double_src_v<_Tp> _CCCL_AND __fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>)
  _CCCL_HOST_DEVICE_API fp_custom(_Tp __d) noexcept
      : __bits_{::cuda::std::bit_cast<__storage_type>(static_cast<__value_type>(__d))}
  {}

  //! @brief Construct from double into a narrower or dynamic format, explicit
  _CCCL_TEMPLATE(class _Tp, typename _Up = _FpType)
  _CCCL_REQUIRES(__fp_custom_is_double_src_v<_Tp> _CCCL_AND(!__fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>))
  _CCCL_HOST_DEVICE_API _CCCL_FP_CUSTOM_EXPLICIT fp_custom(_Tp __d) noexcept
      : __bits_{::cuda::std::bit_cast<__storage_type>(static_cast<__value_type>(__d))}
  {}

  //! @brief Construct from float, implicit where the format holds every float
  //!
  //! The parameter is deduced and constrained to float rather than named float, which is what
  //! keeps it from accepting anything else: deduction runs before conversions, so a double
  //! argument deduces double, fails the constraint, and reaches the constructor above. Named
  //! `float` would take a double through the standard conversion, forcing it through
  //! binary32's exponent range even where the requested format is wider.
  _CCCL_TEMPLATE(class _Tp, typename _Up = _FpType)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp, float> _CCCL_AND __fp_custom_holds_float_v<_Up, _ExpSize, _MantSize>)
  _CCCL_HOST_DEVICE_API fp_custom(_Tp __f) noexcept
      : __bits_{::cuda::std::bit_cast<__storage_type>(static_cast<__value_type>(__f))}
  {}

  //! @brief Construct from float into a narrower or dynamic format, explicit
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES((!__fp_custom_holds_float_v<_Up, _ExpSize, _MantSize>) )
  _CCCL_HOST_DEVICE_API _CCCL_FP_CUSTOM_EXPLICIT fp_custom(float __f) noexcept
      : __bits_{::cuda::std::bit_cast<__storage_type>(static_cast<__value_type>(__f))}
  {}

  //! @brief Construct from any standard integer type (int / long / long long + unsigned),
  //! implicit where the format holds every double
  //!
  //! Routes through the base type, so every width/signedness is handled uniformly and
  //! portably (LP64 and LLP64), and follows the double constructor's explicitness, integers
  //! having double's conversion rank. Wide values may lose precision on the way, as they do
  //! reaching the base type itself. Excludes bool / character types, which the constructors
  //! below cover.
  _CCCL_TEMPLATE(class _Tp, typename _Up = _FpType)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>)
  _CCCL_HOST_DEVICE_API fp_custom(_Tp __i) noexcept
      : __bits_{::cuda::std::bit_cast<__storage_type>(static_cast<__value_type>(__i))}
  {}

  //! @brief Construct from a standard integer type into a narrower or dynamic format, explicit
  _CCCL_TEMPLATE(class _Tp, typename _Up = _FpType)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND(!__fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>))
  _CCCL_HOST_DEVICE_API _CCCL_FP_CUSTOM_EXPLICIT fp_custom(_Tp __i) noexcept
      : __bits_{::cuda::std::bit_cast<__storage_type>(static_cast<__value_type>(__i))}
  {}

  //! @brief Construct from bool or a character type, implicit where the format holds every double
  //!
  //! Excluded from __cccl_is_integer_v, but `1.0 + true` and `1.0 + 'a'` are valid for
  //! double, so mirror that behavior by widening to int and reusing the path above.
  _CCCL_TEMPLATE(class _Tp, typename _Up = _FpType)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND(!::cuda::std::__cccl_is_integer_v<_Tp>)
                   _CCCL_AND __fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>)
  _CCCL_HOST_DEVICE_API fp_custom(_Tp __i) noexcept
      : fp_custom(static_cast<int32_t>(__i))
  {}

  //! @brief Construct from bool or a character type into a narrower or dynamic format, explicit
  _CCCL_TEMPLATE(class _Tp, typename _Up = _FpType)
  _CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp> _CCCL_AND(!::cuda::std::__cccl_is_integer_v<_Tp>)
                   _CCCL_AND(!__fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>))
  _CCCL_HOST_DEVICE_API _CCCL_FP_CUSTOM_EXPLICIT fp_custom(_Tp __i) noexcept
      : fp_custom(static_cast<int32_t>(__i))
  {}

#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1)
  //! @brief Construct from the platform's binary128 type, implicit where the format holds
  //! every binary128 value
  //!
  //! Both deduce and constrain the parameter, for the reason the float constructor above does
  //! and one more: where the binary128 type is `__float128`, a `long double` converts to it and
  //! to double equally well, so a named parameter would make every such argument ambiguous
  //! between the two constructors.
  _CCCL_TEMPLATE(class _Tp, typename _Up = _FpType)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp, __fp_custom_fp128>
                   _CCCL_AND __fp_custom_holds_fp128_v<_Up, _ExpSize, _MantSize>)
  _CCCL_FP_CUSTOM_FP128_API fp_custom(_Tp __q) noexcept
      : __bits_{::cuda::std::bit_cast<__storage_type>(static_cast<__value_type>(__q))}
  {}

  //! @brief Construct from binary128 into a narrower or dynamic format, explicit
  _CCCL_TEMPLATE(class _Tp, typename _Up = _FpType)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp, __fp_custom_fp128>
                   _CCCL_AND(!__fp_custom_holds_fp128_v<_Up, _ExpSize, _MantSize>))
  _CCCL_FP_CUSTOM_FP128_API _CCCL_FP_CUSTOM_EXPLICIT fp_custom(_Tp __q) noexcept
      : __bits_{::cuda::std::bit_cast<__storage_type>(static_cast<__value_type>(__q))}
  {}
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1

  /*
  // The types too wide to arrive through the base type are deleted rather than left absent,
  // so that the diagnostic names the rule. Each mirrors the explicitness of the source it
  // would otherwise travel through, which keeps the copy-initialization candidate set as it
  // was: were they left implicit while the constructors above are explicit, a narrow format
  // would report `T x = 1.0` as an ambiguity between the two deleted 128-bit integers instead
  // of as a conversion that has to be spelled.
  */
#if _CCCL_HAS_INT128()
  //! @brief 128-bit integers are deleted: no base type has the 128 mantissa bits to hold one,
  //! so they would silently truncate
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>)
  _CCCL_HOST_DEVICE_API fp_custom(__int128_t) = delete;
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>)
  _CCCL_HOST_DEVICE_API fp_custom(__uint128_t) = delete;

  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES((!__fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>) )
  _CCCL_HOST_DEVICE_API _CCCL_FP_CUSTOM_EXPLICIT fp_custom(__int128_t) = delete;
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES((!__fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>) )
  _CCCL_HOST_DEVICE_API _CCCL_FP_CUSTOM_EXPLICIT fp_custom(__uint128_t) = delete;
#endif // _CCCL_HAS_INT128()
#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 0) && _CCCL_HAS_FLOAT128()
  //! @brief With no binary128 base type to receive it, __float128 is deleted: it would
  //! silently lose precision through double
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>)
  _CCCL_HOST_DEVICE_API fp_custom(__float128) = delete;
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES((!__fp_custom_holds_double_v<_Up, _ExpSize, _MantSize>) )
  _CCCL_HOST_DEVICE_API _CCCL_FP_CUSTOM_EXPLICIT fp_custom(__float128) = delete;
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 0 && _CCCL_HAS_FLOAT128()

  // === assignment ===
  //! @brief Copy assignment, defaulted so the type stays trivially copyable
  _CCCL_HIDE_FROM_ABI fp_custom& operator=(const fp_custom&) = default;

  //! @brief Assignment to volatile
  //! @note Returns void to avoid the C++20 deprecation of a volatile return type
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API void operator=(const fp_custom& __other) volatile noexcept
  {
    __bits_ = __other.__bits_;
  }

  //! @brief Assignment from volatile
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API fp_custom& operator=(const volatile fp_custom& __other) noexcept
  {
    __bits_ = __other.__bits_;
    return *this;
  }

  //! @brief Assignment from volatile to volatile, e.g. a shared-memory to
  //! shared-memory copy
  template <typename _Dummy = void>
  _CCCL_HOST_DEVICE_API void operator=(const volatile fp_custom& __other) volatile noexcept
  {
    __bits_ = __other.__bits_;
  }

  // === conversions ===
  //! @brief Convert to double (implicit)
  //!
  //! The only conversion out that is implicit and not a template, so it reaches a sink of any
  //! arithmetic type through the standard conversion that follows, which is what keeps a native
  //! format a drop-in. Over a wider base it rounds; the binary128 one below is the exact way out.
  _CCCL_HOST_DEVICE_API operator double() const noexcept
  {
    return static_cast<double>(::cuda::std::bit_cast<__value_type>(__bits_));
  }

#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1)
  //! @brief Convert to the platform's binary128 type, which holds every format and so never
  //! rounds (explicit)
  //!
  //! Explicit is where the rank rule gives way to overload resolution: a second implicit
  //! conversion function to a floating-point type would make a float or long double sink
  //! ambiguous against operator double(), each needing a standard conversion to reach it.
  _CCCL_FP_CUSTOM_FP128_API explicit operator __fp_custom_fp128() const noexcept
  {
    return static_cast<__fp_custom_fp128>(::cuda::std::bit_cast<__value_type>(__bits_));
  }
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1

  /*
  // Conversion to float is implicit where float holds the requested format, explicit where the
  // mantissa bits or the exponent range would be lost. The specifier cannot depend on the class
  // parameters before C++20, hence the constrained templates, whose condition runs through _Up
  // to stay dependent. Implicit, it leaves an overload set holding float and double ambiguous.
  */
  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES(__fp_custom_fits_in_float_v<_Up, _ExpSize, _MantSize>)
  _CCCL_HOST_DEVICE_API operator float() const noexcept
  {
    return static_cast<float>(::cuda::std::bit_cast<__value_type>(__bits_));
  }

  _CCCL_TEMPLATE(typename _Up = _FpType)
  _CCCL_REQUIRES((!__fp_custom_fits_in_float_v<_Up, _ExpSize, _MantSize>) )
  _CCCL_HOST_DEVICE_API explicit operator float() const noexcept
  {
    return static_cast<float>(::cuda::std::bit_cast<__value_type>(__bits_));
  }

  //! @brief Convert to any standard integer type (explicit, truncates toward zero).
  //! Covers int / long / long long + unsigned uniformly; excludes bool / char.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  _CCCL_HOST_DEVICE_API explicit operator _Tp() const noexcept
  {
    return static_cast<_Tp>(::cuda::std::bit_cast<__value_type>(__bits_));
  }

#if _CCCL_HAS_INT128()
  //! @brief See the deleted 128-bit constructors: avoid silent 64-bit truncation
  _CCCL_HOST_DEVICE_API explicit operator __int128_t() const  = delete;
  _CCCL_HOST_DEVICE_API explicit operator __uint128_t() const = delete;
#endif // _CCCL_HAS_INT128()

  // === arithmetic, with precision reduction ===
  //
  // Each operator reduces its operands, calls the base type's round-to-nearest-even
  // primitive, and reduces the result.

  //! @brief Addition with precision reduction
  //!
  //! Operation flow:
  //! 1. Reduce both operands
  //! 2. Perform the native addition on the base type
  //! 3. Reduce the result
  _CCCL_HOST_DEVICE_API fp_custom operator+(const fp_custom& __y) const noexcept
  {
    __storage_type __a = __bits_, __b = __y.__bits_;
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
    const __value_type __sum = ::cuda::experimental::__fp_custom_add(
      ::cuda::std::bit_cast<__value_type>(__a), ::cuda::std::bit_cast<__value_type>(__b));
    __storage_type __r = ::cuda::std::bit_cast<__storage_type>(__sum);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
    return fp_custom(::cuda::std::bit_cast<__value_type>(__r));
  }

  //! @brief Subtraction with precision reduction
  _CCCL_HOST_DEVICE_API fp_custom operator-(const fp_custom& __y) const noexcept
  {
    __storage_type __a = __bits_, __b = __y.__bits_;
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
    const __value_type __diff = ::cuda::experimental::__fp_custom_sub(
      ::cuda::std::bit_cast<__value_type>(__a), ::cuda::std::bit_cast<__value_type>(__b));
    __storage_type __r = ::cuda::std::bit_cast<__storage_type>(__diff);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
    return fp_custom(::cuda::std::bit_cast<__value_type>(__r));
  }

  //! @brief Multiplication with precision reduction
  _CCCL_HOST_DEVICE_API fp_custom operator*(const fp_custom& __y) const noexcept
  {
    __storage_type __a = __bits_, __b = __y.__bits_;
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
    const __value_type __prod = ::cuda::experimental::__fp_custom_mul(
      ::cuda::std::bit_cast<__value_type>(__a), ::cuda::std::bit_cast<__value_type>(__b));
    __storage_type __r = ::cuda::std::bit_cast<__storage_type>(__prod);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
    return fp_custom(::cuda::std::bit_cast<__value_type>(__r));
  }

  //! @brief Division with precision reduction
  _CCCL_HOST_DEVICE_API fp_custom operator/(const fp_custom& __y) const noexcept
  {
    __storage_type __a = __bits_, __b = __y.__bits_;
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
    const __value_type __quot = ::cuda::experimental::__fp_custom_div(
      ::cuda::std::bit_cast<__value_type>(__a), ::cuda::std::bit_cast<__value_type>(__b));
    __storage_type __r = ::cuda::std::bit_cast<__storage_type>(__quot);
    __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
    return fp_custom(::cuda::std::bit_cast<__value_type>(__r));
  }

  //! @brief Unary negation (sign flip)
  //! @note No precision reduction - just flips the sign bit
  _CCCL_HOST_DEVICE_API fp_custom operator-() const noexcept
  {
    constexpr __storage_type __sign_mask = __fp_custom_native_layout<_FpType>::__sign_mask;
    return fp_custom(::cuda::std::bit_cast<__value_type>(static_cast<__storage_type>(__bits_ ^ __sign_mask)));
  }

  // === mixed-type arithmetic ===
  //
  // Hidden friends taking one fp_custom and one arithmetic operand, in either order,
  // so that expressions like `x + 2.0` and `3 * x` work. Instantiations with
  // different sizes are deliberately not accepted here.

  //! @brief Mixed-type addition
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend fp_custom operator+(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) + __as_fp_custom(__y);
  }

  //! @brief Mixed-type subtraction
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend fp_custom operator-(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) - __as_fp_custom(__y);
  }

  //! @brief Mixed-type multiplication
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend fp_custom operator*(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) * __as_fp_custom(__y);
  }

  //! @brief Mixed-type division
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend fp_custom operator/(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) / __as_fp_custom(__y);
  }

  // === compound assignment ===
  //! @brief Add and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator+=(const fp_custom& __other) noexcept
  {
    *this = *this + __other;
    return *this;
  }

  //! @brief Subtract and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator-=(const fp_custom& __other) noexcept
  {
    *this = *this - __other;
    return *this;
  }

  //! @brief Multiply and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator*=(const fp_custom& __other) noexcept
  {
    *this = *this * __other;
    return *this;
  }

  //! @brief Divide and assign
  _CCCL_HOST_DEVICE_API fp_custom& operator/=(const fp_custom& __other) noexcept
  {
    *this = *this / __other;
    return *this;
  }

  // === increment and decrement ===
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
    fp_custom __temp(*this);
    ++(*this);
    return __temp;
  }

  //! @brief Post-decrement
  _CCCL_HOST_DEVICE_API fp_custom operator--(int) noexcept
  {
    fp_custom __temp(*this);
    --(*this);
    return __temp;
  }

  // === comparison ===
  //! @brief Equality comparison
  _CCCL_HOST_DEVICE_API bool operator==(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<__value_type>(__bits_) == ::cuda::std::bit_cast<__value_type>(__y.__bits_);
  }

  //! @brief Inequality comparison
  _CCCL_HOST_DEVICE_API bool operator!=(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<__value_type>(__bits_) != ::cuda::std::bit_cast<__value_type>(__y.__bits_);
  }

  //! @brief Less than comparison
  _CCCL_HOST_DEVICE_API bool operator<(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<__value_type>(__bits_) < ::cuda::std::bit_cast<__value_type>(__y.__bits_);
  }

  //! @brief Greater than comparison
  _CCCL_HOST_DEVICE_API bool operator>(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<__value_type>(__bits_) > ::cuda::std::bit_cast<__value_type>(__y.__bits_);
  }

  //! @brief Less than or equal comparison
  _CCCL_HOST_DEVICE_API bool operator<=(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<__value_type>(__bits_) <= ::cuda::std::bit_cast<__value_type>(__y.__bits_);
  }

  //! @brief Greater than or equal comparison
  _CCCL_HOST_DEVICE_API bool operator>=(const fp_custom& __y) const noexcept
  {
    return ::cuda::std::bit_cast<__value_type>(__bits_) >= ::cuda::std::bit_cast<__value_type>(__y.__bits_);
  }

  //! @name Mixed-Type Comparisons
  //! @{
  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator==(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) == __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator!=(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) != __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator<(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) < __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator>(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) > __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator<=(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) <= __as_fp_custom(__y);
  }

  _CCCL_TEMPLATE(typename _T1, typename _T2)
  _CCCL_REQUIRES(((::cuda::std::is_same_v<_T1, fp_custom> || ::cuda::std::is_same_v<_T2, fp_custom>)
                  && (::cuda::std::is_arithmetic_v<_T1> || ::cuda::std::is_arithmetic_v<_T2>) ))
  _CCCL_HOST_DEVICE_API friend bool operator>=(const _T1& __x, const _T2& __y) noexcept
  {
    return __as_fp_custom(__x) >= __as_fp_custom(__y);
  }
  //! @}

private:
  //! @brief Bring either an fp_custom or an arithmetic operand to fp_custom
  //!
  //! An fp_custom operand is passed through untouched rather than round-tripped
  //! through the base type, so mixed-type operators cannot perturb the bit pattern.
  _CCCL_TEMPLATE(typename _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Tp, fp_custom>)
  [[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API static const fp_custom& __as_fp_custom(const _Tp& __x) noexcept
  {
    return __x;
  }

  _CCCL_TEMPLATE(typename _Tp)
  _CCCL_REQUIRES(::cuda::std::is_arithmetic_v<_Tp>)
  [[nodiscard]] _CCCL_TRIVIAL_HOST_DEVICE_API static fp_custom __as_fp_custom(const _Tp& __x) noexcept
  {
    return fp_custom(static_cast<__value_type>(__x));
  }

  //! @brief Raw IEEE 754 bit representation of the value
  //!
  //! Private, with no raw-bits accessor and no raw-bits constructor: the value is
  //! bit-identical to the base type, so `bit_cast` through that type is the way to
  //! reinterpret a bit pattern as an fp_custom and back.
  __storage_type __bits_;
};

// === math functions ===
//! @brief Square root with precision reduction
//!
//! @param __x Input value
//! @return Square root of x, with the operand and result reduced
//!
//! @note Uses the base type's round-to-nearest square root intrinsic on CUDA,
//! cuda::std::sqrt on host. Over a binary128 base it is host-only, and absent where
//! _CCCL_FP_CUSTOM_FP128_MATH is 0.
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fp_custom<_FpType, _ExpSize, _MantSize>
sqrt(const fp_custom<_FpType, _ExpSize, _MantSize>& __x) noexcept
{
  using _Fp   = __fp_custom_fp_t<_FpType>;
  using _Bits = __fp_custom_bits_t<_FpType>;

  _Bits __a = ::cuda::std::bit_cast<_Bits>(static_cast<_Fp>(__x));
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
  const _Fp __root = ::cuda::experimental::__fp_custom_sqrt(::cuda::std::bit_cast<_Fp>(__a));
  _Bits __r        = ::cuda::std::bit_cast<_Bits>(__root);
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
  return fp_custom<_FpType, _ExpSize, _MantSize>(::cuda::std::bit_cast<_Fp>(__r));
}

//! @brief Fused multiply-add with precision reduction
//!
//! Computes (x * y) + z with a single rounding operation.
//!
//! @param __x First multiplicand
//! @param __y Second multiplicand
//! @param __z Addend
//! @return (x * y) + z, with all operands and the result reduced
//!
//! @note Uses the base type's round-to-nearest FMA intrinsic on CUDA, cuda::std::fma on host,
//! with the same binary128 restrictions as sqrt above.
template <typename _FpType, uint16_t _ExpSize, uint16_t _MantSize>
[[nodiscard]] _CCCL_HOST_DEVICE_API inline fp_custom<_FpType, _ExpSize, _MantSize>
fma(const fp_custom<_FpType, _ExpSize, _MantSize>& __x,
    const fp_custom<_FpType, _ExpSize, _MantSize>& __y,
    const fp_custom<_FpType, _ExpSize, _MantSize>& __z) noexcept
{
  using _Fp   = __fp_custom_fp_t<_FpType>;
  using _Bits = __fp_custom_bits_t<_FpType>;

  _Bits __a = ::cuda::std::bit_cast<_Bits>(static_cast<_Fp>(__x));
  _Bits __b = ::cuda::std::bit_cast<_Bits>(static_cast<_Fp>(__y));
  _Bits __c = ::cuda::std::bit_cast<_Bits>(static_cast<_Fp>(__z));
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__a);
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__b);
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__c);
  const _Fp __muladd = ::cuda::experimental::__fp_custom_fma(
    ::cuda::std::bit_cast<_Fp>(__a), ::cuda::std::bit_cast<_Fp>(__b), ::cuda::std::bit_cast<_Fp>(__c));
  _Bits __r = ::cuda::std::bit_cast<_Bits>(__muladd);
  __fp_custom_reduce<_FpType, _ExpSize, _MantSize>(__r);
  return fp_custom<_FpType, _ExpSize, _MantSize>(::cuda::std::bit_cast<_Fp>(__r));
}

// Trait machinery for the mixed-operand fma below, on top of the __is_fp_custom_v declared
// with the constructor traits above. __has_fp_custom_v is the constraint "at least one operand
// is an fp_custom", which leaves a pure-arithmetic call to the built-in types and, by
// partial ordering, a pure fp_custom call to the more specialized exact-type overload
// above; __fp_custom_pick_t selects the fp_custom type among a set of operands.
//
// The constraint deliberately admits operands of two different fp_custom
// instantiations, which sizes never mix implicitly. Rejecting them here would only
// send the call to ::fma(double, double, double) through the implicit conversion and
// compute at full FP64 precision without a word, so the mix is diagnosed by the
// static_assert in the body instead.
template <class... _Ts>
inline constexpr bool __has_fp_custom_v = (__is_fp_custom_v<_Ts> || ...);

template <class... _Ts>
struct __fp_custom_pick
{
  using type = void;
};
template <class _T0, class... _Ts>
struct __fp_custom_pick<_T0, _Ts...>
{
  using type = ::cuda::std::conditional_t<__is_fp_custom_v<_T0>, _T0, typename __fp_custom_pick<_Ts...>::type>;
};
template <class... _Ts>
using __fp_custom_pick_t = typename __fp_custom_pick<_Ts...>::type;

template <class _Tp, class... _Ts>
inline constexpr bool __fp_custom_same_or_arithmetic_v =
  ((::cuda::std::is_same_v<_Ts, _Tp> || ::cuda::std::is_arithmetic_v<_Ts>) && ...);

//! @brief Fused multiply-add mixing fp_custom with arithmetic operands
//!
//! Without this overload such a call would resolve to `::fma(double, double, double)`
//! through the implicit conversion and quietly compute at full FP64 precision.
_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(__has_fp_custom_v<_T1, _T2, _T3>)
[[nodiscard]] _CCCL_HOST_DEVICE_API inline __fp_custom_pick_t<_T1, _T2, _T3>
fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  using _Tp = __fp_custom_pick_t<_T1, _T2, _T3>;
  static_assert(__fp_custom_same_or_arithmetic_v<_Tp, _T1, _T2, _T3>,
                "fma operands mix two different fp_custom instantiations: exponent and mantissa sizes never "
                "convert implicitly, so convert explicitly to the intended type first");
  return fma(_Tp(__x), _Tp(__y), _Tp(__z));
}

// === type aliases ===
//! @brief fp_custom over double, with the native FP64 field sizes as defaults
//!
//! The type to reach for when emulating a format inside FP64: `fp64_custom<>` is a
//! drop-in for `double` with the precision reduction compiled out entirely, and the
//! sizes of a narrower format are given as `fp64_custom<8, 23>`, or left to runtime
//! with `fp_custom_dynamic_size`.
template <uint16_t _ExpSize = 11, uint16_t _MantSize = 52>
using fp64_custom = fp_custom<double, _ExpSize, _MantSize>;

//! @brief fp_custom over float, with the native FP32 field sizes as defaults
//!
//! The type to reach for when emulating a format inside FP32: `fp32_custom<>` is a
//! drop-in for `float` with the precision reduction compiled out entirely, and the
//! sizes of a narrower format are given as `fp32_custom<8, 10>`, or left to runtime
//! with `fp_custom_dynamic_size`.
template <uint16_t _ExpSize = 8, uint16_t _MantSize = 23>
using fp32_custom = fp_custom<float, _ExpSize, _MantSize>;

#if (_CCCL_FP_CUSTOM_FP128_ENABLE == 1)
//! @brief fp_custom over the platform's binary128 type, with the native FP128 field sizes as
//! defaults
//!
//! The type to reach for when emulating a format inside FP128, FP64 included: `fp128_custom<>`
//! is a drop-in for the base type with the precision reduction compiled out entirely, and
//! the sizes of a narrower format are given as `fp128_custom<11, 52>`, or left to runtime
//! with `fp_custom_dynamic_size`.
template <uint16_t _ExpSize = 15, uint16_t _MantSize = 112>
using fp128_custom = fp_custom<__fp_custom_fp128, _ExpSize, _MantSize>;
#endif // _CCCL_FP_CUSTOM_FP128_ENABLE == 1
} // namespace cuda::experimental

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Overloads of sqrt and fma for fp_custom so the standard spelling cuda::std::sqrt /
// cuda::std::fma selects the reducing implementation. A qualified call suppresses ADL,
// so without these it would silently narrow fp_custom -> double through the implicit
// conversion and compute at full FP64 precision, which is exactly what a precision
// study must not do. These forward to cuda::experimental::sqrt / fma, which unqualified
// and ADL calls already resolve to. The exact-type fma overload wins for pure fp_custom
// calls by partial ordering, while the constrained one handles the mixed case.
template <class _FpType, uint16_t _ExpSize, uint16_t _MantSize>
[[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>
sqrt(const ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>& __x) noexcept
{
  return ::cuda::experimental::sqrt(__x);
}

template <class _FpType, uint16_t _ExpSize, uint16_t _MantSize>
[[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>
fma(const ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>& __x,
    const ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>& __y,
    const ::cuda::experimental::fp_custom<_FpType, _ExpSize, _MantSize>& __z) noexcept
{
  return ::cuda::experimental::fma(__x, __y, __z);
}

_CCCL_TEMPLATE(class _T1, class _T2, class _T3)
_CCCL_REQUIRES(::cuda::experimental::__has_fp_custom_v<_T1, _T2, _T3>)
[[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::experimental::__fp_custom_pick_t<_T1, _T2, _T3>
fma(const _T1& __x, const _T2& __y, const _T3& __z) noexcept
{
  return ::cuda::experimental::fma(__x, __y, __z);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPTOOL_CUSTOM_H
