/*
    ts_accuracy.hpp - FPMP Test Suite Accuracy Testing
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Accuracy testing utilities for verifying correctness of multi-precision floating-point operations
    against quad-precision references. Supports configurable rigor levels and multiple test distributions.
*/

#ifndef __TS_ACCURACY_HPP__
#define __TS_ACCURACY_HPP__

// ts_types.hpp includes ts.hpp and defines all common types
#include "ts_types.hpp"
#include "ts_functions.hpp"
#include "ts_dataset.hpp"
#include "ts_print.hpp"
#include <random>
#include <cmath>
#include <cassert>

#if !defined(__CUDACC__)
#include <omp.h>
#else
#include <curand_kernel.h>
#endif

#ifndef TS_FTZ_DAZ_MODE
    #define TS_FTZ_DAZ_MODE (1)
#endif

// Import commonly used items from ts namespace for cleaner code
using ts::bit_cast;
using ts::is_multiprecision_v;
using ts::mp_component_t;
using ts::is_value_finite;
using ts::get_max_mantissa_bits;
using ts::fpmp2_method_v;
using ts::string_to_value;
#if defined(__CUDACC__)
using ts::atomicMaxDouble;
using ts::atomicAddDouble;
#endif

// ============================================================================
// Helper: Zero out the lo part of a multi-precision value
// For scalar types, returns the value unchanged
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ inline T zero_lo_part(T val)
{
    if constexpr (is_multiprecision_v<T>)
    {
        return T(val.hi(), static_cast<decltype(val.lo())>(0));
    }
    else
    {
        return val;
    }
}

// ============================================================================
// Trait detector: check if a function tag has zero_lo_args trait
// ============================================================================
template<typename Tag, typename = void>
struct has_zero_lo_args : std::false_type {};

template<typename Tag>
struct has_zero_lo_args<Tag, std::void_t<decltype(Tag::zero_lo_args)>> : std::true_type {};

template<typename Tag>
__HOST_DEVICE_DECL__ constexpr unsigned get_zero_lo_args() {
    if constexpr (has_zero_lo_args<Tag>::value) {
        return Tag::zero_lo_args;
    } else {
        return 0;
    }
}

// ============================================================================
// Trait detector: check if a function tag has valid_input_range trait
// ============================================================================
template<typename Tag, typename = void>
struct has_valid_input_range : std::false_type {};

template<typename Tag>
struct has_valid_input_range<Tag, std::void_t<decltype(Tag::has_valid_input_range())>> : std::true_type {};

template<typename Tag, typename T>
__HOST_DEVICE_DECL__ bool is_input_in_valid_range(T val) {
    if constexpr (has_valid_input_range<Tag>::value) {
        double d = static_cast<double>(val);
        return d >= Tag::valid_input_min() && d <= Tag::valid_input_max();
    } else {
        return true;  // No range restriction
    }
}

// ============================================================================
// Helper: Convert a value to fpmp_type for storage in error records
// For 64-bit integers, uses raw bit storage to preserve all bits exactly.
// For 32-bit integers, uses direct double conversion (fits in precision).
// For fp32mp2: the two floats can store 64 bits as raw bit patterns.
// For fp64mp2: the hi double can store 64 bits as a raw bit pattern.
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ fpmp_type to_fpmp_for_storage(T val) {
    using ComponentType = mp_component_t<fpmp_type>;
    
    if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) {
        // For 64-bit integers, store as raw bits to preserve all 64 bits exactly
        // This works for both fp32mp2 (2 floats = 64 bits) and fp64mp2 (use hi only)
        uint64_t bits;
        if constexpr (std::is_same_v<T, int64_t>) {
            bits = static_cast<uint64_t>(val);
        } else {
            bits = val;
        }
        
        if constexpr (std::is_same_v<ComponentType, float>) {
            // fp32mp2: split 64 bits into two 32-bit floats (as raw bits)
            uint32_t lo32 = static_cast<uint32_t>(bits & 0xFFFFFFFFULL);
            uint32_t hi32 = static_cast<uint32_t>(bits >> 32);
            float f_lo, f_hi;
            memcpy(&f_lo, &lo32, sizeof(uint32_t));
            memcpy(&f_hi, &hi32, sizeof(uint32_t));
            return fpmp_type(f_hi, f_lo);  // hi stores upper bits, lo stores lower bits
        } else {
            // fp64mp2: store entire 64 bits in hi (as raw bits)
            double d;
            memcpy(&d, &bits, sizeof(uint64_t));
            return fpmp_type(d, 0.0);
        }
    } else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
        // For 32-bit integers, store as raw bits (fp32mp2 hi component is float, which loses precision)
        if constexpr (std::is_same_v<ComponentType, float>) {
            // fp32mp2: store 32 bits as raw float in hi, lo is zero marker
            uint32_t bits;
            if constexpr (std::is_same_v<T, int32_t>) {
                bits = static_cast<uint32_t>(val);
            } else {
                bits = val;
            }
            float f;
            memcpy(&f, &bits, sizeof(uint32_t));
            // Use a special marker in lo to indicate 32-bit storage (NaN)
            uint32_t marker = 0x7FC00000U;  // Quiet NaN marker
            float f_marker;
            memcpy(&f_marker, &marker, sizeof(uint32_t));
            return fpmp_type(f, f_marker);
        } else {
            // fp64mp2: hi is double, which can store 32-bit integers exactly
            return fpmp_type(static_cast<double>(val), 0.0);
        }
    } else if constexpr (std::is_integral<T>::value) {
        // For other integers, convert via double
        return fpmp_type(static_cast<double>(val));
    } else if constexpr (std::is_same_v<T, bool>) {
        // For booleans, convert to double (0.0 or 1.0)
        return fpmp_type(val ? 1.0 : 0.0);
    } else {
        // For floating-point types, direct conversion.
        //
        // Bridge __ts_fp128 -> __fpmp_fp128 explicitly: on ARM64 device pass
        // these two typedefs resolve to distinct binary128 types (long double
        // vs __float128) with identical bit layout but different C++ identity,
        // so a direct fpmp_type(__ts_fp128) call would be ambiguous against
        // the (double)/(int32_t)/(int64_t)/(uint*_t) constructors. The cast
        // is a no-op on x86_64 and on ARM64 host where both typedefs match.
    #if (FPMP_FP128_ENABLE == 1)
        if constexpr (std::is_same_v<T, __ts_fp128> &&
                      std::is_same_v<mp_component_t<fpmp_type>, double>)
        {
            return fpmp_type(static_cast<cuda::experimental::__fpmp_fp128>(val));
        }
        else
    #endif
        {
            return fpmp_type(val);
        }
    }
}

// ============================================================================
// Helper: Convert a value to fprf_type for storage in error records
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ fprf_type to_fprf_for_storage(T val) {
    if constexpr (std::is_integral<T>::value) {
        // For integers, convert via double to preserve precision
        return static_cast<fprf_type>(static_cast<double>(val));
    } else if constexpr (std::is_same_v<T, bool>) {
        return static_cast<fprf_type>(val ? 1.0 : 0.0);
    } else {
        return static_cast<fprf_type>(val);
    }
}


// ============================================================================
// Helper: Check if a float value is denormal or zero (exponent == 0)
// ============================================================================
__HOST_DEVICE_DECL__ inline bool is_denorm_zero_f32(float val)
{
    return (ts::bit_cast<uint32_t>(val) & 0x7F800000u) == 0;
}

// ============================================================================
// Helper: Check if a double value is denormal or zero (exponent == 0)
// ============================================================================
__HOST_DEVICE_DECL__ inline bool is_denorm_zero_f64(double val)
{
    return (ts::bit_cast<uint64_t>(val) & 0x7FF0000000000000ULL) == 0;
}

#if !defined(__CUDACC__)
// is_denorm_zero_f64/is_denorm_zero_f32 are bit-based; for printing and comparisons we
// also want a cheap "ignore sign of zero" normalization.
#endif

#if (TS_HAS_LIBQUADMATH == 1) || (TS_HAS_LDOUBLE128 == 1)
// ============================================================================
// Helper: Check if a __ts_fp128 value is denormal or zero (exponent == 0)
// IEEE 754 binary128: sign(1) + exponent(15) + mantissa(112) = 128 bits
// ============================================================================
__HOST_DEVICE_DECL__ inline bool is_denorm_zero_f128(__ts_fp128 val)
{
    // Exponent is in bits 112-126 (15 bits), which is bits 48-62 of the high part
    return (ts::bit_cast<u64x2>(val).hi & 0x7FFF000000000000ULL) == 0;
}

#endif

// NOTE: ts_ref_to_fpmp has been moved to ts_types.hpp

// ============================================================================
// Helpers: classify "denorm-ish" results in *component precision*
// For fp32mp2/fp64mp2 we want to detect when the hi/lo components are subnormal
// (exponent == 0) or zero in float/double, even if the reference is computed in
// higher precision.
// ============================================================================
template<typename ComponentType>
__HOST_DEVICE_DECL__ inline bool is_denorm_or_zero_component(ComponentType v)
{
    if constexpr (std::is_same_v<ComponentType, float>)  return is_denorm_zero_f32(v);
    if constexpr (std::is_same_v<ComponentType, double>) return is_denorm_zero_f64(v);
    return false;
}

template<typename T>
__HOST_DEVICE_DECL__ inline bool is_denorm_or_zero_result_in_component_precision(const T& v)
{
    if constexpr (is_multiprecision_v<T>)
    {
        using C = mp_component_t<T>;
        const C hi = v.hi();
        const C lo = v.lo();
        // Consider "denorm-ish" if both components are subnormal/zero in component type.
        return is_denorm_or_zero_component<C>(hi) && is_denorm_or_zero_component<C>(lo);
    }
    else
    {
        using C = mp_component_t<T>;
        return is_denorm_or_zero_component<C>(static_cast<C>(v));
    }
}

// Like above but returns true if ANY component (hi or lo) is denormal/zero.
// Used for input classification: a single denormal component can cause
// accuracy loss even when the other component is normal.
template<typename T>
__HOST_DEVICE_DECL__ inline bool has_any_denorm_or_zero_component(const T& v)
{
    if constexpr (is_multiprecision_v<T>)
    {
        using C = mp_component_t<T>;
        return is_denorm_or_zero_component<C>(v.hi()) || is_denorm_or_zero_component<C>(v.lo());
    }
    else
    {
        using C = mp_component_t<T>;
        return is_denorm_or_zero_component<C>(static_cast<C>(v));
    }
}

// ============================================================================
// Check if the high part of a result is denormal, zero, or in the smallest
// normal binade (exponent == 1, i.e., values in [2^-126, 2^-125) for float
// or [2^-1022, 2^-1021) for double). These "near-denormal" values are prone
// to precision loss as the low component may underflow to denormal.
// ============================================================================
template<typename ComponentType>
__HOST_DEVICE_DECL__ inline bool is_near_denorm_component(ComponentType v)
{
    if constexpr (std::is_same_v<ComponentType, float>)
    {
        // Check if biased exponent is 0 (denormal/zero) or 1 (smallest normal binade)
        uint32_t exp_bits = (ts::bit_cast<uint32_t>(v) & 0x7F800000u) >> 23;
        return exp_bits <= 1;
    }
    else if constexpr (std::is_same_v<ComponentType, double>)
    {
        // Check if biased exponent is 0 (denormal/zero) or 1 (smallest normal binade)
        uint64_t exp_bits = (ts::bit_cast<uint64_t>(v) & 0x7FF0000000000000ULL) >> 52;
        return exp_bits <= 1;
    }
    return false;
}

// ============================================================================
// Check if the high part of a result is in the near-denormal range
// For multi-precision types, checks only the hi component.
// For scalar types, checks the value itself.
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ inline bool is_hi_near_denorm(const T& v)
{
    if constexpr (is_multiprecision_v<T>)
    {
        using C = mp_component_t<T>;
        return is_near_denorm_component<C>(v.hi());
    }
    else
    {
        using C = mp_component_t<T>;
        return is_near_denorm_component<C>(static_cast<C>(v));
    }
}

template<typename T>
__HOST_DEVICE_DECL__ inline bool mp_equal_ignore_zero_sign(const T& a, const T& b)
{
    if constexpr (is_multiprecision_v<T>)
    {
        using C = mp_component_t<T>;
        C ahi = a.hi(), alo = a.lo();
        C bhi = b.hi(), blo = b.lo();
        // Normalize signed zeros
        if (ahi == C(0.0)) ahi = C(0.0);
        if (alo == C(0.0)) alo = C(0.0);
        if (bhi == C(0.0)) bhi = C(0.0);
        if (blo == C(0.0)) blo = C(0.0);
        return (ahi == bhi) && (alo == blo);
    }
    else
    {
        using C = mp_component_t<T>;
        C aa = static_cast<C>(a);
        C bb = static_cast<C>(b);
        if (aa == C(0.0)) aa = C(0.0);
        if (bb == C(0.0)) bb = C(0.0);
        return aa == bb;
    }
}

// ============================================================================
// Helper: Check if a component value is INF or NaN (special value)
// ============================================================================
template<typename ComponentType>
__HOST_DEVICE_DECL__ inline bool is_special_component(ComponentType v)
{
    if constexpr (std::is_same_v<ComponentType, float>)
    {
        // Exponent all 1s means INF or NaN
        return (ts::bit_cast<uint32_t>(v) & 0x7F800000u) == 0x7F800000u;
    }
    else if constexpr (std::is_same_v<ComponentType, double>)
    {
        return (ts::bit_cast<uint64_t>(v) & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL;
    }
    return false;
}

// ============================================================================
// Helper: Check if value contains special (INF/NaN) components
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ inline bool has_special_component(const T& v)
{
    if constexpr (is_multiprecision_v<T>)
    {
        using C = mp_component_t<T>;
        return is_special_component<C>(v.hi()) || is_special_component<C>(v.lo());
    }
    else if constexpr (std::is_integral_v<T>)
    {
        return false;  // Integers don't have special values
    }
    else
    {
        using C = mp_component_t<T>;
        return is_special_component<C>(static_cast<C>(v));
    }
}

// ============================================================================
// Helper: Check if two component values are "matching" special values
// Returns true if both are the same kind of special:
//   - Both +INF → match
//   - Both -INF → match  
//   - Both NaN → match (any NaN matches any NaN)
//   - Otherwise → no match
// ============================================================================
template<typename ComponentType>
__HOST_DEVICE_DECL__ inline bool are_matching_special_components(ComponentType a, ComponentType b)
{
    if constexpr (std::is_same_v<ComponentType, float>)
    {
        uint32_t bits_a = ts::bit_cast<uint32_t>(a);
        uint32_t bits_b = ts::bit_cast<uint32_t>(b);
        
        // Check if both are special (exponent all 1s)
        bool a_special = (bits_a & 0x7F800000u) == 0x7F800000u;
        bool b_special = (bits_b & 0x7F800000u) == 0x7F800000u;
        
        if (!a_special || !b_special) return false;  // At least one is not special
        
        // Check if both are NaN (mantissa != 0)
        bool a_nan = (bits_a & 0x007FFFFFu) != 0;
        bool b_nan = (bits_b & 0x007FFFFFu) != 0;
        
        if (a_nan && b_nan) return true;  // Both NaN → match
        if (a_nan || b_nan) return false; // One NaN, one INF → no match
        
        // Both are INF, check signs match
        return (bits_a & 0x80000000u) == (bits_b & 0x80000000u);
    }
    else if constexpr (std::is_same_v<ComponentType, double>)
    {
        uint64_t bits_a = ts::bit_cast<uint64_t>(a);
        uint64_t bits_b = ts::bit_cast<uint64_t>(b);
        
        // Check if both are special (exponent all 1s)
        bool a_special = (bits_a & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL;
        bool b_special = (bits_b & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL;
        
        if (!a_special || !b_special) return false;  // At least one is not special
        
        // Check if both are NaN (mantissa != 0)
        bool a_nan = (bits_a & 0x000FFFFFFFFFFFFFULL) != 0;
        bool b_nan = (bits_b & 0x000FFFFFFFFFFFFFULL) != 0;
        
        if (a_nan && b_nan) return true;  // Both NaN → match
        if (a_nan || b_nan) return false; // One NaN, one INF → no match
        
        // Both are INF, check signs match
        return (bits_a & 0x8000000000000000ULL) == (bits_b & 0x8000000000000000ULL);
    }
    return false;
}

// ============================================================================
// Helper: Check if two values have matching special components (high part)
// For multiprecision types, only checks the hi component
// Returns true if both have special hi components that match
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ inline bool are_matching_special_values(const T& a, const T& b)
{
    if constexpr (is_multiprecision_v<T>)
    {
        using C = mp_component_t<T>;
        bool a_special = is_special_component<C>(a.hi());
        bool b_special = is_special_component<C>(b.hi());
        
        // If neither is special, they're not "matching special values"
        if (!a_special && !b_special) return false;
        
        // If only one is special, they don't match
        if (a_special != b_special) return false;
        
        // Both hi components are special, check if they match
        return are_matching_special_components<C>(a.hi(), b.hi());
    }
    else if constexpr (std::is_integral_v<T>)
    {
        return false;  // Integers don't have special values
    }
    else
    {
        using C = mp_component_t<T>;
        C ca = static_cast<C>(a);
        C cb = static_cast<C>(b);
        
        bool a_special = is_special_component<C>(ca);
        bool b_special = is_special_component<C>(cb);
        
        if (!a_special && !b_special) return false;
        if (a_special != b_special) return false;
        
        return are_matching_special_components<C>(ca, cb);
    }
}

// ============================================================================
// Helper: Check if component value is near INF (in the largest normal binade)
// For float: exponent 253 or 254 (before 255 = INF)
// For double: exponent 2045 or 2046 (before 2047 = INF)
// ============================================================================
template<typename ComponentType>
__HOST_DEVICE_DECL__ inline bool is_near_inf_component(ComponentType v)
{
    if constexpr (std::is_same_v<ComponentType, float>)
    {
        uint32_t exp_bits = (ts::bit_cast<uint32_t>(v) & 0x7F800000u) >> 23;
        return exp_bits >= 253 && exp_bits <= 254;  // Near max finite
    }
    else if constexpr (std::is_same_v<ComponentType, double>)
    {
        uint64_t exp_bits = (ts::bit_cast<uint64_t>(v) & 0x7FF0000000000000ULL) >> 52;
        return exp_bits >= 2045 && exp_bits <= 2046;  // Near max finite
    }
    return false;
}

// ============================================================================
// Helper: Check if high part of value is near INF
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ inline bool is_hi_near_inf(const T& v)
{
    if constexpr (is_multiprecision_v<T>)
    {
        using C = mp_component_t<T>;
        return is_near_inf_component<C>(v.hi());
    }
    else if constexpr (std::is_integral_v<T>)
    {
        return false;
    }
    else
    {
        using C = mp_component_t<T>;
        return is_near_inf_component<C>(static_cast<C>(v));
    }
}

// ============================================================================
// Helper: Check if any input has special components
// ============================================================================
template<typename T, int arity>
__HOST_DEVICE_DECL__ inline bool any_input_special(const T& a, const T& b, const T& c, const T& d)
{
    bool result = has_special_component(a);
    if constexpr (arity >= 2) result = result || has_special_component(b);
    if constexpr (arity >= 3) result = result || has_special_component(c);
    if constexpr (arity >= 4) result = result || has_special_component(d);
    return result;
}

// ============================================================================
// Helper: Check if any input has a denormal/zero component
// Uses has_any_denorm_or_zero_component (OR logic) so that a single denormal
// component (e.g. denormal lo with normal hi) triggers the classification.
// ============================================================================
template<typename T, int arity>
__HOST_DEVICE_DECL__ inline bool any_input_denormal(const T& a, const T& b, const T& c, const T& d)
{
    bool result = has_any_denorm_or_zero_component(a);
    if constexpr (arity >= 2) result = result || has_any_denorm_or_zero_component(b);
    if constexpr (arity >= 3) result = result || has_any_denorm_or_zero_component(c);
    if constexpr (arity >= 4) result = result || has_any_denorm_or_zero_component(d);
    return result;
}

// ============================================================================
// Helper: Check if any input is near-denormal
// ============================================================================
template<typename T, int arity>
__HOST_DEVICE_DECL__ inline bool any_input_near_denormal(const T& a, const T& b, const T& c, const T& d)
{
    bool result = is_hi_near_denorm(a);
    if constexpr (arity >= 2) result = result || is_hi_near_denorm(b);
    if constexpr (arity >= 3) result = result || is_hi_near_denorm(c);
    if constexpr (arity >= 4) result = result || is_hi_near_denorm(d);
    return result;
}

// ============================================================================
// Helper: Check if any input is near-INF
// ============================================================================
template<typename T, int arity>
__HOST_DEVICE_DECL__ inline bool any_input_near_inf(const T& a, const T& b, const T& c, const T& d)
{
    bool result = is_hi_near_inf(a);
    if constexpr (arity >= 2) result = result || is_hi_near_inf(b);
    if constexpr (arity >= 3) result = result || is_hi_near_inf(c);
    if constexpr (arity >= 4) result = result || is_hi_near_inf(d);
    return result;
}

// ============================================================================
// Detect cancellation: when |result| << |a| + |b| (result much smaller than inputs)
// This happens when subtracting nearly equal values, causing massive
// precision loss. We detect it when the result is much smaller than
// the inputs (by a factor of 2^mantissa_bits or more).
// ============================================================================
template<typename ResultType, typename InputType, int arity>
__HOST_DEVICE_DECL__ inline bool is_cancellation(
    const ResultType& result, const InputType& a, const InputType& b, 
    [[maybe_unused]] const InputType& c, [[maybe_unused]] const InputType& d)
{
    // Only applicable to binary operations (arity >= 2) with floating-point inputs
    if constexpr (arity < 2) return false;
    
    // Not applicable to integer inputs (no cancellation concept)
    if constexpr (std::is_integral_v<InputType>) return false;
    
    // Get absolute values (using hi component for multi-precision)
    double abs_result, abs_a, abs_b;
    
    if constexpr (is_multiprecision_v<ResultType>)
    {
        abs_result = static_cast<double>(result.hi());
    }
    else
    {
        abs_result = static_cast<double>(result);
    }
    
    if constexpr (is_multiprecision_v<InputType>)
    {
        abs_a = static_cast<double>(a.hi());
        abs_b = static_cast<double>(b.hi());
    }
    else
    {
        abs_a = static_cast<double>(a);
        abs_b = static_cast<double>(b);
    }
    
    if (abs_result < 0) abs_result = -abs_result;
    if (abs_a < 0) abs_a = -abs_a;
    if (abs_b < 0) abs_b = -abs_b;
    
    // Check if result is tiny compared to inputs
    // Catastrophic cancellation: |result| < |a + b| * 2^-20
    // This means we lost at least 20 bits of precision due to cancellation
    double sum_inputs = abs_a + abs_b;
    constexpr double cancellation_threshold = 1.0 / (1ULL << 20);  // 2^-20
    
    return (sum_inputs > 0.0) && (abs_result < sum_inputs * cancellation_threshold);
}

// ============================================================================
// Classify accuracy issue into one of the accuracy_class categories
// Returns accuracy_class::normal if within warning threshold
// Returns a warning class (1-9) if above warning threshold but has mitigating factor
// Returns accuracy_class::error if above error threshold with no mitigating factor
//
// The `method` parameter selects per-method tolerance for the gross-error gate
// (see header comment in the body).  The warning/error thresholds are passed
// in by the caller and are also typically derived per-method (component-wide
// thresholds are returned by get_warning_threshold / get_error_threshold,
// and additional per-method scaling can be layered on top by the caller).
// ============================================================================
template<typename TestResultType, typename TestInputType, int arity>
__HOST_DEVICE_DECL__ accuracy_class classify_accuracy_issue(
    double rel_err,
    cuda::experimental::fpmp2_accuracy method,
    const TestResultType& test_result,
    const TestResultType& ref_as_test,
    const TestInputType& a, const TestInputType& b, const TestInputType& c, const TestInputType& d,
    double warning_threshold,
    double error_threshold)
{
    // Normal: within warning threshold
    if (rel_err <= warning_threshold)
    {
        return accuracy_class::normal;
    }

    // === Check for warning conditions (priority order) ===
    //
    // The ladder is split into two halves by a "gross-error" gate:
    //   - OUTPUT-side mitigations come first and apply at any magnitude
    //     (they typically reflect dynamic-range limitations of the
    //     underlying primitives, e.g. `rcp(MAX) → underflow → div
    //     collapses to 0`).
    //   - The gross-error gate then promotes any remaining failure with
    //     `rel_err > T_gross` (sign flip, opposite-sign result with larger
    //     magnitude, or worse) straight to a hard error, before INPUT-side
    //     mitigations and cancellation get a chance to demote it.  This
    //     way sign / quadrant logic bugs can no longer hide under
    //     "input special" / "input denormal" just because an edge-case
    //     input happened to be in play (e.g. atan2(±0, x<0) returning
    //     -π instead of +π, rel_err = 2.0).
    //
    //     The threshold T_gross depends on the operation method:
    //       - method::accurate  →  T_gross = 0.5  (catch any factor-of-1.5
    //                              or worse error: accurate-mode results
    //                              are expected to be within a few ULPs).
    //       - method::def       →  T_gross = 1.0  (catch sign flips and
    //                              cases worse than "result is exactly 0
    //                              or 2x the reference"; allows precision
    //                              degradation up to factor 2).
    //       - method::fast      →  T_gross = 2.0  (only catch errors WORSE
    //                              than a clean sign flip; fast methods
    //                              trade precision for speed and routinely
    //                              hit factor-of-2 underflows in their
    //                              intermediate computations, e.g. fast
    //                              div's `1/b^2` for `b > 2^63`).
    //
    //     Cases with `test = 0` when `ref != 0` (rel_err = 1.0) are still
    //     caught ahead of the gate by the `output_denormal` check above,
    //     so even the fast-mode threshold doesn't lose coverage of the
    //     "result silently underflowed to zero" scenario.
    //   - INPUT-side mitigations and cancellation apply only to the
    //     remaining moderate-magnitude failures.

    // 1) Output is special (INF, NaN)
    if (has_special_component(test_result))
    {
        return accuracy_class::output_special;
    }

    // 2) Output is denormal
    if (is_denorm_or_zero_result_in_component_precision(test_result))
    {
        return accuracy_class::output_denormal;
    }

    // 3) Output is near-denormal
    if (is_hi_near_denorm(test_result))
    {
        return accuracy_class::output_near_denormal;
    }

    // 4) Output is near-INF
    if (is_hi_near_inf(test_result))
    {
        return accuracy_class::output_near_inf;
    }

    // 5) Gross-error gate (see header comment above).  Threshold is
    //    method-dependent: accurate=0.5, def=1.0, fast=2.0.  We only fire
    //    on rel_err strictly greater than the threshold to avoid catching
    //    the boundary case (e.g. exact factor-of-2 underflow at rel_err=1.0
    //    for def methods, or exact sign flip at rel_err=2.0 for fast).
    const double gross_error_threshold = (method == cuda::experimental::fpmp2_accuracy::low)  ? 2.0 :
                                         (method == cuda::experimental::fpmp2_accuracy::high) ? 0.5 :
                                         /* fpmp2_accuracy::def/mid or else */     1.0;
    if (rel_err > gross_error_threshold)
    {
        return accuracy_class::error;
    }

    // 6) Any input is special (INF, NaN)
    if (any_input_special<TestInputType, arity>(a, b, c, d))
    {
        return accuracy_class::input_special;
    }

    // 7) Any input is denormal
    if (any_input_denormal<TestInputType, arity>(a, b, c, d))
    {
        return accuracy_class::input_denormal;
    }

    // 8) Any input is near-denormal
    if (any_input_near_denormal<TestInputType, arity>(a, b, c, d))
    {
        return accuracy_class::input_near_denormal;
    }

    // 9) Any input is near-INF
    if (any_input_near_inf<TestInputType, arity>(a, b, c, d))
    {
        return accuracy_class::input_near_inf;
    }

    // 10) Catastrophic cancellation
    if (is_cancellation<TestResultType, TestInputType, arity>(test_result, a, b, c, d))
    {
        return accuracy_class::cancellation;
    }

    // No warning condition applies
    // If above error threshold, it's an error
    if (rel_err > error_threshold)
    {
        return accuracy_class::error;
    }

    // Between warning and error threshold with no specific condition identified
    // This is an "unknown" warning - we can't identify the cause
    return accuracy_class::unclassified;
}

// ============================================================================
// FTZ (Flush To Zero): Returns zero if input is denormal or zero
// Works for reference types: double, __ts_fp128
// ============================================================================
__HOST_DEVICE_DECL__ inline double ftz(double val)
{
    if (is_denorm_zero_f64(val))
        return 0.0;
    return val;
}

#if (TS_HAS_LIBQUADMATH == 1) || (TS_HAS_LDOUBLE128 == 1)
__HOST_DEVICE_DECL__ inline __ts_fp128 ftz(__ts_fp128 val)
{
    if (is_denorm_zero_f128(val))
        return static_cast<__ts_fp128>(0.0);
    return val;
}
#endif

__HOST_DEVICE_DECL__ inline float ftz(float val)
{
    if (is_denorm_zero_f32(val))
        return 0.0f;
    return val;
}

// ============================================================================
// FTZ for integral types: no flushing needed, return as-is
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ inline
typename std::enable_if<std::is_integral<T>::value, T>::type
ftz(T val)
{
    return val;  // Integers don't have denormals
}

// ============================================================================
// FTZ for multi-precision types: flush each component independently
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ inline
typename std::enable_if<is_multiprecision_v<T>, T>::type
ftz(T val)
{
    using C = mp_component_t<T>;
    C hi = val.hi();
    C lo = val.lo();
    
    if constexpr (std::is_same_v<C, float>)
    {
        hi = ftz(hi);
        lo = ftz(lo);
    }
    else if constexpr (std::is_same_v<C, double>)
    {
        hi = ftz(hi);
        lo = ftz(lo);
    }
    
    return T(hi, lo);
}

// ============================================================================
// Compute relative error between test and reference values
// Uses reference type precision for accurate error computation
// If TS_FTZ_DAZ_MODE is enabled, flushes denormals to zero before comparison
// ============================================================================
template<typename T1, typename T2>
__HOST_DEVICE_DECL__ double compute_rel_error(T1 test_val, T2 ref_val)
{
    T2 test_ref = ts_convert_test_to_ref<T2>(test_val);
    
#if TS_FTZ_DAZ_MODE
    test_ref = ftz(test_ref);
    ref_val  = ftz(ref_val);
#endif

    // Treat +0 and -0 as equal (ignore sign of zero)
    if (test_ref == T2(0.0)) test_ref = T2(0.0);
    if (ref_val  == T2(0.0)) ref_val  = T2(0.0);

    // Compute absolute difference
    double diff;
    if constexpr (std::is_integral<T2>::value) {
        // For integer types, compute in double to avoid integer overflow/truncation
        diff = static_cast<double>(test_ref) - static_cast<double>(ref_val);
        if (diff < 0.0) diff = -diff;
    } else {
        T2 diff_t = test_ref - ref_val;
        if constexpr (std::is_signed<T2>::value) {
            if (diff_t < T2(0)) diff_t = -diff_t;
        }
        diff = static_cast<double>(diff_t);
    }
    
    // Compute absolute denominator
    double denom;
    if constexpr (std::is_integral<T2>::value) {
        denom = (ref_val == T2(0)) ? 1.0 : static_cast<double>(ref_val);
        if (denom < 0.0) denom = -denom;
    } else {
        T2 denom_t = (ref_val == T2(0.0)) ? T2(1.0) : ref_val;
        if constexpr (std::is_signed<T2>::value) {
            if (denom_t < T2(0)) denom_t = -denom_t;
        }
        denom = static_cast<double>(denom_t);
    }
    
    return diff / denom;
}

// ============================================================================
// Compute ULP error (approximate, based on test type precision)
// Uses reference type precision for accurate error computation
// If TS_FTZ_DAZ_MODE is enabled, flushes denormals to zero before comparison
// ============================================================================
template<typename T1, typename T2>
__HOST_DEVICE_DECL__ double compute_ulp_error(T1 test_val, T2 ref_val)
{
    // Determine mantissa bits based on test type
    // fp32mp2_t (~48 bits), fp64mp2_t (~106 bits), float (23), double (52)
    constexpr double mantissa_scale = (sizeof(T1) <= 8) ? (1ULL << 23) : (1ULL << 52);
    
    // Use double precision
    T2 test_ref = ts_convert_test_to_ref<T2>(test_val);
    
#if TS_FTZ_DAZ_MODE
    test_ref = ftz(test_ref);
    ref_val  = ftz(ref_val);
#endif

    // Treat +0 and -0 as equal (ignore sign of zero)
    if (test_ref == T2(0.0)) test_ref = T2(0.0);
    if (ref_val  == T2(0.0)) ref_val  = T2(0.0);

    T2 diff = test_ref - ref_val;
    // Use abs() logic that works for both signed and unsigned types
    if constexpr (std::is_signed<T2>::value) {
        if (diff < 0) diff = -diff;
    }
    T2 denom = (ref_val == T2(0.0)) ? T2(1.0) : ref_val;
    if constexpr (std::is_signed<T2>::value) {
        if (denom < 0) denom = -denom;
    }
    T2 rel_err = diff / denom;
    
    return static_cast<double>(rel_err * mantissa_scale);
}


// NOTE: Input generation functions (generate_input_work, generate_input_normal,
// generate_input_special, generate_input_pattern, etc.) have been moved to ts_dataset.hpp

// ============================================================================
// Unified accuracy kernel - works for both CUDA device and host
// ============================================================================
// Platform-specific type aliases and helper macros
#if defined(__CUDACC__)
    using   AccuracyRngState = curandState_t;
    #define ACCURACY_RNG_INIT(rng, seed_val, worker_id) curand_init(seed_val, worker_id, 0, &rng)
#else
    using   AccuracyRngState = std::mt19937_64;
    #define ACCURACY_RNG_INIT(rng, seed_val, worker_id) rng.seed(seed_val + worker_id)
#endif

template<typename TestFuncTag, 
         typename RefFuncTag>
TS_KERNEL_DECL void accuracy_kernel (accuracy_mode::type   mode,
                                     uint64_t              total_samples,
                                     accuracy_reduction_t* results,
                                     accuracy_error_log_t* error_log,
                                     const fixed_inputs_t<tag_input_t<TestFuncTag>>* fixed_inputs = nullptr)
{
    // ========================================================================
    // Type aliases from function tags
    // ========================================================================
    using TestInputType  = tag_input_t<TestFuncTag>;
    using TestResultType = tag_result_t<TestFuncTag>;
    using RefInputType   = tag_input_t<RefFuncTag>;
    using RefResultType  = tag_result_t<RefFuncTag>;

    // Arity detection using tag's input_type
    constexpr int test_arity = detect_arity<TestFuncTag>();
    constexpr int ref_arity  = detect_arity<RefFuncTag>();
    static_assert(test_arity == ref_arity, "Test and reference functions must have same arity");
    constexpr int arity = test_arity;    

    // ========================================================================
    // Platform-specific thread setup and work distribution
    // ========================================================================
#if defined(__CUDACC__)
    // CUDA: grid-stride loop pattern
    uint64_t worker_id   = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_workers = gridDim.x * blockDim.x;
    uint64_t loop_start  = worker_id;
    uint64_t loop_end    = total_samples;
    uint64_t loop_stride = num_workers;
#else
    // Host: called from within OpenMP parallel region
    uint64_t worker_id   = omp_get_thread_num();
    uint64_t num_workers = omp_get_num_threads();
    uint64_t samples_per_worker = (total_samples + num_workers - 1) / num_workers;
    uint64_t loop_start  = worker_id * samples_per_worker;
    uint64_t loop_end    = std::min(loop_start + samples_per_worker, total_samples);
    uint64_t loop_stride = 1;
    if (loop_start >= total_samples) return;
#endif

    // ========================================================================
    // Thread-local accumulators (common)
    // ========================================================================
    double   local_max_rel_err = 0.0;
    double   local_sum_rel_err = 0.0;
    double   local_max_ulp = 0.0;
    double   local_sum_ulp = 0.0;
    uint64_t local_errs = 0;
    uint64_t local_warnings = 0;
    uint64_t local_valid = 0;
    uint64_t local_skipped = 0;
    
    // Per-class local accumulators
    accuracy_class_stats_t local_class_stats[ACCURACY_CLASS_COUNT] = {};

    // ========================================================================
    // RNG initialization (platform-specific)
    // ========================================================================
    AccuracyRngState rng;
    ACCURACY_RNG_INIT(rng, TS_ACCURACY_SEED, loop_start);

    // ========================================================================
    // Helper lambda: get input value (from fixed_inputs if available, else generate)
    // When fixed_inputs is provided (fixed input mode):
    //   - If valid[arg_idx] is true, use the pre-parsed value
    //   - If valid[arg_idx] is false, return zero (missing argument defaults to 0)
    // When fixed_inputs is nullptr (normal mode):
    //   - Generate random input using the specified mode
    // ========================================================================
    auto get_input = [&](uint64_t sample_idx, int arg_idx) -> TestInputType {
        if (fixed_inputs != nullptr) {
            if (fixed_inputs->valid[arg_idx])
                return fixed_inputs->values[arg_idx];
            // Fixed input mode but this argument not provided - return zero
            return TestInputType{};
        }
        return generate_input<TestInputType, RefInputType, TestFuncTag, AccuracyRngState&>(rng, mode, sample_idx, arg_idx);
    };

    // ========================================================================
    // Main processing loop (common logic)
    // ========================================================================
    for (uint64_t i = loop_start; i < loop_end; i += loop_stride)
    {
        // Get input arguments (from fixed_inputs if provided, else generate)
        TestInputType a_test = get_input(i, 0);
        RefInputType  a_ref  = ts_convert_test_to_ref<RefInputType>(a_test);
        
        [[maybe_unused]] TestInputType b_test{}, c_test{}, d_test{};
        [[maybe_unused]] RefInputType  b_ref{},  c_ref{},  d_ref{};
        
        // Get bitmask of arguments that should have lo=0 (for ACC-like functions)
        constexpr unsigned zero_lo_mask = get_zero_lo_args<TestFuncTag>();
        
        if constexpr (arity >= 2) { 
            b_test = get_input(i, 1); 
            // Zero out lo part if requested by function tag
            if constexpr ((zero_lo_mask & 0x2) != 0) { b_test = zero_lo_part(b_test); }
            b_ref  = ts_convert_test_to_ref<RefInputType>(b_test); 
        }
        if constexpr (arity >= 3) { 
            c_test = get_input(i, 2); 
            if constexpr ((zero_lo_mask & 0x4) != 0) { c_test = zero_lo_part(c_test); }
            c_ref  = ts_convert_test_to_ref<RefInputType>(c_test); 
        }
        if constexpr (arity >= 4) { 
            d_test = get_input(i, 3); 
            if constexpr ((zero_lo_mask & 0x8) != 0) { d_test = zero_lo_part(d_test); }
            d_ref  = ts_convert_test_to_ref<RefInputType>(d_test); 
        }

        // Check for INF/NAN inputs
        const bool has_inf_nan_input = !is_value_finite(a_test) ||
            (arity >= 2 && !is_value_finite(b_test)) ||
            (arity >= 3 && !is_value_finite(c_test)) ||
            (arity >= 4 && !is_value_finite(d_test));

        // Skip inputs outside valid range (e.g., mp2int requires INT32 range)
        if (!is_input_in_valid_range<TestFuncTag>(a_test) ||
            (arity >= 2 && !is_input_in_valid_range<TestFuncTag>(b_test)) ||
            (arity >= 3 && !is_input_in_valid_range<TestFuncTag>(c_test)) ||
            (arity >= 4 && !is_input_in_valid_range<TestFuncTag>(d_test)))
        {
            local_skipped++;
            continue;
        }

        // Call test function using tag's types
        TestResultType test_result = impl_dispatch<TestFuncTag, arity>::call(a_test, b_test, c_test, d_test);
        // Call reference function using tag's types
        RefResultType ref_result = impl_dispatch<RefFuncTag, arity>::call(a_ref, b_ref, c_ref, d_ref);
        #if TS_USE_PRECISE_REFERENCE != 1
            ref_result = ts_convert_test_to_ref<RefResultType>(
                ts_convert_ref_to_test<TestResultType, RefResultType>(ref_result));
        #endif

        // Convert reference result to test type for comparison
        TestResultType ref_as_test = ts_convert_ref_to_test<TestResultType, RefResultType>(ref_result);

        // Check for INF/NAN inputs - only count as warning if outputs DIFFER
        if (has_inf_nan_input)
        {
            // Check if outputs match (both special and matching, or both equal)
            bool outputs_match = are_matching_special_values(test_result, ref_as_test) ||
                                 mp_equal_ignore_zero_sign(test_result, ref_as_test);
            
            if (!outputs_match)
            {
                // Outputs differ despite special inputs - count as input_special warning
                local_warnings++;
                constexpr int cls_idx = static_cast<int>(accuracy_class::input_special);
                local_class_stats[cls_idx].count++;
                
                // Log input_special warnings when __PRINT_WARN__ or __PRINT_ALL__ is enabled
#if defined(__PRINT_WARN__) || defined(__PRINT_ALL__)
                if (error_log != nullptr)
                {
#if defined(__CUDACC__)
                    unsigned long long slot = atomicAdd(&error_log->count, 1ULL);
#else
                    unsigned long long slot = __sync_fetch_and_add(&error_log->count, 1ULL);
#endif
                    if (slot < TS_ACCURACY_ERROR_LOG_SIZE)
                    {
                        accuracy_error_record_t& rec = error_log->records[slot];
                        rec.args[0] = to_fpmp_for_storage(a_test);
                        rec.args[1] = (arity >= 2) ? to_fpmp_for_storage(b_test) : fpmp_type{};
                        rec.args[2] = (arity >= 3) ? to_fpmp_for_storage(c_test) : fpmp_type{};
                        rec.args[3] = (arity >= 4) ? to_fpmp_for_storage(d_test) : fpmp_type{};
                        rec.test_result = to_fpmp_for_storage(test_result);
                        rec.ref_result  = to_fprf_for_storage(ref_result);
                        rec.rel_err     = -1.0;  // No rel_err for special inputs
                        rec.arity       = arity;
                        rec.classification = record_class::warning;
                        rec.acc_class   = accuracy_class::input_special;
                    }
                }
#endif
            }
            // Skip from rel_err statistics regardless (special inputs have undefined behavior)
            local_skipped++;
            continue;
        }

        // Check for INF/NAN results - only count as warning if they DIFFER
        const bool test_special = has_special_component(test_result);
        const bool ref_special  = has_special_component(ref_as_test);
        
        if (test_special || ref_special)
        {
            // Check if both are matching special values (e.g., both +INF, both NaN)
            bool outputs_match = are_matching_special_values(test_result, ref_as_test);
            
            if (!outputs_match)
            {
                // Special outputs that don't match - count as output_special warning
                local_warnings++;
                constexpr int cls_idx = static_cast<int>(accuracy_class::output_special);
                local_class_stats[cls_idx].count++;
                
                // Log output_special warnings when __PRINT_WARN__ or __PRINT_ALL__ is enabled
#if defined(__PRINT_WARN__) || defined(__PRINT_ALL__)
                if (error_log != nullptr)
                {
#if defined(__CUDACC__)
                    unsigned long long slot = atomicAdd(&error_log->count, 1ULL);
#else
                    unsigned long long slot = __sync_fetch_and_add(&error_log->count, 1ULL);
#endif
                    if (slot < TS_ACCURACY_ERROR_LOG_SIZE)
                    {
                        accuracy_error_record_t& rec = error_log->records[slot];
                        rec.args[0] = to_fpmp_for_storage(a_test);
                        rec.args[1] = (arity >= 2) ? to_fpmp_for_storage(b_test) : fpmp_type{};
                        rec.args[2] = (arity >= 3) ? to_fpmp_for_storage(c_test) : fpmp_type{};
                        rec.args[3] = (arity >= 4) ? to_fpmp_for_storage(d_test) : fpmp_type{};
                        rec.test_result = to_fpmp_for_storage(test_result);
                        rec.ref_result  = to_fprf_for_storage(ref_result);
                        rec.rel_err     = -1.0;  // No rel_err for special outputs
                        rec.arity       = arity;
                        rec.classification = record_class::warning;
                        rec.acc_class   = accuracy_class::output_special;
                    }
                }
#endif
            }
            // Skip from rel_err statistics regardless (special outputs have undefined rel_err)
            local_skipped++;
            continue;
        }

        // ----------------------------------------------------------------
        // Denormal handling (component precision):
        // If BOTH test and reference results land in the denormal/zero range
        // of the component type (float/double), and they differ, treat it as
        // a WARNING and exclude it from rel_err / ulp statistics.
        // This prevents huge relative errors like 1.0 from dominating the
        // summary when both answers are effectively subnormal noise.
        // ----------------------------------------------------------------
        const bool test_denorm = is_denorm_or_zero_result_in_component_precision(test_result);
        const bool ref_denorm  = is_denorm_or_zero_result_in_component_precision(ref_as_test);
        if (test_denorm && ref_denorm)
        {
            // If they differ at component precision, count a warning.
            // In all cases, exclude denorm/zero-region results from rel/ulp statistics.
            if (!mp_equal_ignore_zero_sign(test_result, ref_as_test))
            {
                local_warnings++;
                
                // Compute a rough relative error for the record
                double rel_err = compute_rel_error(test_result, ref_result);
                
                // Update per-class statistics for output_denormal
                constexpr int cls_idx = static_cast<int>(accuracy_class::output_denormal);
                local_class_stats[cls_idx].count++;
                local_class_stats[cls_idx].sum_rel_err += rel_err;
                if (rel_err > local_class_stats[cls_idx].max_rel_err)
                    local_class_stats[cls_idx].max_rel_err = rel_err;
                
                // Log denormal warnings when __PRINT_WARN__ or __PRINT_ALL__ is enabled
#if defined(__PRINT_WARN__) || defined(__PRINT_ALL__)
                if (error_log != nullptr)
                {
#if defined(__CUDACC__)
                    unsigned long long slot = atomicAdd(&error_log->count, 1ULL);
#else
                    unsigned long long slot = __sync_fetch_and_add(&error_log->count, 1ULL);
#endif
                    if (slot < TS_ACCURACY_ERROR_LOG_SIZE)
                    {
                        accuracy_error_record_t& rec = error_log->records[slot];
                        rec.args[0] = to_fpmp_for_storage(a_test);
                        rec.args[1] = (arity >= 2) ? to_fpmp_for_storage(b_test) : fpmp_type{};
                        rec.args[2] = (arity >= 3) ? to_fpmp_for_storage(c_test) : fpmp_type{};
                        rec.args[3] = (arity >= 4) ? to_fpmp_for_storage(d_test) : fpmp_type{};
                        rec.test_result = to_fpmp_for_storage(test_result);
                        rec.ref_result  = to_fprf_for_storage(ref_result);
                        rec.rel_err     = rel_err;
                        rec.arity       = arity;
                        rec.classification = record_class::warning;
                        rec.acc_class   = accuracy_class::output_denormal;
                    }
                }
#endif
            }
            continue;
        }

        // Compute errors for valid (finite) results only
        double rel_err = compute_rel_error(test_result, ref_result);
        double ulp_err = compute_ulp_error(test_result, ref_result);
        
        local_sum_rel_err += rel_err;
        local_sum_ulp     += ulp_err;
        local_max_rel_err  = TS_MAX(local_max_rel_err, rel_err);
        local_max_ulp      = TS_MAX(local_max_ulp, ulp_err);
        local_valid++;
        
        // Use component-type-dependent thresholds
        constexpr double error_threshold   = get_error_threshold<mp_component_t<fpmp_type>>();
        constexpr double warning_threshold = get_warning_threshold<mp_component_t<fpmp_type>>();
        // Method (def / fast / accurate) drives per-method gross-error tolerance.
        // Only consumed by the detailed classifier; unused when __RUN_CLASSIFY__==0.
        [[maybe_unused]] constexpr cuda::experimental::fpmp2_accuracy test_method = fpmp2_method_v<fpmp_type>;
        
#if !defined(__RUN_CLASSIFY__) || __RUN_CLASSIFY__ != 0
        // Use the new detailed classification system (can be disabled for faster host runs)
        const accuracy_class acc_cls = classify_accuracy_issue<TestResultType, TestInputType, arity>(
            rel_err, test_method, test_result, ref_as_test, a_test, b_test, c_test, d_test,
            warning_threshold, error_threshold);
        
        // Derive legacy classification from detailed class
        const bool is_error   = (acc_cls == accuracy_class::error);
        const bool is_warning = is_warning_class(acc_cls);
        const record_class rec_class = is_error ? record_class::error : 
                                       is_warning ? record_class::warning : 
                                       record_class::normal;
        
        // Update per-class statistics
        const int cls_idx = static_cast<int>(acc_cls);
        local_class_stats[cls_idx].count++;
        local_class_stats[cls_idx].sum_rel_err += rel_err;
        if (rel_err > local_class_stats[cls_idx].max_rel_err)
            local_class_stats[cls_idx].max_rel_err = rel_err;
#else
        // Simplified classification (faster): just check thresholds
        const bool is_error   = (rel_err > error_threshold);
        const bool is_warning = !is_error && (rel_err > warning_threshold);
        const record_class rec_class = is_error ? record_class::error : 
                                       is_warning ? record_class::warning : 
                                       record_class::normal;
        const accuracy_class acc_cls = is_error ? accuracy_class::error :
                                       is_warning ? accuracy_class::normal : // simplified: no detailed warning classes
                                       accuracy_class::normal;
#endif
        
        // ================================================================
        // Always track max error (regardless of threshold)
        // Also track per-class max errors
        // ================================================================
        if (error_log != nullptr)
        {
            // Helper lambda to write record data
            auto write_record = [&](accuracy_error_record_t& rec) {
                rec.args[0] = to_fpmp_for_storage(a_test);
                rec.args[1] = (arity >= 2) ? to_fpmp_for_storage(b_test) : fpmp_type{};
                rec.args[2] = (arity >= 3) ? to_fpmp_for_storage(c_test) : fpmp_type{};
                rec.args[3] = (arity >= 4) ? to_fpmp_for_storage(d_test) : fpmp_type{};
                rec.test_result = to_fpmp_for_storage(test_result);
                rec.ref_result  = to_fprf_for_storage(ref_result);
                rec.rel_err = rel_err;
                rec.arity   = arity;
                rec.classification = rec_class;
                rec.acc_class = acc_cls;
            };
            
#if defined(__CUDACC__)
            // CUDA: use atomicCAS loop to update max_rel_err and record atomically
            // Global max error
            if (rel_err > error_log->max_rel_err)
            {
                double old_max = error_log->max_rel_err;
                while (rel_err > old_max)
                {
                    unsigned long long* addr = reinterpret_cast<unsigned long long*>(&error_log->max_rel_err);
                    unsigned long long old_bits = __double_as_longlong(old_max);
                    unsigned long long new_bits = __double_as_longlong(rel_err);
                    unsigned long long result = atomicCAS(addr, old_bits, new_bits);
                    
                    if (result == old_bits)
                    {
                        write_record(error_log->max_record);
                        break;
                    }
                    old_max = __longlong_as_double(result);
                }
            }
            
            // Per-class max error (only if classification is enabled)
#if !defined(__RUN_CLASSIFY__) || __RUN_CLASSIFY__ != 0
            if (rel_err > error_log->class_max_rel_err[cls_idx])
            {
                double old_max = error_log->class_max_rel_err[cls_idx];
                while (rel_err > old_max)
                {
                    unsigned long long* addr = reinterpret_cast<unsigned long long*>(&error_log->class_max_rel_err[cls_idx]);
                    unsigned long long old_bits = __double_as_longlong(old_max);
                    unsigned long long new_bits = __double_as_longlong(rel_err);
                    unsigned long long result = atomicCAS(addr, old_bits, new_bits);
                    
                    if (result == old_bits)
                    {
                        write_record(error_log->class_max_records[cls_idx]);
                        break;
                    }
                    old_max = __longlong_as_double(result);
                }
            }
#endif
#else
            // Host: use critical section for thread-safe max update
            #pragma omp critical
            {
                // Global max error
                if (rel_err > error_log->max_rel_err)
                {
                    error_log->max_rel_err = rel_err;
                    write_record(error_log->max_record);
                }
                
                // Per-class max error (only if classification is enabled)
#if !defined(__RUN_CLASSIFY__) || __RUN_CLASSIFY__ != 0
                if (rel_err > error_log->class_max_rel_err[cls_idx])
                {
                    error_log->class_max_rel_err[cls_idx] = rel_err;
                    write_record(error_log->class_max_records[cls_idx]);
                }
#endif
            }
#endif
        }
        
        // Update error/warning counters
        if (is_error)
        {
            local_errs++;
        }
        else if (is_warning)
        {
            local_warnings++;
        }
        
        // ================================================================
        // Thread-safe result logging
        // Controlled by macros:
        //   __PRINT_ALL__  - log all results (errors, warnings, normal)
        //   __PRINT_WARN__ - log errors and warnings
        //   (default)      - log errors only
        // ================================================================
        bool should_log = false;
#if defined(__PRINT_ALL__)
        should_log = true;  // Log everything
#elif defined(__PRINT_WARN__)
        should_log = is_error || is_warning;  // Log errors and warnings
#else
        should_log = is_error;  // Log errors only (default)
#endif
        
        if (should_log && error_log != nullptr)
        {
            // Atomically get next available slot
#if defined(__CUDACC__)
            unsigned long long slot = atomicAdd(&error_log->count, 1ULL);
#else
            unsigned long long slot;
            #pragma omp atomic capture
            slot = error_log->count++;
#endif
            // Only write if we got a valid slot within buffer bounds
            if (slot < TS_ACCURACY_ERROR_LOG_SIZE)
            {
                accuracy_error_record_t& rec = error_log->records[slot];
                
                // Store input arguments (use to_fpmp_for_storage for proper integer handling)
                rec.args[0] = to_fpmp_for_storage(a_test);
                rec.args[1] = (arity >= 2) ? to_fpmp_for_storage(b_test) : fpmp_type{};
                rec.args[2] = (arity >= 3) ? to_fpmp_for_storage(c_test) : fpmp_type{};
                rec.args[3] = (arity >= 4) ? to_fpmp_for_storage(d_test) : fpmp_type{};
                
                // Store results (convert via double for integer types to preserve precision)
                rec.test_result = to_fpmp_for_storage(test_result);
                rec.ref_result  = to_fprf_for_storage(ref_result);
                
                rec.rel_err = rel_err;
                rec.arity   = arity;
                rec.classification = rec_class;
                rec.acc_class = acc_cls;
            }
        }
    }

    // ========================================================================
    // Reduction to shared results (platform-specific)
    // ========================================================================
#if defined(__CUDACC__)
    // CUDA: atomic reduction to global device memory
    atomicMaxDouble(&results->max_rel_err, local_max_rel_err);
    atomicAddDouble(&results->sum_rel_err, local_sum_rel_err);
    atomicMaxDouble(&results->max_ulp, local_max_ulp);
    atomicAddDouble(&results->sum_ulp, local_sum_ulp);
    atomicAdd(&results->total_errs, local_errs);
    atomicAdd(&results->total_warnings, local_warnings);
    atomicAdd(&results->valid_count, local_valid);
    atomicAdd(&results->skipped_inf_nan, local_skipped);
    
    // Per-class statistics reduction (only if classification is enabled)
#if !defined(__RUN_CLASSIFY__) || __RUN_CLASSIFY__ != 0
    for (int c = 0; c < ACCURACY_CLASS_COUNT; c++)
    {
        atomicAdd(&results->class_stats[c].count, local_class_stats[c].count);
        atomicAddDouble(&results->class_stats[c].sum_rel_err, local_class_stats[c].sum_rel_err);
        atomicMaxDouble(&results->class_stats[c].max_rel_err, local_class_stats[c].max_rel_err);
    }
#endif
#else
    // Host: OpenMP critical section for thread-safe reduction
    #pragma omp critical
    {
        results->max_rel_err = std::max(results->max_rel_err, local_max_rel_err);
        results->sum_rel_err += local_sum_rel_err;
        results->max_ulp = std::max(results->max_ulp, local_max_ulp);
        results->sum_ulp += local_sum_ulp;
        results->total_errs += local_errs;
        results->total_warnings += local_warnings;
        results->valid_count += local_valid;
        results->skipped_inf_nan += local_skipped;
        
        // Per-class statistics reduction (only if classification is enabled)
#if !defined(__RUN_CLASSIFY__) || __RUN_CLASSIFY__ != 0
        for (int c = 0; c < ACCURACY_CLASS_COUNT; c++)
        {
            results->class_stats[c].count += local_class_stats[c].count;
            results->class_stats[c].sum_rel_err += local_class_stats[c].sum_rel_err;
            results->class_stats[c].max_rel_err = std::max(results->class_stats[c].max_rel_err, 
                                                           local_class_stats[c].max_rel_err);
        }
#endif
    }
#endif
}

// ============================================================================
// Unified measure_accuracy function
// ============================================================================
template<typename TestFuncTag, 
         typename RefFuncTag>
ts_accuracy_result_t measure_accuracy(accuracy_mode mode = {accuracy_mode::work, TS_ACCURACY_RIGOR_WORK, "work"})
{
    // Type aliases
    using TestInputType  = tag_input_t<TestFuncTag>;
    using TestResultType = tag_result_t<TestFuncTag>;

    // Get rigor from mode struct and calculate total samples
    uint64_t total_samples = ((uint64_t)1) << ((uint64_t)mode.rigor);

    #if defined(__FIXED_INPUTS__)
        total_samples = 1;
    #endif
    
    // Common reduction results structure for both host and CUDA
    accuracy_reduction_t h_results = {};
    
    // Error log for capturing first N errors
    accuracy_error_log_t h_error_log = {};
    h_error_log.max_rel_err = -1.0;  // -1.0 means "not set" (any valid error >= 0.0 will update it)
    for (int c = 0; c < ACCURACY_CLASS_COUNT; c++)
    {
        h_error_log.class_max_rel_err[c] = -1.0;  // Initialize per-class max errors
    }

    // ========================================================================
    // Fixed inputs: parse strings on host before launching kernel
    // ========================================================================
    fixed_inputs_t<TestInputType> h_fixed_inputs = {};
    
#if defined(__FIXED_INPUTS__)
    // Helper to stringify macro values
    #define __FIXED_STRINGIFY__(x) #x
    #define __FIXED_TOSTRING__(x) __FIXED_STRINGIFY__(x)
    
    #if defined(__A1__)
        h_fixed_inputs.values[0] = string_to_value<TestInputType>(std::string(__FIXED_TOSTRING__(__A1__)));
        h_fixed_inputs.valid[0] = true;
    #endif
    #if defined(__A2__)
        h_fixed_inputs.values[1] = string_to_value<TestInputType>(std::string(__FIXED_TOSTRING__(__A2__)));
        h_fixed_inputs.valid[1] = true;
    #endif
    #if defined(__A3__)
        h_fixed_inputs.values[2] = string_to_value<TestInputType>(std::string(__FIXED_TOSTRING__(__A3__)));
        h_fixed_inputs.valid[2] = true;
    #endif
    #if defined(__A4__)
        h_fixed_inputs.values[3] = string_to_value<TestInputType>(std::string(__FIXED_TOSTRING__(__A4__)));
        h_fixed_inputs.valid[3] = true;
    #endif
#endif // __FIXED_INPUTS__

#if defined(__CUDACC__)
    // CUDA path: allocate device memory, launch kernel, copy back
    uint64_t blocks_needed = (total_samples + TS_ACCURACY_THREADS_PER_BLOCK - 1) / TS_ACCURACY_THREADS_PER_BLOCK;
    int num_blocks = (int)min(blocks_needed, (uint64_t)65536);
    
    accuracy_reduction_t* d_results;
    cudaMalloc(&d_results, sizeof(accuracy_reduction_t));
    cudaMemset(d_results, 0, sizeof(accuracy_reduction_t));
    
    accuracy_error_log_t* d_error_log;
    cudaMalloc(&d_error_log, sizeof(accuracy_error_log_t));
    cudaMemset(d_error_log, 0, sizeof(accuracy_error_log_t));
    // Initialize max_rel_err to -1.0 to indicate "not set" (any valid error >= 0.0 will update it)
    double init_max = -1.0;
    cudaMemcpy(&d_error_log->max_rel_err, &init_max, sizeof(double), cudaMemcpyHostToDevice);
    // Initialize per-class max errors to -1.0
    for (int c = 0; c < ACCURACY_CLASS_COUNT; c++)
    {
        cudaMemcpy(&d_error_log->class_max_rel_err[c], &init_max, sizeof(double), cudaMemcpyHostToDevice);
    }
    
    // Allocate and copy fixed inputs to device (if any are set)
    fixed_inputs_t<TestInputType>* d_fixed_inputs = nullptr;
#if defined(__FIXED_INPUTS__)
    cudaMalloc(&d_fixed_inputs, sizeof(fixed_inputs_t<TestInputType>));
    cudaMemcpy(d_fixed_inputs, &h_fixed_inputs, sizeof(fixed_inputs_t<TestInputType>), cudaMemcpyHostToDevice);
#endif
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    accuracy_kernel<TestFuncTag, RefFuncTag>
        <<<num_blocks, TS_ACCURACY_THREADS_PER_BLOCK>>>
        (mode.mode, total_samples, d_results, d_error_log, d_fixed_inputs);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(&h_results, d_results, sizeof(accuracy_reduction_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_error_log, d_error_log, sizeof(accuracy_error_log_t), cudaMemcpyDeviceToHost);
    cudaFree(d_results);
    cudaFree(d_error_log);
#if defined(__FIXED_INPUTS__)
    cudaFree(d_fixed_inputs);
#endif
#else
    // Host path: launch kernel from OpenMP parallel region
    // Pass pointer to fixed inputs (nullptr if not in fixed input mode)
    const fixed_inputs_t<TestInputType>* fixed_ptr = nullptr;
#if defined(__FIXED_INPUTS__)
    fixed_ptr = &h_fixed_inputs;
#endif
    #pragma omp parallel
    {
        accuracy_kernel<TestFuncTag, RefFuncTag>
            (mode.mode, total_samples, &h_results, &h_error_log, fixed_ptr);
    }
#endif

    // Build result structure (common for both paths)
    using RefResultType = tag_result_t<RefFuncTag>;
    
    ts_accuracy_result_t result = {};
    result.total_ops      = total_samples;
    result.valid_ops      = h_results.valid_count;
    result.skipped_inf_nan = h_results.skipped_inf_nan;
    result.max_rel_err    = h_results.max_rel_err;
    result.avg_rel_err    = (h_results.valid_count > 0) ? h_results.sum_rel_err / h_results.valid_count : 0.0;
    result.max_ulp        = h_results.max_ulp;
    result.avg_ulp        = (h_results.valid_count > 0) ? h_results.sum_ulp / h_results.valid_count : 0.0;
    result.total_errs     = h_results.total_errs;
    result.total_warnings = h_results.total_warnings;
    result.max_mantissa_bits = get_max_mantissa_bits<TestResultType>();
    constexpr int ref_bits = get_max_mantissa_bits<RefResultType>();
    int max_bits = std::min(ref_bits, result.max_mantissa_bits);
    result.correct_bits   = (h_results.valid_count > 0) ? compute_correct_bits(result.max_rel_err, max_bits) : -1;

    // Copy max error record (always tracked, regardless of threshold)
    // max_rel_err starts at -1.0; if it's >= 0.0, a valid comparison was made
    result.max_error_record = h_error_log.max_record;
    result.has_max_error = (h_error_log.max_rel_err >= 0.0);
    
    // Copy error log to result (capped at buffer size)
    result.error_log_count = h_error_log.count;
    uint64_t copy_count = std::min(h_error_log.count, (unsigned long long)TS_ACCURACY_ERROR_LOG_SIZE);
    for (uint64_t i = 0; i < copy_count; i++)
    {
        result.error_log[i] = h_error_log.records[i];
    }

    // Copy per-class statistics
    for (int c = 0; c < ACCURACY_CLASS_COUNT; c++)
    {
        result.class_results[c].count = h_results.class_stats[c].count;
        result.class_results[c].max_rel_err = h_results.class_stats[c].max_rel_err;
        result.class_results[c].avg_rel_err = (h_results.class_stats[c].count > 0) 
            ? h_results.class_stats[c].sum_rel_err / h_results.class_stats[c].count 
            : 0.0;
        
        // Compute correct bits - special handling for special value classes
        // For input_special, output_special, and output_denormal where no rel_err is computed,
        // these represent mismatches so correct_bits should be 0, not max_bits
        accuracy_class cls = static_cast<accuracy_class>(c);
        if (h_results.class_stats[c].count == 0)
        {
            result.class_results[c].correct_bits = -1;  // No samples
        }
        else if (cls == accuracy_class::input_special || 
                 cls == accuracy_class::output_special ||
                 (cls == accuracy_class::output_denormal && h_results.class_stats[c].max_rel_err == 0.0))
        {
            // Special values with no rel_err computed are mismatches - 0 correct bits
            result.class_results[c].correct_bits = 0;
        }
        else
        {
            result.class_results[c].correct_bits = compute_correct_bits(h_results.class_stats[c].max_rel_err, max_bits);
        }
        
        result.class_results[c].max_record = h_error_log.class_max_records[c];
        result.class_results[c].has_max_record = (h_error_log.class_max_rel_err[c] >= 0.0);
    }

    return result;
}

// NOTE: Printing functions (print_value_line, print_binary_mp, print_int_value_line,
// print_uint_value_line, print_bool_value_line, get_integer_arg_type, print_error_record,
// print_accuracy_log) have been moved to ts_print.hpp

#endif // __TS_ACCURACY_HPP__
