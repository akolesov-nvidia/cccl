/*
    ts_dataset.hpp - FPMP Test Suite Dataset Generation
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Dataset generation utilities for accuracy testing. Provides various input generation strategies:
    - work:    Uniform distribution over working range
    - normal:  Gaussian distribution around mean
    - special: Random mix of IEEE-754 special values (inf, nan, denorm, zero)
    - pattern: Exhaustive bit-pattern based testing
*/

#ifndef __TS_DATASET_HPP__
#define __TS_DATASET_HPP__

// ts_types.hpp includes ts.hpp and defines all common types
#include "ts_types.hpp"
#include "ts_functions.hpp"
#include <random>
#include <cmath>

#if !defined(__CUDACC__)
    // libquadmath ships only with GCC's runtime on x86 (32/64-bit).
    // Skip the include on ARM/ARM64/RISC-V/PowerPC, MSVC, and Windows.
    #if (defined(__x86_64__) || defined(_M_X64) || \
         defined(__i386__)   || defined(_M_IX86)) \
        && !defined(_MSC_VER) && !defined(_WIN32)
        #include <quadmath.h>  // For sqrtq, logq, cosq in quad-precision input generation
    #endif
#else
#include <curand_kernel.h>
#endif

// Import commonly used items from ts namespace for cleaner code
using ts::bit_cast;
using ts::is_multiprecision_v;
using ts::mp_component_t;

// ============================================================================
// Key points for single precision floating point numbers (IEEE 754 binary32)
// ============================================================================
#define FP32_NEG_INF        0xFF800000u  // -Infinity
#define FP32_NEG_MAX_NORM   0xFF7FFFFFu  // -3.40282e+38 (Most negative normalized)
#define FP32_NEG_ONE        0xBF800000u  // -1.0
#define FP32_NEG_MIN_NORM   0x80800000u  // -1.17549e-38 (Smallest negative normalized)
#define FP32_NEG_MAX_DENORM 0x807FFFFFu  // Largest negative denormalized
#define FP32_NEG_MIN_DENORM 0x80000001u  // Smallest negative denormalized
#define FP32_NEG_ZERO       0x80000000u  // -0.0
#define FP32_POS_ZERO       0x00000000u  // +0.0
#define FP32_POS_MIN_DENORM 0x00000001u  // Smallest positive denormalized
#define FP32_POS_MAX_DENORM 0x007FFFFFu  // Largest positive denormalized
#define FP32_POS_MIN_NORM   0x00800000u  // +1.17549e-38 (Smallest positive normalized)
#define FP32_POS_ONE        0x3F800000u  // +1.0
#define FP32_POS_MAX_NORM   0x7F7FFFFFu  // +3.40282e+38 (Most positive normalized)
#define FP32_POS_INF        0x7F800000u  // +Infinity
#define FP32_SNAN           0x7FA00000u  // Signaling NaN
#define FP32_QNAN           0x7FC00000u  // Quiet NaN

// FP32 values for integer conversion tests (representable integers)
#define FP32_NEG_LLONG_MAX  0xDF000000u  // -9223372036854775808 (-2^63, closest representable)
#define FP32_NEG_INT_MAX    0xCF000000u  // -2147483648 (-2^31)
#define FP32_POS_INT_MAX    0x4EFFFFFF  // +2147483520 (closest to 2^31-1 in float)
#define FP32_POS_LLONG_MAX  0x5EFFFFFF  // +9223371487098961920 (closest to 2^63-1 in float)
#define FP32_POS_UINT_MAX   0x4F800000u  // +4294967296 (2^32, overflow for uint32)
#define FP32_POS_ULLONG_MAX 0x5F800000u  // +18446744073709551616 (2^64, overflow for uint64)

// ============================================================================
// Key points for double precision floating point numbers
// ============================================================================
#define FP64_NEG_INF        0xfff0000000000000  // -Infinity
#define FP64_NEG_MAX_NORM   0xffefffffffffffff  // -1.79769e+308 (Most negative normalized)
#define FP64_NEG_ONE        0xbff0000000000000  // -1.0
#define FP64_NEG_MIN_NORM   0x8010000000000000  // -2.22507e-308 (Smallest negative normalized)
#define FP64_NEG_MAX_DENORM 0x800fffffffffffff  // -2.22507e-308 (Largest negative denormalized)
#define FP64_NEG_MIN_DENORM 0x8000000000000001  // -4.94066e-324 (Smallest negative denormalized)
#define FP64_NEG_ZERO       0x8000000000000000  // -0.0
#define FP64_POS_ZERO       0x0000000000000000  // +0.0
#define FP64_POS_MIN_DENORM 0x0000000000000001  // +4.94066e-324 (Smallest positive denormalized)
#define FP64_POS_MAX_DENORM 0x000fffffffffffff  // +2.22507e-308 (Largest positive denormalized)
#define FP64_POS_MIN_NORM   0x0010000000000000  // +2.22507e-308 (Smallest positive normalized)
#define FP64_POS_ONE        0x3ff0000000000000  // +1.0
#define FP64_POS_MAX_NORM   0x7fefffffffffffff  // +1.79769e+308 (Most positive normalized)
#define FP64_POS_INF        0x7ff0000000000000  // +Infinity
#define FP64_SNAN           0x7ff7ffffffffffff  // Signaling NaN
#define FP64_QNAN           0x7fffffffffffffff  // Quiet NaN

// FP64 values for integer conversion tests (representable integers)
#define FP64_NEG_LLONG_MAX  0xC3E0000000000000  // -9223372036854775808 (-2^63)
#define FP64_NEG_INT_MAX    0xC1E0000000200000  // -2147483649 (just below -2^31)
#define FP64_NEG_INT_MIN    0xC1E0000000000000  // -2147483648 (-2^31)
#define FP64_POS_INT_MAX    0x41DFFFFFFFC00000  // +2147483647 (2^31-1)
#define FP64_POS_UINT_MAX   0x41EFFFFFFFE00000  // +4294967295 (2^32-1)
#define FP64_POS_LLONG_MAX  0x43DFFFFFFFFFFFFF  // +9223372036854774784 (closest to 2^63-1)
#define FP64_POS_ULLONG_MAX 0x43EFFFFFFFFFFFFF  // +18446744073709549568 (closest to 2^64-1)

// ============================================================================
// Key points for 32-bit signed integers
// ============================================================================
#define INT32_ZERO          0                   // Zero
#define INT32_ONE           1                   // One
#define INT32_NEG_ONE       (-1)                // Negative one
#define INT32_TWO           2                   // Two
#define INT32_NEG_TWO       (-2)                // Negative two
#define INT32_MAXIMUM       0x7FFFFFFF          // Maximum positive (2147483647)
#define INT32_MINIMUM       (-0x7FFFFFFF - 1)   // Minimum negative (-2147483648)
#define INT32_MAX_M1        0x7FFFFFFE          // Maximum - 1
#define INT32_MIN_P1        (-0x7FFFFFFF)       // Minimum + 1
#define INT32_HALF_MAX      0x3FFFFFFF          // Half max
#define INT32_HALF_MIN      (-0x40000000)       // Half min
#define INT32_SMALL_POS     100                 // Small positive
#define INT32_SMALL_NEG     (-100)              // Small negative
#define INT32_MILLION       1000000             // Million
#define INT32_NEG_MILLION   (-1000000)          // Negative million
#define INT32_NEAR_MAX      0x7FFFFFF0          // Near max with low bits clear

// ============================================================================
// Key points for 64-bit signed integers
// ============================================================================
#define INT64_ZERO          0LL                             // Zero
#define INT64_ONE           1LL                             // One
#define INT64_NEG_ONE       (-1LL)                          // Negative one
#define INT64_TWO           2LL                             // Two
#define INT64_NEG_TWO       (-2LL)                          // Negative two
#define INT64_MAXIMUM       0x7FFFFFFFFFFFFFFFLL            // Maximum positive
#define INT64_MINIMUM       (-0x7FFFFFFFFFFFFFFFLL - 1)     // Minimum negative
#define INT64_MAX_M1        0x7FFFFFFFFFFFFFFELL            // Maximum - 1
#define INT64_MIN_P1        (-0x7FFFFFFFFFFFFFFELL)         // Minimum + 1
#define INT64_HALF_MAX      0x3FFFFFFFFFFFFFFFLL            // Half max
#define INT64_HALF_MIN      (-0x4000000000000000LL)         // Half min
#define INT64_SMALL_POS     100LL                           // Small positive
#define INT64_SMALL_NEG     (-100LL)                        // Small negative
#define INT64_TRILLION      1000000000000LL                 // Trillion
#define INT64_NEG_TRILLION  (-1000000000000LL)              // Negative trillion
#define INT64_NEAR_MAX      0x7FFFFFFFFFFFFFF0LL            // Near max with low bits clear

// ============================================================================
// Key points for 32-bit unsigned integers
// ============================================================================
#define UINT32_ZERO         0U                  // Zero
#define UINT32_ONE          1U                  // One
#define UINT32_TWO          2U                  // Two
#define UINT32_MAXIMUM      0xFFFFFFFFU         // Maximum
#define UINT32_MAX_M1       0xFFFFFFFEU         // Maximum - 1
#define UINT32_HALF_MAX     0x7FFFFFFFU         // Half max (MSB clear)
#define UINT32_HALF_MAX_P1  0x80000000U         // Half max + 1 (MSB set)
#define UINT32_SMALL        100U                // Small value
#define UINT32_THOUSAND     1000U               // Thousand
#define UINT32_MILLION      1000000U            // Million
#define UINT32_BILLION      1000000000U         // Billion
#define UINT32_MSB_SET      0x80000000U         // MSB set
#define UINT32_MSB_CLR      0x7FFFFFFFU         // MSB clear, rest set
#define UINT32_UPPER_HALF   0xFFFF0000U         // Upper half set
#define UINT32_LOWER_HALF   0x0000FFFFU         // Lower half set
#define UINT32_ALT_BITS     0xAAAAAAAAU         // Alternating bits (10...)

// ============================================================================
// Key points for 64-bit unsigned integers
// ============================================================================
#define UINT64_ZERO         0ULL                            // Zero
#define UINT64_ONE          1ULL                            // One
#define UINT64_TWO          2ULL                            // Two
#define UINT64_MAXIMUM      0xFFFFFFFFFFFFFFFFULL           // Maximum
#define UINT64_MAX_M1       0xFFFFFFFFFFFFFFFEULL           // Maximum - 1
#define UINT64_HALF_MAX     0x7FFFFFFFFFFFFFFFULL           // Half max (MSB clear)
#define UINT64_HALF_MAX_P1  0x8000000000000000ULL           // Half max + 1 (MSB set)
#define UINT64_SMALL        100ULL                          // Small value
#define UINT64_TRILLION     1000000000000ULL                // Trillion
#define UINT64_MSB_SET      0x8000000000000000ULL           // MSB set
#define UINT64_MSB_CLR      0x7FFFFFFFFFFFFFFFULL           // MSB clear, rest set
#define UINT64_UPPER_32     0xFFFFFFFF00000000ULL           // Upper 32 bits set
#define UINT64_LOWER_32     0x00000000FFFFFFFFULL           // Lower 32 bits set
#define UINT64_ALT_10       0xAAAAAAAAAAAAAAAAULL           // Alternating bits (10...)
#define UINT64_ALT_01       0x5555555555555555ULL           // Alternating bits (01...)
#define UINT64_NEAR_MAX     0xFFFFFFFFFFFFFF00ULL           // Near max with low byte clear

// ============================================================================
// Accuracy measurement mode - stores mode type, rigor, and name
// ============================================================================
struct accuracy_mode
{
    enum type { 
        work,    // Uniform distribution over work_beg to work_end (wide coverage)
        normal,  // Gaussian distribution around normal_mean with normal_stddev (focused testing)
        special,  // Random mix of special FP values (inf, nan, denorm, zero) + random work value
        pattern,  // Exhaustive testing of all possible input combinations
    };
    
    type        mode;       // Mode type
    int         rigor;      // 2^rigor samples
    const char* name;       // Human-readable name
    
    // Constructor with all parameters
    constexpr accuracy_mode(type m, int r, const char* n) : mode(m), rigor(r), name(n) {}
};

// ============================================================================
// Convert reference type to test type (handles RefType -> fpmp_type)
// For fp32mp2: uses double as intermediate (matches FPMP library behavior)
// For fp64mp2: uses __ts_fp128 as intermediate
// ============================================================================
template<typename TestType, typename RefType>
__HOST_DEVICE_DECL__ inline TestType ts_convert_ref_to_test(RefType ref_val)
{
    if constexpr (is_multiprecision_v<TestType>)
    {
        using ComponentType = mp_component_t<TestType>;
        
        if constexpr (std::is_same_v<ComponentType, float>)
        {
            // fp32mp2: use double as intermediate precision (matches FPMP library)
            double ref_dbl = static_cast<double>(ref_val);
            float hi = static_cast<float>(ref_dbl);
            float lo = static_cast<float>(ref_dbl - static_cast<double>(hi));
            return TestType(hi, lo);
        }
        else if constexpr (std::is_same_v<ComponentType, double>)
        {
            // fp64mp2: use __ts_fp128 as intermediate precision
            __ts_fp128 ref_q = static_cast<__ts_fp128>(ref_val);
            double hi = static_cast<double>(ref_q);
            double lo = static_cast<double>(ref_q - static_cast<__ts_fp128>(hi));
            return TestType(hi, lo);
        }
    }
    else
    {
        // For scalar types, direct construction works
        return TestType(ref_val);
    }
}

// ============================================================================
// Convert test type to reference type (handles fpmp_type -> __ts_fp128)
// ============================================================================
template<typename RefType, typename TestType>
__HOST_DEVICE_DECL__ inline RefType ts_convert_test_to_ref(TestType test_val)
{
    if constexpr (std::is_same_v<RefType, __ts_fp128> && is_multiprecision_v<TestType>)
    {
        // Manual conversion from multi-precision type to __ts_fp128
        return static_cast<__ts_fp128>(test_val.hi()) + static_cast<__ts_fp128>(test_val.lo());
    }
    else if constexpr (std::is_same_v<RefType, double> && is_multiprecision_v<TestType>)
    {
        // Manual conversion from multi-precision type to double
        // Avoids using the library's operator double() so that mp2fp tests
        // can detect bugs in __fpmp2_to_double independently
        return static_cast<double>(test_val.hi()) + static_cast<double>(test_val.lo());
    }
    else
    {
        return static_cast<RefType>(test_val);
    }
}

// ============================================================================
// Unified input generation for accuracy testing
// ============================================================================
// Generate input value based on accuracy mode and function tag properties
// - work:   Uniform distribution between work_beg and work_end from TestFuncTag
// Uses double for range calculations (avoids integer overflow for work_beg/work_end)
// For integer types: generates in double, then casts to integer
// For floating-point types: uses RefType for extended precision if available
template<typename TestType, typename RefType, typename TestFuncTag, typename StateType>
__HOST_DEVICE_DECL__ TestType generate_input_work(StateType& state)
{
    TestType res;
    
    // For integer input types, use double for range calculation to avoid overflow
    // (e.g., work_end=4e9 would overflow uint32_t)
    if constexpr (std::is_integral_v<TestType>)
    {
        double beg = TestFuncTag::work_beg();
        double end = TestFuncTag::work_end();
#if defined(__CUDACC__)
        double u = curand_uniform_double(&state);
#else
        std::uniform_real_distribution<double> dist01(0.0, 1.0);
        double u = dist01(state);
#endif
        double val = beg + u * (end - beg);
        res = static_cast<TestType>(val);
    }
    else
    {
        // Floating-point types: use RefType for extended precision
        RefType u;
        RefType beg = static_cast<RefType>(TestFuncTag::work_beg());
        RefType end = static_cast<RefType>(TestFuncTag::work_end());
#if defined(__CUDACC__)
        double u_hi = curand_uniform_double(&state);
        if constexpr (std::is_same_v<RefType, __ts_fp128>) 
        {
            double u_lo = curand_uniform_double(&state) * 0x1p-53;
            u = static_cast<RefType>(u_hi) + static_cast<RefType>(u_lo);
        } 
        else 
        {
            u = static_cast<RefType>(u_hi);
        }
#else
        std::uniform_real_distribution<double> dist01(0.0, 1.0);
        double u_hi = dist01(state);
        if constexpr (std::is_same_v<RefType, __ts_fp128>) 
        {
            double u_lo = dist01(state) * 0x1p-53;
            u = static_cast<RefType>(u_hi) + static_cast<RefType>(u_lo);
        } 
        else 
        {
            u = static_cast<RefType>(u_hi);
        }
#endif
        res = ts_convert_ref_to_test<TestType, RefType>(beg + u * (end - beg));
    }
    return res;
}

// ============================================================================
// Generate Gaussian-distributed input using Box-Muller transform
// Uses normal_mean() and normal_stddev() from TestFuncTag
// For integer types: generates in double, then casts to integer
// ============================================================================
template<typename TestType, typename RefType, typename TestFuncTag, typename StateType>
__HOST_DEVICE_DECL__ TestType generate_input_normal(StateType& state)
{
    TestType res;
    
    // For integer input types, use double for calculation to avoid overflow
    if constexpr (std::is_integral_v<TestType>)
    {
        double mean   = TestFuncTag::normal_mean();
        double stddev = TestFuncTag::normal_stddev();
#if defined(__CUDACC__)
        double n = curand_normal_double(&state);
#else
        std::normal_distribution<double> dist(0.0, 1.0);
        double n = dist(state);
#endif
        double val = mean + n * stddev;
        res = static_cast<TestType>(val);
    }
    else
    {
        // Floating-point types: use RefType for extended precision
        RefType mean   = static_cast<RefType>(TestFuncTag::normal_mean());
        RefType stddev = static_cast<RefType>(TestFuncTag::normal_stddev());

#if defined(__CUDACC__)
        double n_hi = curand_normal_double(&state);
        if constexpr (std::is_same_v<RefType, __ts_fp128>)
        {
            double n_lo = curand_normal_double(&state) * 0x1p-53;
            RefType n = static_cast<RefType>(n_hi) + static_cast<RefType>(n_lo);
            res = ts_convert_ref_to_test<TestType, RefType>(mean + n * stddev);
        }
        else
        {
            res = ts_convert_ref_to_test<TestType, RefType>(mean + static_cast<RefType>(n_hi) * stddev);
        }
#else
        std::normal_distribution<double> dist(0.0, 1.0);
        double n_hi = dist(state);
        if constexpr (std::is_same_v<RefType, __ts_fp128>)
        {
            double n_lo = dist(state) * 0x1p-53;
            RefType n = static_cast<RefType>(n_hi) + static_cast<RefType>(n_lo);
            res = ts_convert_ref_to_test<TestType, RefType>(mean + n * stddev);
        }
        else
        {
            res = ts_convert_ref_to_test<TestType, RefType>(mean + static_cast<RefType>(n_hi) * stddev);
        }
#endif
    }
    return res;
}

// ============================================================================
// Template: Generate special floating-point value by index
// Tables are defined inline to work in both host and device code
// ============================================================================
template<typename FpType>
__HOST_DEVICE_DECL__ FpType generate_special_fp_scalar(int index)
{
    if constexpr (std::is_same_v<FpType, float>)
    {
        // Special FP32 values: inf, nan, zero, denorm, normalized boundaries
        constexpr uint32_t special_fp32[] = {
            FP32_NEG_INF,        // 0: -Infinity
            FP32_NEG_MAX_NORM,   // 1: Most negative normalized
            FP32_NEG_ONE,        // 2: -1.0
            FP32_NEG_MIN_NORM,   // 3: Smallest negative normalized
            FP32_NEG_MAX_DENORM, // 4: Largest negative denormalized
            FP32_NEG_MIN_DENORM, // 5: Smallest negative denormalized
            FP32_NEG_ZERO,       // 6: -0.0
            FP32_POS_ZERO,       // 7: +0.0
            FP32_POS_MIN_DENORM, // 8: Smallest positive denormalized
            FP32_POS_MAX_DENORM, // 9: Largest positive denormalized
            FP32_POS_MIN_NORM,   // 10: Smallest positive normalized
            FP32_POS_ONE,        // 11: +1.0
            FP32_POS_MAX_NORM,   // 12: Most positive normalized
            FP32_POS_INF,        // 13: +Infinity
            FP32_SNAN,           // 14: Signaling NaN
            FP32_QNAN            // 15: Quiet NaN
        };
        float f;
        uint32_t bits = special_fp32[index % 16];
        memcpy(&f, &bits, sizeof(f));
        return f;
    }
    else if constexpr (std::is_same_v<FpType, double>)
    {
        // Special FP64 values: inf, nan, zero, denorm, normalized boundaries
        constexpr uint64_t special_fp64[] = {
            FP64_NEG_INF,        // 0: -Infinity
            FP64_NEG_MAX_NORM,   // 1: Most negative normalized
            FP64_NEG_ONE,        // 2: -1.0
            FP64_NEG_MIN_NORM,   // 3: Smallest negative normalized
            FP64_NEG_MAX_DENORM, // 4: Largest negative denormalized
            FP64_NEG_MIN_DENORM, // 5: Smallest negative denormalized
            FP64_NEG_ZERO,       // 6: -0.0
            FP64_POS_ZERO,       // 7: +0.0
            FP64_POS_MIN_DENORM, // 8: Smallest positive denormalized
            FP64_POS_MAX_DENORM, // 9: Largest positive denormalized
            FP64_POS_MIN_NORM,   // 10: Smallest positive normalized
            FP64_POS_ONE,        // 11: +1.0
            FP64_POS_MAX_NORM,   // 12: Most positive normalized
            FP64_POS_INF,        // 13: +Infinity
            FP64_SNAN,           // 14: Signaling NaN
            FP64_QNAN            // 15: Quiet NaN
        };
        double d;
        uint64_t bits = special_fp64[index % 16];
        memcpy(&d, &bits, sizeof(d));
        return d;
    }
    else
    {
        // Fallback for other FP types: use double and cast
        return static_cast<FpType>(generate_special_fp_scalar<double>(index));
    }
}

// Legacy wrappers for backwards compatibility
__HOST_DEVICE_DECL__ inline float generate_special_fp32_scalar(int index) { return generate_special_fp_scalar<float>(index); }
__HOST_DEVICE_DECL__ inline double generate_special_fp64_scalar(int index) { return generate_special_fp_scalar<double>(index); }

// Number of special floating-point values in the table
constexpr int SPECIAL_FP_COUNT = 16;

// Number of special values for compact table display (excludes sNaN)
constexpr int SPECIAL_TABLE_COUNT = 15;

// Mapping from table index to special value index (skips sNaN at index 14)
__HOST_DECL__ inline int get_special_table_index(int table_idx)
{
    // Indices 0-13 map directly, index 14 maps to qNaN (15)
    return (table_idx < 14) ? table_idx : 15;
}

// Short names for special values (for table headers)
__HOST_DECL__ inline const char* get_special_fp_name(int index)
{
    static const char* names[] = {
        "-INF ",      // 0: -Infinity
        "-maxN",      // 1: Most negative normalized
        "-1   ",      // 2: -1.0
        "-minN",      // 3: Smallest negative normalized
        "-maxD",       // 4: Largest negative denormalized
        "-minD",       // 5: Smallest negative denormalized
        "-0   ",        // 6: -0.0
        "+0   ",        // 7: +0.0
        "+minD",       // 8: Smallest positive denormalized
        "+maxD",       // 9: Largest positive denormalized
        "+minN",      // 10: Smallest positive normalized
        "+1   ",      // 11: +1.0
        "+maxN",      // 12: Most positive normalized
        "+INF ",      // 13: +Infinity
        "SNAN ",      // 14: Signaling NaN
        "QNAN "       // 15: Quiet NaN
    };
    return names[index % SPECIAL_FP_COUNT];
}

// ============================================================================
// Special values for integer output functions (mp2int, mp2ll, mp2uint, mp2ull)
// Excludes denormals/min_norm, includes integer boundary values
// ============================================================================
constexpr int SPECIAL_INT_OUTPUT_COUNT = 12;

// Generate special FP value for integer output tests (by index)
template<typename FpType>
__HOST_DEVICE_DECL__ FpType generate_special_int_output_scalar(int index)
{
    if constexpr (std::is_same_v<FpType, float>)
    {
        // FP32 values relevant for integer conversion
        constexpr uint32_t table[] = {
            FP32_NEG_INF,        // 0: -Infinity
            FP32_NEG_LLONG_MAX,  // 1: -2^63 (long long min)
            FP32_NEG_INT_MAX,    // 2: -2^31 (int min)
            FP32_NEG_ONE,        // 3: -1.0
            FP32_NEG_ZERO,       // 4: -0.0
            FP32_POS_ZERO,       // 5: +0.0
            FP32_POS_ONE,        // 6: +1.0
            FP32_POS_INT_MAX,    // 7: ~2^31-1 (int max)
            FP32_POS_UINT_MAX,   // 8: 2^32 (uint max + 1)
            FP32_POS_LLONG_MAX,  // 9: ~2^63-1 (llong max)
            FP32_POS_INF,        // 10: +Infinity
            FP32_QNAN            // 11: Quiet NaN
        };
        float f;
        uint32_t bits = table[index % SPECIAL_INT_OUTPUT_COUNT];
        memcpy(&f, &bits, sizeof(f));
        return f;
    }
    else if constexpr (std::is_same_v<FpType, double>)
    {
        // FP64 values relevant for integer conversion
        constexpr uint64_t table[] = {
            FP64_NEG_INF,        // 0: -Infinity
            FP64_NEG_LLONG_MAX,  // 1: -2^63 (long long min)
            FP64_NEG_INT_MIN,    // 2: -2^31 (int min)
            FP64_NEG_ONE,        // 3: -1.0
            FP64_NEG_ZERO,       // 4: -0.0
            FP64_POS_ZERO,       // 5: +0.0
            FP64_POS_ONE,        // 6: +1.0
            FP64_POS_INT_MAX,    // 7: 2^31-1 (int max)
            FP64_POS_UINT_MAX,   // 8: 2^32-1 (uint max)
            FP64_POS_LLONG_MAX,  // 9: ~2^63-1 (llong max)
            FP64_POS_INF,        // 10: +Infinity
            FP64_QNAN            // 11: Quiet NaN
        };
        double d;
        uint64_t bits = table[index % SPECIAL_INT_OUTPUT_COUNT];
        memcpy(&d, &bits, sizeof(d));
        return d;
    }
    else
    {
        return static_cast<FpType>(generate_special_int_output_scalar<double>(index));
    }
}

// Short names for integer output special values
__HOST_DECL__ inline const char* get_special_int_output_name(int index)
{
    static const char* names[] = {
        "-INF ",      // 0: -Infinity
        "-LMAX",      // 1: -2^63 (long long min)
        "-IMAX",      // 2: -2^31 (int min)
        "-1   ",      // 3: -1.0
        "-0   ",      // 4: -0.0
        "+0   ",      // 5: +0.0
        "+1   ",      // 6: +1.0
        "+IMAX",      // 7: int max (~2^31-1)
        "+UMAX",      // 8: uint max (2^32-1 or 2^32)
        "+LMAX",      // 9: llong max (~2^63-1)
        "+INF ",      // 10: +Infinity
        "QNAN "       // 11: Quiet NaN
    };
    return names[index % SPECIAL_INT_OUTPUT_COUNT];
}

// ============================================================================
// Special integer input values for int2mp, ll2mp, uint2mp, ull2mp
// ============================================================================
constexpr int SPECIAL_INT_INPUT_COUNT = 16;

// Short names for integer input special values (signed)
__HOST_DECL__ inline const char* get_special_int32_name(int index)
{
    static const char* names[] = {
        "0    ",      // 0: Zero
        "+1   ",      // 1: One
        "-1   ",      // 2: Negative one
        "+2   ",      // 3: Two
        "-2   ",      // 4: Negative two
        "+MAX ",      // 5: Maximum positive (2^31-1)
        "-MAX ",      // 6: Minimum negative (-2^31)
        "+Mx-1",      // 7: Max - 1
        "-Mx+1",      // 8: Min + 1
        "+half",      // 9: Half max
        "-half",      // 10: Half min
        "+100 ",      // 11: Small positive
        "-100 ",      // 12: Small negative
        "+1M  ",      // 13: Million
        "-1M  ",      // 14: Negative million
        "~MAX "       // 15: Near max
    };
    return names[index % SPECIAL_INT_INPUT_COUNT];
}

__HOST_DECL__ inline const char* get_special_int64_name(int index)
{
    static const char* names[] = {
        "0    ",      // 0: Zero
        "+1   ",      // 1: One
        "-1   ",      // 2: Negative one
        "+2   ",      // 3: Two
        "-2   ",      // 4: Negative two
        "+MAX ",      // 5: Maximum positive (2^63-1)
        "-MAX ",      // 6: Minimum negative (-2^63)
        "+Mx-1",      // 7: Max - 1
        "-Mx+1",      // 8: Min + 1
        "+half",      // 9: Half max
        "-half",      // 10: Half min
        "+100 ",      // 11: Small positive
        "-100 ",      // 12: Small negative
        "+1T  ",      // 13: Trillion
        "-1T  ",      // 14: Negative trillion
        "~MAX "       // 15: Near max
    };
    return names[index % SPECIAL_INT_INPUT_COUNT];
}

__HOST_DECL__ inline const char* get_special_uint32_name(int index)
{
    static const char* names[] = {
        "0    ",      // 0: Zero
        "1    ",      // 1: One
        "2    ",      // 2: Two
        "MAX  ",      // 3: Maximum (2^32-1)
        "Mx-1 ",      // 4: Max - 1
        "half ",      // 5: Half max (MSB clear)
        "hlf+1",      // 6: Half max + 1 (MSB set)
        "100  ",      // 7: Small value
        "1K   ",      // 8: Thousand
        "1M   ",      // 9: Million
        "1B   ",      // 10: Billion
        "MSB  ",      // 11: MSB set
        "~MSB ",      // 12: MSB clear, rest set
        "hi16 ",      // 13: Upper half set
        "lo16 ",      // 14: Lower half set
        "0xAA "       // 15: Alternating bits
    };
    return names[index % SPECIAL_INT_INPUT_COUNT];
}

__HOST_DECL__ inline const char* get_special_uint64_name(int index)
{
    static const char* names[] = {
        "0    ",      // 0: Zero
        "1    ",      // 1: One
        "2    ",      // 2: Two
        "MAX  ",      // 3: Maximum (2^64-1)
        "Mx-1 ",      // 4: Max - 1
        "half ",      // 5: Half max (MSB clear)
        "hlf+1",      // 6: Half max + 1 (MSB set)
        "100  ",      // 7: Small value
        "1T   ",      // 8: Trillion
        "MSB  ",      // 9: MSB set
        "~MSB ",      // 10: MSB clear, rest set
        "hi32 ",      // 11: Upper 32 bits set
        "lo32 ",      // 12: Lower 32 bits set
        "0xAA ",      // 13: Alternating bits (10...)
        "0x55 ",      // 14: Alternating bits (01...)
        "~MAX "       // 15: Near max
    };
    return names[index % SPECIAL_INT_INPUT_COUNT];
}

// ============================================================================
// Generate a random index for special value selection (0-16)
// ============================================================================
template<typename StateType>
__HOST_DEVICE_DECL__ int generate_special_index(StateType& state)
{
    constexpr int num_options = 17;  // 16 special + 1 random work value
#if defined(__CUDACC__)
    return (int)(curand_uniform(&state) * num_options) % num_options;
#else
    std::uniform_int_distribution<int> dist(0, num_options - 1);
    return dist(state);
#endif
}

// ============================================================================
// Template: Generate special integer value by index
// Tables are defined inline to work in both host and device code
// ============================================================================
template<typename IntType>
__HOST_DEVICE_DECL__ IntType generate_special_int_scalar(int index)
{
    if constexpr (std::is_same_v<IntType, int32_t>)
    {
        // Special values for int32_t: 0, ±1, ±2, min, max, boundaries, common values
        constexpr int32_t table[] = {
            INT32_ZERO,             // 0: Zero
            INT32_ONE,              // 1: One
            INT32_NEG_ONE,          // 2: Negative one
            INT32_TWO,              // 3: Two
            INT32_NEG_TWO,          // 4: Negative two
            INT32_MAXIMUM,          // 5: Maximum positive
            INT32_MINIMUM,          // 6: Minimum negative
            INT32_MAX_M1,           // 7: Max - 1
            INT32_MIN_P1,           // 8: Min + 1
            INT32_HALF_MAX,         // 9: Half max
            INT32_HALF_MIN,         // 10: Half min
            INT32_SMALL_POS,        // 11: Small positive
            INT32_SMALL_NEG,        // 12: Small negative
            INT32_MILLION,          // 13: Million
            INT32_NEG_MILLION,      // 14: Negative million
            INT32_NEAR_MAX          // 15: Near max with low bits clear
        };
        return table[index % 16];
    }
    else if constexpr (std::is_same_v<IntType, int64_t>)
    {
        // Special values for int64_t: 0, ±1, ±2, min, max, boundaries, common values
        constexpr int64_t table[] = {
            INT64_ZERO,             // 0: Zero
            INT64_ONE,              // 1: One
            INT64_NEG_ONE,          // 2: Negative one
            INT64_TWO,              // 3: Two
            INT64_NEG_TWO,          // 4: Negative two
            INT64_MAXIMUM,          // 5: Maximum positive
            INT64_MINIMUM,          // 6: Minimum negative
            INT64_MAX_M1,           // 7: Max - 1
            INT64_MIN_P1,           // 8: Min + 1
            INT64_HALF_MAX,         // 9: Half max
            INT64_HALF_MIN,         // 10: Half min
            INT64_SMALL_POS,        // 11: Small positive
            INT64_SMALL_NEG,        // 12: Small negative
            INT64_TRILLION,         // 13: Trillion
            INT64_NEG_TRILLION,     // 14: Negative trillion
            INT64_NEAR_MAX          // 15: Near max with low bits clear
        };
        return table[index % 16];
    }
    else if constexpr (std::is_same_v<IntType, uint32_t>)
    {
        // Special values for uint32_t: 0, 1, 2, max, boundaries, bit patterns
        constexpr uint32_t table[] = {
            UINT32_ZERO,            // 0: Zero
            UINT32_ONE,             // 1: One
            UINT32_TWO,             // 2: Two
            UINT32_MAXIMUM,         // 3: Maximum
            UINT32_MAX_M1,          // 4: Max - 1
            UINT32_HALF_MAX,        // 5: Half max (MSB clear)
            UINT32_HALF_MAX_P1,     // 6: Half max + 1 (MSB set)
            UINT32_SMALL,           // 7: Small value
            UINT32_THOUSAND,        // 8: Thousand
            UINT32_MILLION,         // 9: Million
            UINT32_BILLION,         // 10: Billion
            UINT32_MSB_SET,         // 11: MSB set
            UINT32_MSB_CLR,         // 12: MSB clear, rest set
            UINT32_UPPER_HALF,      // 13: Upper half set
            UINT32_LOWER_HALF,      // 14: Lower half set
            UINT32_ALT_BITS         // 15: Alternating bits
        };
        return table[index % 16];
    }
    else if constexpr (std::is_same_v<IntType, uint64_t>)
    {
        // Special values for uint64_t: 0, 1, 2, max, boundaries, bit patterns
        constexpr uint64_t table[] = {
            UINT64_ZERO,            // 0: Zero
            UINT64_ONE,             // 1: One
            UINT64_TWO,             // 2: Two
            UINT64_MAXIMUM,         // 3: Maximum
            UINT64_MAX_M1,          // 4: Max - 1
            UINT64_HALF_MAX,        // 5: Half max (MSB clear)
            UINT64_HALF_MAX_P1,     // 6: Half max + 1 (MSB set)
            UINT64_SMALL,           // 7: Small value
            UINT64_TRILLION,        // 8: Trillion
            UINT64_MSB_SET,         // 9: MSB set
            UINT64_MSB_CLR,         // 10: MSB clear, rest set
            UINT64_UPPER_32,        // 11: Upper 32 bits set
            UINT64_LOWER_32,        // 12: Lower 32 bits set
            UINT64_ALT_10,          // 13: Alternating bits (10...)
            UINT64_ALT_01,          // 14: Alternating bits (01...)
            UINT64_NEAR_MAX         // 15: Near max with low byte clear
        };
        return table[index % 16];
    }
    else
    {
        // Fallback for other integer types: use int64_t and cast
        return static_cast<IntType>(generate_special_int_scalar<int64_t>(index));
    }
}

// ============================================================================
// Generate input with random mix of special FP values
// ============================================================================
template<typename TestType, typename RefType, typename TestFuncTag, typename StateType, bool per_component = false>
__HOST_DEVICE_DECL__ TestType generate_input_special(StateType& state, [[maybe_unused]] int arg_idx = 0)
{
    // Handle integer types specially to avoid undefined behavior from FP->int casts
    if constexpr (std::is_integral_v<TestType>)
    {
        int index = generate_special_index(state);
        // Use table-based lookup for special integer values (index 0-15)
        // For index >= 16, generate random work value
        if (index >= 16) { return generate_input_work<TestType, RefType, TestFuncTag, StateType>(state); }
        return generate_special_int_scalar<TestType>(index);
    }
    else if constexpr (is_multiprecision_v<TestType> && per_component)
    {
        // Multi-precision type with per-component generation:
        // generate separate special values for hi and lo parts
        using ComponentType = mp_component_t<TestType>;
        
        // Random indices for hi and lo parts (independent selection)
        int idx_hi = generate_special_index(state);
        int idx_lo = generate_special_index(state);
        
        ComponentType hi_val, lo_val;
        
        // Use template for both float and double components
        if (idx_hi >= 16) { TestType tmp = generate_input_work<TestType, RefType, TestFuncTag, StateType>(state); hi_val = tmp.hi(); } 
        else              { hi_val = generate_special_fp_scalar<ComponentType>(idx_hi); }
        
        if (idx_lo >= 16) { TestType tmp = generate_input_work<TestType, RefType, TestFuncTag, StateType>(state); lo_val = tmp.lo(); } 
        else              { lo_val = generate_special_fp_scalar<ComponentType>(idx_lo); }
        
        return TestType(hi_val, lo_val);
    }
    else // scalar floating-point type or multi-precision without per-component
    {
        // Scalar type OR multi-precision without per-component:
        // generate single special value and convert to TestType
        int index = generate_special_index(state);
        if (index >= 16) { return generate_input_work<TestType, RefType, TestFuncTag, StateType>(state); }
        return static_cast<TestType>(generate_special_fp_scalar<TestType>(index));
    } // scalar type or multi-precision without per-component
}

// ============================================================================
// Generate a random uint64_t value with a specified number of bits
// ============================================================================
template<typename StateType>
__HOST_DEVICE_DECL__ uint64_t generate_random_uint64(StateType& state, int bit_count)
{
    // Clamp bit_count to [1, 64]
    if (bit_count <= 0) bit_count = 1;
    if (bit_count > 64) bit_count = 64;
#if defined(__CUDACC__)
    // CUDA: Use curand for random 32-bit values
    uint64_t result = 0;
    if (bit_count <= 32) {
        uint32_t r = curand(&state);
        result = static_cast<uint64_t>(r) & ((bit_count == 64) ? ~0ull : ((1ull << bit_count) - 1));
    } else {
        // Need both lower and upper halves
        uint32_t r1 = curand(&state);
        uint32_t r2 = curand(&state);
        result = (static_cast<uint64_t>(r1) | (static_cast<uint64_t>(r2) << 32));
        result = (bit_count == 64) ? result : (result & ((1ull << bit_count) - 1));
    }
    return result;
#else
    // Host: Use C++ standard random generator (uniform_int_distribution)
    uint64_t mask = (bit_count == 64) ? ~0ull : ((1ull << bit_count) - 1);
    std::uniform_int_distribution<uint64_t> dist(0, mask);
    return dist(state);
#endif
}

// ============================================================================
// generate_input_pattern_int: Generate integer test input from bit pattern
// ============================================================================
template<typename StateType, typename TestType>
__HOST_DEVICE_DECL__ TestType generate_input_pattern_int(StateType& state, uint64_t bit_pattern, int bit_pattern_size)
{
    constexpr int arg_size = sizeof(TestType) * 8;
    // Initially set all bits to random bits
    TestType val = static_cast<TestType>(generate_random_uint64(state, arg_size));    

    // bit_pattern_size is a runtime parameter, so use regular const
    const int msb_size = (bit_pattern_size > arg_size) ? arg_size : bit_pattern_size;
    const int msb_shift = (msb_size < arg_size) ? (arg_size - msb_size) : 0;
    TestType msb_bits = static_cast<TestType>(bit_pattern) << msb_shift;
    TestType msb_mask = static_cast<TestType>(~TestType(0)) << msb_shift;
    val = msb_bits | (val & ~msb_mask);
    return val;
}

// ============================================================================
// generate_input_pattern_fp: Generate floating-point test input from bit pattern
// ============================================================================
template<typename StateType, typename ComponentType>
__HOST_DEVICE_DECL__ ComponentType generate_input_pattern_fp(StateType& state, 
                                                        uint64_t bit_pattern, 
                                                        int bit_pattern_size,
                                                        uint32_t* exp_bits       = nullptr,
                                                        uint32_t  exp_fixed_bits = 0xffffffff)
{
    // Use unsigned integer type matching TestType size for bit manipulation
    using BitsType = std::conditional_t<std::is_same_v<ComponentType, float>, uint32_t, uint64_t>;

    constexpr int arg_size = sizeof(ComponentType) * 8;

    constexpr int sign_size = 1;
    constexpr int exp_size  = (std::is_same_v<ComponentType, float>) ? 8 : 11;
    constexpr int mant_size = (std::is_same_v<ComponentType, float>) ? 23 : 52;

    constexpr int sign_shift = (arg_size - sign_size);
    constexpr int exp_shift  = (arg_size - sign_size - exp_size);

    constexpr BitsType sign_mask = ((BitsType(1ull) << sign_size) - 1) << sign_shift;
    constexpr BitsType exp_mask  = ((BitsType(1ull) << exp_size)  - 1) << exp_shift;

    int bits_remaining = bit_pattern_size;

    // Initially set all bits to random bits
    BitsType val_bits = static_cast<BitsType>(generate_random_uint64(state, arg_size));

    // 1. Extract sign from bit_pattern and blend it to the sign bit of the value
    if (bits_remaining >= sign_size) 
    {
        BitsType sign_bits = (static_cast<BitsType>(bit_pattern)) << sign_shift;
        val_bits           = (sign_bits & sign_mask) | (val_bits & (~(sign_mask)));

        bit_pattern   >>= sign_size;
        bits_remaining -= sign_size;
    }

    // 2. Set exponent field:
    // If exp_fixed_bits is not 0xffffffff, use the fixed bits.
    // Otherwise extract exponent from bit_pattern and blend it to the exponent bits of the value
    bool use_fixed_exp    = (exp_fixed_bits != 0xffffffff);
    BitsType exp_bits_val = (use_fixed_exp)?static_cast<BitsType>(exp_fixed_bits):
                                            static_cast<BitsType>(bit_pattern);
    if ((bits_remaining >= exp_size) || use_fixed_exp)
    {
        exp_bits_val = exp_bits_val << exp_shift;
        val_bits     = (exp_bits_val & exp_mask) | (val_bits & (~(exp_mask)));

        if (!use_fixed_exp)
        {
            bit_pattern   >>= exp_size;
            bits_remaining -= exp_size;
        }
    }

    // 3. Split remaining bits into most and least significant mantissa bits ~equally
    int msb_bits_used = (bits_remaining + 1) / 2;
    msb_bits_used     = (msb_bits_used > mant_size) ? mant_size : msb_bits_used;
    int lsb_bits_used = bits_remaining - msb_bits_used;
    lsb_bits_used     = ((msb_bits_used + lsb_bits_used) <= mant_size) ? lsb_bits_used : (mant_size - msb_bits_used);

    // 4. Extract most significant mantissa bits from bit_pattern 
    // and blend them to the most significant mantissa bits of the value
    if (msb_bits_used > 0)
    {
        int msb_shift     = (arg_size - sign_size - exp_size - msb_bits_used);
        BitsType msb_mask = ((BitsType(1ull) << msb_bits_used) - 1) << msb_shift;
        BitsType msb_bits = (static_cast<BitsType>(bit_pattern)) << msb_shift;
        // Blend MSB bits to the value
        val_bits          = (msb_bits & msb_mask) | (val_bits & (~(msb_mask)));
        // Shift bit_pattern by msb_bits_used to get the remaining bits
        bit_pattern     >>= msb_bits_used;
        // Subtract the number of bits used from bits_remaining
        bits_remaining   -= msb_bits_used;
    }

    // 5. Extract least significant mantissa bits from bit_pattern 
    // and blend them to the mantissa tail.
    if (lsb_bits_used > 0)
    {
        BitsType lsb_mask = (BitsType(1ull) << lsb_bits_used) - 1;
        BitsType lsb_bits = static_cast<BitsType>(bit_pattern);
        // Blend LSB bits to the value
        val_bits          = (lsb_bits & lsb_mask) | (val_bits & (~lsb_mask));
    }

    // 6. Extract exponent bits from the value and store them in exp_bits if provided
    if (exp_bits != nullptr) { *exp_bits = static_cast<uint32_t>((val_bits & exp_mask) >> exp_shift); }

    // Convert bits to floating-point value.  Explicit `ts::` qualification
    // — like the fp128 sibling below — because NVCC's two-stage name
    // lookup inside templates does not always honor the file-scope
    // `using ts::bit_cast;` brought in at the top of this header.
    return ts::bit_cast<ComponentType>(val_bits);
}

// ============================================================================
// generate_input_pattern_fp128: Generate IEEE 754 binary128 test input from a
// bit pattern, exercising sign / exponent / mantissa fields directly.
//
// Mirrors generate_input_pattern_fp but with a 128-bit BitsType so the full
// 112-bit mantissa is reachable. Layout: [sign:1][exp:15][mantissa:112].
//
// Gated on __SIZEOF_INT128__ since fp128 support coexists with native 128-bit
// integer support on the platforms the test suite targets (x86_64 GCC +
// libquadmath, ARM64 / s390x / PPC long double 128b).
// ============================================================================
#if defined(__SIZEOF_INT128__)
template<typename StateType>
__HOST_DEVICE_DECL__ __ts_fp128 generate_input_pattern_fp128(StateType& state,
                                                             uint64_t   bit_pattern,
                                                             int        bit_pattern_size,
                                                             uint32_t*  exp_bits       = nullptr,
                                                             uint32_t   exp_fixed_bits = 0xffffffff)
{
    using BitsType = __uint128_t;

    constexpr int arg_size  = 128;
    constexpr int sign_size = 1;
    constexpr int exp_size  = 15;
    constexpr int mant_size = 112;

    constexpr int sign_shift = arg_size - sign_size;             // 127
    constexpr int exp_shift  = arg_size - sign_size - exp_size;  // 112

    // Masks are 'const' (not 'constexpr') to sidestep any NVCC strictness
    // around constexpr arithmetic on the __uint128_t extension type; values
    // are trivially compile-time foldable by the optimizer.
    const BitsType sign_mask = ((BitsType(1) << sign_size) - 1) << sign_shift;
    const BitsType exp_mask  = ((BitsType(1) << exp_size)  - 1) << exp_shift;

    int bits_remaining = bit_pattern_size;

    // Initially set all bits to random; two uint64_t draws fill 128 bits.
    BitsType val_bits = (static_cast<BitsType>(generate_random_uint64(state, 64)) << 64)
                      |  static_cast<BitsType>(generate_random_uint64(state, 64));

    // 1. Extract sign from bit_pattern and blend it to the sign bit of the value
    if (bits_remaining >= sign_size)
    {
        BitsType sign_bits = (static_cast<BitsType>(bit_pattern)) << sign_shift;
        val_bits           = (sign_bits & sign_mask) | (val_bits & (~sign_mask));

        bit_pattern   >>= sign_size;
        bits_remaining -= sign_size;
    }

    // 2. Set exponent field:
    // If exp_fixed_bits is not 0xffffffff, use the fixed bits.
    // Otherwise extract exponent from bit_pattern and blend it to the exponent bits.
    bool use_fixed_exp    = (exp_fixed_bits != 0xffffffff);
    BitsType exp_bits_val = use_fixed_exp ? static_cast<BitsType>(exp_fixed_bits)
                                          : static_cast<BitsType>(bit_pattern);
    if ((bits_remaining >= exp_size) || use_fixed_exp)
    {
        exp_bits_val = exp_bits_val << exp_shift;
        val_bits     = (exp_bits_val & exp_mask) | (val_bits & (~exp_mask));

        if (!use_fixed_exp)
        {
            bit_pattern   >>= exp_size;
            bits_remaining -= exp_size;
        }
    }

    // 3. Split remaining bits into most and least significant mantissa bits ~equally
    int msb_bits_used = (bits_remaining + 1) / 2;
    msb_bits_used     = (msb_bits_used > mant_size) ? mant_size : msb_bits_used;
    int lsb_bits_used = bits_remaining - msb_bits_used;
    lsb_bits_used     = ((msb_bits_used + lsb_bits_used) <= mant_size) ? lsb_bits_used
                                                                       : (mant_size - msb_bits_used);

    // 4. Most significant mantissa bits (top of mantissa field)
    if (msb_bits_used > 0)
    {
        int      msb_shift_pos = arg_size - sign_size - exp_size - msb_bits_used;
        BitsType msb_mask_v    = ((BitsType(1) << msb_bits_used) - 1) << msb_shift_pos;
        BitsType msb_bits_v    = (static_cast<BitsType>(bit_pattern)) << msb_shift_pos;
        val_bits               = (msb_bits_v & msb_mask_v) | (val_bits & (~msb_mask_v));

        bit_pattern   >>= msb_bits_used;
        bits_remaining -= msb_bits_used;
    }

    // 5. Least significant mantissa bits (bottom of mantissa field)
    if (lsb_bits_used > 0)
    {
        BitsType lsb_mask_v = (BitsType(1) << lsb_bits_used) - 1;
        BitsType lsb_bits_v = static_cast<BitsType>(bit_pattern);
        val_bits            = (lsb_bits_v & lsb_mask_v) | (val_bits & (~lsb_mask_v));
    }

    // 6. Extract exponent bits from the value and store them in exp_bits if provided
    if (exp_bits != nullptr) { *exp_bits = static_cast<uint32_t>((val_bits & exp_mask) >> exp_shift); }

    // Convert 128 raw bits to a binary128 floating-point value.
    // Fully-qualified to bypass NVCC's parse-time unqualified lookup, which
    // (unlike the dependent-template-arg call sites in this file) does not
    // pick up the file-scope `using ts::bit_cast;` declaration here.
    return ts::bit_cast<__ts_fp128>(val_bits);
}
#endif // __SIZEOF_INT128__

// ============================================================================
// generate_input_pattern_mp: Generate multi-precision test input from bit pattern
// ============================================================================
template<typename StateType, typename TestType>
__HOST_DEVICE_DECL__ TestType generate_input_pattern_mp(StateType& state, 
                                                        uint64_t bit_pattern, 
                                                        int bit_pattern_size)                                                    
{
    using ComponentType = mp_component_t<TestType>;

    constexpr int component_size = sizeof(ComponentType) * 8;
    constexpr int exp_size       = (component_size == 32) ? 8 : 11;
    constexpr int exp_bias       = (component_size == 32) ? 127 : 1023;
    constexpr int mant_size      = (component_size == 32) ? 23 : 52;    
    ComponentType hi_val = 0, lo_val = 0;

    // Get number of bits grabbed from bit_pattern for hi and lo parts
    // Divide bit_pattern_size by 2 plus exponent size to get the number of bits for hi part
    // check if it is not greater than component size
    int      hi_size = (bit_pattern_size + 1) / 2 + exp_size; hi_size = (hi_size > component_size) ? component_size : hi_size;
    // SUbtract the number of bits for hi part from bit_pattern_size to get the number of bits for lo part
    // check if it is not greater than component size
    int      lo_size = bit_pattern_size - hi_size; lo_size = (lo_size > component_size) ? component_size : lo_size;
    uint64_t hi_bits = bit_pattern;
    // Shift bit_pattern by hi_size to get the bits for lo part
    uint64_t lo_bits = bit_pattern >> hi_size;

    // Generate high part and extract exponent
    uint32_t hi_exp;
    hi_val = generate_input_pattern_fp<StateType, ComponentType>(state, hi_bits, hi_size, &hi_exp);

    // Calculate exponent for low part to ensure proper normalization
    // Offset by (mant_size + 2) instead of just mant_size to guarantee |lo| < ulp(hi)/2
    // This provides a safety margin to avoid hi/lo range overlap
    constexpr int32_t lo_exp_offset = mant_size + 2;
    int32_t lo_exp = static_cast<int32_t>(hi_exp) - lo_exp_offset;

    // Low part is a normal floating-point value, generate it if exponent is not less than 1
    if (lo_exp >= 1)
    {
        lo_val = generate_input_pattern_fp<StateType, ComponentType>(state, lo_bits, lo_size, nullptr, lo_exp);
    }
    // Low part is a subnormal floating-point value, generate it if exponent is less than 1 
    // and greater than (1 - mant_size), which is the smallest representable subnormal exponent
    else if ((lo_exp < 1) && (lo_exp > 1 - static_cast<int32_t>(mant_size)))
    {
        // Generate with biased exponent = exp_bias (value in [1.0, 2.0)), then scale down
        // to achieve subnormal representation with target exponent lo_exp
        lo_val = generate_input_pattern_fp<StateType, ComponentType>(state, lo_bits, lo_size, nullptr, exp_bias);
        lo_val = TS_LDEXP(lo_val, lo_exp - exp_bias);
    }
    else
    {
        lo_val = 0;
    }

    TestType val = TestType(hi_val, lo_val);
    return val;
}

// ============================================================================
// generate_input_pattern: Top-level pattern-based input generator
// ============================================================================
template<typename TestType, typename RefType, typename TestFuncTag, typename StateType>
__HOST_DEVICE_DECL__ TestType generate_input_pattern(StateType& state,       // RNG state
                                                        uint64_t   bits    = 0, // Bits pattern
                                                        int        arg_idx = 0  // Function argument index (0-3)
                                                        )
{
    TestType val = 0;
        // Arity detection - number of function arguments (1-4)
    constexpr int arity = detect_arity<TestFuncTag>();
    // Total number of bits to use for the function input (use pattern rigor constant)
    constexpr int total_bits_count = TS_ACCURACY_RIGOR_PATTERN;
    // Number of bits to use for the current function argument
    constexpr int bit_pattern_size = total_bits_count / arity;
    // Bits pattern to use for the current function argument
    uint64_t bit_pattern = (bits >> (arg_idx * bit_pattern_size)) & ((1ull << bit_pattern_size) - 1);

    // Multi-precision type with per-component generation:
    if constexpr (is_multiprecision_v<TestType>)
    {
        val = generate_input_pattern_mp<StateType, TestType>(state, bit_pattern, bit_pattern_size);
    }
    // Scalar type or multi-precision without per-component:
    else
    {
        // Floating-point types: float / double:
        if constexpr ((std::is_same_v<TestType, float>) || (std::is_same_v<TestType, double>))
        {
            val = generate_input_pattern_fp<StateType, TestType>(state, bit_pattern, bit_pattern_size);
        }
        // Quad-precision (__float128 / long double 128b): full IEEE 754
        // binary128 bit-pattern generation, exercising the entire 112-bit
        // mantissa. Used as input for the fp128 -> fp64mp2 / fp32mp2
        // conversion benchmarks.
        else if constexpr (std::is_same_v<TestType, __ts_fp128>)
        {
        #if defined(__SIZEOF_INT128__)
            val = generate_input_pattern_fp128<StateType>(state, bit_pattern, bit_pattern_size);
        #else
            // Fallback for the (unlikely) case of fp128 support without
            // native 128-bit integers: widen a double-shaped pattern.
            double d_val = generate_input_pattern_fp<StateType, double>(state, bit_pattern, bit_pattern_size);
            val = static_cast<TestType>(d_val);
        #endif
        }
        // Integer types:
        else
        {
            val = generate_input_pattern_int<StateType, TestType>(state, bit_pattern, bit_pattern_size);
        }
    }

    return val;
} // end of generate_input_pattern

// ============================================================================
// Unified input generator: dispatches to appropriate mode-specific generator
// ============================================================================
template<typename TestType, typename RefType, typename TestFuncTag, typename StateType>
__HOST_DEVICE_DECL__ TestType generate_input(StateType& state, accuracy_mode::type mode, uint64_t bits = 0, int arg_idx = 0)
{
    if      (mode == accuracy_mode::work)       { return generate_input_work<TestType, RefType, TestFuncTag, StateType>(state); }
    else if (mode == accuracy_mode::normal)     { return generate_input_normal<TestType, RefType, TestFuncTag, StateType>(state); }
    else if (mode == accuracy_mode::special)    { return generate_input_special<TestType, RefType, TestFuncTag, StateType>(state, arg_idx); }
    else if (mode == accuracy_mode::pattern)    { return generate_input_pattern<TestType, RefType, TestFuncTag, StateType>(state, bits, arg_idx); }
    else                                        { return TestType{}; }
}

#endif // __TS_DATASET_HPP__
