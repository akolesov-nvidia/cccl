/*
    ts_functions.hpp - FPMP Test Suite Function Tags and Definitions
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Function tag definitions for arithmetic operations tested by the suite. Each tag defines input/output
    types, work ranges, and the operation functor.
*/

#ifndef __TS_FUNCTIONS_HPP__  
#define __TS_FUNCTIONS_HPP__

#if defined __DEVICE_FUNC__
    // Standalone compilation only - ts_types.hpp includes ts.hpp
    #include "ts_types.hpp"
    #include "ts_utils.hpp"
#endif

// Import commonly used items from ts namespace for cleaner code
using ts::fpmp2_base_type_t;

// ============================================================================
// Function tags with explicit input_type and result_type
// ============================================================================
// Each function tag defines:
//   - input_type:  type of input arguments
//   - result_type: type of return value
//   - work_beg/work_end: uniform distribution range (wide coverage)
//   - normal_mean/normal_stddev: Gaussian distribution parameters (focused testing)

template<typename T> struct _add_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a + b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _sub_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a - b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

// ACC (Accumulate): Optimized single-component accumulation into mp2
// 
// Tests the ACC operator which accumulates a single float/double into mp2.
// The second argument (b) should have lo=0 to represent a single-component value.
// This is enforced via the zero_lo_args trait (see accuracy kernel).
//
// - mp2 test: uses b.hi() via operator+= (optimized ACC)
// - __ts_fp128 ref: uses b directly (which equals hi since lo=0)
template<typename T> struct _acc_ 
{
    using input_type  = T;
    using result_type = T;
    
    // Trait: arguments that should have their lo part zeroed (bitmask)
    // Bit 0 = arg 0, bit 1 = arg 1, etc.
    // For ACC: arg 1 (b) should have lo=0
    static constexpr unsigned zero_lo_args = 0x2;  // bit 1 = arg index 1
    
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { 
        if constexpr (std::is_same_v<T, __ts_fp128> || 
                      std::is_same_v<T, float> || 
                      std::is_same_v<T, double>) {
            // Scalar types: simple addition
            return a + b;
        } else {
            // mp2 type: use optimized ACC operator
            result_type acc = a;
            acc += b.hi();  // Uses __nv_fpmp2_acc via operator+=
            return acc;
        }
    }
    
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _mul_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a * b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

// ============================================================================
// FMA (Fused Multiply-Add): x * y + z
// ============================================================================
template<typename T> struct _fma_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x, input_type y, input_type z) const { 
        if constexpr (std::is_same_v<T, __ts_fp128>) {
            return __TS_FMAQ(x, y, z);
        } else if constexpr (std::is_same_v<T, float>) {
            return fmaf(x, y, z);
        } else if constexpr (std::is_same_v<T, double>) {
            return fma(x, y, z);
        } else {
            // mp2 type: use library fma function
            return fma(x, y, z);
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e8; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e8; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

// ============================================================================
// MAD (Multiply-Add with Rounding): x * y + z
// Note: For mp2 types, this is similar to FMA but may have different rounding
// ============================================================================
template<typename T> struct _mad_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x, input_type y, input_type z) const { 
        if constexpr (std::is_same_v<T, __ts_fp128>) {
            return __TS_FMAQ(x, y, z);
        } else if constexpr (std::is_same_v<T, float>) {
            return fmaf(x, y, z);
        } else if constexpr (std::is_same_v<T, double>) {
            return fma(x, y, z);
        } else {
            // mp2 type: use library mad function
            return mad(x, y, z);
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e8; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e8; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _div_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a / b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _mp2int_ 
{
    using input_type  = T;
    using result_type = int32_t;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const { return (int32_t)(a); }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1.6e7; }  // Within fp32mp2's hi component exact range (2^24)
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.6e7; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 5.0e6; }
    // Input validation: for integer conversion, limit to where hi component can exactly represent the integer
    // fp32mp2: 2^24 ≈ 16.78e6 (float's exact integer range for hi component)
    // fp64mp2: 2^53 (double's exact integer range for hi component)
    __HOST_DEVICE_DECL__ static constexpr bool has_valid_input_range() { return true; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_min() { return -16777216.0; }  // -2^24 (fp32mp2 hi exact integer limit)
    __HOST_DEVICE_DECL__ static constexpr double valid_input_max() { return  16777216.0; }  //  2^24 (fp32mp2 hi exact integer limit)
};

template<typename T> struct _int2mp_ 
{
    using input_type  = int32_t;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const { 
        if constexpr (std::is_same_v<T, double>) {
            // Reference: compute exactly as fp32mp2 does (hi + lo as floats) for 48-bit match
            float hi = fpmp::int2fp_rz<float>(a);
            float lo = fpmp::int2fp_rz<float>(a - fpmp::fp2int_rz(hi));
            return static_cast<double>(hi) + static_cast<double>(lo);
        } else if constexpr (std::is_same_v<T, float>) {
            // Scalar float: round toward zero (int32_t values > 2^24 don't fit exactly)
            return fpmp::int2fp_rz<float>(a);
        } else {
            // Multi-precision types: use constructor
            return (T)(a);
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -2e9; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 2e9; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e6; }  // int32_t focused range
};

// ============================================================================
// Unsigned int32 conversions
// ============================================================================
template<typename T> struct _mp2uint_ 
{
    using input_type  = T;
    using result_type = uint32_t;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const { return (uint32_t)(a); }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.6e7; }  // Within fp32mp2's hi component exact range (2^24)
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 8.0e6; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 5.0e6; }
    // Input validation: for integer conversion, limit to where hi component can exactly represent the integer
    // fp32mp2: 2^24 ≈ 16.78e6 (float's exact integer range for hi component)
    // fp64mp2: 2^53 (double's exact integer range for hi component)
    __HOST_DEVICE_DECL__ static constexpr bool has_valid_input_range() { return true; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_min() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_max() { return 16777216.0; }  // 2^24 (fp32mp2 hi exact integer limit)
};

template<typename T> struct _uint2mp_ 
{
    using input_type  = uint32_t;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const { 
        if constexpr (std::is_same_v<T, double>) {
            // Reference: compute exactly as fp32mp2 does (hi + lo as floats) for 48-bit match
            float hi = fpmp::uint2fp_rz<float>(a);
            uint32_t residual = a - fpmp::fp2uint_rz(hi);
            float lo = fpmp::uint2fp_rz<float>(residual);
            return static_cast<double>(hi) + static_cast<double>(lo);
        } else if constexpr (std::is_same_v<T, float>) {
            // Scalar float: round toward zero (uint32_t values > 2^24 don't fit exactly)
            return fpmp::uint2fp_rz<float>(a);
        } else {
            // Multi-precision types: use constructor
            return (T)(a);
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 4e9; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 2e9; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e9; }
};

// ============================================================================
// Signed int64 conversions
// ============================================================================
template<typename T> struct _mp2ll_ 
{
    using input_type  = T;
    using result_type = int64_t;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const { return (int64_t)(a); }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1.6e7; }  // Within fp32mp2's hi component exact range (2^24)
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.6e7; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 5.0e6; }
    // Input validation: for integer conversion, limit to where hi component can exactly represent the integer
    // fp32mp2: 2^24 ≈ 16.78e6 (float's exact integer range for hi component)
    // fp64mp2: 2^53 (double's exact integer range for hi component)
    // Use 2^24 as the conservative limit that works for fp32mp2
    __HOST_DEVICE_DECL__ static constexpr bool has_valid_input_range() { return true; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_min() { return -16777216.0; }  // -2^24 (fp32mp2 hi exact integer limit)
    __HOST_DEVICE_DECL__ static constexpr double valid_input_max() { return  16777216.0; }  //  2^24 (fp32mp2 hi exact integer limit)
};

template<typename T> struct _ll2mp_ 
{
    using input_type  = int64_t;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const { 
        if constexpr (std::is_same_v<T, double>) {
            // Reference: compute exactly as fp32mp2 does (hi + lo as floats) for 48-bit match
            float hi = fpmp::ll2fp_rz<float>(a);
            float lo = fpmp::ll2fp_rz<float>(a - fpmp::fp2ll_rz(hi));
            return static_cast<double>(hi) + static_cast<double>(lo);
        } else if constexpr (std::is_same_v<T, float>) {
            // Scalar float: round toward zero
            return fpmp::ll2fp_rz<float>(a);
        } else {
            // Multi-precision types: use constructor
            return (T)(a);
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -9e18; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 9e18; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e15; }
};

// ============================================================================
// Unsigned int64 conversions
// ============================================================================
template<typename T> struct _mp2ull_ 
{
    using input_type  = T;
    using result_type = uint64_t;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const { return (uint64_t)(a); }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.6e7; }  // Within fp32mp2's hi component exact range (2^24)
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 8.0e6; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 5.0e6; }
    // Input validation: for integer conversion, limit to where hi component can exactly represent the integer
    // fp32mp2: 2^24 ≈ 16.78e6 (float's exact integer range for hi component)
    // fp64mp2: 2^53 (double's exact integer range for hi component)
    // Use 2^24 as the conservative limit that works for fp32mp2
    __HOST_DEVICE_DECL__ static constexpr bool has_valid_input_range() { return true; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_min() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_max() { return 16777216.0; }  // 2^24 (fp32mp2 hi exact integer limit)
};

template<typename T> struct _ull2mp_ 
{
    using input_type  = uint64_t;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const { 
        if constexpr (std::is_same_v<T, double>) {
            // Reference: compute exactly as fp32mp2 does (hi + lo as floats) for 48-bit match
            float hi = fpmp::ull2fp_rz<float>(a);
            uint64_t residual = a - fpmp::fp2ull_rz(hi);
            float lo = fpmp::ull2fp_rz<float>(residual);
            return static_cast<double>(hi) + static_cast<double>(lo);
        } else if constexpr (std::is_same_v<T, float>) {
            // Scalar float: round toward zero
            return fpmp::ull2fp_rz<float>(a);
        } else {
            // Multi-precision types: use constructor
            return (T)(a);
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.8e19; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 9e18; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 4e18; }
};

// ============================================================================
// Multi-precision -> next-wider scalar conversion
// ============================================================================
//
// Symmetric counterpart to `_fp2mp_<T>`: the result type is tied to the
// build's REF_TYPE (the next-wider scalar above the mp2 component), so the
// conversion under test is non-trivial in every build:
//
//   fp32mp2 build  -> input = fp32mp2, output = double
//                     (mirrors __nv_fpmp2_to_double<float>)
//   fp64mp2 build  -> input = fp64mp2, output = __ts_fp128
//                     (mirrors __nv_fpmp2_to_quad<double>)
//
// Without REF_TYPE on the output side, the fp64mp2 case would degenerate to
// `fp64mp2 -> double`, which collapses to a single `hi + lo` `DADD` (lossless
// only at double precision, not at fp128) and benchmarks effectively nothing.
template<typename T> struct _mp2fp_
{
    using input_type  = T;
    using result_type = REF_TYPE;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        if constexpr (std::is_same_v<T, REF_TYPE>) {
            // High-precision accuracy reference: replicate the mp2 library's
            // bit-pattern by splitting through the BASE_TYPE component and
            // recombining at REF_TYPE precision.
            //   fp32mp2 build: float  hi/lo, double sum  (~ __nv_fpmp2_to_double)
            //   fp64mp2 build: double hi/lo, fp128  sum  (~ __nv_fpmp2_to_quad)
            using C = BASE_TYPE;
            C hi = static_cast<C>(a);
            C lo = static_cast<C>(a - static_cast<input_type>(hi));
            return static_cast<result_type>(hi) + static_cast<result_type>(lo);
        } else if constexpr (std::is_same_v<T, BASE_TYPE>) {
            // Base scalar column: trivial widening cast (BASE_TYPE -> REF_TYPE).
            return static_cast<result_type>(a);
        } else {
            // Multi-precision source: invoke the mp2 -> REF_TYPE conversion,
            // which dispatches to __nv_fpmp2_to_double (fp32mp2) via
            // operator double(), or __nv_fpmp2_to_quad (fp64mp2) via
            // operator __fpmp_fp128() based on T's component type.
            //
            // For fp64mp2 (REF_TYPE is __ts_fp128) bridge through __fpmp_fp128
            // explicitly: on ARM64 device pass, __ts_fp128 (long double) and
            // __fpmp_fp128 (__float128) are distinct binary128 types with
            // identical bit layout but different C++ identity, so a direct
            // static_cast<__ts_fp128>(a) wouldn't see the
            // operator __fpmp_fp128() conversion. The cast is a no-op on
            // x86_64 and on ARM64 host where both typedefs resolve to the
            // same scalar.
        #if (FPMP_FP128_ENABLE == 1)
            if constexpr (std::is_same_v<result_type, __ts_fp128>) {
                return static_cast<result_type>(static_cast<cuda::experimental::__fpmp_fp128>(a));
            } else
        #endif
            {
                return static_cast<result_type>(a);
            }
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e32; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e32; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e8; }
};

// Precision-aware conversion test:
//
// The input type is tied to the build's REF_TYPE (the next-wider scalar
// above the mp2 component), so the conversion under test is non-trivial in
// every build:
//
//   fp32mp2 build  -> input = double,      output = fp32mp2
//                     (mirrors __nv_fpmp2_from_double<float>)
//   fp64mp2 build  -> input = __ts_fp128,  output = fp64mp2
//                     (mirrors __nv_fpmp2_from_quad<double>)
//
// Without this, the fp64mp2 case would degenerate to a `double -> fp64mp2`
// cast, which collapses to `hi = a; lo = 0` and benchmarks effectively
// nothing.
template<typename T> struct _fp2mp_
{
    using input_type  = REF_TYPE;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        if constexpr (std::is_same_v<T, REF_TYPE>) {
            // High-precision accuracy reference: replicate the mp2 library's
            // bit-pattern by splitting through the BASE_TYPE component and
            // recombining at REF_TYPE precision.
            //   fp32mp2 build: float hi/lo, double sum  (~ __nv_fpmp2_from_double)
            //   fp64mp2 build: double hi/lo, fp128 sum  (~ __nv_fpmp2_from_quad)
            using C = BASE_TYPE;
            C hi = static_cast<C>(a);
            C lo = static_cast<C>(a - static_cast<input_type>(hi));
            return static_cast<input_type>(hi) + static_cast<input_type>(lo);
        } else if constexpr (std::is_same_v<T, BASE_TYPE>) {
            // Base scalar column: lossy narrow cast (REF_TYPE -> BASE_TYPE).
            return static_cast<T>(a);
        } else {
            // Multi-precision target: invoke the mp2 constructor, which
            // dispatches to __nv_fpmp2_from_double (fp32mp2) or
            // __nv_fpmp2_from_quad (fp64mp2) based on T's component type.
            //
            // For fp64mp2 (input is __ts_fp128) bridge through __fpmp_fp128
            // explicitly: on ARM64 device pass, __ts_fp128 (long double) and
            // __fpmp_fp128 (__float128) are distinct binary128 types with
            // identical bit layout but different C++ identity, so a direct
            // T(a) would be ambiguous against the (double)/(int*)/(uint*)
            // constructors. The cast is a no-op on x86_64 and on ARM64 host
            // where both typedefs resolve to the same scalar.
        #if (FPMP_FP128_ENABLE == 1)
            if constexpr (std::is_same_v<input_type, __ts_fp128>) {
                return T(static_cast<cuda::experimental::__fpmp_fp128>(a));
            } else
        #endif
            {
                return T(a);
            }
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e32; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e32; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e8; }
};

// ============================================================================
// Cast-based performance references for conversion functions.
//
// These are **bit-equivalent** to the non-optimized default-mode path of
// the corresponding __nv_fpmp2_* implementations. Two consequences:
//
//   1. Accuracy parity. The reference produces the same bits as the baseline
//      across the whole input domain (including the large-value branches
//      where a naive `(int)(hi + lo)` would silently lose precision). So the
//      A/B comparison is fair — both sides solve the same problem.
//
//   2. Frozen cost. The cast tag stays anchored at the non-optimized cost
//      profile even when the implementation later enables an optimized
//      variant (e.g. __FPMP_USE_OPT_TO_INT__ == 1). The fp32mp2/cast ratio
//      then directly reports the speedup of that optimization. While no
//      optimization is enabled, fp32mp2/cast is expected to be ~ -1.00x.
//
// Without these, the perf framework falls back to the fp64/fp128 reference,
// whose throughput on consumer GPUs (~1:64 fp64:fp32) dwarfs any real
// algorithmic improvement and produces misleading speedup numbers.
// ============================================================================
// Cast reference for _mp2fp_, bit-equivalent to the non-opt path of
// __nv_fpmp2_to_double (fp32mp2 build) or __nv_fpmp2_to_quad (fp64mp2 build):
// widen each BASE_TYPE component to REF_TYPE and add. Keeping result_type ==
// REF_TYPE here mirrors `_mp2fp_<T>` (the symmetric counterpart of
// `_fp2mp_<T>`) so the perf benchmark exercises the *same* output precision
// as the test under measurement — otherwise fp64mp2/cast would collapse to
// a single `DADD` and dwarf any meaningful fp128-width comparison.
template<typename T> struct _mp2fp_ref_
{
    using input_type  = T;
    using result_type = REF_TYPE;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        return static_cast<result_type>(a.hi()) + static_cast<result_type>(a.lo());
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e32; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e32; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e8; }
};

// Cast reference for _fp2mp_, bit-equivalent to the non-opt path of
// __nv_fpmp2_from_double (fp32mp2 build) or __nv_fpmp2_from_quad (fp64mp2
// build): split the REF_TYPE input into two BASE_TYPE components via cast
// and subtract.
template<typename T> struct _fp2mp_ref_
{
    using input_type  = REF_TYPE;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        using C = decltype(std::declval<T>().hi());
        C hi = static_cast<C>(a);
        C lo = static_cast<C>(a - static_cast<input_type>(hi));
        return T(hi, lo);
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e32; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e32; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e8; }
};

// ----------------------------------------------------------------------------
// (hi, lo) -> int / uint / ll / ull conversions.
// Bit-equivalent to the non-optimized __nv_fpmp2_to_{int,uint,ll,ull}: a
// threshold check on hi (2^24 for float / 2^53 for double) selects between
// (a) fast small-value path: add_rz(hi, lo) followed by fp2*_rz, and
// (b) precise large-value path: integer addition of fp2*_rz(hi) and lo.
// ----------------------------------------------------------------------------
template<typename T> struct _mp2int_ref_
{
    using input_type  = T;
    using result_type = int32_t;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        using C = decltype(std::declval<T>().hi());
        C x_hi = a.hi();
        C x_lo = a.lo();
        C abs_hi    = fpmp::internal_fabs(x_hi);
        C threshold = std::is_same<C, float>::value ? 0x1.0p24f : 0x1.0p53;
        if (abs_hi < threshold) {
            C res = fpmp::add_rz(x_hi, x_lo);
            return fpmp::fp2int_rz(res);
        } else {
            int32_t hi_int = fpmp::fp2int_rz(x_hi);
            int32_t lo_int = fpmp::fp2int_rz(x_lo);
            return hi_int + lo_int;
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1.6e7; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.6e7; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 5.0e6; }
};

template<typename T> struct _mp2uint_ref_
{
    using input_type  = T;
    using result_type = uint32_t;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        using C = decltype(std::declval<T>().hi());
        C x_hi = a.hi();
        C x_lo = a.lo();
        C threshold = std::is_same<C, float>::value ? 0x1.0p24f : 0x1.0p53;
        if (x_hi < threshold) {
            C res = fpmp::add_rz(x_hi, x_lo);
            return fpmp::fp2uint_rz(res);
        } else {
            uint32_t hi_uint = fpmp::fp2uint_rz(x_hi);
            int32_t  lo_int  = fpmp::fp2int_rz(x_lo);
            return hi_uint + lo_int;
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.6e7; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 8.0e6; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 5.0e6; }
};

template<typename T> struct _mp2ll_ref_
{
    using input_type  = T;
    using result_type = int64_t;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        using C = decltype(std::declval<T>().hi());
        C x_hi = a.hi();
        C x_lo = a.lo();
        C abs_hi    = fpmp::internal_fabs(x_hi);
        C threshold = std::is_same<C, float>::value ? 0x1.0p24f : 0x1.0p53;
        if (abs_hi < threshold) {
            C res = fpmp::add_rz(x_hi, x_lo);
            return fpmp::fp2ll_rz(res);
        } else {
            int64_t hi_ll = fpmp::fp2ll_rz(x_hi);
            int64_t lo_ll = fpmp::fp2ll_rz(x_lo);
            return hi_ll + lo_ll;
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1.6e7; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.6e7; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 5.0e6; }
};

template<typename T> struct _mp2ull_ref_
{
    using input_type  = T;
    using result_type = uint64_t;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        using C = decltype(std::declval<T>().hi());
        C x_hi = a.hi();
        C x_lo = a.lo();
        C threshold = std::is_same<C, float>::value ? 0x1.0p24f : 0x1.0p53;
        if (x_hi < threshold) {
            C res = fpmp::add_rz(x_hi, x_lo);
            return fpmp::fp2ull_rz(res);
        } else {
            uint64_t hi_ull = fpmp::fp2ull_rz(x_hi);
            int64_t  lo_ll  = fpmp::fp2ll_rz(x_lo);
            return hi_ull + lo_ll;
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.6e7; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 8.0e6; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 5.0e6; }
};

// ----------------------------------------------------------------------------
// int / uint / ll / ull -> (hi, lo) conversions.
// Cast-based path: matches the body of __nv_fpmp2_from_{int,uint,ll,ull}
// (round-toward-zero conversions in the component precision, no fp64 widening).
// ----------------------------------------------------------------------------
template<typename T> struct _int2mp_ref_
{
    using input_type  = int32_t;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        using C = decltype(std::declval<T>().hi());
        C hi = fpmp::int2fp_rz<C>(a);
        C lo = fpmp::int2fp_rz<C>(a - fpmp::fp2int_rz(hi));
        return T(hi, lo);
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -2e9; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 2e9; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e6; }
};

template<typename T> struct _uint2mp_ref_
{
    using input_type  = uint32_t;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        using C = decltype(std::declval<T>().hi());
        C hi = fpmp::uint2fp_rz<C>(a);
        // Signed residual: hi may round above a, so use int32_t to avoid uint underflow.
        int32_t residual = static_cast<int32_t>(a) - static_cast<int32_t>(fpmp::fp2uint_rz(hi));
        C lo = fpmp::int2fp_rz<C>(residual);
        return T(hi, lo);
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 4e9; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 2e9; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e9; }
};

template<typename T> struct _ll2mp_ref_
{
    using input_type  = int64_t;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        using C = decltype(std::declval<T>().hi());
        C hi = fpmp::ll2fp_rz<C>(a);
        C lo = fpmp::ll2fp_rz<C>(a - fpmp::fp2ll_rz(hi));
        return T(hi, lo);
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -9e18; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 9e18; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e15; }
};

template<typename T> struct _ull2mp_ref_
{
    using input_type  = uint64_t;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a) const {
        using C = decltype(std::declval<T>().hi());
        C hi = fpmp::ull2fp_rz<C>(a);
        // Residual is non-negative since ull2fp_rz rounds toward zero (hi <= a).
        uint64_t residual = a - fpmp::fp2ull_rz(hi);
        C lo = fpmp::ull2fp_rz<C>(residual);
        return T(hi, lo);
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.8e19; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 9e18; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 4e18; }
};

// ============================================================================
// Trait to select performance reference tag.
// For conversion functions: use cast-based reference with same mp2 types.
// For all other functions: use the default (fp64/fp128) reference.
// ============================================================================
template<typename TestTag, typename DefaultRefTag>
struct perf_ref_tag { using type = DefaultRefTag; };

template<typename T, typename D>
struct perf_ref_tag<_mp2fp_<T>, D> { using type = _mp2fp_ref_<T>; };

template<typename T, typename D>
struct perf_ref_tag<_fp2mp_<T>, D> { using type = _fp2mp_ref_<T>; };

template<typename T, typename D>
struct perf_ref_tag<_mp2int_<T>, D> { using type = _mp2int_ref_<T>; };

template<typename T, typename D>
struct perf_ref_tag<_mp2uint_<T>, D> { using type = _mp2uint_ref_<T>; };

template<typename T, typename D>
struct perf_ref_tag<_mp2ll_<T>, D> { using type = _mp2ll_ref_<T>; };

template<typename T, typename D>
struct perf_ref_tag<_mp2ull_<T>, D> { using type = _mp2ull_ref_<T>; };

template<typename T, typename D>
struct perf_ref_tag<_int2mp_<T>, D> { using type = _int2mp_ref_<T>; };

template<typename T, typename D>
struct perf_ref_tag<_uint2mp_<T>, D> { using type = _uint2mp_ref_<T>; };

template<typename T, typename D>
struct perf_ref_tag<_ll2mp_<T>, D> { using type = _ll2mp_ref_<T>; };

template<typename T, typename D>
struct perf_ref_tag<_ull2mp_<T>, D> { using type = _ull2mp_ref_<T>; };

template<typename TestTag, typename DefaultRefTag>
using perf_ref_tag_t = typename perf_ref_tag<TestTag, DefaultRefTag>::type;

// ============================================================================
// Comparison operators
// ============================================================================
template<typename T> struct _ne_  // Not equal
{
    using input_type  = T;
    using result_type = bool;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a != b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e24; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e24; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _lt_  // Less than
{
    using input_type  = T;
    using result_type = bool;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a < b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e24; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e24; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _le_  // Less than or equal
{
    using input_type  = T;
    using result_type = bool;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a <= b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e24; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e24; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _gt_  // Greater than
{
    using input_type  = T;
    using result_type = bool;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a > b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e24; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e24; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _ge_  // Greater than or equal
{
    using input_type  = T;
    using result_type = bool;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a >= b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e24; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e24; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _eq_ 
{
    using input_type  = T;
    using result_type = bool;
    __HOST_DEVICE_DECL__ result_type operator()(input_type a, input_type b) const { return a == b; }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e24; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e24; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _sqrt_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) {
            return __TS_SQRTQ(x);
        }
        else { return sqrt(x); }
    }    
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e32; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 1.0; }   // Positive only
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.5; }   // Stays positive (mean - 2*stddev > 0)
};

template<typename T> struct _rsqrt_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) {
            return (T)1.0/__TS_SQRTQ(x);
        }
        else 
        { 
            #if defined(__CUDA_ARCH__)
                return rsqrt(x);
            #else
                return 1.0/sqrt(x); 
            #endif
        }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e32; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 1.0; }   // Positive only
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.5; }   // Stays positive
};

template<typename T> struct _floor_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
        {
        #if defined(__CUDA_ARCH__) && defined(FPMP_CUDA_FP128_INTRINSICS) && \
            (defined(__FLOAT128_CPP_SPELLING_ENABLED__) || defined(__FLOAT128_C_SPELLING_ENABLED__))
            // True binary128 floor -- mirrors the library (fpmp_math.hpp). Casting
            // the fp128 reference down to double first would drop the low bits and
            // round to the wrong integer for |x| > 2^52 (the work range reaches
            // 1e16), making the reference *less* precise than the library.
            return __nv_fp128_floor(x);
        #elif defined(__CUDA_ARCH__)
            return (__ts_fp128)::floor((double)x);
        #elif (TS_HAS_LIBQUADMATH == 1)
            return floorq(x);
        #else
            return floorl(x);
        #endif
        }
        else { return floor(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _ceil_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
        {
        #if defined(__CUDA_ARCH__) && defined(FPMP_CUDA_FP128_INTRINSICS) && \
            (defined(__FLOAT128_CPP_SPELLING_ENABLED__) || defined(__FLOAT128_C_SPELLING_ENABLED__))
            // True binary128 ceil -- mirrors the library; see _floor_ for rationale.
            return __nv_fp128_ceil(x);
        #elif defined(__CUDA_ARCH__)
            return (__ts_fp128)::ceil((double)x);
        #elif (TS_HAS_LIBQUADMATH == 1)
            return ceilq(x);
        #else
            return ceill(x);
        #endif
        }
        else { return ceil(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _round_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
        {
        #if defined(__CUDA_ARCH__) && defined(FPMP_CUDA_FP128_INTRINSICS) && \
            (defined(__FLOAT128_CPP_SPELLING_ENABLED__) || defined(__FLOAT128_C_SPELLING_ENABLED__))
            // True binary128 round -- mirrors the library; see _floor_ for rationale.
            return __nv_fp128_round(x);
        #elif defined(__CUDA_ARCH__)
            return (__ts_fp128)::round((double)x);
        #elif (TS_HAS_LIBQUADMATH == 1)
            return roundq(x);
        #else
            return roundl(x);
        #endif
        }
        else { return round(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _trunc_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
        {
        #if defined(__CUDA_ARCH__) && defined(FPMP_CUDA_FP128_INTRINSICS) && \
            (defined(__FLOAT128_CPP_SPELLING_ENABLED__) || defined(__FLOAT128_C_SPELLING_ENABLED__))
            // True binary128 trunc -- mirrors the library; see _floor_ for rationale.
            return __nv_fp128_trunc(x);
        #elif defined(__CUDA_ARCH__)
            return (__ts_fp128)::trunc((double)x);
        #elif (TS_HAS_LIBQUADMATH == 1)
            return truncq(x);
        #else
            return truncl(x);
        #endif
        }
        else { return trunc(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _rint_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
        {
        #if defined(__CUDA_ARCH__) && defined(FPMP_CUDA_FP128_INTRINSICS) && \
            (defined(__FLOAT128_CPP_SPELLING_ENABLED__) || defined(__FLOAT128_C_SPELLING_ENABLED__))
            // True binary128 rint -- mirrors the library; see _floor_ for rationale.
            return __nv_fp128_rint(x);
        #elif defined(__CUDA_ARCH__)
            return (__ts_fp128)::rint((double)x);
        #elif (TS_HAS_LIBQUADMATH == 1)
            return rintq(x);
        #else
            return rintl(x);
        #endif
        }
        else { return rint(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _nearbyint_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
        {
        #if defined(__CUDA_ARCH__) && defined(FPMP_CUDA_FP128_INTRINSICS) && \
            (defined(__FLOAT128_CPP_SPELLING_ENABLED__) || defined(__FLOAT128_C_SPELLING_ENABLED__))
            // nearbyint == rint numerically (rint only additionally raises the
            // inexact flag); use native binary128 rint to mirror the library.
            return __nv_fp128_rint(x);
        #elif defined(__CUDA_ARCH__)
            return (__ts_fp128)::nearbyint((double)x);
        #elif (TS_HAS_LIBQUADMATH == 1)
            return nearbyintq(x);
        #else
            return nearbyintl(x);
        #endif
        }
        else { return nearbyint(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

template<typename T> struct _exp_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_EXPQ(x); }
        else { return exp(x); }
    }
    
    // Use base type to determine domain: double/__ts_fp128 base -> [-640,640], float base -> [-64,64]
    using base_t = fpmp2_base_type_t<T>;
    static constexpr bool is_wide_domain = std::is_same_v<base_t, double> || std::is_same_v<base_t, __ts_fp128>;
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return is_wide_domain ? -640.0 : -64.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return is_wide_domain ? 640.0 : 64.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 1.0; }   // exp(1) = e ≈ 2.718
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 2.0; }   // Reasonable range around 1
};

template<typename T> struct _exp2_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
            { return __TS_EXP2Q(x); }
        else { return exp2(x); }
    }
    // Lower bound chosen so the fp32mp2 lo limb of the result stays in
    // fp32 normal range: ulp(hi) > FLT_MIN ⇒ hi > 2⁻¹⁰³, i.e. x > -103.
    // We leave a small guard band (-90) to match the safe regime of
    // _exp_ ([-64, 64], whose result range is [exp(-64), exp(64)] ≈
    // [1.6e-28, 6.2e27]; here 2⁻⁹⁰ ≈ 8e-28 sits comfortably above
    // 2⁻¹⁰³).  Upper bound symmetric, well inside exp2's overflow
    // boundary at +128.  Wide-domain (fp64/fp128) keeps the broader
    // range because its own denormal floor (2⁻¹⁰²²·²⁻⁵² = 2⁻¹⁰⁷⁴) is
    // far away.
    using base_t = fpmp2_base_type_t<T>;
    static constexpr bool is_wide_domain = std::is_same_v<base_t, double> || std::is_same_v<base_t, __ts_fp128>;
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return is_wide_domain ? -900.0 : -90.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return is_wide_domain ? 900.0 : 90.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 3.0; }
};

template<typename T> struct _exp10_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
            { return __TS_EXP10Q(x); }
        else
        {
            #if defined(__CUDA_ARCH__)
                return exp10(x);
            #else
                /* libm has no portable exp10 for non-fp128 types. */
                return pow(static_cast<T>(10), x);
            #endif
        }
    }
    // Lower bound chosen so the fp32mp2 lo limb of the result stays
    // in fp32 normal range: 10^x > 2⁻¹⁰³ ⇒ x > -31.  Upper bound stays
    // near the fp32 exp10 overflow boundary (log10(FLT_MAX) ≈ 38.5):
    // for large positive x the result is well in normal range, so the
    // lo limb has no denormal issue.
    using base_t = fpmp2_base_type_t<T>;
    static constexpr bool is_wide_domain = std::is_same_v<base_t, double> || std::is_same_v<base_t, __ts_fp128>;
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return is_wide_domain ? -280.0 : -28.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return is_wide_domain ? 280.0 : 38.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1.0; }
};

template<typename T> struct _expm1_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
            { return __TS_EXPM1Q(x); }
        else { return expm1(x); }
    }
    // Center around 0 (the regime where expm1 is non-trivial vs. exp-1),
    // but cover up to fp32 exp overflow boundary so the large-|x| branch
    // is exercised too.
    using base_t = fpmp2_base_type_t<T>;
    static constexpr bool is_wide_domain = std::is_same_v<base_t, double> || std::is_same_v<base_t, __ts_fp128>;
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return is_wide_domain ? -640.0 : -64.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return is_wide_domain ? 640.0 : 64.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.3; }
};

// ============================================================================
// Exponent manipulation: ldexp(x, n) = x · 2^n,  scalbn(x, n) ≡ ldexp(x, n)
// ============================================================================
// Note on (T, int) plumbing:
//   The test framework requires every argument to share the tag's
//   `input_type`, so we declare both args as T and cast the second one to
//   `int` inside operator() via `ldexp_int(T)`.  Casts are guarded against
//   NaN / very large values so neither the impl nor the reference hits UB
//   when the special / pattern datasets generate corner-case inputs.
//   Saturation at ±300 is information-preserving for ldexp (matches the
//   dedicated impl's own clamp): any |n| beyond that point already
//   produces ±inf or ±0 for every finite fp32 input.
template<typename T>
__HOST_DEVICE_DECL__ inline int ldexp_int(const T& v)
{
    /* Collapse the input to a single double *using the full value*.
     * For mp2 types this invokes __nv_fpmp2_to_double(hi, lo) so the
     * lo limb participates in the rounding to int; for scalar types it
     * is a plain conversion.  Without this, the impl path (which sees
     * the mp2 value directly) and the reference path (which sees the
     * losslessly-widened double/fp128 value) could disagree on the int
     * `n` by 1 whenever the lo limb tipped (hi + lo) across an integer
     * boundary — and a one-off in `n` doubles or halves the ldexp
     * result, manifesting as the catastrophic rel_err ≈ 1 we saw on
     * device runs while host runs (fp64 ref) happened to match. */
    const double d = static_cast<double>(v);
    if (d != d)        return 0;     /* NaN  → 0 (skip nonsense) */
    if (d >   300.0)   return  300;  /* saturate big positive */
    if (d <  -300.0)   return -300;  /* saturate big negative */
    return static_cast<int>(d);
}

template<typename T> struct _ldexp_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x, input_type n_t) const
    {
        const int n = ldexp_int(n_t);
        if constexpr (std::is_same_v<T, __ts_fp128>) { return __TS_LDEXPQ(x, n); }
        else                                         { return ldexp(x, n); }
    }
    /* Both arguments share this range.  x in [-200, 200] keeps it in the
     * fp32 normal band; n cast to int spans the full saturation window so
     * the 3-piece bit-cast scaling, the clamp, and the
     * overflow/underflow shortcuts all get exercised. */
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -200.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return  200.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 20.0; }
};

template<typename T> struct _scalbn_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x, input_type n_t) const
    {
        const int n = ldexp_int(n_t);
        if constexpr (std::is_same_v<T, __ts_fp128>) { return __TS_SCALBNQ(x, n); }
        else                                         { return scalbn(x, n); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -200.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return  200.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 20.0; }
};

// ============================================================================
// Logarithmic functions
// ============================================================================
template<typename T> struct _log_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_LOGQ(x); }
        else { return log(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 1e-16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 1.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.5; }
};

template<typename T> struct _log2_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_LOG2Q(x); }
        else { return log2(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 1e-16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 1.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.5; }
};

template<typename T> struct _log10_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_LOG10Q(x); }
        else { return log10(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 1e-16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 1.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.5; }
};

template<typename T> struct _log1p_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_LOG1PQ(x); }
        else { return log1p(x); }
    }
    // Domain: x > -1; focus on small values where log1p(x) ≈ x
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -0.99; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e6; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.1; }
};

// ============================================================================
// Power functions
// ============================================================================
template<typename T> struct _pow_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x, input_type y) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_POWQ(x, y); }
        else { return pow(x, y); }
    }
    // Base positive, exponent moderate to avoid overflow
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 0.01; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 100.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 2.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1.0; }
    // Perf input: stride 7e-7 keeps `a` in [0.8, 1.17] across the 524288-thread
    // sweep.  The perf harness hardcodes `b = 1.5 + 0.0007*tid` (no override
    // hook for the 2nd arg), so b reaches ~368 at the high end.  With
    // |log(a)| <= 0.236, we have:
    //     b * log(a) in [-82, +57]   (well inside exp's active range [-87, +88])
    // so the dedicated exp does real polynomial work for every thread instead
    // of hitting its +Inf overflow shortcut (which would otherwise consume
    // ~93% of the sweep with the default a-range, since b*log(a) reaches
    // ~1463 at the high end and saturates exp from tid >~ 35k onward).
    __HOST_DEVICE_DECL__ static constexpr float perf_start_val() { return 0.8f; }
    __HOST_DEVICE_DECL__ static constexpr float perf_stride()    { return 7e-7f; }
};

template<typename T> struct _cbrt_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_CBRTQ(x); }
        else { return cbrt(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

// rcbrt(x) = 1 / cbrt(x).  For mp2 types we dispatch to the dedicated
// fp32mp2/fp64mp2 implementation via ADL; for the scalar reference types
// (double, float, __ts_fp128) we synthesise it as 1 / cbrt(x).
template<typename T> struct _rcbrt_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>)
            { return T(1) / __TS_CBRTQ(x); }
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
            { return T(1) / cbrt(x); }
        else
            { return rcbrt(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e16; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e16; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1e3; }
};

// ============================================================================
// Trigonometric functions
// ============================================================================
template<typename T> struct _sin_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_SINQ(x); }
        else { return sin(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e4; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e4; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 3.14159; }
};

template<typename T> struct _cos_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_COSQ(x); }
        else { return cos(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e4; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e4; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 3.14159; }
};

template<typename T> struct _tan_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
            { return __TS_TANQ(x); }
        else { return tan(x); }
    }
    // Same broad range as sin/cos.  tan blows up at x = π/2 + k·π, but those
    // are isolated points; the pattern sweep handles ±∞ in the classification
    // table just like the sin/cos near-saturation entries.
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e4; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e4; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 3.14159; }
};

// ============================================================================
// Inverse trigonometric functions
// ============================================================================
template<typename T> struct _asin_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_ASINQ(x); }
        else { return asin(x); }
    }
    // Domain: [-1, 1]
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.5; }
};

template<typename T> struct _acos_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_ACOSQ(x); }
        else { return acos(x); }
    }
    // Domain: [-1, 1]
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.5; }
};

template<typename T> struct _atan_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_ATANQ(x); }
        else { return atan(x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e8; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e8; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1.0; }
};

template<typename T> struct _atan2_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type y, input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_ATAN2Q(y, x); }
        else { return atan2(y, x); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e8; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e8; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1.0; }
};

// ============================================================================
// Remainder family: fmod(x, y), remainder(x, y)
// ============================================================================
// Both arguments share the tag's input_type T and the same work range.
// The range is kept moderate so the integer long-division reduction sees a
// realistic spread of exponent gaps (and the divisor is rarely a tiny
// denormal, which would only stress the slow path without adding coverage).
template<typename T> struct _fmod_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x, input_type y) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>) { return __TS_FMODQ(x, y); }
        else { return fmod(x, y); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1000.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return  1000.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 100.0; }
};

template<typename T> struct _remainder_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x, input_type y) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>) { return __TS_REMAINDERQ(x, y); }
        else { return remainder(x, y); }
    }
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1000.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return  1000.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 100.0; }
};

// ============================================================================
// Hyperbolic functions
// ============================================================================
template<typename T> struct _sinh_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_SINHQ(x); }
        else { return sinh(x); }
    }
    // Moderate range to avoid overflow (sinh grows as exp)
    using base_t = fpmp2_base_type_t<T>;
    static constexpr bool is_wide_domain = std::is_same_v<base_t, double> || std::is_same_v<base_t, __ts_fp128>;
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return is_wide_domain ? -640.0 : -64.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return is_wide_domain ? 640.0 : 64.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 2.0; }
};

template<typename T> struct _cosh_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_COSHQ(x); }
        else { return cosh(x); }
    }
    // Moderate range to avoid overflow (cosh grows as exp)
    using base_t = fpmp2_base_type_t<T>;
    static constexpr bool is_wide_domain = std::is_same_v<base_t, double> || std::is_same_v<base_t, __ts_fp128>;
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return is_wide_domain ? -640.0 : -64.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return is_wide_domain ? 640.0 : 64.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 2.0; }
};

template<typename T> struct _tanh_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_TANHQ(x); }
        else { return tanh(x); }
    }
    // tanh saturates at ±1 for large |x|; wider range is fine
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -20.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 20.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 2.0; }
    // Perf input: stride 0.00001 keeps max ~5.25, which is:
    //   - well below the fp32mp2 saturation threshold ~17.5
    //     (otherwise ~68% of throughput threads would hit the trivial
    //      `return +-1` branch and inflate the timing),
    //   - covering both algorithmic branches: small `|x| < 0.6554117`
    //     (polynomial-only path, ~12% of threads) and large
    //     `|x| >= 0.6554117` (exp-based path, ~88% of threads).
    // Matches erf's perf range for direct comparability.
    __HOST_DEVICE_DECL__ static constexpr float perf_start_val() { return 0.01f; }
    __HOST_DEVICE_DECL__ static constexpr float perf_stride()    { return 0.00001f; }
};

// ============================================================================
// Inverse hyperbolic functions
// ============================================================================
template<typename T> struct _asinh_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
            { return __TS_ASINHQ(x); }
        else { return asinh(x); }
    }
    // asinh grows logarithmically; wide range OK without overflow worry.
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -1e6; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return  1e6; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1.0; }
};

template<typename T> struct _acosh_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
            { return __TS_ACOSHQ(x); }
        else { return acosh(x); }
    }
    // Domain: x >= 1.  Cover the well-conditioned range plus the
    // small-(x-1) regime where the log1p form matters most.
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 1.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1e6; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 1.5; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1.0; }
};

template<typename T> struct _atanh_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
            { return __TS_ATANHQ(x); }
        else { return atanh(x); }
    }
    // Domain: |x| < 1.  Stay slightly inside ±1 to avoid the
    // log1p(+inf) saturation branch dominating the work dataset.
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -0.99; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return  0.99; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.3; }
    // Perf input: stride 7e-7 with start 0.1 keeps the per-thread sweep
    // inside (0.1, ~0.36) for typical L40S/A100 throughput launches
    // (~360K threads * 7e-7 ≈ 0.25 added to start).  The default
    // (start=1.0, stride=1e-4) sits entirely outside the |x|<1 domain
    // and would route every thread through the NaN fast-exit, giving
    // a meaningless +∞x speedup vs. atanhf.
    //   - start=0.1 lands above the 0.25 polynomial branch threshold,
    //     so the canonical perf number reflects the (more expensive)
    //     log1p path that dominates real workloads.
    //   - the polynomial branch is exercised separately by the work /
    //     normal accuracy datasets (which span the full domain).
    __HOST_DEVICE_DECL__ static constexpr float perf_start_val() { return 0.1f; }
    __HOST_DEVICE_DECL__ static constexpr float perf_stride()    { return 7e-7f; }
};

// ============================================================================
// Error functions
// ============================================================================
template<typename T> struct _erf_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_ERFQ(x); }
        else { return erf(x); }
    }
    // erf saturates at ±1 for |x| > ~5.92
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -6.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 6.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1.0; }
    // Perf input: stride 0.00001 keeps max ~5.24 below saturation (~5.92)
    __HOST_DEVICE_DECL__ static constexpr float perf_start_val() { return 0.01f; }
    __HOST_DEVICE_DECL__ static constexpr float perf_stride()    { return 0.00001f; }
};

template<typename T> struct _erfc_ 
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
            { return __TS_ERFCQ(x); }
        else { return erfc(x); }
    }
    // erfc = 1 - erf; interesting near 0 and for positive x (erfc→0)
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return -6.0; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 6.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 1.0; }
    // Perf input: stride 0.00001 keeps max ~5.25, which is:
    //   - well below the fp32mp2 erfc saturation threshold ~27.5
    //     (otherwise ~50% of throughput threads would hit the trivial
    //      `return 0` branch and inflate the timing),
    //   - within the meaningful work range [-6, 6] where erfc is non-trivial
    //     (erfc(5.25) ~ 3e-13, erfc(6) ~ 2e-17 ~ noise floor).
    // Matches erf's perf range for direct comparability.
    __HOST_DEVICE_DECL__ static constexpr float perf_start_val() { return 0.01f; }
    __HOST_DEVICE_DECL__ static constexpr float perf_stride()    { return 0.00001f; }
};

// ============================================================================
// Special functions
// ============================================================================
template<typename T> struct _boys_f0_
{
    using input_type  = T;
    using result_type = T;
    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const
    {
        if constexpr (std::is_same_v<T, __ts_fp128>)
        {
            double xd = (double)x;
            if (xd < 1e-15) return (__ts_fp128)1.0;
            double sx = sqrt(xd);
            return (__ts_fp128)(0.5 * sqrt(3.14159265358979323846 / xd) * erf(sx));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            if (x < 1e-15) return 1.0;
            double sx = sqrt(x);
            return 0.5 * sqrt(3.14159265358979323846 / x) * erf(sx);
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            double xd = (double)x;
            if (xd < 1e-15) return 1.0f;
            double sx = sqrt(xd);
            return (float)(0.5 * sqrt(3.14159265358979323846 / xd) * erf(sx));
        }
        else { return boys_f0(x); }
    }
    // Domain: x >= 0; avoid singularity at x = 0
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 1e-10; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 50.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 10.0; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 5.0; }
    __HOST_DEVICE_DECL__ static constexpr bool has_valid_input_range() { return true; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_min() { return 1e-10; }
    // Perf input: stride 0.00006 keeps max ~31.5, which is:
    //   - well below the dedicated fp32mp2 trivial-branch cliff at
    //     a > 34.38 where boys_f0 reduces to `sqrt(pi)/2 * rsqrt(a)`
    //     (otherwise ~36% of throughput threads would skip the
    //      polynomial work entirely and inflate the timing),
    //   - spanning all three real algorithmic branches:
    //       branch 1 (a < 4)         : ~13% of threads (deg-16 comp Horner)
    //       branch 2 (4 <= a < 11.46): ~24% of threads (deg-19 ff Horner)
    //       branch 3 (11.46 <= a)    : ~63% of threads (rsqrt + deg-18 comp Horner)
    //     giving a representative cost weighted toward the heaviest
    //     polynomial branch (the typical max-cost path).
    __HOST_DEVICE_DECL__ static constexpr float perf_start_val() { return 0.01f; }
    __HOST_DEVICE_DECL__ static constexpr float perf_stride()    { return 0.00006f; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_max() { return 50.0; }
};

// ============================================================================
// Probability functions
// ============================================================================
template<typename T> struct _normcdfinv_ 
{
    using input_type  = T;
    using result_type = T;

    static __HOST_DEVICE_DECL__ double _ref_poly(double p)
    {
        if (p <= 0.0 || p >= 1.0) return nan("");
        double a = fma(2.0, p, -1.0);
        double w = -log(fmax(fma(a, -a, 1.0), 0x1.0p-1022));

        double tc = w - 3.125;
        double rc;
        rc =                    -3.6444120640178197e-21;
        rc = fma(rc, tc, -1.6850591381820166e-19);
        rc = fma(rc, tc,  1.2858480715256400e-18);
        rc = fma(rc, tc,  1.1157877678025181e-17);
        rc = fma(rc, tc, -1.3331716628546209e-16);
        rc = fma(rc, tc,  2.0972767875968562e-17);
        rc = fma(rc, tc,  6.6376381343583238e-15);
        rc = fma(rc, tc, -4.0545662729752069e-14);
        rc = fma(rc, tc, -8.1519341976054722e-14);
        rc = fma(rc, tc,  2.6335093153082323e-12);
        rc = fma(rc, tc, -1.2975133253453532e-11);
        rc = fma(rc, tc, -5.4154120542946279e-11);
        rc = fma(rc, tc,  1.0512122733215323e-09);
        rc = fma(rc, tc, -4.1126339803469837e-09);
        rc = fma(rc, tc, -2.9070369957882005e-08);
        rc = fma(rc, tc,  4.2347877827932404e-07);
        rc = fma(rc, tc, -1.3654692000834679e-06);
        rc = fma(rc, tc, -1.3882523362786469e-05);
        rc = fma(rc, tc,  1.8673420803405714e-04);
        rc = fma(rc, tc, -7.4070253416626698e-04);
        rc = fma(rc, tc, -6.0336708714301491e-03);
        rc = fma(rc, tc,  2.4015818242558962e-01);
        rc = fma(rc, tc,  1.6536545626831027e+00);

        double tt = sqrt(w) - 3.25;
        double rt;
        rt =                     2.2137376921775787e-09;
        rt = fma(rt, tt,  9.0756561938885391e-08);
        rt = fma(rt, tt, -2.7517406297064545e-07);
        rt = fma(rt, tt,  1.8239629214389228e-08);
        rt = fma(rt, tt,  1.5027403968909828e-06);
        rt = fma(rt, tt, -4.0138675269815460e-06);
        rt = fma(rt, tt,  2.9234449089955446e-06);
        rt = fma(rt, tt,  1.2475304481671779e-05);
        rt = fma(rt, tt, -4.7318229009055734e-05);
        rt = fma(rt, tt,  6.8284851459573175e-05);
        rt = fma(rt, tt,  2.4031110387097894e-05);
        rt = fma(rt, tt, -3.5503752036284748e-04);
        rt = fma(rt, tt,  9.5328937973738050e-04);
        rt = fma(rt, tt, -1.6882755560235047e-03);
        rt = fma(rt, tt,  2.4914420961078508e-03);
        rt = fma(rt, tt, -3.7512085075692412e-03);
        rt = fma(rt, tt,  5.3709145535900636e-03);
        rt = fma(rt, tt,  1.0052589676941592e+00);
        rt = fma(rt, tt,  3.0838856104922208e+00);

        // Hardcoded value since M_SQRT2 is not guaranteed to be defined on all platforms
        constexpr double sqrt2_v = 1.41421356237309504880;
        return ((w < 6.125) ? rc : rt) * a * sqrt2_v;
    }

    __HOST_DEVICE_DECL__ result_type operator()(input_type x) const 
    { 
        if constexpr (std::is_same_v<T, __ts_fp128>) 
        {
        #ifdef __CUDACC__
            return (__ts_fp128)::normcdfinv((double)x);
        #else
            return (__ts_fp128)_ref_poly((double)x);
        #endif
        }
        else if constexpr (std::is_same_v<T, double>) 
        {
        #ifdef __CUDACC__
            return ::normcdfinv(x);
        #else
            return _ref_poly(x);
        #endif
        }
        else if constexpr (std::is_same_v<T, float>)
        {
        #ifdef __CUDACC__
            return ::normcdfinvf(x);
        #else
            return (float)_ref_poly((double)x);
        #endif
        }
        else { return normcdfinv(x); }
    }
    // Domain: p in (0, 1); avoid extremes where result → ±∞
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return 1e-6; }
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return 1.0 - 1e-6; }
    __HOST_DEVICE_DECL__ static constexpr double normal_mean()   { return 0.5; }
    __HOST_DEVICE_DECL__ static constexpr double normal_stddev() { return 0.2; }
    __HOST_DEVICE_DECL__ static constexpr float perf_start_val() { return 0.5f; }
    __HOST_DEVICE_DECL__ static constexpr float perf_stride()    { return 0.0000001f; }
    // Restrict pattern-mode bit-pattern sweep to the working domain so we
    // don't bin ~75% of fp32 inputs (everything outside (0,1)) as
    // output_special when the function legitimately returns NaN/±INF there.
    __HOST_DEVICE_DECL__ static constexpr bool has_valid_input_range() { return true; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_min() { return 1e-6; }
    __HOST_DEVICE_DECL__ static constexpr double valid_input_max() { return 1.0 - 1e-6; }
};

// ============================================================================
// Common function wrapper
// ============================================================================
template<typename Tag> struct function 
{
    Tag tag{}; 
    template<typename... Args> __HOST_DEVICE_DECL__ constexpr auto operator()(Args&&... args) const
        -> decltype(tag(std::forward<Args>(args)...)) { return tag(std::forward<Args>(args)...);}
    __HOST_DEVICE_DECL__ static constexpr double work_beg() { return Tag::work_beg();}
    __HOST_DEVICE_DECL__ static constexpr double work_end() { return Tag::work_end();}
};

// ============================================================================
// Compile-time arity detection using tag's input_type
// ============================================================================
template<typename Tag, typename = void>
struct has_input_type : std::false_type {};

template<typename Tag>
struct has_input_type<Tag, std::void_t<typename Tag::input_type>> : std::true_type {};

// Primary template: use tag's input_type if available
template<typename Tag, typename FallbackArg = void>
__HOST_DEVICE_DECL__ constexpr int detect_arity() {
    using In = std::conditional_t<has_input_type<Tag>::value, typename Tag::input_type, FallbackArg>;
    if constexpr (std::is_invocable_v<Tag, In, In, In, In>) {return 4;} 
    else if constexpr (std::is_invocable_v<Tag, In, In, In>) {return 3;} 
    else if constexpr (std::is_invocable_v<Tag, In, In>) {return 2;} 
    else if constexpr (std::is_invocable_v<Tag, In>) {return 1;}
    return 0;
}

// ============================================================================
// Type traits for function tags
// ============================================================================
template<typename Tag>
using tag_input_t = typename Tag::input_type;

template<typename Tag>
using tag_result_t = typename Tag::result_type;

// ============================================================================
// Implementation dispatcher based on detected arity
// Uses tag's input_type and result_type
// ============================================================================
template<typename Tag, int Arity> struct impl_dispatch;

// 1-arity implementation
template<typename Tag> struct impl_dispatch<Tag, 1> {
    using input_t  = tag_input_t<Tag>;
    using result_t = tag_result_t<Tag>;
    static __HOST_DEVICE_DECL__ result_t call(input_t x, input_t, input_t, input_t) { function<Tag> f; return f(x); }
};
// 2-arity implementation
template<typename Tag> struct impl_dispatch<Tag, 2> {
    using input_t  = tag_input_t<Tag>;
    using result_t = tag_result_t<Tag>;
    static __HOST_DEVICE_DECL__ result_t call(input_t x, input_t y, input_t, input_t) { function<Tag> f; return f(x, y); }
};
// 3-arity implementation
template<typename Tag> struct impl_dispatch<Tag, 3> {
    using input_t  = tag_input_t<Tag>;
    using result_t = tag_result_t<Tag>;
    static __HOST_DEVICE_DECL__ result_t call(input_t x, input_t y, input_t z, input_t) { function<Tag> f; return f(x, y, z); }
};
// 4-arity implementation
template<typename Tag> struct impl_dispatch<Tag, 4> {
    using input_t  = tag_input_t<Tag>;
    using result_t = tag_result_t<Tag>;
    static __HOST_DEVICE_DECL__ result_t call(input_t x, input_t y, input_t z, input_t w) { function<Tag> f; return f(x, y, z, w); }
};

/*
* ============================================================================
* Device function implementation for PTX/SASS generation and analysis
* ============================================================================
*/
#if defined __DEVICE_FUNC__

    // Use tag's input_type and result_type directly
    using fp_input_type  = tag_input_t<FUNC_TAG>;
    using fp_result_type = tag_result_t<FUNC_TAG>;

    extern "C" __device__ fp_result_type FUNC_IMPL(fp_input_type x, fp_input_type y = fp_input_type(0), fp_input_type z = fp_input_type(0), fp_input_type w = fp_input_type(0))
    {
        constexpr int arity = detect_arity<FUNC_TAG>();
        return impl_dispatch<FUNC_TAG, arity>::call(x, y, z, w);
    }

    /*
     * SASS-step kernel wrapper.
     *
     * The TS SASS rule needs ptxas to run as a final code-generator
     * (not in --compile-only mode) so it reports `Used N registers`
     * for FUNC_IMPL.  ptxas only does physical register allocation
     * when it sees a `__global__` entry, so this wrapper provides a
     * minimal launchable kernel that calls FUNC_IMPL.  The wrapper is
     * gated behind __TS_SASS_KERNEL__ so it is invisible to the
     * regular test executable build.
     */
#if defined __TS_SASS_KERNEL__
    extern "C" __global__ void __ts_sass_kernel(fp_input_type x,
                                                fp_input_type y,
                                                fp_input_type z,
                                                fp_input_type w,
                                                fp_result_type* out)
    {
        *out = FUNC_IMPL(x, y, z, w);
    }
#endif

#endif


#endif // __TS_FUNCTIONS_HPP__
