/*
    ts_types.hpp - FPMP Test Suite Type Definitions
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Type definitions, enumerations, and type traits used throughout the test suite.
    This file consolidates all common types used by accuracy and performance testing.
*/

#ifndef __TS_TYPES_HPP__
#define __TS_TYPES_HPP__

#include "ts.hpp"

namespace ts
{
    // Stream selection for printf_stream
    enum struct stream
    {
        def       = 0,
        stdout    = 0,    
        stderr    = 1,
        file      = 2,
        null      = 3,
    };

    // Console output redirection (compile-time constant set via Makefile)
    constexpr stream console = stream::__CONSOLE__;

    // CSV log file (set at runtime if .csv argument provided)
    FILE* logfile = nullptr;

} // end of namespace ts

// ============================================================================
// Performance Types
// ============================================================================

// Bit manipulation helper types (prevent compiler optimization)
struct u32x2 { uint32_t lo, hi; };
struct u32x4 { uint32_t p0, p1, p2, p3; };

// Performance measurement mode
enum class perf_mode { throughput, latency };

// Timing result structure (shared between host and device)
struct ts_timing_result_t 
{
    double time_ms;           // Average time in milliseconds
    double gflops;            // Operations per second (GFLOPS)
    double evals_per_clk_sm;  // Evaluations per clock per SM (or core for host)
    double clocks_per_eval;   // Clocks per evaluation (for latency)
    long long total_ops;      // Total number of operations measured
    int num_blocks;           // num_blocks (GPU) or num_threads (host)
    int threads_per_block;    // threads_per_block (GPU) or 1 (host)
    int reps;
    int iterations;
    int sm_count;             // Number of SMs (GPU) or cores (host)
    int clock_rate_khz;       // Clock rate in kHz
}; // ts_timing_result_t

// Comparison result structure (shared between host and device)
struct ts_timing_comparison_result_t 
{
    ts_timing_result_t test_result;
    ts_timing_result_t ref_result;
    double speedup;  // ref_time / test_time (>1 means test is faster)
}; // ts_timing_comparison_result_t

// ============================================================================
// Accuracy Types
// ============================================================================

// Number of accuracy warning classes (for array sizing)
constexpr int ACCURACY_CLASS_COUNT = 12;  // 0=normal, 1-10=warnings, 11=error

// Detailed accuracy classification for warnings and errors
// Warning classes (1-9) are treated as warnings, not errors
// The order matters: lower numbers have higher priority in classification
enum class accuracy_class : int
{
    normal                    = 0,   // Normal: within warning threshold
    
    // === Warning classes (1-10) ===
    output_special            = 1,   // Output is incorrect special value (INF, NaN)
    input_special             = 2,   // Incorrect result when input is special (INF, NaN)
    output_denormal           = 3,   // Output is incorrect denormal (exceeds warning threshold)
    input_denormal            = 4,   // Incorrect result when input is denormal (exceeds warning threshold)
    output_near_denormal      = 5,   // Output is close to denormal and exceeds warning threshold
    input_near_denormal       = 6,   // Input is close to denormal and result exceeds warning threshold
    output_near_inf           = 7,   // Output is close to INF and exceeds warning threshold
    input_near_inf            = 8,   // Input is close to INF and result exceeds warning threshold
    cancellation              = 9,   // Cancellation case (result much smaller than inputs)
    unclassified              = 10,  // Warning without identified cause (between warning and error threshold)
    
    // === Error class ===
    error                     = 11   // Error: exceeds error threshold without any warning condition
};

// Get human-readable name for accuracy class
inline const char* accuracy_class_name(accuracy_class cls)
{
    switch (cls)
    {
        case accuracy_class::normal:                    return "normal (OK)";
        case accuracy_class::output_special:            return "output special";
        case accuracy_class::input_special:             return "input special";
        case accuracy_class::output_denormal:           return "output denormal";
        case accuracy_class::input_denormal:            return "input denormal";
        case accuracy_class::output_near_denormal:      return "output near denormal";
        case accuracy_class::input_near_denormal:       return "input near denormal";
        case accuracy_class::output_near_inf:           return "output near inf";
        case accuracy_class::input_near_inf:            return "input near inf";
        case accuracy_class::cancellation:              return "cancellation";
        case accuracy_class::unclassified:              return "unclassified";
        case accuracy_class::error:                     return "error (FAIL)";
        default:                                        return "unknown";
    }
}

// Get short name for accuracy class (for compact output)
inline const char* accuracy_class_short_name(accuracy_class cls)
{
    switch (cls)
    {
        case accuracy_class::normal:                    return "OK";
        case accuracy_class::output_special:            return "OUT_SPL";
        case accuracy_class::input_special:             return "IN_SPL";
        case accuracy_class::output_denormal:           return "OUT_DEN";
        case accuracy_class::input_denormal:            return "IN_DEN";
        case accuracy_class::output_near_denormal:      return "OUT_NDEN";
        case accuracy_class::input_near_denormal:       return "IN_NDEN";
        case accuracy_class::output_near_inf:           return "OUT_NINF";
        case accuracy_class::input_near_inf:            return "IN_NINF";
        case accuracy_class::cancellation:              return "CANCEL";
        case accuracy_class::unclassified:              return "UNCLASS";
        case accuracy_class::error:                     return "ERR";
        default:                                        return "???";
    }
}

// Check if accuracy class is a warning (not normal, not error)
__HOST_DEVICE_DECL__ inline bool is_warning_class(accuracy_class cls)
{
    int c = static_cast<int>(cls);
    return c >= 1 && c <= 10;  // warning classes: 1-10 (including unclassified)
}

// Legacy record_class for backward compatibility
enum class record_class : int
{
    error   = 0,  // Error: exceeds error threshold with normal high part
    warning = 1,  // Warning: exceeds threshold with near-denormal high part, or exceeds warning threshold
    normal  = 2   // Normal: within warning threshold
};

// Per-class statistics structure
struct accuracy_class_stats_t
{
    unsigned long long count;       // Number of cases in this class
    double             max_rel_err; // Maximum relative error in this class
    double             sum_rel_err; // Sum of relative errors (for average)
};

// Error record structure - stores details of a single error case
// Uses native fpmp_type and fprf_type for full precision storage
struct accuracy_error_record_t
{
    // Input arguments in native test type
    fpmp_type args[4];
    
    // Test result in native test type
    fpmp_type test_result;
    
    // Reference result in native reference type
    fprf_type ref_result;
    
    double         rel_err;           // Relative error
    int            arity;             // Number of arguments used
    record_class   classification;    // Legacy: Error, warning, or normal
    accuracy_class acc_class;         // Detailed accuracy classification
};

// Per-class result summary
struct accuracy_class_result_t
{
    uint64_t count;           // Number of cases in this class
    double   max_rel_err;     // Maximum relative error in this class
    double   avg_rel_err;     // Average relative error in this class
    int      correct_bits;    // Minimum correct bits for this class
    
    // Max error record for this class
    accuracy_error_record_t max_record;
    bool                    has_max_record;
};

// Accuracy result structure
struct ts_accuracy_result_t 
{
    double   max_rel_err;       // Maximum relative error
    double   avg_rel_err;       // Average relative error
    double   max_ulp;           // Maximum ULP error
    double   avg_ulp;           // Average ULP error
    int      correct_bits;      // Minimum correct bits (based on max error)
    int      max_mantissa_bits; // Maximum possible mantissa bits for test type
    uint64_t total_ops;         // Total operations tested
    uint64_t valid_ops;         // Valid (finite) operations tested
    uint64_t total_errs;        // Number of errors (rel_err > threshold)
    uint64_t total_warnings;    // Number of warnings
    uint64_t skipped_inf_nan;   // Number of skipped INF/NAN results
    
    // Maximum error record (always tracked, even below threshold)
    accuracy_error_record_t max_error_record;
    bool                    has_max_error;    // True if any valid comparison was made
    
    // Error log: first N errors captured during testing (above threshold only)
    accuracy_error_record_t error_log[TS_ACCURACY_ERROR_LOG_SIZE];
    uint64_t                error_log_count;  // Actual number of logged errors (may exceed buffer size)
    
    // Per-class statistics (indexed by accuracy_class enum value)
    accuracy_class_result_t class_results[ACCURACY_CLASS_COUNT];
};

// Common reduction results structure (used by both host and CUDA)
struct accuracy_reduction_t
{
    double max_rel_err;
    double sum_rel_err;
    double max_ulp;
    double sum_ulp;
    unsigned long long total_errs;      // unsigned long long for CUDA atomics compatibility
    unsigned long long total_warnings;  // warnings: rel_err > warning_threshold but <= error_threshold
    unsigned long long valid_count;
    unsigned long long skipped_inf_nan;
    
    // Per-class statistics (indexed by accuracy_class enum value)
    accuracy_class_stats_t class_stats[ACCURACY_CLASS_COUNT];
};

// Error log structure with atomic counter for thread-safe logging
struct accuracy_error_log_t
{
    accuracy_error_record_t records[TS_ACCURACY_ERROR_LOG_SIZE];
    unsigned long long      count;  // Atomic counter for next available slot
    
    // Max error tracking (always updated, regardless of threshold)
    accuracy_error_record_t max_record;
    double                  max_rel_err;  // For atomic compare-and-swap
    
    // Per-class max error tracking
    accuracy_error_record_t class_max_records[ACCURACY_CLASS_COUNT];
    double                  class_max_rel_err[ACCURACY_CLASS_COUNT];  // For atomic CAS
};

// Fixed inputs structure for passing pre-parsed values to kernel
// When __FIXED_INPUTS__ is enabled, strings are parsed on host and values
// are passed to the kernel via this structure. This allows GPU execution
// with user-specified inputs.
template<typename TestInputType>
struct fixed_inputs_t
{
    TestInputType values[4];  // Up to 4 input arguments
    bool          valid[4];   // Which inputs were provided
};

// Helper struct for 128-bit bit_cast
struct u64x2 { uint64_t lo, hi; };

// Get error threshold based on component type (use template specialization for CUDA compatibility)
template<typename ComponentType> struct accuracy_error_threshold_impl        { static constexpr double value = TS_ACCURACY_ERROR_THRESHOLD_FP64; };
template<>                       struct accuracy_error_threshold_impl<float> { static constexpr double value = TS_ACCURACY_ERROR_THRESHOLD_FP32; };

// Get warning threshold based on component type
template<typename ComponentType> struct accuracy_warning_threshold_impl        { static constexpr double value = TS_ACCURACY_WARNING_THRESHOLD_FP64; };
template<>                       struct accuracy_warning_threshold_impl<float> { static constexpr double value = TS_ACCURACY_WARNING_THRESHOLD_FP32; };

template<typename ComponentType> __HOST_DEVICE_DECL__ constexpr double get_error_threshold()   { return accuracy_error_threshold_impl<ComponentType>::value; }
template<typename ComponentType> __HOST_DEVICE_DECL__ constexpr double get_warning_threshold() { return accuracy_warning_threshold_impl<ComponentType>::value; }



#endif // __TS_TYPES_HPP__
