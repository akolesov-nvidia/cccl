/*
    ts.cpp - FPMP Test Suite Main Entry Point
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Main entry point for the FPMP test suite. This file orchestrates accuracy and performance testing
    for all multi-precision floating-point types and operations.

    Note: Work in progress.
*/

// ts_types.hpp includes ts.hpp, which defines FPMP_TYPE_ID and other macros
#include "ts_types.hpp"
#include "ts_utils.hpp"

#include "ts_utils.hpp"
#include "ts_functions.hpp"
#include "ts_print.hpp"
#include "ts_performance.hpp"
#include "ts_accuracy.hpp"
#include <cmath>
#include <cstdio>

// Import commonly used items from ts namespace for cleaner code
using ts::stream;
using ts::printf_stream;
using ts::write_csv_header;

// Main function
int main (int argc, char *argv[])
{
    if (argc > 1)
    {
        if (strstr(argv[1], ".csv"))
        {
            // Check if file exists and has content
            FILE* check_file = fopen(argv[1], "r");
            bool need_header = (check_file == nullptr);
            if (check_file)
            {
                fseek(check_file, 0, SEEK_END);
                need_header = (ftell(check_file) == 0);
                fclose(check_file);
            }
            
            ts::logfile = fopen(argv[1], "a");
            if (!ts::logfile) 
            {
                printf_stream(stream::stderr, "Error opening log file %s\n", argv[1]);
                return 1;
            }
            
            // Write CSV header if file is new or empty
            if (need_header)
            {
                write_csv_header(ts::logfile);
            }
        }
    }

    CUDA_INIT();

    // Define function tags for current function
    using func_tag_tst  = GLUE3(_,__FUNC__,_)<fpmp_type>;   // Multi-precision type (fp32mp2 or fp64mp2)
    using func_tag_ref  = GLUE3(_,__FUNC__,_)<fprf_type>;   // Reference type (double or __ts_fp128)
    using func_tag_base = GLUE3(_,__FUNC__,_)<BASE_TYPE>;   // Base scalar type (float or double)

    // Performance reference: for conversion functions (mp2fp, fp2mp), use a cast-based
    // reference with the SAME mp2 types so the benchmark loop is identical and only
    // the function body differs. For all other functions, use the standard fp64 reference.
    using func_tag_perf_ref = perf_ref_tag_t<func_tag_tst, func_tag_ref>;
    constexpr bool has_dedicated_perf_ref = !std::is_same_v<func_tag_perf_ref, func_tag_ref>;
    using perf_ref_arg_type = std::conditional_t<has_dedicated_perf_ref, fpmp_type, fprf_type>;

    // ============================================================================
    // Run performance measurements first (needed for header info)
    // ============================================================================
    // Get device info (always, even if timing is disabled)
#if defined(__CUDACC__)
    int clock_rate_khz;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, cuda_dev);
    int clock_mhz = clock_rate_khz / 1000;
    int threads_to_show = TS_ACCURACY_THREADS_PER_BLOCK;  // Same as TS_TIMING_THREADS_PER_BLOCK
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, cuda_dev);
    int sm_count = props.multiProcessorCount;
#else
    int clock_mhz = static_cast<int>(get_clock_rate() / 1000);  // get_clock_rate returns kHz
    int threads_to_show = omp_get_max_threads();
    int sm_count = threads_to_show;  // On host, "SM count" is OMP thread count
#endif

#if (!defined(__RUN_TIMING__) || __RUN_TIMING__ != 0) && !defined(__FIXED_INPUTS__)
    // Test vs reference (mp2 vs cast-based ref for conversions, fp64/fp128 for others)
    auto throughput = compare_throughput<func_tag_tst, fpmp_type, func_tag_perf_ref, perf_ref_arg_type>();
    auto latency = compare_latency<func_tag_tst, fpmp_type, func_tag_perf_ref, perf_ref_arg_type>();
    
    // Base scalar type performance (fp32 or fp64)
    auto throughput_base = compare_throughput<func_tag_base, BASE_TYPE, func_tag_tst, fpmp_type>();
    auto latency_base = compare_latency<func_tag_base, BASE_TYPE, func_tag_tst, fpmp_type>();
#endif

    // ============================================================================
    // Print boxed header with function info and device details
    // ============================================================================
    printf_stream(stream::stdout, "\n==================================================================================\n");
    printf_stream(stream::stdout, "%s <%s> (%s) @  %s [ %d MHz, %d thr/blk, %d SM ]\n",
                      ABC(__FUNC__), ABC(__METHOD__), ABC(__TYPE__),
                      CUDA_DEVICE_NAME(), clock_mhz, threads_to_show, sm_count);
    printf_stream(stream::stdout, "==================================================================================\n");

    // ============================================================================
    // Accuracy testing
    // ============================================================================
#if !defined(__RUN_ACCURACY__) || __RUN_ACCURACY__ != 0

    // Dataset filtering configuration
#if defined(__FIXED_INPUTS__)
    #undef __DATASET_WORK__
    #undef __DATASET_NORMAL__
    #undef __DATASET_PATTERN__
    #define __DATASET_SPECIAL__
#elif !defined(__DATASET_WORK__) && !defined(__DATASET_NORMAL__) && !defined(__DATASET_SPECIAL__) && !defined(__DATASET_PATTERN__)
    #define __DATASET_WORK__
    #define __DATASET_NORMAL__
    #define __DATASET_SPECIAL__
    #define __DATASET_PATTERN__
#endif

    const accuracy_mode accuracy_modes[] = { 
#if defined(__DATASET_WORK__)
        { accuracy_mode::work,    TS_ACCURACY_RIGOR_WORK,    "work"    },
#endif
#if defined(__DATASET_NORMAL__)
        { accuracy_mode::normal,  TS_ACCURACY_RIGOR_NORMAL,  "normal"  },
#endif
#if defined(__DATASET_SPECIAL__)
        { accuracy_mode::special, TS_ACCURACY_RIGOR_SPECIAL, "special" },
#endif
#if defined(__DATASET_PATTERN__)
        { accuracy_mode::pattern, TS_ACCURACY_RIGOR_PATTERN, "pattern" },
#endif
    };
    const int num_modes = sizeof(accuracy_modes) / sizeof(accuracy_modes[0]);

    // Get max bits for the result type (function-specific, not just type-specific)
    // For comparison ops (eq, ne, lt, le, gt, ge): result is bool → 1 bit
    // For integer conversions (mp2int, mp2ll, mp2uint, mp2ull): result bits match integer size
    // For arithmetic ops: result is same as input fpmp type
    using TestResultType = typename func_tag_tst::result_type;
    constexpr int result_max_bits = ts::get_max_mantissa_bits<TestResultType>();
    
    // Print accuracy table
    char bits_header[16];
    snprintf(bits_header, sizeof(bits_header), "Bits/%d", result_max_bits);
    printf_stream(stream::stdout, "%-10s| %-9s| %-9s| %-48s\n", 
                      "Dataset", "Rel err", bits_header, "Accuracy");
    printf_stream(stream::stdout, "----------+----------+----------+-------------------------------------------------\n");

    // Track work dataset accuracy for CSV logging
    ts_accuracy_result_t work_accuracy = {};
    bool have_work_accuracy = false;

    for (int i = 0; i < num_modes; i++)
    {
        auto accuracy = measure_accuracy<func_tag_tst, func_tag_ref>(accuracy_modes[i]);
        
        // Save work dataset accuracy for CSV logging
        if (accuracy_modes[i].mode == accuracy_mode::work)
        {
            work_accuracy = accuracy;
            have_work_accuracy = true;
        }
        
        // Print special values table for special dataset (skip in fixed-input mode to avoid extra function calls)
#if !defined(__FIXED_INPUTS__)
        if (accuracy_modes[i].mode == accuracy_mode::special)
        {
            print_special_values_table<func_tag_tst, fpmp_type>(stderr, ABC(__FUNC__), ABC(__METHOD__), "special");
        }
#endif
        
        // Use global max_rel_err for summary - shows worst-case accuracy across ALL samples
        // (Classification stats are shown separately in the detailed log)
        double display_rel_err = accuracy.max_rel_err;
        int display_bits = accuracy.correct_bits;
        
        if (display_bits >= 0)
        {
            // Determine status and format message
            const char* status = "OK";
            char status_detail[64] = "";
            
            if (accuracy.total_errs > 0)
            {
                status = "FAIL";
                double err_pct = (accuracy.total_errs * 100.0) / accuracy.total_ops;
                double warn_pct = (accuracy.total_warnings * 100.0) / accuracy.total_ops;
                // Use scientific notation for very small percentages
                if (err_pct >= 0.01)
                    snprintf(status_detail, sizeof(status_detail), " (err %.2f%%, warn %.2f%%)", err_pct, warn_pct);
                else
                    snprintf(status_detail, sizeof(status_detail), " (err %.0e%%, warn %.0e%%)", err_pct, warn_pct);
            }
            else if (accuracy.total_warnings > 0)
            {
                status = "WARN";
                double warn_pct = (accuracy.total_warnings * 100.0) / accuracy.total_ops;
                // Use scientific notation for very small percentages
                if (warn_pct >= 0.01)
                    snprintf(status_detail, sizeof(status_detail), " (warn %.2f%%)", warn_pct);
                else
                    snprintf(status_detail, sizeof(status_detail), " (warn %.0e%%)", warn_pct);
            }
            
            printf_stream(stream::stdout, "%-10s| %.2e | %-9d| %s%s\n",
                              accuracy_modes[i].name,
                              display_rel_err,
                              display_bits,
                              status,
                              status_detail);
        }
        else
        {
            printf_stream(stream::stdout, "%-10s| %-9s| %-9s| %s\n",
                              accuracy_modes[i].name,
                              "N/A",
                              "N/A",
                              "-- (all INF/NAN)");
        }
        
        // Print error logs if configured
#if defined(__PRINT_NONE__)
        (void)accuracy;
#else
        constexpr bool print_errors =
#if defined(__PRINT_ALL__) || defined(__PRINT_FAIL__)
            true;
#else
            false;
#endif

        constexpr bool print_warnings =
#if defined(__PRINT_ALL__) || defined(__PRINT_WARN__)
            true;
#else
            false;
#endif

        const bool want_detailed_log =
            (print_errors   && (accuracy.total_errs > 0 || accuracy.error_log_count > 0)) ||
            (print_warnings && (accuracy.total_warnings > 0));

        if (accuracy.has_max_error || want_detailed_log)
        {
            print_accuracy_log(accuracy, ABC(__FUNC__) " <" ABC(__METHOD__) ">", accuracy_modes[i].name);
        }
        
        // Print per-class statistics always (to stderr log file)
        // Only available when CLASSIFY is enabled
#if !defined(__RUN_CLASSIFY__) || __RUN_CLASSIFY__ != 0
        print_class_statistics(accuracy, ABC(__FUNC__) " <" ABC(__METHOD__) ">", accuracy_modes[i].name, stderr);
#endif
#endif
    }

#else
    // Accuracy testing disabled
    ts_accuracy_result_t work_accuracy = {};
    bool have_work_accuracy = false;
#endif // __RUN_ACCURACY__

    // ============================================================================
    // Print performance table
    // ============================================================================
#if (!defined(__RUN_TIMING__) || __RUN_TIMING__ != 0) && !defined(__FIXED_INPUTS__)
    // Calculate speedups for display
    // Convention: ratio <1 = slowdown (shown as negative), ratio >1 = speedup (shown as positive)
    
    // For throughput (higher = better): mp2/base or mp2/ref
    double base_ratio_tput = throughput.test_result.evals_per_clk_sm / throughput_base.test_result.evals_per_clk_sm;
    double ref_ratio_tput = throughput.test_result.evals_per_clk_sm / throughput.ref_result.evals_per_clk_sm;
    
    // For latency (lower = better): mp2/base <1 means mp2 is slower (more clocks)
    double base_ratio_lat = latency.test_result.clocks_per_eval / latency_base.test_result.clocks_per_eval;
    // mp2/ref: <1 means mp2 slower (more clocks), >1 means mp2 faster (fewer clocks)
    double ref_ratio_lat = latency.test_result.clocks_per_eval / latency.ref_result.clocks_per_eval;
    
    const char* perf_ref_name = has_dedicated_perf_ref ? "cast" : ABC(REF_TYPE_NAME);
    printf_stream(stream::stdout, "----------+----------+----------+----------+--------------+-----------------------\n");
    printf_stream(stream::stdout, "%-10s| %-8s | %-8s | %-8s | %-12s | %s/%s\n",
                      "Perf", ABC(BASE_TYPE_NAME), ABC(__TYPE__), perf_ref_name, 
                      ABC(BASE_TYPE_NAME) "/" ABC(__TYPE__), ABC(__TYPE__), perf_ref_name);
    printf_stream(stream::stdout, "----------+----------+----------+----------+--------------+-----------------------\n");
    
    // Format a speedup ratio into a fixed-width field. Applies the sign rule
    // (ratio <1 = slowdown, shown negative; >1 = speedup, shown positive) and
    // prints "n/a" for non-finite values -- the host timer reports 0 clocks on
    // some platforms (e.g. ARM64), which would otherwise yield "+nanx"/"+infx".
    auto fmt_ratio = [](double r, char* buf, size_t n) -> const char* {
        if (std::isfinite(r)) {
            double s = (r < 1.0) ? -r : r;
            std::snprintf(buf, n, "%+11.2fx", s);
        } else {
            std::snprintf(buf, n, "%12s", "n/a");
        }
        return buf;
    };
    char rbuf1[24], rbuf2[24];

    // ev/clk/SM row (evaluations per clock per SM - higher is better)
    // ratio <1 = slowdown (negative), ratio >1 = speedup (positive)
    printf_stream(stream::stdout, "%-10s| %-8.2f | %-8.2f | %-8.2f | %s | %s\n",
                      "ev/clk/SM",
                      throughput_base.test_result.evals_per_clk_sm,
                      throughput.test_result.evals_per_clk_sm,
                      throughput.ref_result.evals_per_clk_sm,
                      fmt_ratio(base_ratio_tput, rbuf1, sizeof(rbuf1)),
                      fmt_ratio(ref_ratio_tput,  rbuf2, sizeof(rbuf2)));
    
    // GFLOPS row
    printf_stream(stream::stdout, "%-10s| %-8.1f | %-8.1f | %-8.1f | %s | %s\n",
                      "GFLOPS",
                      throughput_base.test_result.gflops,
                      throughput.test_result.gflops,
                      throughput.ref_result.gflops,
                      fmt_ratio(base_ratio_tput, rbuf1, sizeof(rbuf1)),
                      fmt_ratio(ref_ratio_tput,  rbuf2, sizeof(rbuf2)));
    
    // clk/eval row (clocks per evaluation - lower is better)
    // Invert ratio for display, then same sign rule: <1 = negative, >1 = positive
    double base_inv_lat = 1.0 / base_ratio_lat;
    double ref_inv_lat  = 1.0 / ref_ratio_lat;
    printf_stream(stream::stdout, "%-10s| %-8.2f | %-8.2f | %-8.2f | %s | %s\n",
                      "clk/ev",
                      latency_base.test_result.clocks_per_eval,
                      latency.test_result.clocks_per_eval,
                      latency.ref_result.clocks_per_eval,
                      fmt_ratio(base_inv_lat, rbuf1, sizeof(rbuf1)),
                      fmt_ratio(ref_inv_lat,  rbuf2, sizeof(rbuf2)));
#endif

    printf_stream(stream::stdout, "==================================================================================\n");
    fflush(stdout);

    // ============================================================================
    // Write CSV log entry if log file is open
    // ============================================================================
    if (ts::logfile)
    {
        // Determine work dataset status
        const char* work_status = "N/A";
        double work_rel_err = 0.0;
        int work_bits = 0;
        
        if (have_work_accuracy)
        {
            if (work_accuracy.correct_bits >= 0)
            {
                work_status = (work_accuracy.total_errs > 0) ? "FAIL" : "OK";
                work_rel_err = work_accuracy.max_rel_err;
                work_bits = work_accuracy.correct_bits;
            }
        }
        
#if (!defined(__RUN_TIMING__) || __RUN_TIMING__ != 0) && !defined(__FIXED_INPUTS__)
        // Calculate ratios for CSV (same as display)
        double csv_base_ratio_tput = throughput.test_result.evals_per_clk_sm / throughput_base.test_result.evals_per_clk_sm;
        double csv_ref_ratio_tput = throughput.test_result.evals_per_clk_sm / throughput.ref_result.evals_per_clk_sm;
        double csv_base_ratio_lat = latency.test_result.clocks_per_eval / latency_base.test_result.clocks_per_eval;
        double csv_ref_ratio_lat = latency.test_result.clocks_per_eval / latency.ref_result.clocks_per_eval;
        
        // Format ratios the same way as display: <1 = negative, >1 = positive
        double ev_clk_ratio_base = (csv_base_ratio_tput < 1.0) ? -csv_base_ratio_tput : csv_base_ratio_tput;
        double ev_clk_ratio_ref  = (csv_ref_ratio_tput < 1.0) ? -csv_ref_ratio_tput : csv_ref_ratio_tput;
        double csv_base_inv_lat = 1.0 / csv_base_ratio_lat;
        double csv_ref_inv_lat  = 1.0 / csv_ref_ratio_lat;
        double clk_ev_ratio_base = (csv_base_inv_lat < 1.0) ? -csv_base_inv_lat : csv_base_inv_lat;
        double clk_ev_ratio_ref  = (csv_ref_inv_lat < 1.0)  ? -csv_ref_inv_lat  : csv_ref_inv_lat;
        
        // CSV format: function,type,method,work_rel_err,work_bits,work_status,
        //             base_gflops,test_gflops,ref_gflops,
        //             base_ev_clk,test_ev_clk,ref_ev_clk,ev_clk_ratio_base,ev_clk_ratio_ref,
        //             base_clk_ev,test_clk_ev,ref_clk_ev,clk_ev_ratio_base,clk_ev_ratio_ref
        fprintf(ts::logfile, "%s,%s,%s,%.6e,%d,%s,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                ABC(__FUNC__),
                ABC(__TYPE__),
                ABC(__METHOD__),
                work_rel_err,
                work_bits,
                work_status,
                throughput_base.test_result.gflops,
                throughput.test_result.gflops,
                throughput.ref_result.gflops,
                throughput_base.test_result.evals_per_clk_sm,
                throughput.test_result.evals_per_clk_sm,
                throughput.ref_result.evals_per_clk_sm,
                ev_clk_ratio_base,
                ev_clk_ratio_ref,
                latency_base.test_result.clocks_per_eval,
                latency.test_result.clocks_per_eval,
                latency.ref_result.clocks_per_eval,
                clk_ev_ratio_base,
                clk_ev_ratio_ref);
#else
        // No timing data - just log accuracy
        fprintf(ts::logfile, "%s,%s,%s,%.6e,%d,%s,,,,,,,,,,,,,\n",
                ABC(__FUNC__),
                ABC(__TYPE__),
                ABC(__METHOD__),
                work_rel_err,
                work_bits,
                work_status);
#endif
        fflush(ts::logfile);
    }

    // Close the log file
    if (ts::logfile) fclose(ts::logfile);

    return 0;
}
