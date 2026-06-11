// TS headers
#include "ts.hpp"
#include "ts_utils.hpp"
#include "ts_types.hpp"
#include "ts_functions.hpp"
#include "ts_fixed.hpp"
#include "ts_print.hpp"
#include "ts_datasets.hpp"
#include "ts_run.hpp"
#include "ts_analyze.hpp"

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

            ts::config.logfile = fopen(argv[1], "a");
            if (!ts::config.logfile) 
            {
                ts::printf_stream(ts::stream::stderr, "Error opening log file %s\n", argv[1]);
                return 1;
            }

            // Write CSV header if file is new or empty
            if (need_header)
            {
                fprintf(ts::config.logfile,
                    "function,rounding,method,type,dataset,"
                    "throughput_tst,throughput_ref,throughput_ratio,"
                    "latency_tst,latency_ref,latency_ratio,"
                    "result,device\n");
                fflush(ts::config.logfile);
            }
        }
    }

    int64_t  current_errors   = 0;
    int64_t  current_warnings = 0;
    int64_t  total_errors     = 0;
    int64_t  total_warnings   = 0;
    uint64_t total_len        = 0;
    int32_t  total_bits       = 0;
    
    ts::config.mask_value = argc < 8 ? 1.0 : (double)argc;

    // Create the function and reference function
    using emu_function     = FUNC_NAME<ARGTYPE_EMU,     RESTYPE_EMU,     ts::config.rm>;
    using native_function  = FUNC_NAME<ARGTYPE_NATIVE,  RESTYPE_NATIVE,  ts::config.rm>;

    CUDA_INIT();

    // Create the dataset_array
    ts::dataset_array_t<emu_function, native_function> dataset_array(ts::config.max_len);

    // Print the header
    ts::printf_stream(ts::stream::stdout, "------------------------------------------------------\n");
    ts::printf_stream(ts::stream::stdout, "%s <%s, %s> (%s / native) @ %s [%d thr/blk]\n",  
    ABC(__FUNC__), ABC(__ROUNDING__), ABC(__METHOD__), ABC(__TYPE__), CUDA_DEVICE_NAME(), __THREADS_PER_BLOCK__);
    ts::printf_stream(ts::stream::stdout, "------------------------------------------------------\n");

    // Loop through all datasets
    for (ts::dataset_t& dataset : ts::all_datasets)
    {
        uint64_t dataset_len;
        current_errors  = -1;
        current_warnings = -1;
        ts::timing_results_t cur_timings = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int32_t dataset_bits = -1;

        // Skip datasets that are not included in the configuration
        if (!is_dataset_included(dataset)) continue;

        // Print the dataset_array name
        ts::printf_stream(ts::stream::stdout, "%14s: ", dataset.name.c_str());

        // Fill the dataset_array with the current dataset_array type
        dataset_len = fill_dataset<emu_function, native_function>(dataset_array, dataset);

        if(run_accuracy_for_dataset(dataset))
        {
            // Run the function and check the accuracy
            run_accuracy<emu_function, native_function>(dataset_array, dataset);

            // Analyze the results and print the final report
            analyze_results(dataset_array, dataset, current_errors, current_warnings, dataset_bits);
            // Accumulate the results
            total_errors   += current_errors;
            total_warnings += current_warnings;
        }

        // Print the results
        std::string result = ts::print_results(current_errors, current_warnings, dataset_len, dataset_bits);
        ts::printf_stream(ts::stream::stdout, "%-37s", result.c_str());
        fflush(stdout);

        // Run the function performance measurements, only for selected datasets
        if(run_timing_for_dataset(dataset))
        {
            // Run the function performance measurements
            cur_timings = run_timing<emu_function, native_function>(dataset_array, dataset);

            // Print results for the dataset_array 
            ts::printf_stream(ts::stream::stdout, 
                "| ev/sm/ck=%7.3lg / %-7.3lg (%5.2lgx), ck/ev=%5ld / %-5ld (%5.2lgx)",
                cur_timings.throughput_tst, cur_timings.throughput_ref, 
                cur_timings.throughput_tst / cur_timings.throughput_ref,
                (uint64_t)cur_timings.latency_tst, (uint64_t)cur_timings.latency_ref,    
                cur_timings.latency_ref / cur_timings.latency_tst);
            fflush(stdout);
        }
        else
        {
            ts::printf_stream(ts::stream::stdout, "|");
        }
        
        if (ts::config.logfile)
        {
            double throughput_ratio = (cur_timings.throughput_ref > 0.0) 
                ? cur_timings.throughput_tst / cur_timings.throughput_ref : 0.0;
            double latency_ratio = (cur_timings.latency_tst > 0.0) 
                ? cur_timings.latency_ref / cur_timings.latency_tst : 0.0;
            ts::printf_stream(ts::stream::file, 
                    "%s,%s,%s,%s,%s,%.3lg,%.3lg,%.3lg,%.3lg,%.3lg,%.3lg,%s,%s\n", 
                    ABC(__FUNC__), ABC(__ROUNDING__), ABC(__METHOD__), ABC(__TYPE__), dataset.name.c_str(), 
                    cur_timings.throughput_tst, cur_timings.throughput_ref, throughput_ratio,
                    cur_timings.latency_tst, cur_timings.latency_ref, latency_ratio,
                    result.c_str(), CUDA_DEVICE_NAME());
        }        
        
        ts::printf_stream(ts::stream::stdout, "\n");
        total_len += dataset.accuracy_len;
        total_bits = (dataset_bits >= 0)?std::max(total_bits, dataset_bits):total_bits;

    } // end of dataset loop

    // Print the final report
    ts::printf_stream(ts::stream::stdout, "======================================================\n");
    std::string total_result = ts::print_results(total_errors, total_warnings, total_len, total_bits);
    ts::printf_stream(ts::stream::stdout, "TOTAL: %s\n\n", total_result.c_str());

    // Close the log file
    if (ts::config.logfile) fclose(ts::config.logfile);

    return 0;
}

