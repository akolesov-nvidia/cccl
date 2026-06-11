/*
    ts_performance.hpp - FPMP Test Suite Performance Testing
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Performance benchmarking utilities for measuring throughput and latency of multi-precision
    floating-point operations on both CPU and GPU.
*/

#ifndef __TS_PERFORMANCE_HPP__
#define __TS_PERFORMANCE_HPP__

// ts_types.hpp includes ts.hpp and defines all common types
#include "ts_types.hpp"
#include "ts_functions.hpp"
#include <chrono>
#if !defined(__CUDACC__)
#include <omp.h>
#endif

// ============================================================================
// Configuration defaults for timing (shared between host and device)
// ============================================================================
#ifndef TS_TIMING_THREADS_PER_BLOCK
    #define TS_TIMING_THREADS_PER_BLOCK (256)
#endif

#ifndef TS_TIMING_NUM_BLOCKS
    #define TS_TIMING_NUM_BLOCKS (2048)
#endif

#ifndef TS_TIMING_REPS
    #define TS_TIMING_REPS (1024)
#endif

#ifndef TS_TIMING_UNROLL
    #define TS_TIMING_UNROLL (16)
#endif


#ifndef TS_TIMING_ITERATIONS
    #define TS_TIMING_ITERATIONS (8)
#endif

#ifndef TS_TIMING_HOST_WORK_ITEMS
    #define TS_TIMING_HOST_WORK_ITEMS (1024)  // Total parallel work items
#endif

// NOTE: The following types have been moved to ts_types.hpp:
//   - u32x2, u32x4, perf_mode, ts_timing_result_t, ts_timing_comparison_result_t

// ============================================================================
// Per-function performance input traits (SFINAE-based defaults)
// Functions can override by defining perf_start_val() / perf_stride() members.
// ============================================================================
template<typename T, typename = void>
struct has_perf_stride : std::false_type {};
template<typename T>
struct has_perf_stride<T, std::void_t<decltype(T::perf_stride())>> : std::true_type {};

template<typename FuncTag>
__HOST_DEVICE_DECL__ constexpr float get_perf_stride() {
    if constexpr (has_perf_stride<FuncTag>::value)
        return FuncTag::perf_stride();
    else
        return 0.0001f;
}

template<typename FuncTag>
__HOST_DEVICE_DECL__ constexpr float get_perf_start_val() {
    if constexpr (has_perf_stride<FuncTag>::value)
        return FuncTag::perf_start_val();
    else
        return 1.0f;
}

// ============================================================================
// Update bits function to prevent compiler optimization using chain dependency
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ void ts_update_bits(T a, T& r) 
{
    constexpr int sz = sizeof(T);
    if constexpr (sz == 4) 
    {
        uint32_t a_bits = ts::bit_cast<uint32_t>(a);
        uint32_t r_bits = ts::bit_cast<uint32_t>(r);
        r_bits = r_bits ^ (a_bits & 0x00000001u);
        r = ts::bit_cast<T>(r_bits);
    } 
    else if constexpr (sz == 8) 
    {
        u32x2 a_bits = ts::bit_cast<u32x2>(a);
        u32x2 r_bits = ts::bit_cast<u32x2>(r);
        r_bits.lo = r_bits.lo ^ (a_bits.lo & 0x00000001u);
        r_bits.hi = r_bits.hi ^ (a_bits.hi & 0x00000001u);
        r = ts::bit_cast<T>(r_bits);
    } 
    else if constexpr (sz == 16) 
    {
        u32x4 a_bits = ts::bit_cast<u32x4>(a);
        u32x4 r_bits = ts::bit_cast<u32x4>(r);
        r_bits.p0 = r_bits.p0 ^ (a_bits.p0 & 0x00000001u);
        r_bits.p1 = r_bits.p1 ^ (a_bits.p1 & 0x00000001u);
        r_bits.p2 = r_bits.p2 ^ (a_bits.p2 & 0x00000001u);
        r_bits.p3 = r_bits.p3 ^ (a_bits.p3 & 0x00000001u);
        r = ts::bit_cast<T>(r_bits);
    }
} // ts_update_bits

// (ts::bit_cast is used below for raw bit copy from result to input type,
//  bypassing conversion constructors in the performance feedback loop)

// ============================================================================
// Host-only functions (disabled when compiling with CUDA)
// ============================================================================
#if !defined(__CUDACC__)

// ============================================================================
// Host-side unified performance measurement kernel
// Mode: throughput = OpenMP parallel, latency = single-threaded sequential
// ============================================================================
#pragma GCC push_options
#pragma GCC optimize ("O3,unroll-loops")
template<perf_mode Mode, 
         typename FuncTag, 
         typename ArgType, 
         int Arity>
__attribute__((noinline)) void run_perf_host (float start_val, 
                                              int num_work_items, 
                                              volatile unsigned char* result) 
{
    // Use tag's input_type for arguments
    using input_t  = tag_input_t<FuncTag>;
    using result_t = tag_result_t<FuncTag>;
    
    auto kernel = [&](int tid) 
    {
        constexpr float stride = get_perf_stride<FuncTag>();
        input_t a = input_t(start_val + (float)tid * stride + 0.0002f);
        if constexpr (std::is_same_v<input_t, double>) {
            uint64_t bits = ts::bit_cast<uint64_t>(a);
            bits |= 0x1ULL;
            a = ts::bit_cast<double>(bits);
        }
        [[maybe_unused]] input_t b{}, c{}, d{};
        if constexpr (Arity >= 2) b = input_t(1.5f + (float)tid * 0.0007f);
        if constexpr (Arity >= 3) c = input_t(0.5f);
        if constexpr (Arity >= 4) d = input_t(0.25f);
        
        result_t r{};

        for (int i = 0; i < TS_TIMING_REPS; i++) 
        {
            r = impl_dispatch<FuncTag, Arity>::call(a, b, c, d);
            if constexpr (std::is_same_v<result_t, input_t>) 
            {
                ts_update_bits(r, a);
            } 
            else 
            {
                input_t tmp = ts::bit_cast<input_t>(r);
                ts_update_bits(tmp, a);
            }
        }
        // Prevent optimization
        unsigned char tmp = result[tid % 64];
        tmp += *((unsigned char*)&r);
        result[tid % 64] = tmp;
    };

    if constexpr (Mode == perf_mode::throughput) 
    {
        #pragma omp parallel for schedule(static)
        for (int tid = 0; tid < num_work_items; tid++) 
        {
            kernel(tid);
        }
    } 
    else 
    {
        // Latency mode: single iteration, tid = 0
        kernel(0);
    }
} // run_perf_host
#pragma GCC pop_options

// ============================================================================
// Unified measure function for host (throughput or latency)
// ============================================================================
template<perf_mode Mode, 
         typename FuncTag, 
         typename ArgType>
ts_timing_result_t measure_host( float start_val    = 1.0f,
                                 int num_work_items = TS_TIMING_HOST_WORK_ITEMS,
                                 int iterations     = TS_TIMING_ITERATIONS,
                                 int clock_rate_khz = 0 /* 0 = auto-detect */)
{
    constexpr int arity = detect_arity<FuncTag>();
    static_assert(arity >= 1 && arity <= 4, "Function arity must be between 1 and 4");

    volatile unsigned char result[64] = {0};
    
    // For latency mode, only 1 work item
    const int actual_work_items = (Mode == perf_mode::latency) ? 1 : num_work_items;

    // Warmup
    run_perf_host<Mode, FuncTag, ArgType, arity>(start_val, actual_work_items, result);

    // Timed runs
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) 
    {
        run_perf_host<Mode, FuncTag, ArgType, arity>(start_val, actual_work_items, result);
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    double milliseconds = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (clock_rate_khz == 0) clock_rate_khz = (int)get_clock_rate();

    ts_timing_result_t res;
    res.time_ms = milliseconds / iterations;
    res.total_ops = (long long)actual_work_items * TS_TIMING_REPS;
    res.gflops = res.total_ops / (res.time_ms * 1e6);
    
    if (clock_rate_khz > 0) 
    {
        double total_clocks  = res.time_ms * clock_rate_khz;
        res.evals_per_clk_sm = res.total_ops / total_clocks;
        res.clocks_per_eval  = total_clocks / res.total_ops;
    } 
    else 
    {
        res.evals_per_clk_sm = 0;
        res.clocks_per_eval  = 0;
    }
    
    if constexpr (Mode == perf_mode::throughput) 
    {
        int omp_threads       = omp_get_max_threads();
        res.num_blocks        = actual_work_items;
        res.threads_per_block = omp_threads;
        res.sm_count          = omp_threads;
    } 
    else 
    {
        res.num_blocks        = 1;
        res.threads_per_block = 1;
        res.sm_count          = 1;
    }

    res.reps = TS_TIMING_REPS;
    res.iterations = iterations;
    res.clock_rate_khz = clock_rate_khz;

    return res;
}

// Convenience wrappers for backward compatibility
template<typename FuncTag, 
         typename ArgType>
ts_timing_result_t measure_throughput_host (float start_val    = 1.0f, 
                                            int num_work_items = TS_TIMING_HOST_WORK_ITEMS, 
                                            int iterations     = TS_TIMING_ITERATIONS, 
                                            int clock_rate_khz = 0) 
{
    return measure_host<perf_mode::throughput, FuncTag, ArgType> (start_val, 
                                                                  num_work_items, 
                                                                  iterations, 
                                                                  clock_rate_khz);
}

template<typename FuncTag, typename ArgType>
ts_timing_result_t measure_latency_host (float start_val    = 1.0f, 
                                         int iterations     = TS_TIMING_ITERATIONS, 
                                         int clock_rate_khz = 0) 
{
    return measure_host<perf_mode::latency, FuncTag, ArgType> (start_val, 
                                                               1, 
                                                               iterations, 
                                                               clock_rate_khz);
}

// ============================================================================
// Unified compare function for host (throughput or latency)
// ============================================================================
template<perf_mode Mode, 
         typename TestFuncTag, 
         typename TestType, 
         typename RefFuncTag, 
         typename RefType>
ts_timing_comparison_result_t compare_host (float start_val    = 1.0f,
                                            int num_work_items = TS_TIMING_HOST_WORK_ITEMS,
                                            int iterations     = TS_TIMING_ITERATIONS)
{
    // Read clock rate once to ensure consistent metrics between test and ref
    int clock_rate_khz = (int)get_clock_rate();
    
    ts_timing_comparison_result_t result;
    result.test_result = measure_host<Mode, TestFuncTag, TestType>( start_val, num_work_items, iterations, clock_rate_khz);
    result.ref_result  = measure_host<Mode, RefFuncTag, RefType>  ( start_val, num_work_items, iterations, clock_rate_khz);
    
    if constexpr (Mode == perf_mode::throughput) 
    {
        result.speedup = result.ref_result.time_ms / result.test_result.time_ms;
    } 
    else
    {
        result.speedup = result.ref_result.clocks_per_eval / result.test_result.clocks_per_eval;
    }

    return result;
}

// Convenience wrappers for backward compatibility
template<typename TestFuncTag, 
         typename TestType, 
         typename RefFuncTag, 
         typename RefType>
ts_timing_comparison_result_t compare_throughput_host (float start_val    = 1.0f, 
                                                       int num_work_items = TS_TIMING_HOST_WORK_ITEMS, 
                                                       int iterations     = TS_TIMING_ITERATIONS) 
{
    return compare_host<perf_mode::throughput, TestFuncTag, TestType, RefFuncTag, RefType> (start_val, 
                                                                                            num_work_items, 
                                                                                            iterations);
}

template<typename TestFuncTag, 
         typename TestType, 
         typename RefFuncTag, 
         typename RefType>
ts_timing_comparison_result_t compare_latency_host (float start_val = 1.0f, 
                                                    int iterations  = TS_TIMING_ITERATIONS) 
{
    return compare_host<perf_mode::latency, TestFuncTag, TestType, RefFuncTag, RefType> (start_val, 
                                                                                         1, 
                                                                                         iterations);
}

#endif // !__CUDACC__ (host-only functions)


// ============================================================================
// CUDA-specific performance measurement
// ============================================================================
#if defined(__CUDACC__)

// ============================================================================
// CUDA unified performance kernel
// Mode: throughput = parallel threads, latency = single thread
// ============================================================================
template<perf_mode Mode, 
         typename FuncTag, 
         typename ArgType, 
         int Arity>
__global__ void run_perf_cuda (float start_val, 
                               unsigned char* result, 
                               uint32_t never = 0) 
{
    static_assert(Arity >= 1 && Arity <= 4, "Arity must be between 1 and 4");
    
    // Use tag's input_type for arguments
    using input_t  = tag_input_t<FuncTag>;
    using result_t = tag_result_t<FuncTag>;
    
    int tid = (Mode == perf_mode::throughput) ? (blockIdx.x * blockDim.x + threadIdx.x) : 0;
    constexpr float stride = get_perf_stride<FuncTag>();
    input_t a = input_t(start_val + (float)tid * stride + 0.0002f);
    if constexpr (std::is_same_v<input_t, double>) {
        uint64_t bits = ts::bit_cast<uint64_t>(a);
        bits |= 0x1ULL;
        a = ts::bit_cast<double>(bits);
    }

    [[maybe_unused]] input_t b{}, c{}, d{};

    if constexpr (Arity >= 2) b = input_t(1.5f + (float)tid * 0.0007f);
    if constexpr (Arity >= 3) c = input_t(0.5f);
    if constexpr (Arity >= 4) d = input_t(0.25f);
    
    result_t r{};

    if constexpr (Mode == perf_mode::throughput) 
    {
        constexpr int u = TS_TIMING_UNROLL;
        #pragma unroll u
        for (int i = 0; i < TS_TIMING_REPS; i++) 
        {
            r = impl_dispatch<FuncTag, Arity>::call(a, b, c, d);
            if constexpr (std::is_same_v<result_t, input_t>) 
            {
                ts_update_bits(r, a);
            } 
            else 
            {
                input_t tmp = ts::bit_cast<input_t>(r);
                ts_update_bits(tmp, a);
            }
        }
    } 
    else 
    {
        for (int i = 0; i < TS_TIMING_REPS; i++) 
        {
            r = impl_dispatch<FuncTag, Arity>::call(a, b, c, d);
            if constexpr (std::is_same_v<result_t, input_t>) 
            {
                ts_update_bits(r, a);
            } 
            else 
            {
                input_t tmp = ts::bit_cast<input_t>(r);
                ts_update_bits(tmp, a);
            }
        }
    }

    if (never) 
    {
        unsigned char sum = 0;
        unsigned char* ptr = (unsigned char*)&r;
        for (int b = 0; b < sizeof(r); b++) sum += ptr[b];
        result[tid] = sum;
    }
} // run_perf_cuda

// ============================================================================
// CUDA unified kernel launcher
// ============================================================================
template<perf_mode Mode, 
         typename FuncTag, 
         typename ArgType, 
         int Arity>
struct perf_kernel_launcher_cuda 
{
    static void launch (int num_blocks, 
                        int threads_per_block, 
                        float start_val, 
                        unsigned char* result, 
                        uint32_t never) 
    {
        if constexpr (Mode == perf_mode::throughput) 
        {
            run_perf_cuda<Mode, FuncTag, ArgType, Arity><<<num_blocks, threads_per_block>>>(start_val, result, never);
        } 
        else 
        {
            run_perf_cuda<Mode, FuncTag, ArgType, Arity><<<1, 1>>>(start_val, result, never);
        }
    }
};

// ============================================================================
// Unified measure function for CUDA (throughput or latency)
// ============================================================================
template<perf_mode Mode, 
         typename FuncTag, 
         typename ArgType>
ts_timing_result_t measure_cuda( float start_val       = 1.0f,
                                 int num_blocks        = TS_TIMING_NUM_BLOCKS,
                                 int threads_per_block = TS_TIMING_THREADS_PER_BLOCK,
                                 int iterations        = TS_TIMING_ITERATIONS)
{
    constexpr int arity = detect_arity<FuncTag>();
    static_assert(arity >= 1 && arity <= 4, "Function arity must be between 1 and 4");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int sm_count = props.multiProcessorCount;
    int clock_rate_khz;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device);

    // For latency mode, only 1 thread
    const int actual_blocks = (Mode == perf_mode::latency) ? 1 : num_blocks;
    const int actual_threads = (Mode == perf_mode::latency) ? 1 : threads_per_block;

    unsigned char* d_result = nullptr;
    cudaMalloc(&d_result, actual_blocks * actual_threads);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    perf_kernel_launcher_cuda<Mode, FuncTag, ArgType, arity>::launch (actual_blocks, 
                                                                      actual_threads, 
                                                                      start_val, 
                                                                      d_result, 
                                                                      0);
    cudaDeviceSynchronize();

    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) 
    {
        perf_kernel_launcher_cuda<Mode, FuncTag, ArgType, arity>::launch (actual_blocks, 
                                                                          actual_threads, 
                                                                          start_val, 
                                                                          d_result, 
                                                                          0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_result);

    ts_timing_result_t result;
    result.time_ms = milliseconds / iterations;
    result.total_ops = (long long)actual_blocks * actual_threads * TS_TIMING_REPS;
    result.gflops = result.total_ops / (result.time_ms * 1e6);
    
    double total_clocks = result.time_ms * clock_rate_khz;
    if constexpr (Mode == perf_mode::throughput) 
    {
        result.evals_per_clk_sm = result.total_ops / (total_clocks * sm_count);
    } 
    else 
    {
        result.evals_per_clk_sm = result.total_ops / total_clocks;
    }

    result.clocks_per_eval = total_clocks / result.total_ops;
    result.num_blocks = actual_blocks;
    result.threads_per_block = actual_threads;
    result.reps = TS_TIMING_REPS;
    result.iterations = iterations;
    result.sm_count = sm_count;
    result.clock_rate_khz = clock_rate_khz;

    return result;
}

// Convenience wrappers for backward compatibility
template<typename FuncTag, 
         typename ArgType>
ts_timing_result_t measure_throughput_cuda (float start_val       = 1.0f, 
                                            int num_blocks        = TS_TIMING_NUM_BLOCKS,
                                            int threads_per_block = TS_TIMING_THREADS_PER_BLOCK, 
                                            int iterations        = TS_TIMING_ITERATIONS) 
{
    return measure_cuda<perf_mode::throughput, FuncTag, ArgType> (start_val, 
                                                                  num_blocks, 
                                                                  threads_per_block, 
                                                                  iterations);
}

template<typename FuncTag, 
         typename ArgType>
ts_timing_result_t measure_latency_cuda (float start_val = 1.0f, 
                                         int iterations  = TS_TIMING_ITERATIONS) 
{
    return measure_cuda<perf_mode::latency, FuncTag, ArgType> (start_val, 
                                                               1, 
                                                               1, 
                                                               iterations);
}

// ============================================================================
// Unified compare function for CUDA (throughput or latency)
// ============================================================================
template<perf_mode Mode, 
         typename TestFuncTag, 
         typename TestType, 
         typename RefFuncTag, 
         typename RefType>
ts_timing_comparison_result_t compare_cuda (float start_val       = 1.0f,
                                            int num_blocks        = TS_TIMING_NUM_BLOCKS,
                                            int threads_per_block = TS_TIMING_THREADS_PER_BLOCK,
                                            int iterations        = TS_TIMING_ITERATIONS)
{
    ts_timing_comparison_result_t result;
    result.test_result = measure_cuda<Mode, TestFuncTag, TestType>( start_val, num_blocks, threads_per_block, iterations);
    result.ref_result  = measure_cuda<Mode, RefFuncTag, RefType>  ( start_val, num_blocks, threads_per_block, iterations);
    
    if constexpr (Mode == perf_mode::throughput) 
    {
        result.speedup = result.ref_result.time_ms / result.test_result.time_ms;
    } 
    else 
    {
        result.speedup = result.ref_result.clocks_per_eval / result.test_result.clocks_per_eval;
    }
    return result;
}

// Convenience wrappers for backward compatibility
template<typename TestFuncTag, 
         typename TestType, 
         typename RefFuncTag, 
         typename RefType>
ts_timing_comparison_result_t compare_throughput_cuda (float start_val       = 1.0f, 
                                                       int num_blocks        = TS_TIMING_NUM_BLOCKS,
                                                       int threads_per_block = TS_TIMING_THREADS_PER_BLOCK, 
                                                       int iterations        = TS_TIMING_ITERATIONS) 
{
    return compare_cuda<perf_mode::throughput, TestFuncTag, TestType, RefFuncTag, RefType> (start_val, 
                                                                                            num_blocks, 
                                                                                            threads_per_block, 
                                                                                            iterations);
}

template<typename TestFuncTag, 
         typename TestType, 
         typename RefFuncTag, 
         typename RefType>
ts_timing_comparison_result_t compare_latency_cuda (float start_val = 1.0f, 
                                                    int iterations  = TS_TIMING_ITERATIONS) 
{
    return compare_cuda<perf_mode::latency, TestFuncTag, TestType, RefFuncTag, RefType> (start_val, 
                                                                                         1, 
                                                                                         1, 
                                                                                         iterations);
}

#endif // __CUDACC__

#if !defined(__CUDACC__)
    template<typename TestFuncTag, 
             typename TestType, 
             typename RefFuncTag, 
             typename RefType> 
    ts_timing_comparison_result_t compare_throughput(){ 
        constexpr float sv = get_perf_start_val<TestFuncTag>();
        return compare_throughput_host<TestFuncTag, TestType, RefFuncTag, RefType>(sv); } 
    template<typename TestFuncTag, 
             typename TestType, 
             typename RefFuncTag, 
             typename RefType> 
    ts_timing_comparison_result_t compare_latency()   { 
        constexpr float sv = get_perf_start_val<TestFuncTag>();
        return compare_latency_host<TestFuncTag, TestType, RefFuncTag, RefType>(sv); }
#else
    template<typename TestFuncTag, 
             typename TestType, 
             typename RefFuncTag, 
             typename RefType> 
    ts_timing_comparison_result_t compare_throughput() { 
        constexpr float sv = get_perf_start_val<TestFuncTag>();
        return compare_throughput_cuda<TestFuncTag, TestType, RefFuncTag, RefType>(sv); } 
    template<typename TestFuncTag, 
             typename TestType, 
             typename RefFuncTag, 
             typename RefType> 
    ts_timing_comparison_result_t compare_latency()    { 
        constexpr float sv = get_perf_start_val<TestFuncTag>();
        return compare_latency_cuda<TestFuncTag, TestType, RefFuncTag, RefType>(sv); }  
#endif // __CUDACC__

#endif // __TS_PERFORMANCE_HPP__ 
