#ifndef __TS_DEBUG_HPP__
#define __TS_DEBUG_HPP__
/*
    ts_debug.hpp - Debug print utilities for multi-precision values
    ========================================================================
    Provides debug print macros for quick inspection of floating-point
    values during development/debugging. Output uses printf (stdout) with
    both decimal and C99 hexadecimal (%a) format.

    Multi-precision:
        PRINTMP32(x)  - fp32mp2 value (hi/lo as native float)
        PRINTMP64(x)  - fp64mp2 value (hi/lo as native double)
    Scalar:
        PRINTFP32(x)  - single float value
        PRINTFP64(x)  - single double value

    Debug workflow:
        1. #include "ts_debug.hpp" in the file you're debugging
        2. Add PRINTFP32/PRINTMP32/etc. calls at points of interest
        3. Run with fixed inputs: make tsmp FUNC=add A1=1:0 A2=2:0
           (A1/A2 enables fixed-input mode: 1 sample, no special table)
        4. Remove includes and print calls when done

    Safe to use inside __host__ __device__ functions.
    Device printf supports %a since compute capability 2.0.
*/

#if defined(__FIXED_INPUTS__)

    #include <cstdio>

    // ============================================================================
    // Helper functions using printf only (works on both host and device).
    // ============================================================================

    #ifdef __CUDACC__
    #define __FPMP_DBG_QUAL__ __host__ __device__
    #else
    #define __FPMP_DBG_QUAL__
    #endif

    __FPMP_DBG_QUAL__ inline void _fpmp_dbg_fp32(const char* name, float v) {
        printf("\t\t[FP32] %s: %+.9e (%a)\n", name, v, (double)v);
    }

    __FPMP_DBG_QUAL__ inline void _fpmp_dbg_fp64(const char* name, double v) {
        printf("\t\t[FP64] %s: %+.18e (%a)\n", name, v, v);
    }

    __FPMP_DBG_QUAL__ inline void _fpmp_dbg_mp32(const char* name, float hi, float lo) {
        printf("\t\t[MP32] %s: hi = %+.9e (%a), lo = %+.9e (%a)\n",
                name, hi, (double)hi, lo, (double)lo);
    }

    __FPMP_DBG_QUAL__ inline void _fpmp_dbg_mp64(const char* name, double hi, double lo) {
        printf("\t\t[MP64] %s: hi = %+.18e (%a), lo = %+.18e (%a)\n",
                name, hi, hi, lo, lo);
    }

    // Print a single float value
    #define PRINTFP32(x) _fpmp_dbg_fp32(#x, (x))

    // Print a single double value
    #define PRINTFP64(x) _fpmp_dbg_fp64(#x, (x))

    // Print fp32mp2 value: float hi/lo without conversion to double
    #define PRINTMP32(x) _fpmp_dbg_mp32(#x, (x).hi(), (x).lo())

    // Print fp64mp2 value: double hi/lo natively
    #define PRINTMP64(x) _fpmp_dbg_mp64(#x, (x).hi(), (x).lo())

#else

    #define PRINTFP32(x)
    #define PRINTFP64(x)
    #define PRINTMP32(x)
    #define PRINTMP64(x)

#endif

#endif // __TS_DEBUG_HPP__
