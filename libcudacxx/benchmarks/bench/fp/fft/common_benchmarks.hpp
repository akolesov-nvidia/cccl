/*
    common_benchmarks.hpp - Common Definitions for FFT Benchmarks
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Common macros, type definitions, and utility functions shared across FFT benchmark files.
*/

#ifndef __COMMON_BENCHMARKS_HPP__
#define __COMMON_BENCHMARKS_HPP__

#include <cstddef>
#include <cstdint>
#include <cstring>

// Macro to concatenate two tokens
#define GLUE_HELPER(a,b) a##b
#define GLUE(a,b) GLUE_HELPER(a,b)

// Macro to convert a macro value to a string
#define STRINGIFY(x) #x
#define ABC(x) STRINGIFY(x)

#define double_id      (1)
#define efloat64_t_id  (2)
#define fpbits64_t_id  (3)
#define __TYPE_ID__ GLUE(__TYPE__,_id)


#ifndef __TARGET__
    #ifdef __CUDACC__
        #define __TARGET__      __device__
        #define __HOST_DEVICE__ __host__ __device__
    #else
        #define __TARGET__
        #define __HOST_DEVICE__
    #endif
#endif

// Custom implementation for C++17 and earlier
template<typename T, typename R>
__HOST_DEVICE__ T bit_cast(const R value) {
    T dst;
    // memcpy implementation
    std::memcpy(static_cast<void*>(&dst), static_cast<const void*>(&value), sizeof(T));
    return dst;
}

#endif // __COMMON_BENCHMARKS_HPP__
