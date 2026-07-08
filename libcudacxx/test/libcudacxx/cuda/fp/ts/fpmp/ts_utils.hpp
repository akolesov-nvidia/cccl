/*
    ts_utils.hpp - FPMP Test Suite Utility Functions
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Utility functions and macros for CUDA error checking and common operations
    used throughout the test suite.
*/

#ifndef __TS_UTILS_HPP__
#define __TS_UTILS_HPP__

#if defined __CUDACC__ // CUDA only

    // Macro to catch CUDA errors in CUDA runtime calls
    #define CSC(call)                                                     \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (cudaSuccess != err) {                                         \
            fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err) );        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

    // Class to measure the time of a function using CUDA events
    class Timer 
    {
        public:
        Timer();
        virtual ~Timer();
        void start();
        float stop();
        private:
        cudaEvent_t event_start, event_stop;
    };

    Timer::Timer() 
    {
        CSC(cudaEventCreate(&event_start));
        CSC(cudaEventCreate(&event_stop));
    }

    void Timer::start() 
    {
        CSC(cudaEventRecord(event_start, 0));
    }

    float Timer::stop() 
    {
        float time;
        CSC(cudaEventRecord(event_stop, 0));
        CSC(cudaEventSynchronize(event_stop));
        CSC(cudaEventElapsedTime(&time, event_start, event_stop));
        return time;
    }

    Timer::~Timer() 
    {
        CSC(cudaEventDestroy(event_start));
        CSC(cudaEventDestroy(event_stop));
    }

    #ifndef DEFAULT_DEV
        #define DEFAULT_DEV (0)
    #endif

    #define CUDA_INIT() \
        CSC (cudaFree (0)); \
        int cuda_dev = DEFAULT_DEV; \
        struct cudaDeviceProp cuda_props; \
        CSC (cudaSetDevice (cuda_dev)); \
        CSC (cudaGetDeviceProperties (&cuda_props, cuda_dev));
    #define CUDA_DEVICE_NAME() cuda_props.name

#else // host only

    #define CSC(call) 
    #define CUDA_INIT()
    #define CUDA_DEVICE_NAME() "host"

#endif // __CUDACC__

// Function to get the current CPU clock rate in kHz (available for both CUDA and host)
static inline double get_clock_rate()
{
    FILE* fp = fopen("/proc/cpuinfo", "r");
    if (!fp) return 0.0;
    
    char line[256];
    double clock_rate = 0.0;
    
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "cpu MHz", 7) == 0) {
            char* freq = strchr(line, ':');
            if (freq) {
                clock_rate = atof(freq + 1) * 1000.0; // Convert MHz to kHz
                break;
            }
        }
    }
    
    fclose(fp);
    return clock_rate;
}

// Multi-precision types come from <cuda/fpmp> (cuda::experimental, brought into
// scope in ts.hpp), which is always included before this header:
// fpmp2<FpType, fpmp2_accuracy>.

namespace ts
{
    // Custom bit_cast implementation for C++17 and earlier
    // Handles different sizes: truncates if dst < src, duplicates if dst > src
    template<typename T, typename R>
    __HOST_DEVICE_DECL__ T bit_cast(const R value) 
    {
        T dst;
        constexpr size_t dst_sz = sizeof(T);
        constexpr size_t src_sz = sizeof(R);
        
        if constexpr (dst_sz == src_sz) {
            // Same size: direct copy
            std::memcpy(static_cast<void*>(&dst), static_cast<const void*>(&value), dst_sz);
        } else if constexpr (dst_sz < src_sz) {
            // Destination smaller: copy first portion of source
            std::memcpy(static_cast<void*>(&dst), static_cast<const void*>(&value), dst_sz);
        } else {
            // Destination larger: duplicate source pattern to fill destination
            unsigned char* dst_ptr = reinterpret_cast<unsigned char*>(&dst);
            const unsigned char* src_ptr = reinterpret_cast<const unsigned char*>(&value);
            for (size_t i = 0; i < dst_sz; i++) {
                dst_ptr[i] = src_ptr[i % src_sz];
            }
        }
        return dst;
    }

    // Type trait to extract base type from multi-precision types
    template<typename T> struct fpmp2_base_type { using type = T; };
    template<typename FpType, cuda::experimental::fpmp2_accuracy met>
    struct fpmp2_base_type<cuda::experimental::fpmp2<FpType, met>> { using type = FpType; };
    template<typename T> using fpmp2_base_type_t = typename fpmp2_base_type<T>::type;

    // Type trait to extract method from multi-precision types.
    // Defaults to cuda::experimental::fpmp2_accuracy::def for scalar / non-fpmp2 types.
    template<typename T> struct fpmp2_method { static constexpr cuda::experimental::fpmp2_accuracy value = cuda::experimental::fpmp2_accuracy::def; };
    template<typename FpType, cuda::experimental::fpmp2_accuracy met>
    struct fpmp2_method<cuda::experimental::fpmp2<FpType, met>> { static constexpr cuda::experimental::fpmp2_accuracy value = met; };
    template<typename T> inline constexpr cuda::experimental::fpmp2_accuracy fpmp2_method_v = fpmp2_method<T>::value;

// ============================================================================
// Get maximum mantissa bits for a type
// For fpmp2 types: base_mantissa * 2 (float-float=48, double-double=106)
// For scalar types: standard mantissa sizes
// ============================================================================
template<typename T>
constexpr int get_max_mantissa_bits()
{
    // Handle integer types first
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
        return 8;
    } else if constexpr (std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t>) {
        return 16;
    } else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
        return 32;
    } else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        return 64;
    } else if constexpr (std::is_same_v<T, bool>) {
        return 1;
    }
    // Handle scalar floating-point types before mp2 base-type checks
    // (fpmp2_base_type_t<double> == double, which would incorrectly match the mp2 branch)
    if constexpr (std::is_same_v<T, __ts_fp128>) {
        return 113; // quad precision
    } else if constexpr (std::is_same_v<T, double>) {
        return 53;  // double
    } else if constexpr (std::is_same_v<T, float>) {
        return 24;  // float
    }
    // Handle multi-precision types
    using base_t = fpmp2_base_type_t<T>;
    if constexpr (std::is_same_v<base_t, float>) {
        return 48;  // float-float: 24 + 24 = 48 bits
    } else if constexpr (std::is_same_v<base_t, double>) {
        return 106; // double-double: 53 + 53 = 106 bits
    } else {
        return 64;  // default
    }
} // get_max_mantissa_bits

// ============================================================================
// Check if a value is finite (handles __ts_fp128 and multi-precision types)
// ============================================================================
template<typename T>
__HOST_DEVICE_DECL__ inline bool is_value_finite(T val)
{
    double d = static_cast<double>(val);
    #if defined(__CUDA_ARCH__)
        return isfinite(d);
    #else
        return std::isfinite(d);
    #endif
}

// ============================================================================
// CUDA atomic operations for accuracy measurement
// ============================================================================
#if defined(__CUDACC__)

// Atomic max for double using CAS
__device__ inline void atomicMaxDouble(double* addr, double val)
{
    unsigned long long* addr_as_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_as_ull;
    unsigned long long assumed;
    
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (old_val >= val) break;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// Atomic add for double (built-in for CC 6.0+, emulated otherwise)
__device__ inline void atomicAddDouble(double* addr, double val)
{
#if __CUDA_ARCH__ >= 600
    atomicAdd(addr, val);
#else
    unsigned long long* addr_as_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_as_ull;
    unsigned long long assumed;
    
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, 
                        __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
#endif
}

#endif // defined(__CUDACC__)

// ============================================================================
// Type trait to detect multi-precision types (types with hi() and lo() methods)
// ============================================================================
template<typename T, typename = void>
struct is_multiprecision : std::false_type {};

template<typename T>
struct is_multiprecision<T, std::void_t<decltype(std::declval<T>().hi()), decltype(std::declval<T>().lo())>> 
    : std::true_type {};

template<typename T>
inline constexpr bool is_multiprecision_v = is_multiprecision<T>::value;

// Get the component type of a multi-precision type
template<typename T, typename = void>
struct mp_component_type { using type = T; };  // Default: type itself for scalars

template<typename T>
struct mp_component_type<T, std::enable_if_t<is_multiprecision_v<T>>> {
    using type = decltype(std::declval<T>().hi());
};

template<typename T>
using mp_component_t = typename mp_component_type<T>::type;

} // end of namespace ts

// ============================================================================
// Convert reference result to fpmp_type for comparison/display
// Works for both:
//   - fp32mp2 (FpmpType=__fpmp2_t<float,...>, RefType=double)
//   - fp64mp2 (FpmpType=__fpmp2_t<double,...>, RefType=__ts_fp128)
// ============================================================================
template<typename FpmpType, typename RefType>
__HOST_DEVICE_DECL__ inline FpmpType ts_ref_to_fpmp(RefType ref)
{
    using FpType = decltype(FpmpType().hi());
    FpType hi = static_cast<FpType>(ref);
    FpType lo = static_cast<FpType>(ref - static_cast<RefType>(hi));
    return FpmpType(hi, lo);
}

// ============================================================================
// Compute number of correct bits from relative error
// max_bits: maximum measurable bits (depends on reference type precision)
//           - 53 for double reference
//           - 113 for __ts_fp128 reference
// ============================================================================
__HOST_DEVICE_DECL__ inline int compute_correct_bits(double rel_err, int max_bits = 113)
{
    // Handle zero error: return max measurable bits
    if (rel_err == 0.0) return max_bits;
    if (rel_err < 0.0) rel_err = -rel_err;
    return TS_MAX(0, TS_MIN(max_bits, (int)(-TS_LOG2(rel_err))));
}

#endif // __TS_UTILS_HPP__
