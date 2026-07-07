#ifndef __TS_HPP__
#define __TS_HPP__

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <random>
#include <iomanip>

    // Macro to concatenate tokens
    #if !defined __FPEMU_GLUE_IMPL__
    #define __FPEMU_GLUE_IMPL__
    #define GLUE_HELPER(a,b) a##b
    #define GLUE(a,b) GLUE_HELPER(a,b)
    #define GLUE2(a,b) GLUE(a,b)
    #define GLUE3(a,b,c) GLUE(GLUE2(a,b),c)
    #define GLUE4(a,b,c,d) GLUE(GLUE3(a,b,c),d)
    #define GLUE5(a,b,c,d,e) GLUE(GLUE4(a,b,c,d),e)
    #define GLUE6(a,b,c,d,e,f) GLUE(GLUE5(a,b,c,d,e),f)
    #define GLUE7(a,b,c,d,e,f,g) GLUE(GLUE6(a,b,c,d,e,f),g)
    #define GLUE8(a,b,c,d,e,f,g,h) GLUE(GLUE7(a,b,c,d,e,f,g),h)
    #endif // __FPEMU_GLUE_IMPL__

    // Macro to convert a macro value to a string
    #define STRINGIFY(x) #x
    #define ABC(x) STRINGIFY(x)

    #define fp64emu_t_id           (0)
    #define fp64emu_unpacked_t_id  (1)
    #define EMU_TYPE GLUE2(__TYPE__,_id)

    #include <cuda/fpemu>

    // The CCCL FP SDK puts these directly in cuda::experimental:
    // fp64emu_t<fp64emu_accuracy>, fp64emu_unpacked_t<...>, the fpbits64*
    // layouts and the __fp64emu_* builtins. Pull the namespace in so the
    // existing (now unqualified) references resolve.
    using namespace cuda::experimental;

    #define FUNC_NAME        GLUE2(__FUNC__,_function)
    #define FUNC_IMPL        GLUE2(__FUNC__,_device_impl)
    #define FIXED_NAME       GLUE2(__FUNC__,_fixed)

    // Helper macro to convert METHOD token to enum value
    // We can't use token pasting with ::, so we use a helper macro that expands the token
    // This uses token pasting to select the right helper macro based on __METHOD__ value
    // We need an extra level of indirection to force __METHOD__ to expand before pasting
    #define METHOD_ENUM_HELPER_high fp64emu_accuracy::high
    #define METHOD_ENUM_HELPER_mid  fp64emu_accuracy::mid
    #define METHOD_ENUM_HELPER_low  fp64emu_accuracy::low
    #define METHOD_ENUM_HELPER_def  fp64emu_accuracy::def
    #define METHOD_ENUM_HELPER_INDIRECT(m) METHOD_ENUM_HELPER_##m
    #define METHOD_ENUM_HELPER(m) METHOD_ENUM_HELPER_INDIRECT(m)
    #define METHOD_ENUM METHOD_ENUM_HELPER(__METHOD__)

    #if EMU_TYPE == fp64emu_unpacked_t_id
        #define ARGTYPE_EMU fp64emu_unpacked_t<METHOD_ENUM>
        #define RESTYPE_EMU fp64emu_unpacked_t<METHOD_ENUM>
        #define ARGTYPE_NATIVE double
        #define RESTYPE_NATIVE double
    #else // fp64emu_t_id
        #define ARGTYPE_EMU fp64emu_t<METHOD_ENUM>
        #define RESTYPE_EMU fp64emu_t<METHOD_ENUM>
        #define ARGTYPE_NATIVE double
        #define RESTYPE_NATIVE double
    #endif

    #define add_args   2
    #define dadd_args  2
    #define mul_args   2
    #define dmul_args  2
    #define sub_args   2
    #define dsub_args  2
    #define div_args   2
    #define ddiv_args  2
    #define fma_args   3
    #define mad_args   3
    #define dfma_args  3
    #define dot_args   4
    #define cmul_args  4
    #define rsqrt_args 1
    #define sqrt_args  1
    #define dsqrt_args 1
    #define FUNC_ARGS  GLUE2(__FUNC__,_args)

    #if defined __CUDACC__ // CUDA only
        // Number of threads per block
        #ifndef __THREADS_PER_BLOCK__
        #define __THREADS_PER_BLOCK__  (512)
        #endif
        #define __HOST_DEVICE_DECL__  __host__ __device__
        #define __GLOBAL_DECL__       __global__ __launch_bounds__(__THREADS_PER_BLOCK__)
        #define __DEVICE_DECL__       __device__
        #define __HOST_DECL__         __host__
        #define __INLINE__            __always_inline
        #define __USE_CUDA_BUILTINS__
    #else // host only
        // Number of threads per block
        #ifndef __THREADS_PER_BLOCK__
        #define __THREADS_PER_BLOCK__  (1)
        #endif
        #define __HOST_DEVICE_DECL__
        #define __GLOBAL_DECL__
        #define __DEVICE_DECL__
        #define __HOST_DECL__
        #define __INLINE__ inline
        #undef  __USE_CUDA_BUILTINS__
    #endif // __CUDACC__

    #ifndef __FUNC__
        #define __FUNC__     fma
    #endif
    #ifndef __ROUNDING__
        #define __ROUNDING__   rn
    #endif
    // METHOD is the only external parameter - ACC and RANGE are internal
    #ifndef __METHOD__
        // Default to def if METHOD not specified
        #define __METHOD__ def
    #endif
    #ifndef __MEAN__
        #define __MEAN__       0.0
    #endif
    #ifndef __STDDEV__
        #define __STDDEV__     1.0
    #endif
    #ifndef __SEED__
        #define __SEED__       123456789
    #endif

    #ifndef __ACCURACY_LEN__
            #define __ACCURACY_LEN__    10000000
    #endif

    #ifndef __HTRLEN__
            #define __HTR_LEN__    1024
    #endif

    #ifdef __CUDACC__
        #ifndef __LATENCY_LEN__
                #define __LATENCY_LEN__ 512
        #endif
        #ifndef __THROUGHPUT_LEN__
                #define __THROUGHPUT_LEN__ (64*1024)
        #endif
        #define __ITERATIONS__ 16
        #if defined __FUNC_REPEATS__
            #define __REPEATS__    __FUNC_REPEATS__
        #else
            #define __REPEATS__    4096
        #endif
    #else
        #ifndef __LATENCY_LEN__
                #define __LATENCY_LEN__ 1024
        #endif
        #ifndef __THROUGHPUT_LEN__
                #define __THROUGHPUT_LEN__ (128*1024)
        #endif
        #define __ITERATIONS__ 8
        #if defined __FUNC_REPEATS__
            #define __REPEATS__    __FUNC_REPEATS__
        #else
            #define __REPEATS__    512
        #endif
    #endif

    #ifndef __MAX_LEN__
        #define __MAX_LEN1__ (__ACCURACY_LEN__ > __LATENCY_LEN__ ? __ACCURACY_LEN__ : __LATENCY_LEN__)
        #define __MAX_LEN__  (__MAX_LEN1__ > __THROUGHPUT_LEN__ ? __MAX_LEN1__ : __THROUGHPUT_LEN__)
    #endif

    #ifndef __NAN_PAYLOAD__
        #define  __NAN_PAYLOAD__ 0
    #else
        #define  __NAN_PAYLOAD__ 1
    #endif

    #ifndef __ZERO_SIGN__
        #define  __ZERO_SIGN__ 1
    #else
        #define  __ZERO_SIGN__ 0
    #endif

    #ifndef __PRINT_LIMIT__
        #define __PRINT_LIMIT__ 16
    #endif

    #if (!(defined __PRINT_FAIL__)) && \
        (!(defined __PRINT_OK__)) && \
        (!(defined __PRINT_WARN__)) && \
        (!(defined __PRINT_ALL__)) && \
        (!(defined __PRINT_NONE__))

        #define __PRINT_FAIL__ 1
    #endif

    #ifndef __RUN_TIMING__
        #define __RUN_TIMING__ 0
    #endif

    #ifndef __CONSOLE__
        #define __CONSOLE__ def
    #endif

    #ifndef __TOTAL_TIMING__
        #define __TOTAL_TIMING__ uniform
    #endif

    #if (defined (__A1__)) || (defined (__A2__)) || (defined (__A3__))
        #undef  __PRINT_ALL__
        #define __PRINT_ALL__
        #define __FIXED_INPUTS__
    #endif

    #if defined(__FIXED_INPUTS__) && defined(__RUN_TIMING__)
        #define __FIXED_INPUTS_TIMING__
    #endif

    /*
    // Accuracy thresholds for the accuracy mode
    // measured in approximate incorrect mantissa bits
    */
    #define CR_ACC_THRESHOLD (0)
    #define HA_ACC_THRESHOLD (2)
    #define LA_ACC_THRESHOLD (29)

    #undef MIN
    #undef MAX
    #undef ABS
    #define MIN(a,b) ((a) < (b) ? (a) : (b))
    #define MAX(a,b) ((a) > (b) ? (a) : (b))
    #define ABS(a) ((a) < 0) ? -(a) : (a)

    #ifndef _CCCL_FP32_BIAS
        #define _CCCL_FP32_BIAS     127
    #endif
    #ifndef FP32_EXP_MIN
        #define FP32_EXP_MIN -126
    #endif
    #ifndef FP32_EXP_MAX
        #define FP32_EXP_MAX  127
    #endif
    #ifndef _CCCL_FP64_BIAS
        #define _CCCL_FP64_BIAS     1023
    #endif
    #ifndef FP64_EXP_MIN
        #define FP64_EXP_MIN -1022
    #endif
    #ifndef FP64_EXP_MAX
        #define FP64_EXP_MAX  1023
    #endif

#endif // __TS_HPP__
