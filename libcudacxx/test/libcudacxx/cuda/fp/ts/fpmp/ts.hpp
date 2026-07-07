/*
    ts.hpp - FPMP Test Suite Core Definitions
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Core definitions, macros, and configuration structures for the FPMP test suite.
*/

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

    // Pull in the __nv_fp128_* declarations early (CUDA builds only) so the
    // header's spelling-availability macros (_CCCL_FLOAT128_CPP_SPELLING_ENABLED /
    // __FLOAT128_C_SPELLING_ENABLED__) are visible for both the __ts_fp128
    // typedef and the native-reference override below. The header self-guards
    // (it only declares overloads where the host fp128 spelling exists) and has
    // its own include guard, so a later #include is a harmless no-op.
    #if defined(__CUDA_ARCH__) && (defined(__aarch64__) || defined(_M_ARM64)) \
        && defined(_CCCL_FPMP_CUDA_FP128_INTRINSICS) \
        && !(defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER_MAJOR__) \
             && ((__GNUC__ > 13) || (__GNUC__ == 13 && __GNUC_MINOR__ >= 1))) \
        && !defined(_CCCL_FLOAT128_CPP_SPELLING_ENABLED)
        #define _CCCL_FLOAT128_CPP_SPELLING_ENABLED
    #endif
    #if defined(__CUDACC__)
        #include "crt/device_fp128_functions.h"
    #endif

    // Macro to concatenate tokens
    // (legacy naming: some macros still use the historical __EFP64_* prefix)
    #if !defined __EFP64_GLUE_IMPL__
        #define __EFP64_GLUE_IMPL__
        #define GLUE_HELPER(a,b) a##b
        #define GLUE(a,b) GLUE_HELPER(a,b)
        #define GLUE2(a,b) GLUE(a,b)
        #define GLUE3(a,b,c) GLUE(GLUE2(a,b),c)
        #define GLUE4(a,b,c,d) GLUE(GLUE3(a,b,c),d)
        #define GLUE5(a,b,c,d,e) GLUE(GLUE4(a,b,c,d),e)
        #define GLUE6(a,b,c,d,e,f) GLUE(GLUE5(a,b,c,d,e),f)
        #define GLUE7(a,b,c,d,e,f,g) GLUE(GLUE6(a,b,c,d,e,f),g)
        #define GLUE8(a,b,c,d,e,f,g,h) GLUE(GLUE7(a,b,c,d,e,f,g),h)
    #endif // __EFP64_GLUE_IMPL__

    // Macro to convert a macro value to a string
    #define STRINGIFY(x) #x
    #define ABC(x) STRINGIFY(x)


    #ifndef __FUNC__
        #define __FUNC__       mul
    #endif
    #ifndef __TYPE__
        #define __TYPE__       fp32mp2
    #endif
    #ifndef __METHOD__
        #define __METHOD__     def
    #endif    

    #define fp32mp2_id          (0)
    #define fp64mp2_id          (1)
    #define fp32mp3_id          (2)
    #define fp64mp3_id          (3)
    #define FPMP_TYPE_ID       GLUE2(__TYPE__,_id)

    /*
     * Platform detection for 128-bit floating-point type (__ts_fp128)
     * ----------------------------------------------------------------
     * This typedef provides a quad-precision type for test reference values:
     *   - x86/x86_64 with GCC (non-Windows): __float128 via libquadmath
     *   - ARM64/s390x/PowerPC IEEE128: long double (128-bit IEEE 754)
     *   - Other platforms: compilation will fail for fp64mp2 tests
     */
    #ifndef TS_HAS_LIBQUADMATH
        #if (defined(__x86_64__) || defined(_M_X64) || \
             defined(__i386__)   || defined(_M_IX86)) \
            && !defined(_MSC_VER) && !defined(_WIN32)
            #define TS_HAS_LIBQUADMATH 1
        #else
            #define TS_HAS_LIBQUADMATH 0
        #endif
    #endif

    #ifndef TS_HAS_LDOUBLE128
        #if defined(__aarch64__) || defined(_M_ARM64) || \
            defined(__s390x__) || \
            defined(__LONG_DOUBLE_IEEE128__)
            #define TS_HAS_LDOUBLE128 1
        #else
            #define TS_HAS_LDOUBLE128 0
        #endif
    #endif

    // x86 with libquadmath:        use __float128
    // ARM64 CUDA build (GCC>=13.1): use _Float128  (distinct binary128 type whose
    //                              mangling matches the __nv_fp128_* builtins)
    // ARM64/s390x host otherwise:  use long double (128-bit IEEE)
    //
    // Note on ARM64: __ts_fp128 surfaces in __global__ kernel template arguments
    // (the fp128 perf column and the accuracy reference tag), which cudafe
    // re-emits verbatim into the host-side launch stub. The spelling must
    // therefore be parseable by the AArch64 host compiler AND, if it is to call
    // __nv_fp128_*, mangle to the type the device builtins are emitted under
    // (__float128 == 'g'). On host GCC < 13.1 there is no single spelling that
    // does both: __float128 is unparseable in the stub, while _Float128 is just
    // an alias of long double ('e') and links to a device symbol that does not
    // exist. We therefore use long double there (the device reference is capped
    // at ~53 bits -- a genuine 128-bit reference requires either GCC >= 13.1 or a
    // host-computed reference path). GCC >= 13.1 makes _Float128 a distinct type
    // that both parses and links, so __FLOAT128_C_SPELLING_ENABLED__ gates it.
    // Constructor calls __nv_fpmp2_t<double>(__ts_fp128) still bridge through
    // static_cast<__fpmp_fp128>(...) at the call sites.
    #if (TS_HAS_LIBQUADMATH == 1)
        typedef __float128 __ts_fp128;
    #elif (TS_HAS_LDOUBLE128 == 1)
        #if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__CUDACC__) && defined(__FLOAT128_C_SPELLING_ENABLED__)
            typedef _Float128 __ts_fp128;
        #else
            typedef long double __ts_fp128;
        #endif
    #endif

    /*
     * Platform-dispatched quad-precision math function wrappers for test suite
     * -------------------------------------------------------------------------
     * These macros dispatch to the appropriate math functions based on platform.
     * Branch order matters -- the first matching one wins:
     *   - libm *l: host with TS_HAS_LDOUBLE128 when libquadmath is not used
     *   - CUDA device with extended __nv_fp128_* (_CCCL_FPMP_CUDA_FP128_INTRINSICS +
     *     an fp128 spelling): true 128-bit reference for all that have an
     *     intrinsic, double-widen only for atan2/cbrt/erf/erfc/scalbn. Covers
     *     both x86_64 and AArch64; this is the path that gives true 128-bit refs.
     *   - CUDA device AArch64 (no extended intrinsics): double math widened to __ts_fp128
     *   - CUDA device (x86_64, … no extended intrinsics): base __nv_fp128_* where available; cbrt uses double
     *   - x86 with libquadmath: *q functions (expq, sinq, etc.)
     */
    #if (TS_HAS_LIBQUADMATH == 1) || (TS_HAS_LDOUBLE128 == 1)
        #if (TS_HAS_LDOUBLE128 == 1) && !defined(__CUDA_ARCH__) && (TS_HAS_LIBQUADMATH == 0)
            #define __TS_FMAQ(x,y,z)     fmal((x),(y),(z))
            #define __TS_SQRTQ(x)        sqrtl(x)
            #define __TS_EXPQ(x)         expl(x)
            #define __TS_EXP2Q(x)        exp2l(x)
            #define __TS_EXP10Q(x)       ((__ts_fp128)powl(10.0L, (x)))
            #define __TS_EXPM1Q(x)       expm1l(x)
            #define __TS_LOGQ(x)         logl(x)
            #define __TS_LOG2Q(x)        log2l(x)
            #define __TS_LOG10Q(x)       log10l(x)
            #define __TS_LOG1PQ(x)       log1pl(x)
            #define __TS_POWQ(x,y)       powl((x),(y))
            #define __TS_ATAN2Q(y,x)     atan2l((y),(x))
            #define __TS_SINQ(x)         sinl(x)
            #define __TS_COSQ(x)         cosl(x)
            #define __TS_TANQ(x)         tanl(x)
            #define __TS_ASINQ(x)        asinl(x)
            #define __TS_ACOSQ(x)        acosl(x)
            #define __TS_ATANQ(x)        atanl(x)
            #define __TS_SINHQ(x)        sinhl(x)
            #define __TS_COSHQ(x)        coshl(x)
            #define __TS_TANHQ(x)        tanhl(x)
            #define __TS_ASINHQ(x)       asinhl(x)
            #define __TS_ACOSHQ(x)       acoshl(x)
            #define __TS_ATANHQ(x)       atanhl(x)
            #define __TS_ERFQ(x)         erfl(x)
            #define __TS_ERFCQ(x)        erfcl(x)
            #define __TS_CBRTQ(x)        cbrtl(x)
            #define __TS_LDEXPQ(x,n)     ldexpl((x),(n))
            #define __TS_SCALBNQ(x,n)    scalbnl((x),(n))
            #define __TS_FMODQ(x,y)      fmodl((x),(y))
            #define __TS_REMAINDERQ(x,y) remainderl((x),(y))
        #elif defined(__CUDA_ARCH__) && defined(_CCCL_FPMP_CUDA_FP128_INTRINSICS) && \
              (defined(_CCCL_FLOAT128_CPP_SPELLING_ENABLED) || defined(__FLOAT128_C_SPELLING_ENABLED__))
            // CUDA device with the *extended* __nv_fp128_* intrinsics guaranteed
            // present (_CCCL_FPMP_CUDA_FP128_INTRINSICS is set by the test build whenever
            // host fp128 spelling the crt header could declare them under (__float128 on x86_64, or
            // _Float128 on AArch64 with GCC >= 13.1). One branch serves both
            // x86_64 and AArch64: a true 128-bit reference for every function that
            // has a native intrinsic, and a double-widen reference only for the
            // few that do not (atan2, cbrt, erf, erfc, scalbn). This mirrors the
            // library dispatch in fpmp_math.h and replaces the older "base
            // defaults + #undef/#define override" two-step (the spelling macros
            // are already resolved here, since the crt header is included above).
            // When the intrinsics/spelling are NOT guaranteed, the conservative
            // device branches below stand so the build still compiles.
            #define __TS_FMAQ(x,y,z)     __nv_fp128_fma((x),(y),(z))
            #define __TS_SQRTQ(x)        __nv_fp128_sqrt(x)
            #define __TS_EXPQ(x)         __nv_fp128_exp(x)
            #define __TS_EXP2Q(x)        __nv_fp128_exp2(x)
            #define __TS_EXP10Q(x)       __nv_fp128_exp10(x)
            #define __TS_EXPM1Q(x)       __nv_fp128_expm1(x)
            #define __TS_LOGQ(x)         __nv_fp128_log(x)
            #define __TS_LOG2Q(x)        __nv_fp128_log2(x)
            #define __TS_LOG10Q(x)       __nv_fp128_log10(x)
            #define __TS_LOG1PQ(x)       __nv_fp128_log1p(x)
            #define __TS_POWQ(x,y)       __nv_fp128_pow((x),(y))
            #define __TS_SINQ(x)         __nv_fp128_sin(x)
            #define __TS_COSQ(x)         __nv_fp128_cos(x)
            #define __TS_TANQ(x)         __nv_fp128_tan(x)
            #define __TS_ASINQ(x)        __nv_fp128_asin(x)
            #define __TS_ACOSQ(x)        __nv_fp128_acos(x)
            #define __TS_ATANQ(x)        __nv_fp128_atan(x)
            #define __TS_SINHQ(x)        __nv_fp128_sinh(x)
            #define __TS_COSHQ(x)        __nv_fp128_cosh(x)
            #define __TS_TANHQ(x)        __nv_fp128_tanh(x)
            #define __TS_ASINHQ(x)       __nv_fp128_asinh(x)
            #define __TS_ACOSHQ(x)       __nv_fp128_acosh(x)
            #define __TS_ATANHQ(x)       __nv_fp128_atanh(x)
            #define __TS_LDEXPQ(x,n)     __nv_fp128_ldexp((x),(n))
            #define __TS_FMODQ(x,y)      __nv_fp128_fmod((x),(y))
            #define __TS_REMAINDERQ(x,y) __nv_fp128_remainder((x),(y))
            // No native __nv_fp128_cbrt: reconstruct from pow with a true fp128
            // 1/3 and restore the sign (pow rejects negative bases). Mirrors the
            // library reference (_CCCL_FPMP_CBRTQ in fpmp_math.h) so the cbrt
            // reference is ~fp128-accurate instead of being capped at ~53 bits.
            #define __TS_CBRTQ(x)        __nv_fp128_copysign(__nv_fp128_pow(__nv_fp128_fabs(x), (__ts_fp128)1 / (__ts_fp128)3), (x))
            // No native __nv_fp128_* for these -- widen through double.
            #define __TS_ATAN2Q(y,x)     ((__ts_fp128)atan2((double)(y), (double)(x)))
            #define __TS_ERFQ(x)         ((__ts_fp128)erf((double)(x)))
            #define __TS_ERFCQ(x)        ((__ts_fp128)erfc((double)(x)))
            #define __TS_SCALBNQ(x,n)    ((__ts_fp128)scalbn((double)(x), (n)))
        #elif defined(__CUDA_ARCH__) && (defined(__aarch64__) || defined(_M_ARM64))
            #define __TS_FMAQ(x,y,z)     ((__ts_fp128)fma((double)(x), (double)(y), (double)(z)))
            #define __TS_SQRTQ(x)        ((__ts_fp128)sqrt((double)(x)))
            #define __TS_EXPQ(x)         ((__ts_fp128)exp((double)(x)))
            #define __TS_EXP2Q(x)        ((__ts_fp128)exp2((double)(x)))
            #define __TS_EXP10Q(x)       ((__ts_fp128)exp10((double)(x)))
            #define __TS_EXPM1Q(x)       ((__ts_fp128)expm1((double)(x)))
            #define __TS_LOGQ(x)         ((__ts_fp128)log((double)(x)))
            #define __TS_LOG2Q(x)        ((__ts_fp128)log2((double)(x)))
            #define __TS_LOG10Q(x)       ((__ts_fp128)log10((double)(x)))
            #define __TS_LOG1PQ(x)       ((__ts_fp128)log1p((double)(x)))
            #define __TS_POWQ(x,y)       ((__ts_fp128)pow((double)(x), (double)(y)))
            #define __TS_ATAN2Q(y,x)     ((__ts_fp128)atan2((double)(y), (double)(x)))
            #define __TS_SINQ(x)         ((__ts_fp128)sin((double)(x)))
            #define __TS_COSQ(x)         ((__ts_fp128)cos((double)(x)))
            #define __TS_TANQ(x)         ((__ts_fp128)tan((double)(x)))
            #define __TS_ASINQ(x)        ((__ts_fp128)asin((double)(x)))
            #define __TS_ACOSQ(x)        ((__ts_fp128)acos((double)(x)))
            #define __TS_ATANQ(x)        ((__ts_fp128)atan((double)(x)))
            #define __TS_SINHQ(x)        ((__ts_fp128)sinh((double)(x)))
            #define __TS_COSHQ(x)        ((__ts_fp128)cosh((double)(x)))
            #define __TS_TANHQ(x)        ((__ts_fp128)tanh((double)(x)))
            #define __TS_ASINHQ(x)       ((__ts_fp128)asinh((double)(x)))
            #define __TS_ACOSHQ(x)       ((__ts_fp128)acosh((double)(x)))
            #define __TS_ATANHQ(x)       ((__ts_fp128)atanh((double)(x)))
            #define __TS_ERFQ(x)         ((__ts_fp128)erf((double)(x)))
            #define __TS_ERFCQ(x)        ((__ts_fp128)erfc((double)(x)))
            #define __TS_CBRTQ(x)        ((__ts_fp128)cbrt((double)(x)))
            #define __TS_LDEXPQ(x,n)     ((__ts_fp128)ldexp((double)(x), (n)))
            #define __TS_SCALBNQ(x,n)    ((__ts_fp128)scalbn((double)(x), (n)))
            #define __TS_FMODQ(x,y)      ((__ts_fp128)fmod((double)(x), (double)(y)))
            #define __TS_REMAINDERQ(x,y) ((__ts_fp128)remainder((double)(x), (double)(y)))
        #elif defined(__CUDA_ARCH__)
            // CUDA device (not AArch64): __nv_fp128_* from CUDA headers
            #define __TS_FMAQ(x,y,z)     __nv_fp128_fma((x),(y),(z))
            #define __TS_SQRTQ(x)        __nv_fp128_sqrt(x)
            #define __TS_EXPQ(x)         __nv_fp128_exp(x)
            #define __TS_EXP2Q(x)        __nv_fp128_exp2(x)
            /* No __nv_fp128_exp10 in CUDA fp128 intrinsics; widen via double. */
            #define __TS_EXP10Q(x)       ((__ts_fp128)exp10((double)(x)))
            #define __TS_EXPM1Q(x)       __nv_fp128_expm1(x)
            #define __TS_LOGQ(x)         __nv_fp128_log(x)
            #define __TS_LOG2Q(x)        __nv_fp128_log2(x)
            #define __TS_LOG10Q(x)       __nv_fp128_log10(x)
            #define __TS_LOG1PQ(x)       __nv_fp128_log1p(x)
            #define __TS_POWQ(x,y)       __nv_fp128_pow((x),(y))
            /* No __nv_fp128_atan2 in CUDA fp128 intrinsics; widen via double. */
            #define __TS_ATAN2Q(y,x)     ((__ts_fp128)atan2((double)(y), (double)(x)))
            #define __TS_SINQ(x)         __nv_fp128_sin(x)
            #define __TS_COSQ(x)         __nv_fp128_cos(x)
            #define __TS_TANQ(x)         __nv_fp128_tan(x)
            #define __TS_ASINQ(x)        __nv_fp128_asin(x)
            #define __TS_ACOSQ(x)        __nv_fp128_acos(x)
            #define __TS_ATANQ(x)        __nv_fp128_atan(x)
            #define __TS_SINHQ(x)        __nv_fp128_sinh(x)
            #define __TS_COSHQ(x)        __nv_fp128_cosh(x)
            #define __TS_TANHQ(x)        __nv_fp128_tanh(x)
            #define __TS_ASINHQ(x)       __nv_fp128_asinh(x)
            #define __TS_ACOSHQ(x)       __nv_fp128_acosh(x)
            #define __TS_ATANHQ(x)       __nv_fp128_atanh(x)
            // No fp128 erf/erfc in CUDA yet; fall back to double precision
            #define __TS_ERFQ(x)         erf((double)(x))
            #define __TS_ERFCQ(x)        erfc((double)(x))
            #define __TS_CBRTQ(x)        ((__ts_fp128)cbrt((double)(x)))
            /* No __nv_fp128_ldexp / scalbn in CUDA fp128 intrinsics; widen via double. */
            #define __TS_LDEXPQ(x,n)     ((__ts_fp128)ldexp((double)(x), (n)))
            #define __TS_SCALBNQ(x,n)    ((__ts_fp128)scalbn((double)(x), (n)))
            /* fmod / remainder DO have native fp128 intrinsics; use them so the
             * reference keeps full 106-bit accuracy (a double-widened reference
             * would only validate to ~53 bits and reject the accurate library). */
            #define __TS_FMODQ(x,y)      __nv_fp128_fmod((x),(y))
            #define __TS_REMAINDERQ(x,y) __nv_fp128_remainder((x),(y))
        #elif (TS_HAS_LIBQUADMATH == 1)
            #define __TS_FMAQ(x,y,z)     fmaq((x),(y),(z))
            #define __TS_SQRTQ(x)        sqrtq(x)
            #define __TS_EXPQ(x)         expq(x)
            #define __TS_EXP2Q(x)        exp2q(x)
            /* libquadmath has no exp10q; synthesize via powq. */
            #define __TS_EXP10Q(x)       powq((__float128)10.0, (x))
            #define __TS_EXPM1Q(x)       expm1q(x)
            #define __TS_LOGQ(x)         logq(x)
            #define __TS_LOG2Q(x)        log2q(x)
            #define __TS_LOG10Q(x)       log10q(x)
            #define __TS_LOG1PQ(x)       log1pq(x)
            #define __TS_POWQ(x,y)       powq((x),(y))
            #define __TS_ATAN2Q(y,x)     atan2q((y),(x))
            #define __TS_SINQ(x)         sinq(x)
            #define __TS_COSQ(x)         cosq(x)
            #define __TS_TANQ(x)         tanq(x)
            #define __TS_ASINQ(x)        asinq(x)
            #define __TS_ACOSQ(x)        acosq(x)
            #define __TS_ATANQ(x)        atanq(x)
            #define __TS_SINHQ(x)        sinhq(x)
            #define __TS_COSHQ(x)        coshq(x)
            #define __TS_TANHQ(x)        tanhq(x)
            #define __TS_ASINHQ(x)       asinhq(x)
            #define __TS_ACOSHQ(x)       acoshq(x)
            #define __TS_ATANHQ(x)       atanhq(x)
            #define __TS_ERFQ(x)         erfq(x)
            #define __TS_ERFCQ(x)        erfcq(x)
            #define __TS_CBRTQ(x)        cbrtq(x)
            #define __TS_LDEXPQ(x,n)     ldexpq((x),(n))
            #define __TS_SCALBNQ(x,n)    scalbnq((x),(n))
            #define __TS_FMODQ(x,y)      fmodq((x),(y))
            #define __TS_REMAINDERQ(x,y) remainderq((x),(y))
        #endif

    #endif

    #if FPMP_TYPE_ID == fp32mp2_id
        #define FPMP_TYPE cuda::experimental::fpmp2_t<float,cuda::experimental::fpmp2_accuracy::__METHOD__>
        #define REF_TYPE  double
        #define BASE_TYPE float
        #define BASE_TYPE_NAME fp32
        #define REF_TYPE_NAME  fp64
        #define MANTISSA_BITS 46
    #elif FPMP_TYPE_ID == fp64mp2_id
        #if (TS_HAS_LIBQUADMATH == 1) || (TS_HAS_LDOUBLE128 == 1)
            #define FPMP_TYPE cuda::experimental::fpmp2_t<double,cuda::experimental::fpmp2_accuracy::__METHOD__>
            #define REF_TYPE  __ts_fp128
            #define BASE_TYPE double
            #define BASE_TYPE_NAME fp64
            #define REF_TYPE_NAME  fp128
            #define MANTISSA_BITS 104
        #else
            #error "fp64mp2 tests require 128-bit float support (libquadmath on x86 or 128-bit long double on ARM64)"
        #endif
    #else
        #error "Unsupported multi-precision floating-point type"
    #endif

    #if defined __CUDACC__
        #define __EFP64_INTERNAL_DECL__ static __forceinline__ __host__ __device__
    #else
        #define __EFP64_INTERNAL_DECL__ static inline
    #endif

    #define FUNC_TAG         GLUE3(_,__FUNC__,_)<fp_arg_type>
    #define FUNC_NAME        GLUE2(__FUNC__,_function)
    #define FUNC_IMPL        GLUE2(__FUNC__,_device_impl)

    // Note: Function arity is now auto-detected in ts_functions.hpp
    // using std::is_invocable_v, so no manual *_args definitions needed

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

    // NOTE: Legacy macros (__MEAN__, __STDDEV__, __SEED__, __ACCURACY_LEN__, __LATENCY_LEN__,
    // __THROUGHPUT_LEN__, __MAX_LEN__, __ITERATIONS__, __REPEATS__, __NAN_PAYLOAD__,
    // __ZERO_SIGN__, __PRINT_LIMIT__, __HTR_LEN__) have been removed as they were unused.
    // They were previously used by ts::config_t in ts_types.hpp but that code was cleaned up.

    #if (!(defined __PRINT_FAIL__)) && \
        (!(defined __PRINT_OK__)) && \
        (!(defined __PRINT_WARN__)) && \
        (!(defined __PRINT_ALL__)) && \
        (!(defined __PRINT_NONE__))

        #define __PRINT_FAIL__ 1
    #endif

    #ifndef __RUN_TIMING__
        #define __RUN_TIMING__ 1
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

// Debug print utilities (always available; add PRINTFP32/PRINTMP32 calls to fpmp_impl.h as needed)
#include "ts_debug.hpp"

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

    #ifndef FP128_BIAS
        #define FP128_BIAS     16383
    #endif
    #ifndef FP128_EXP_MIN
        #define FP128_EXP_MIN -16382
    #endif
    #ifndef FP128_EXP_MAX
        #define FP128_EXP_MAX  16383
    #endif


// Default rigor for all datasets (can be overridden per-dataset below)
// CUDA: 2^24 = ~16M samples (fast on GPU)
// Host: 2^20 = ~1M samples (reasonable speed with OpenMP)
#ifndef TS_ACCURACY_RIGOR
    #if defined(__CUDACC__)
        #define TS_ACCURACY_RIGOR (24)  // 2^24 = ~16M samples (GPU default)
    #else
        #define TS_ACCURACY_RIGOR (20)  // 2^20 = ~1M samples (host default)
    #endif
#endif

// Cap rigor to prevent unreasonable run times on host only
// Host: 2^28 max (~256M), CUDA: 2^48 max (allows multi-hour runs)
#if defined(__CUDACC__)
    #define TS_ACCURACY_RIGOR_MAX (48)  // ~280T samples, allows very long runs
#else
    #define TS_ACCURACY_RIGOR_MAX (30)  // ~256M samples, ~minutes on host with OMP
#endif

#if TS_ACCURACY_RIGOR > TS_ACCURACY_RIGOR_MAX
    #undef TS_ACCURACY_RIGOR
    #define TS_ACCURACY_RIGOR TS_ACCURACY_RIGOR_MAX
#endif

// Per-dataset rigor settings (default to TS_ACCURACY_RIGOR if not defined)
#ifndef TS_ACCURACY_RIGOR_WORK
    #define TS_ACCURACY_RIGOR_WORK TS_ACCURACY_RIGOR
#endif
#ifndef TS_ACCURACY_RIGOR_NORMAL
    #define TS_ACCURACY_RIGOR_NORMAL TS_ACCURACY_RIGOR
#endif
// Fixed input mode: when __FIXED_INPUTS__ is defined, force special dataset to run only once
#if defined(__FIXED_INPUTS__)
    #ifdef TS_ACCURACY_RIGOR_SPECIAL
        #undef TS_ACCURACY_RIGOR_SPECIAL
    #endif
    #define TS_ACCURACY_RIGOR_SPECIAL 0  // 2^0 = 1 sample
#elif !defined(TS_ACCURACY_RIGOR_SPECIAL)
    #define TS_ACCURACY_RIGOR_SPECIAL TS_ACCURACY_RIGOR
#endif
#ifndef TS_ACCURACY_RIGOR_PATTERN
    // For fp32mp2 on GPU: use 32 bits for exhaustive 32-bit integer coverage
    // For all other types/targets: use default 24 bits
    #if defined(__CUDACC__) && (FPMP_TYPE_ID == fp32mp2_id) && (TS_ACCURACY_RIGOR < 32)
        #define TS_ACCURACY_RIGOR_PATTERN (32)
    #else
        #define TS_ACCURACY_RIGOR_PATTERN TS_ACCURACY_RIGOR
    #endif
#endif

// Cap per-dataset rigors
#if TS_ACCURACY_RIGOR_WORK > TS_ACCURACY_RIGOR_MAX
    #undef TS_ACCURACY_RIGOR_WORK
    #define TS_ACCURACY_RIGOR_WORK TS_ACCURACY_RIGOR_MAX
#endif
#if TS_ACCURACY_RIGOR_NORMAL > TS_ACCURACY_RIGOR_MAX
    #undef TS_ACCURACY_RIGOR_NORMAL
    #define TS_ACCURACY_RIGOR_NORMAL TS_ACCURACY_RIGOR_MAX
#endif
#if TS_ACCURACY_RIGOR_SPECIAL > TS_ACCURACY_RIGOR_MAX
    #undef TS_ACCURACY_RIGOR_SPECIAL
    #define TS_ACCURACY_RIGOR_SPECIAL TS_ACCURACY_RIGOR_MAX
#endif
#if TS_ACCURACY_RIGOR_PATTERN > TS_ACCURACY_RIGOR_MAX
    #undef TS_ACCURACY_RIGOR_PATTERN
    #define TS_ACCURACY_RIGOR_PATTERN TS_ACCURACY_RIGOR_MAX
#endif

#ifndef TS_ACCURACY_SEED
    #define TS_ACCURACY_SEED (12345)
#endif

#ifndef TS_ACCURACY_THREADS_PER_BLOCK
    #define TS_ACCURACY_THREADS_PER_BLOCK (256)
#endif

#ifndef TS_USE_PRECISE_REFERENCE
    #define TS_USE_PRECISE_REFERENCE (0)
#endif

#ifndef TS_ACCURACY_ERROR_LOG_SIZE
    #define TS_ACCURACY_ERROR_LOG_SIZE (100)
#endif

// Default thresholds (can be overridden)
// Error threshold: significant precision loss (basically lost all extended precision benefit)
// Warning threshold: some precision loss but still reasonable for most use cases
#ifndef TS_ACCURACY_ERROR_THRESHOLD_FP32
    #define TS_ACCURACY_ERROR_THRESHOLD_FP32 (1e-7)   // For fp32mp2: error if lost all extended precision
#endif
#ifndef TS_ACCURACY_WARNING_THRESHOLD_FP32
    #define TS_ACCURACY_WARNING_THRESHOLD_FP32 (1e-13) // For fp32mp2: warning for moderate precision loss
#endif

#ifndef TS_ACCURACY_ERROR_THRESHOLD_FP64
    #define TS_ACCURACY_ERROR_THRESHOLD_FP64 (1e-15)  // For fp64mp2: error if lost all extended precision
#endif
#ifndef TS_ACCURACY_WARNING_THRESHOLD_FP64
    #define TS_ACCURACY_WARNING_THRESHOLD_FP64 (1e-28) // For fp64mp2: warning for moderate precision loss
#endif

#if defined(__CUDACC__)
    #define TS_MAX(a,b)      max(a,b)
    #define TS_MIN(a,b)      min(a,b)
    #define TS_ABS(a)        abs(a)
    #define TS_LOG2(x)       log2(x)
    #define TS_KERNEL_DECL   __global__
    static __forceinline__ __HOST_DEVICE_DECL__ float  my_frexp(float x, int* e)  { return frexpf(x, e); }
    static __forceinline__ __HOST_DEVICE_DECL__ double my_frexp(double x, int* e) { return frexp(x, e); }    
    static __forceinline__ __HOST_DEVICE_DECL__ float  my_ldexp(float x, int exp) { return ldexpf(x, exp); }
    static __forceinline__ __HOST_DEVICE_DECL__ double my_ldexp(double x, int exp) { return ldexp(x, exp); }
    #define TS_FREXP(x, exp)    my_frexp(x, exp)
    #define TS_LDEXP(x, exp)    my_ldexp(x, exp)
#else
    #define TS_MAX(a,b)      std::max(a,b)
    #define TS_MIN(a,b)      std::min(a,b)
    #define TS_ABS(a)        std::abs(a)
    #define TS_LOG2(x)       std::log2(x)
    #define TS_KERNEL_DECL
    #define TS_FREXP(x, exp) std::frexp(x, exp)
    #define TS_LDEXP(x, exp) std::ldexp(x, exp)
#endif

#if FPMP_TYPE_ID == fp64mp2_id
    /* CUDA device (x86_64 and AArch64): the __nv_fp128_* reference intrinsics
     * live only in this header. It self-guards on the host fp128 spelling, so
     * it is safe (and required for the extended set) on AArch64 too. */
    #if defined(__CUDA_ARCH__)
        #include "crt/device_fp128_functions.h"
    #elif (TS_HAS_LIBQUADMATH == 1)
        // x86 host: libquadmath
        #include <quadmath.h>
    #elif (TS_HAS_LDOUBLE128 == 1)
        // ARM64/s390x host: long double is 128-bit IEEE
        #include <cmath>
    #endif
#endif

    #include <cuda/fpmp>
    #include <cuda/fpmp_math>

    // The ts harness historically referred to the library namespace as `fpmp`
    // and the type as __nv_fpmp2_t<FpType, fpmp::method::X>. The CCCL FP SDK
    // lives in cuda::experimental::fpmp with class fpmp2_t<FpType,
    // fpmp2_accuracy::X>; alias the namespace so the existing references resolve.
    namespace fpmp = cuda::experimental::fpmp;

    using fpmp_type = FPMP_TYPE;
    using fprf_type = REF_TYPE;
    
    // fp_arg_type: used by FUNC_TAG macro for assembly generation
    // When __NATIVE_IMPL__ is defined, use reference type for .hi files
    // Otherwise use multi-precision type for .mp files
#if defined __NATIVE_IMPL__
    using fp_arg_type = fprf_type;
#else
    using fp_arg_type = fpmp_type;
#endif

#endif // __TS_HPP__
