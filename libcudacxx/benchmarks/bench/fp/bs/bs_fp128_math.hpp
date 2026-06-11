/*
    bs_fp128_math.hpp - Quad-precision math overloads for Black-Scholes
    ===================================================================
    Provides free-function overloads of sqrt, log, exp, erf, erfc, fabs for
    the quad-precision type (__bs_fp128) so that bs_price<__bs_fp128>
    compiles on both host and device.

    Platform support:
      - CUDA device: __float128 or _Float128 via CUDA intrinsics (sm_100+)
      - x86 host with libquadmath: __float128 via sqrtq/logq/expq/erfq/erfcq/fabsq
      - ARM64/s390x host: long double via standard <cmath> (no overloads needed)

    Guard:  only active when __FPMP_FP128_ENABLED__ == 1.
*/
#ifndef BS_FP128_MATH_HPP
#define BS_FP128_MATH_HPP

#if __FPMP_FP128_ENABLED__ == 1

/* ------------------------------------------------------------------ */
/* Platform detection for 128-bit float type                            */
/* ------------------------------------------------------------------ */
#ifndef BS_HAS_LIBQUADMATH
    #if (defined(__x86_64__) || defined(_M_X64) || \
         defined(__i386__)   || defined(_M_IX86)) \
        && !defined(_MSC_VER) && !defined(_WIN32)
        #define BS_HAS_LIBQUADMATH 1
    #else
        #define BS_HAS_LIBQUADMATH 0
    #endif
#endif

#ifndef BS_HAS_LDOUBLE128
    #if defined(__aarch64__) || defined(_M_ARM64) || \
        defined(__s390x__) || \
        defined(__LONG_DOUBLE_IEEE128__)
        #define BS_HAS_LDOUBLE128 1
    #else
        #define BS_HAS_LDOUBLE128 0
    #endif
#endif

#if defined(__CUDACC__) && (BS_HAS_LIBQUADMATH == 0)
    typedef _Float128 __bs_fp128;
#elif (BS_HAS_LIBQUADMATH == 1)
    typedef __float128 __bs_fp128;
#elif (BS_HAS_LDOUBLE128 == 1)
    typedef long double __bs_fp128;
#else
    #error "No 128-bit float type available for BS benchmark"
#endif

/* ------------------------------------------------------------------ */
/* Compiler-specific setup                                              */
/* ------------------------------------------------------------------ */
#if defined(__CUDACC__)
  #include "crt/device_fp128_functions.h"

  #if (BS_HAS_LIBQUADMATH == 1)
  extern "C" {
      __bs_fp128 sqrtq(__bs_fp128);
      __bs_fp128 logq (__bs_fp128);
      __bs_fp128 expq (__bs_fp128);
      __bs_fp128 erfq (__bs_fp128);
      __bs_fp128 erfcq(__bs_fp128);
      __bs_fp128 fabsq(__bs_fp128);
  }
  #endif

  #define BS_FP128_FUNC __host__ __device__ inline
#elif (BS_HAS_LIBQUADMATH == 1)
  #include <quadmath.h>
  #define BS_FP128_FUNC inline
#elif (BS_HAS_LDOUBLE128 == 1)
  #include <cmath>
  #define BS_FP128_FUNC inline
#endif

/*
 * Math overloads for __bs_fp128 (x86 libquadmath / CUDA device).
 * On ARM64 host builds with long double, standard <cmath> overloads are used directly.
 */
#if defined(__CUDACC__) || (BS_HAS_LIBQUADMATH == 1)

/* ------------------------------------------------------------------ */
/* sqrt, log, exp, fabs                                                 */
/* ------------------------------------------------------------------ */
BS_FP128_FUNC __bs_fp128 sqrt(__bs_fp128 x) {
#if defined(__CUDA_ARCH__)
    return __nv_fp128_sqrt(x);
#elif (BS_HAS_LIBQUADMATH == 1)
    return sqrtq(x);
#else
    return (__bs_fp128)sqrtl((long double)x);
#endif
}

BS_FP128_FUNC __bs_fp128 log(__bs_fp128 x) {
#if defined(__CUDA_ARCH__)
    return __nv_fp128_log(x);
#elif (BS_HAS_LIBQUADMATH == 1)
    return logq(x);
#else
    return (__bs_fp128)logl((long double)x);
#endif
}

BS_FP128_FUNC __bs_fp128 exp(__bs_fp128 x) {
#if defined(__CUDA_ARCH__)
    return __nv_fp128_exp(x);
#elif (BS_HAS_LIBQUADMATH == 1)
    return expq(x);
#else
    return (__bs_fp128)expl((long double)x);
#endif
}

BS_FP128_FUNC __bs_fp128 fabs(__bs_fp128 x) {
#if defined(__CUDA_ARCH__)
    return __nv_fp128_fabs(x);
#elif (BS_HAS_LIBQUADMATH == 1)
    return fabsq(x);
#else
    return (__bs_fp128)fabsl((long double)x);
#endif
}

/* ------------------------------------------------------------------ */
/* erf(__bs_fp128)                                                      */
/*                                                                      */
/* Device: non-alternating series (DLMF 7.6.2)                         */
/*   erf(x) = (2/sqrt(pi)) * exp(-x^2)                                 */
/*            * sum_{n=0}^{N} 2^n * x^{2n+1} / (2n+1)!!                */
/* All terms positive => no catastrophic cancellation.                  */
/* Recurrence: a_0 = x,  a_{n+1} = a_n * 2x^2 / (2n+3).               */
/*                                                                      */
/* Host: erfq from libquadmath.                                         */
/* ------------------------------------------------------------------ */
BS_FP128_FUNC __bs_fp128 erf(__bs_fp128 x) {
#if defined(__CUDA_ARCH__)
    const __bs_fp128 zero = (__bs_fp128)0.0;
    const __bs_fp128 one  = (__bs_fp128)1.0;
    const __bs_fp128 two  = (__bs_fp128)2.0;

    if (x == zero) return zero;

    __bs_fp128 ax = __nv_fp128_fabs(x);

    /* erfc(9) < 1e-35 < quad epsilon; erf = +/-1 exactly */
    if (ax >= (__bs_fp128)9.0)
        return x > zero ? one : -one;

    /* pi via double-double: pi = pi_hi + pi_lo */
    const __bs_fp128 pi_q =
        (__bs_fp128)3.141592653589793 + (__bs_fp128)1.2246467991473532e-16;
    const __bs_fp128 two_over_sqrt_pi = two / __nv_fp128_sqrt(pi_q);

    __bs_fp128 x2     = ax * ax;
    __bs_fp128 two_x2 = two * x2;
    __bs_fp128 term   = ax;
    __bs_fp128 sum    = ax;

    for (int n = 0; n < 250; ++n) {
        term = term * two_x2 / (__bs_fp128)(2 * n + 3);
        sum  = sum + term;
        if (term < sum * (__bs_fp128)1e-36) break;
    }

    __bs_fp128 result = two_over_sqrt_pi * __nv_fp128_exp(-x2) * sum;
    return x > zero ? result : -result;
#elif (BS_HAS_LIBQUADMATH == 1)
    return erfq(x);
#else
    return (__bs_fp128)erfl((long double)x);
#endif
}

BS_FP128_FUNC __bs_fp128 erfc(__bs_fp128 x) {
#if defined(__CUDA_ARCH__)
    const __bs_fp128 zero = (__bs_fp128)0.0;
    const __bs_fp128 one  = (__bs_fp128)1.0;
    const __bs_fp128 two  = (__bs_fp128)2.0;

    if (x == zero) return one;
    if (x < zero) return two - erfc(-x);
    if (x >= (__bs_fp128)9.0) return zero;
    return one - erf(x);
#elif (BS_HAS_LIBQUADMATH == 1)
    return erfcq(x);
#else
    return (__bs_fp128)erfcl((long double)x);
#endif
}

#undef BS_FP128_FUNC
#endif /* __CUDACC__ || BS_HAS_LIBQUADMATH */

#endif /* __FPMP_FP128_ENABLED__ == 1 */
#endif /* BS_FP128_MATH_HPP */
