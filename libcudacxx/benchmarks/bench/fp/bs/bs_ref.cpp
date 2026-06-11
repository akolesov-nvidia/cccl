/*
    bs_ref.cpp - Float128 reference Black-Scholes price and accuracy
    ================================================================
    Always compiled with g++ (never nvcc) so that quad-precision math
    is available regardless of the benchmark target (GPU or CPU).

    Platform support:
      - x86 with libquadmath: __float128 via sqrtq/logq/expq/erfcq/fabsq
      - ARM64/s390x: long double via sqrtl/logl/expl/erfcl/fabsl

    Provides:
      bs_ref_price_f128  - Black-Scholes price computed entirely in quad precision
      bs_accuracy_f128   - Computes max and average relative error in quad precision
                           to avoid cancellation noise in comparisons.

    Enabled by FP128_REF=1 (the default) which defines __USE_FP128_REFERENCE__.
*/

// Platform detection
#if (defined(__x86_64__) || defined(_M_X64) || \
     defined(__i386__)   || defined(_M_IX86)) \
    && !defined(_MSC_VER) && !defined(_WIN32)
    #define BS_REF_HAS_LIBQUADMATH 1
    #include <quadmath.h>
    typedef __float128 __bs_ref_fp128;
#elif defined(__aarch64__) || defined(_M_ARM64) || \
      defined(__s390x__) || defined(__LONG_DOUBLE_IEEE128__)
    #define BS_REF_HAS_LIBQUADMATH 0
    #define BS_REF_HAS_LDOUBLE128 1
    #include <cmath>
    typedef long double __bs_ref_fp128;
#else
    #define BS_REF_HAS_LIBQUADMATH 0
    #define BS_REF_HAS_LDOUBLE128 0
#endif

#if (BS_REF_HAS_LIBQUADMATH == 1) || (BS_REF_HAS_LDOUBLE128 == 1)

// Platform-dispatched math function macros
#if (BS_REF_HAS_LIBQUADMATH == 1)
    #define BS_REF_SQRTQ(x)  sqrtq(x)
    #define BS_REF_LOGQ(x)   logq(x)
    #define BS_REF_EXPQ(x)   expq(x)
    #define BS_REF_ERFCQ(x)  erfcq(x)
    #define BS_REF_FABSQ(x)  fabsq(x)
#else
    #define BS_REF_SQRTQ(x)  sqrtl(x)
    #define BS_REF_LOGQ(x)   logl(x)
    #define BS_REF_EXPQ(x)   expl(x)
    #define BS_REF_ERFCQ(x)  erfcl(x)
    #define BS_REF_FABSQ(x)  fabsl(x)
#endif

double bs_ref_price_f128(double S, double K, double r,
                         double q, double sigma, double Tmat,
                         bool is_call)
{
    const __bs_ref_fp128 qhalf = (__bs_ref_fp128)0.5;
    const __bs_ref_fp128 sqrt1_2 = BS_REF_SQRTQ(qhalf);

    __bs_ref_fp128 qS   = (__bs_ref_fp128)S;
    __bs_ref_fp128 qK   = (__bs_ref_fp128)K;
    __bs_ref_fp128 qr   = (__bs_ref_fp128)r;
    __bs_ref_fp128 qq   = (__bs_ref_fp128)q;
    __bs_ref_fp128 qsig = (__bs_ref_fp128)sigma;
    __bs_ref_fp128 qT   = (__bs_ref_fp128)Tmat;

    __bs_ref_fp128 sqrtT  = BS_REF_SQRTQ(qT);
    __bs_ref_fp128 vsqrtT = qsig * sqrtT;
    __bs_ref_fp128 d1     = (BS_REF_LOGQ(qS / qK) + (qr - qq + qhalf * qsig * qsig) * qT) / vsqrtT;
    __bs_ref_fp128 d2     = d1 - vsqrtT;
    __bs_ref_fp128 Nd1    = qhalf * BS_REF_ERFCQ(-d1 * sqrt1_2);
    __bs_ref_fp128 Nd2    = qhalf * BS_REF_ERFCQ(-d2 * sqrt1_2);
    __bs_ref_fp128 disc_r = BS_REF_EXPQ(-qr * qT);
    __bs_ref_fp128 disc_q = BS_REF_EXPQ(-qq * qT);

    __bs_ref_fp128 price;
    if (is_call)
        price = qS * disc_q * Nd1 - qK * disc_r * Nd2;
    else
        price = qK * disc_r * (qhalf * BS_REF_ERFCQ(d2 * sqrt1_2)) -
                qS * disc_q * (qhalf * BS_REF_ERFCQ(d1 * sqrt1_2));
    return (double)price;
}

void bs_accuracy_f128(const double* prices, const double* ref_prices,
                      int N,
                      double* out_max_re, double* out_avg_re)
{
    __bs_ref_fp128 max_re = (__bs_ref_fp128)0.0;
    __bs_ref_fp128 sum_re = (__bs_ref_fp128)0.0;
    int n = 0;

    for (int i = 0; i < N; ++i) {
        __bs_ref_fp128 ref = (__bs_ref_fp128)ref_prices[i];
        if (BS_REF_FABSQ(ref) < (__bs_ref_fp128)1e-12) continue;
        __bs_ref_fp128 re = BS_REF_FABSQ(((__bs_ref_fp128)prices[i] - ref) / ref);
        if (re > max_re) max_re = re;
        sum_re += re;
        ++n;
    }

    *out_max_re = (double)max_re;
    *out_avg_re = (n > 0) ? (double)(sum_re / (__bs_ref_fp128)n) : 0.0;
}

#endif // BS_REF_HAS_LIBQUADMATH || BS_REF_HAS_LDOUBLE128
