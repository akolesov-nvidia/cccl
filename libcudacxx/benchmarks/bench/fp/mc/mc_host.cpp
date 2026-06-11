/*
    mc_host.cpp - Host-only helpers for the Monte Carlo benchmark
    =============================================================
    Always compiled with g++ (never nvcc) so that:
      - quad-precision math is available for the BS reference, and
      - host-only helpers stay out of the GPU translation unit.

    Platform support:
      - x86 with libquadmath: __float128 via sqrtq/logq/expq/erfq
      - ARM64/s390x: long double via sqrtl/logl/expl/erfl

    Provides:
      mc_ref_price_f128   - Black-Scholes analytical price computed entirely
                            in quad precision.  Used as the reference for
                            MC accuracy comparisons.
      mc_host_normcdfinv  - Peter Acklam's rational approximation of the
                            inverse normal CDF.  Converts a uniform variate
                            u in (0,1) to a standard normal variate.
                            Max relative error < 1.15e-9.
*/

// Platform detection
#if (defined(__x86_64__) || defined(_M_X64) || \
     defined(__i386__)   || defined(_M_IX86)) \
    && !defined(_MSC_VER) && !defined(_WIN32)
    #define MC_HOST_HAS_LIBQUADMATH 1
    #include <quadmath.h>
    typedef __float128 __mc_fp128;
#elif defined(__aarch64__) || defined(_M_ARM64) || \
      defined(__s390x__) || defined(__LONG_DOUBLE_IEEE128__)
    #define MC_HOST_HAS_LIBQUADMATH 0
    #define MC_HOST_HAS_LDOUBLE128 1
    #include <cmath>
    typedef long double __mc_fp128;
#else
    #define MC_HOST_HAS_LIBQUADMATH 0
    #define MC_HOST_HAS_LDOUBLE128 0
#endif
#include <math.h>

/* ------------------------------------------------------------------ */
/* Float128 Black-Scholes analytical price                            */
/* ------------------------------------------------------------------ */
#if (MC_HOST_HAS_LIBQUADMATH == 1) || (MC_HOST_HAS_LDOUBLE128 == 1)

// Platform-dispatched math function macros
#if (MC_HOST_HAS_LIBQUADMATH == 1)
    #define MC_SQRTQ(x)  sqrtq(x)
    #define MC_LOGQ(x)   logq(x)
    #define MC_EXPQ(x)   expq(x)
    #define MC_ERFQ(x)   erfq(x)
#else
    #define MC_SQRTQ(x)  sqrtl(x)
    #define MC_LOGQ(x)   logl(x)
    #define MC_EXPQ(x)   expl(x)
    #define MC_ERFQ(x)   erfl(x)
#endif

double mc_ref_price_f128(double S, double K, double r,
                         double q, double sigma, double Tmat,
                         bool is_call)
{
    const __mc_fp128 qhalf   = (__mc_fp128)0.5;
    const __mc_fp128 qone    = (__mc_fp128)1.0;
    const __mc_fp128 sqrt1_2 = MC_SQRTQ(qhalf);

    __mc_fp128 qS   = (__mc_fp128)S;
    __mc_fp128 qK   = (__mc_fp128)K;
    __mc_fp128 qr   = (__mc_fp128)r;
    __mc_fp128 qq   = (__mc_fp128)q;
    __mc_fp128 qsig = (__mc_fp128)sigma;
    __mc_fp128 qT   = (__mc_fp128)Tmat;

    __mc_fp128 sqrtT  = MC_SQRTQ(qT);
    __mc_fp128 vsqrtT = qsig * sqrtT;
    __mc_fp128 d1     = (MC_LOGQ(qS / qK) + (qr - qq + qhalf * qsig * qsig) * qT) / vsqrtT;
    __mc_fp128 d2     = d1 - vsqrtT;
    __mc_fp128 Nd1    = qhalf * (qone + MC_ERFQ(d1 * sqrt1_2));
    __mc_fp128 Nd2    = qhalf * (qone + MC_ERFQ(d2 * sqrt1_2));
    __mc_fp128 disc_r = MC_EXPQ(-qr * qT);
    __mc_fp128 disc_q = MC_EXPQ(-qq * qT);

    __mc_fp128 price;
    if (is_call)
        price = qS * disc_q * Nd1 - qK * disc_r * Nd2;
    else
        price = qK * disc_r * (qone - Nd2) - qS * disc_q * (qone - Nd1);
    return (double)price;
}
#endif // MC_HOST_HAS_LIBQUADMATH || MC_HOST_HAS_LDOUBLE128

/* ------------------------------------------------------------------ */
/* Inverse normal CDF (Acklam's rational approximation)               */
/* ------------------------------------------------------------------ */
double mc_host_normcdfinv(double p)
{
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };

    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;

    double q, r;
    if (p < p_low) {
        q = sqrt(-2.0 * log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}
