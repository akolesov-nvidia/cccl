//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FP_FPMP_MATH_H
#define _CUDA___FP_FPMP_MATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
/*
    fpmp_math.h - Math Extensions for fpmp2 Types
    ======================================================================================================
    This header provides transcendental mathematical functions for fpmp2 types
    (fp32mp2 = double-float, fp64mp2 = double-double) beyond core arithmetic.
    Include this header after fpmp.h to enable mathematical functions.

    All dedicated fp32mp2 implementations use pure float-float arithmetic
    (no double-precision operations), making them suitable for GPU architectures
    where fp64 throughput is limited.

    Functions Provided:
    -------------------------------------------------------------------------
    Exponential/Logarithmic:
    - exp(x)    : Exponential function (e^x) - dedicated fp32mp2
    - log(x)    : Natural logarithm ln(x) - dedicated fp32mp2
    - log2(x)   : Base-2 logarithm - dedicated fp32mp2
    - log10(x)  : Base-10 logarithm - dedicated fp32mp2
    - log1p(x)  : Natural logarithm of (1+x) - dedicated fp32mp2
    - exp2(x)   : Base-2 exponential (2^x) - dedicated fp32mp2
    - exp10(x)  : Base-10 exponential (10^x) - dedicated fp32mp2
    - expm1(x)  : e^x - 1 - dedicated fp32mp2
    - logb(x)   : Extract exponent - placeholder

    Power:
    - pow(x,y)  : Power function x^y - dedicated fp32mp2
    - cbrt(x)   : Cube root - dedicated fp32mp2
    - rcbrt(x)  : Reciprocal cube root 1/cbrt(x) - dedicated fp32mp2

    Trigonometric:
    - sin(x)    : Sine - dedicated fp32mp2
    - cos(x)    : Cosine - dedicated fp32mp2
    - tan(x)    : Tangent - dedicated fp32mp2
    - sincos(x) : Simultaneous sine and cosine - dedicated fp32mp2
    - sinpi(x)  : sin(pix) - placeholder (host: sin(x*pi))
    - cospi(x)  : cos(pix) - placeholder (host: cos(x*pi))
    - sincospi(x): Simultaneous sin(pix) and cos(pix) - placeholder
    - asin(x)   : Arcsine - dedicated fp32mp2
    - acos(x)   : Arccosine - dedicated fp32mp2
    - atan(x)   : Arctangent - dedicated fp32mp2
    - atan2(y,x): Two-argument arctangent - dedicated fp32mp2

    Hyperbolic:
    - sinh(x)   : Hyperbolic sine - dedicated fp32mp2
    - cosh(x)   : Hyperbolic cosine - dedicated fp32mp2
    - tanh(x)   : Hyperbolic tangent - dedicated fp32mp2
    - acosh(x)  : Inverse hyperbolic cosine - dedicated fp32mp2
    - asinh(x)  : Inverse hyperbolic sine - dedicated fp32mp2
    - atanh(x)  : Inverse hyperbolic tangent - dedicated fp32mp2

    Error Functions:
    - erf(x)    : Error function - dedicated fp32mp2
    - erfc(x)   : Complementary error function - dedicated fp32mp2
    - erfcinv(x): Inverse complementary error function - placeholder (device only)
    - erfinv(x) : Inverse error function - placeholder (device only)
    - erfcx(x)  : Scaled complementary error function - placeholder (device only)

    Special Functions:
    - boys_f0(x): Boys function of zeroth order F_0(x) - dedicated fp32mp2
                  F_0(x) = 0.5*sqrt(pi/x)*erf(sqrtx),  F_0(0) = 1

    Probability / Statistics:
    - normcdfinv(p) : Inverse normal CDF Phi^-1(p) - dedicated fp32mp2
    - normcdf(x)    : Normal CDF Phi(x) - placeholder (host: 0.5*erfc(-x/sqrt2))
    
    Non-standard helper funcitons to convert integer uniform random numbers
    to Gaussian distributed random numbers:
    - icdf(uint32)  : Integer uniform -> Gaussian via normcdfinv (32-bit input)
    - icdf(uint64)  : Integer uniform -> Gaussian via normcdfinv (64-bit input)

    Gamma Functions:
    - lgamma(x) : Log-gamma function - placeholder
    - tgamma(x) : True gamma function - placeholder

    Bessel Functions (POSIX / CUDA):
    - j0(x)     : Bessel function of first kind, order 0 - placeholder
    - j1(x)     : Bessel function of first kind, order 1 - placeholder
    - jn(n,x)   : Bessel function of first kind, order n - placeholder
    - y0(x)     : Bessel function of second kind, order 0 - placeholder
    - y1(x)     : Bessel function of second kind, order 1 - placeholder
    - yn(n,x)   : Bessel function of second kind, order n - placeholder
    - cyl_bessel_i0(x) : Modified Bessel function of first kind, order 0 - placeholder
    - cyl_bessel_i1(x) : Modified Bessel function of first kind, order 1 - placeholder

    Rounding:
    - ceil(x)   : Round up to nearest integer - dedicated fp32mp2 optimization
    - floor(x)  : Round down to nearest integer - dedicated fp32mp2 optimization
    - trunc(x)  : Round toward zero - dedicated fp32mp2 optimization
    - round(x)  : Round to nearest integer - dedicated fp32mp2 optimization
    - rint(x)   : Round using current rounding mode - placeholder
    - nearbyint(x): Round using current rounding mode (no FE exception) - placeholder
    - lrint(x)  : Round to long int - placeholder
    - lround(x) : Round to long int (away from zero) - placeholder
    - llrint(x) : Round to long long int - placeholder
    - llround(x): Round to long long int (away from zero) - placeholder

    Floating-Point Manipulation:
    - fabs(x)   : Absolute value - dedicated fp32mp2 optimization (operates on (hi, lo) directly)
    - copysign(x,y): Copy sign of y to x - placeholder
    - ldexp(x,n): x * 2^n - dedicated fp32mp2
    - scalbn(x,n): x * FLT_RADIX^n - dedicated fp32mp2 (forwards to ldexp)
    - scalbln(x,n): x * FLT_RADIX^n (long exponent) - placeholder
    - frexp(x,*n): Extract mantissa and exponent - placeholder
    - modf(x,*i): Split into integer and fractional parts - placeholder
    - logb(x)   : Extract exponent - placeholder
    - ilogb(x)  : Extract exponent as int - placeholder
    - nextafter(x,y): Next representable value - placeholder

    Min/Max/Difference:
    - fmax(x,y) : Maximum - dedicated fp32mp2 optimization (lexicographic compare on (hi, lo), NaN-aware)
    - fmin(x,y) : Minimum - dedicated fp32mp2 optimization (lexicographic compare on (hi, lo), NaN-aware)
    - max(x,y)  : Maximum - dedicated fp32mp2 optimization (std::max-like, first-arg tie/unordered behavior)
    - min(x,y)  : Minimum - dedicated fp32mp2 optimization (std::min-like, first-arg tie/unordered behavior)
    - fdim(x,y) : Positive difference - placeholder

    Remainder:
    - fmod(x,y) : Floating-point remainder - dedicated fp32mp2
    - remainder(x,y): IEEE remainder - dedicated fp32mp2
    - remquo(x,y,*q): Remainder with quotient bits - placeholder

    Distance:
    - hypot(x,y): Hypotenuse sqrt(x^2+y^2) - placeholder
    - rhypot(x,y): Reciprocal hypotenuse - placeholder (host: 1/hypot)

    Vector Norms:
    - norm3d(a,b,c)    : 3D Euclidean norm sqrt(a^2+b^2+c^2) - placeholder
    - norm4d(a,b,c,d)  : 4D Euclidean norm sqrt(a^2+b^2+c^2+d^2) - placeholder
    - rnorm3d(a,b,c)   : Reciprocal 3D norm - placeholder
    - rnorm4d(a,b,c,d) : Reciprocal 4D norm - placeholder

    Classification (portable prefixed API + conditional standard overloads):
    - fpmp_isfinite(x): Test for finite value - placeholder
    - fpmp_isinf(x)   : Test for infinity - placeholder
    - fpmp_isnan(x)   : Test for NaN - placeholder
    - fpmp_signbit(x) : Test sign bit - placeholder
      Standard names isfinite/isinf/isnan/signbit are exposed only when the
      corresponding macro is not defined.

    Warp Shuffle (CUDA-only, modern __shfl_sync family): the fpmp2_t overloads
    of __shfl_sync / __shfl_xor_sync / __shfl_down_sync / __shfl_up_sync are
    thread-cooperation primitives, not math, so they live in the core header
    <cuda/__fp/fpmp.h> (available via <cuda/fpmp>), not here.

    Dedicated Implementation Details:
    -------------------------------------------------------------------------
    exp(x) for fp32mp2:
      Argument reduction: x = n*ln2 + r, |r| < ln2/2.
      Core: 14-term Taylor series for exp(r) in fp32mp2.
      Reconstruction: result * 2^n via IEEE-754 exponent manipulation.
      Accuracy: ~10^-10 - 10^-11 relative error.

    log(x) for fp32mp2:
      Range reduction: x = m*2^e, m  in  [1, sqrt2).
      Core: log(m) = 2*atanh((m-1)/(m+1)) via degree-8 minimax polynomial.
      Reconstruction: log(x) = log(m) + e*ln2 with fp32mp2 ln2 constant.
      Handles denormals via pre-scaling.

    log1p(x) for fp32mp2:
      Forwards to log(1+x) after computing 1+x via fp32mp2 2-sum, which
      preserves the small-x lo (the lo carries x exactly when |x| <= 1
      causes float hi cancellation).  log()'s accurate sub for (m-1)
      then recovers full fp32mp2 precision in the asinh-form Horner
      core, yielding the a - a^2/2 + a^3/3 - ... expansion implicitly.
      Special cases (x = -1 -> -inf, x < -1 -> NaN, +-inf) handled
      explicitly before forwarding.

    log2(x), log10(x) for fp32mp2:
      Composition over the dedicated fp32mp2 natural log:
        log2(x)  = log(x) * (1/ln 2)
        log10(x) = log(x) * (1/ln 10)
      with the reciprocal stored as a single fp32mp2 (hi+lo) constant.
      The one ff-multiply costs ~1 ulp on the lo limb; combined with
      the ~46-bit precision of __nv_fpmp2_log this still leaves >44
      bits of accuracy across the whole representable input range,
      matching the fp32mp2 noise floor.  All domain special cases
      (x<=0, +-inf, NaN) are handled inside __nv_fpmp2_log.

    exp2(x), exp10(x) for fp32mp2:
      Dedicated implementations following libdevice's "single base-2
      split" strategy (`__internal_accurate_expf_1p93ulp` paired with
      MUFU.EX2 on the reduced argument).  Pseudocode:
        t      = x * log2(base)        [exp2: t = x; exp10: t = x * log2 10]
        n      = round(t.hi)           [exact integer = binary exponent]
        r      = t - n                 [|r.hi| <= 0.5]
        2^r    via the inlined base-2 Taylor kernel
               `__nv_fp32mp2_exp2_kernel` with coefficients
               a_k = (ln 2)^k / k!     [no r * ln 2 detour, no
                                        natural-log reduction inside]
        result = 2^n * 2^r             [via split-exponent helper]
      The single integer split happens in *base-2* units, so the 2^n
      factor drops out exactly and the kernel never touches a value
      outside [-0.5, 0.5].  Compared to the earlier composition
      exp(x * ln base) -- which forced __nv_fpmp2_exp to re-derive
      n_internal from the already-amplified product, stacking two
      reduction errors -- the dedicated path keeps only one ffloat
      multiplication on the input side (and zero for exp2).  Net:
      `exp2` matches the composed path within 1 bit on the `work`
      dataset while cutting fp32mp2 call cost; `exp10` matches the
      composed path's 39-bit `work` accuracy at lower instruction
      count.  Overflow / underflow shortcuts (|x| at the float
      exponent boundary) avoid the wasted polynomial / scaling work
      when the result is already a rounded +-inf / 0.

    expm1(x) for fp32mp2:
      Strategy mirrors log1p:
        |x_hi| < 1/2:  direct Taylor series
                       expm1(x) = x + x^2 * (1/2 + x/6 + x^2/24 + ...)
                       evaluated as 12 mixed-precision Horner terms
                       (M = 4 split, top 4 plain float, bottom 8 ff).
        |x_hi| >= 1/2: exp(x) - 1 via fp32mp2 accurate sub against
                       the constant 1.0; the subtraction loses <= 1
                       bit because |exp(x) - 1| >= 0.4 in this band.
      The polynomial branch covers ~25 % of a normal-around-0 input
      distribution; warp divergence cost stays modest.  Truncation
      noise at the branch point: omitted x^13 term ~= 0.5^13/13!
      ~= 1.96*10^-14, well below fp32mp2 ulp at expm1(1/2).
      Special cases: NaN->NaN, +inf->+inf, -inf->-1, x=0->0.

    asinh(x), acosh(x), atanh(x) for fp32mp2:
      Inverse hyperbolic family -- all three reduce to a single log1p
      call with cancellation-safe arithmetic forms:
        asinh(x) = sign(x)*log1p(|x| + x^2/(sqrt(x^2+1)+1))
        acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))     for x >= 1
        atanh(x) = 0.5*sign(x)*log1p(2|x|/(1-|x|))      for |x| >= 0.25
        atanh(x) = x*(1 + y*P(y)),  y = x^2,             for |x| < 0.25
                   P(y) = 1/3 + y/5 + y^2/7 + ... + y^11/25
      The rationalized arguments avoid the (1 - cos)-style cancellation
      that breaks the textbook formulas around x = 0 (asinh, atanh)
      and x = 1 (acosh).  acosh's `(x-1)*(x+1)` factorization sidesteps
      the catastrophic cancellation that `x^2-1` would suffer near 1.
      atanh's polynomial branch covers ~25 % of the typical work range:
      it bypasses the divide-driven precision loss in the log1p form
      around 0 while keeping warp divergence small.  Large-|x| paths
      (asinh, acosh) switch to log(2|x|) above 2^25 -- early enough
      that the lo-limb ulp accumulation in the (x^2+1)/(x-1)(x+1) chain
      doesn't degrade precision, and the dropped 1/(4x^2) correction
      sits below fp32mp2 ulp throughout the asymptotic region.

    ldexp(x, n) for fp32mp2:
      result = x * 2^n built directly in fp32 via bit-cast of the
      biased exponent.  No fp64 round-trip -- important on GPUs where
      double-precision throughput is 1:32 of float.  The 2^n factor is
      split into three pieces  2^k * 2^k * 2^(n - 2k)  (k = n/3) so each
      factor's biased exponent stays inside the fp32 normal range
      [1, 254]; this avoids the spurious 0*inf = NaN that a single
      saturated scale would produce when x = 0 and n is large.  n is
      pre-clamped to +-300, which is wider than any input range that can
      actually round to something other than +-0 or +-inf: 2^300 overflows
      every fp32 (denormals included), and 2^-300 underflows every
      finite fp32.  Special cases (NaN/+-inf/+-0) propagate naturally through
      the three multiplications with the lo limb cleared.

    fmod(x, y), remainder(x, y) for fp32mp2:
      Integer-mantissa long division in the style of libdevice's
      __nv_fmod.  Each operand is decomposed into a 64-bit mantissa M
      and a binary exponent E (value = M * 2^E, M normalized to
      [2^52, 2^53)), then  Mx * 2^(Ex-Ey) mod My  is evaluated with
      exact uint64 arithmetic, chunking the left shift by 10 bits so the
      running dividend never exceeds 2^63 (~D/10 iterations).  No
      fp64 is touched, so the routine keeps full fp32 throughput.  The
      53-bit window matches IEEE double's significand: a renormalized
      fp32mp2 carries 24 (hi) + 24 (lo) significant bits whose full span
      is ~49 bits, so the uint64 mantissa is exact and the result equals
      the former ::fmod(double,double) round-trip bit-for-bit - but
      without any fp64 instructions.  (A 48-bit window rounded away the
      bottom of lo, which fmod's inherent cancellation then amplified to
      ~27-bit results.)  remainder adds the round-to-
      nearest-even step (compare 2*ia vs My, ties broken by the parity of
      the accumulated quotient) and the |x| < |y| short-circuit.  Special
      cases: NaN or x = +-inf or y = 0 -> NaN; y = +-inf -> x.

    erf(x) for fp32mp2:
      erf(x) = -expm1(-|x|*P(|x|)).
      Core: degree-24 Remez polynomial P for the argument, followed by
      expm1 via argument reduction + polynomial, all in fp32mp2.

    erfc(x) for fp32mp2:
      erfc(x) = erfcx(|x|)*exp(-x^2).
      Core: erfcx approximated by a degree-22 Chebyshev polynomial in
      the transformed variable t = 1/(1+|x|), combined with a dedicated
      exp(-x^2) evaluation using a degree-8 polynomial, all in fp32mp2.
      Uses the identity erfc(-x) = 2 - erfc(x) for negative arguments.

    normcdfinv(p) for fp32mp2:
      Rational approximation (Mike Giles coefficients).
      Variable: w = -log(4p(1-p)), computed via fp32mp2 log.
      Central branch (w < 6.125): degree-22 Horner polynomial in (w - 3.125).
      Tail 1 branch (6.125 <= w < 16): degree-18 Horner polynomial in (sqrtw - 3.25).
      Tail 2 branch (w >= 16): degree-24 Horner polynomial in (sqrtw - 7.25),
        covering all representable float inputs including denormals,
        with full fp32mp2 precision (~46+ bits) across the entire range.
      Returns +-infinity for p <= 0 or p >= 1 (standard mathematical convention).
      The icdf() wrappers clamp to +-FLT_MAX for safe Gaussian variate generation.

    icdf(uint32_t x) for fp32mp2:
      Converts a 32-bit integer uniform RNG output to a Gaussian sample.
      Mirrors x around 2^31 to map into (0, 0.5] (cuRAND convention).
      Computes p = (x + 0.5)/2^32 as an exact fp32mp2 value by splitting
      x into two 16-bit halves, then calls normcdfinv(p).

    icdf(uint64_t x) for fp32mp2:
      64-bit variant.  Keeps the top 48 bits (matching fp32mp2 precision),
      mirrors around 2^47, computes p = (x + 0.5)/2^48 by splitting into
      two 24-bit halves, then calls normcdfinv(p).

    sin(x), cos(x), sincos(x) for fp32mp2:
      Argument reduction: x = n*(pi/2) + r, |r| <= pi/4.
      Three paths based on |x|:
        Tiny (|x| < pi/4):  no reduction needed.
        Fast (|x| < 2^20):  Cody-Waite with 3-piece pi/2 (~70 bits),
          error-tracked via two_mult_fma + two_sum.
        Large (|x| >= 2^20): Payne-Hanek using integer 2/pi table (160 bits),
          returning fp32mp2 via extended-precision fixed-point conversion
          without any fp64 operations.
      Core: sin(r) via 8-term Taylor (x through x^15),
            cos(r) via 9-term Taylor (1 through x^16),
            both evaluated in fp32mp2 Horner form.
      sincos computes both kernels; sin/cos call sincos internally.
      Quadrant mapping via n mod 4 with sign/swap adjustment.

    Placeholder functions:
    - Delegate to standard double-precision system functions
    - Intended as API stubs; they do not provide full multi-precision accuracy yet

    fp64mp2 specializations:
    - When FPMP_FP128_MATH_FALLBACK=1, route through __float128 (libquadmath
      on host, CUDA fp128 intrinsics on device) for ~113-bit accuracy.
    - When FPMP_FP128_MATH_FALLBACK=0, fall back to double-precision math.
    - normcdfinv<double> uses CUDA erfcinv on device, the fp32mp2 polynomial
      on host (no standard erfcinv available).

    Configuration Macros:
    -------------------------------------------------------------------------
    - FPMP_FP64MP2_ENABLE: When 1 (default), enables fp64mp2 math function specializations.
      Set to 0 to disable double-double support and reduce code size.
    - FPMP_FP128_ENABLE: Automatically detected from compiler/CUDA. When 1, enables
      __float128 type support for conversions. Can be set to 0 to disable.
    - FPMP_FP128_MATH_FALLBACK: Controls fp64mp2 transcendental function accuracy:
      * When 1: Uses quad-precision (__float128) via libquadmath (host) or CUDA fp128
        intrinsics (device). Provides ~113-bit accuracy but requires libquadmath linkage,
        slower compilation, and larger code size.
      * When 0 (default): Falls back to double-precision math. Faster builds, smaller
        code, no extra library dependencies, but accuracy limited to ~53-bit mantissa
        for transcendental functions.
    - __FPMP_LARGE_TRIG_FP64_FALLBACK__: Controls fp32mp2 sin/cos/sincos/tan for large
      arguments (|x| >= 2^20):
      * When 0 (default): Uses dedicated pure-fp32mp2 Payne-Hanek argument reduction
        (~46 bits in the reduced argument; final precision can be lower for tan near
        singularities, where any small input quantization is amplified by tan').
        No fp64 operations, larger code.
      * When 1: Falls back to system fp64 sin/cos/tan. Smaller code path on the hot
        loop, accuracy limited by fp64 (~52 bits for sin/cos; tan still suffers
        near-singularity amplification on the fp32mp2 input).
      Note: this macro does NOT change the small-argument (|x| < 2^20) Cody-Waite
      path, which is always used and never depends on fp64.
*/
#include <cuda/__fp/fpmp.h>
#include <cuda/std/cmath>
#include <cmath>
#include <cassert>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

#if !(defined __FPMP_USE_LIB__)

/*
 * Polynomial evaluation helpers
 */
namespace fpmp
{
    /*********************************************************************
    * Mixed-precision (split-M) Horner polynomial evaluation
    * (internal building block -- namespace `fpmp`)
    *
    *   p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[N-1]*x^(N-1)
    *
    * with x given as fpmp2_t<FpType, met> and the coefficient
    * table `c[]` packed as fpmp2_t<FpType, met> in ascending order
    * of degree (c[0] = constant term, c[N-1] = leading coefficient).
    *
    * The template parameter M controls the precision split:
    *
    *   - The M HIGHEST-degree coefficients
    *         c[N-1], c[N-2], ..., c[N-M]
    *     are treated as plain FpType constants. Their `.lo()` parts
    *     are assumed to be zero (which is the natural state when the
    *     coefficient is built from a single FpType literal via the
    *     implicit `fpmp2_t(FpType)` ctor; using a ffloat literal
    *     whose `.lo()` happens to be zero -- e.g. for layout
    *     consistency -- works just as well, the `.lo()` is simply
    *     ignored in this phase). The leading M iterations run in
    *     pure FpType arithmetic:
    *         v_f = v_f * x.hi() + c[k].hi()
    *     using the "op" form r = a*b + c (no fma/mad), matching the
    *     pattern used by the hand-written math kernels (e.g. erfc).
    *
    *   - The remaining N - M LOWER-degree coefficients
    *         c[N-M-1], c[N-M-2], ..., c[0]
    *     are evaluated in full ff (float-float) arithmetic. The
    *     transition step
    *         v = v_f * x.hi() + c[N-M-1]
    *     mirrors the float*float + ff layout used in `__nv_fpmp2_erfc`
    *     (the float product gets promoted to ff via the mixed-arithmetic
    *     operator, then the remaining iterations are plain ff Horner).
    *
    * Special cases:
    *   - M == 0     : pure ff Horner (no FpType phase).
    *   - M == N     : pure FpType Horner; the FpType accumulator is
    *                  promoted to ff_t (lo == 0) at the return point.
    *
    * Use this routine as the "B" side of an A/B switch against
    * `poly_horner_comp`; the call site is identical:
    *     ffloat v = fpmp::poly_horner_mixed<M>(x, c);  // mixed standard
    *     ffloat v = fpmp::poly_horner_comp    (x, c);  // compensated
    * or dispatch via `fpmp::poly_eval<strategy, M>(x, c)` below.
    *
    * Template params:
    *   M      : number of high-degree coefficients to evaluate in
    *            plain FpType arithmetic (0 <= M <= N). Must be
    *            supplied explicitly at the call site.
    *   N      : number of coefficients (= array length, deduced).
    *            Polynomial degree is N - 1.
    *   FpType : float or double (deduced from arguments).
    *   met    : fpmp arithmetic accuracy level (deduced from arguments).
    *********************************************************************/
    template<int M, int N, typename FpType, fpmp2_accuracy met>
    __FPMP_API_DECL__ fpmp2_t<FpType, met>
    poly_horner_mixed(const fpmp2_t<FpType, met>& x,
                      const fpmp2_t<FpType, met> (&c)[N])
    {
        static_assert(N >= 2, "poly_horner_mixed requires at least 2 coefficients (degree >= 1)");
        static_assert(M >= 0, "poly_horner_mixed: M must be non-negative");
        static_assert(M <= N, "poly_horner_mixed: M must not exceed N");

        using ff_t = fpmp2_t<FpType, met>;

        if constexpr (M == 0)
        {
            // Pure ff Horner -- no FpType phase.
            ff_t v = c[N - 1];
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int k = N - 2; k >= 0; --k) {
                v = v * x + c[k];
            }
            return v;
        }
        else
        {
            // FpType phase: M iterations consuming c[N-1] ... c[N-M].
            const FpType xh = x.hi();
            FpType v_f      = c[N - 1].hi();
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int k = N - 2; k >= N - M; --k) {
                v_f = v_f * xh + c[k].hi();
            }

            if constexpr (M == N)
            {
                // No ff phase at all -- promote the FpType result.
                return ff_t(v_f);
            }
            else
            {
                // Transition step: (float * float) + ff -> ff
                // (the mixed-type operator+ promotes the FpType product
                // to ff_t with .lo() == 0 before adding c[N-M-1].)
                ff_t v = v_f * xh + c[N - M - 1];
            #if defined(__CUDA_ARCH__)
                #pragma unroll
            #endif
                for (int k = N - M - 2; k >= 0; --k) {
                    v = v * x + c[k];
                }
                return v;
            }
        }
    } // poly_horner_mixed

    /*********************************************************************
    * Compensated Horner polynomial evaluation for fp32mp2 / fp64mp2
    * (internal building block -- namespace `fpmp`)
    *
    *   p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[N-1]*x^(N-1)
    *
    * with x and c[k] given as fpmp2_t<FpType, met> (ascending order
    * of degree, c[0] = constant term, c[N-1] = leading coefficient).
    * The bulk of the work runs in single-precision FpType arithmetic
    * with an error-tracking ("compensated") Horner inner loop, then two
    * correction sweeps fold in the c[k].lo terms and the x.lo * p'(x.hi)
    * cross term.
    *
    * Layout:
    *   Phase 0:  (optional)  plain FpType Horner over the M highest-degree
    *             coefficients c[N-1].hi() ... c[N-M].hi() -- error not
    *             tracked. Use M > 0 to skip compensation on iterations
    *             whose values are small enough that the rounding error
    *             they would contribute is dominated by the polynomial
    *             truncation noise; mirrors the M-split of
    *             `poly_horner_mixed<M>`. Top M coefficients are required
    *             to have c[k].lo() == 0 (the natural state for plain
    *             FpType literals built via `fpmp2_t(FpType)` ctor).
    *   Phase 1:  compensated Horner over the remaining (N-M) coefficients
    *             c[N-M-1] ... c[0] -- running FpType acc + FpType err such
    *             that acc + err equals Sum_{k<=N-M-1} c[k].hi * x.hi^k + acc0
    *             (where acc0 is Phase 0's output) to ~2*FpType precision
    *             (Graillat-Langlois-Louvet, "Compensated Horner", 2005).
    *   Phase 2a: + Sum_{k<=N-M-1} c[k].lo * x.hi^k   (plain FpType Horner;
    *             top M iterations are skipped, their .lo() == 0).
    *   Phase 2b: + x.lo * p'(x.hi)                (plain FpType Horner
    *             over the full derivative -- all N-1 terms regardless
    *             of M, because the high-degree derivative terms carry
    *             the x.lo correction signal, not rounding noise).
    *   Phase 3:  fast_two_sum(acc, err+corr) -> (hi, lo) ffloat
    *
    * Coefficients with c[k].lo == 0 (e.g. those built from a pure
    * FpType constant via the implicit `fpmp2_t(FpType)` ctor)
    * fold cleanly: their Phase 2a iterations are no-ops, and Phase 2b's
    * `(FpType)k * c[k].hi()` constant evaluates at compile time inside
    * the unrolled loop. This makes the helper a uniform way to express
    * mixed-precision polynomials.
    *
    * Special cases:
    *   - M == 0 : full compensated Horner over all N coefficients
    *              (bit-identical to the un-split implementation).
    *   - M == N : pure plain FpType Horner over all N coefficients
    *              with the x.lo * p'(x.hi) cross-term correction --
    *              cheaper than full compensated, more accurate than
    *              `poly_horner_mixed<N>` (which drops the cross term).
    *
    * Template params:
    *   M      : number of HIGH-degree coefficients to evaluate in
    *            plain FpType (no error tracking).  0 <= M <= N.
    *            Defaults to 0 (= full compensated Horner).
    *   N      : number of coefficients (= array length, polynomial
    *            degree is N-1). Deduced from the coefficient array.
    *   FpType : float or double (deduced from arguments)
    *   met    : fpmp arithmetic accuracy level (deduced from arguments)
    *********************************************************************/
    template<int M = 0, int N, typename FpType, fpmp2_accuracy met>
    __FPMP_API_DECL__ fpmp2_t<FpType, met>
    poly_horner_comp(const fpmp2_t<FpType, met>& x,
                     const fpmp2_t<FpType, met> (&c)[N])
    {
        static_assert(N >= 2, "poly_horner_comp requires at least 2 coefficients (degree >= 1)");
        static_assert(M >= 0, "poly_horner_comp: M must be non-negative");
        static_assert(M <= N, "poly_horner_comp: M must not exceed N");

        const FpType xh = x.hi();
        const FpType xl = x.lo();

        // === Phase 0: M-1 plain FpType Horner steps (no error tracking) ===
        FpType acc = c[N - 1].hi();
        if constexpr (M >= 2) {
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int k = N - 2; k >= N - M; --k) {
                acc = acc * xh + c[k].hi();
            }
        }

        // === Phase 1: N-M compensated Horner steps ===
        FpType err = static_cast<FpType>(0);
        if constexpr (M < N) {
            // For M == 0 the init handled c[N-1], so compensated loop
            // starts at c[N-2]; for M >= 1 Phase 0 handled c[N-1]..c[N-M],
            // so compensated loop starts at c[N-M-1].
            constexpr int comp_start = (M == 0) ? (N - 2) : (N - M - 1);
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int k = comp_start; k >= 0; --k)
            {
                const FpType ckh = c[k].hi();

                // two_mult_fma: P + pi == xh * acc  (exact)
                FpType P  = mul_rn(xh, acc);
                FpType pi = fma_rn(xh, acc, -P);

                // two_sum: S + sg == P + ckh  (exact, no magnitude assumption)
                FpType S  = add_rn(P, ckh);
                FpType bb = sub_rn(S, P);
                FpType t  = sub_rn(S, bb);
                FpType u  = sub_rn(P, t);
                FpType v  = sub_rn(ckh, bb);
                FpType sg = add_rn(u, v);

                err = fma_rn(xh, err, add_rn(pi, sg));
                acc = S;
            }
        }

        // === Phase 2a: contribution of c[k].lo (top M iterations skipped) ===
        FpType corr = static_cast<FpType>(0);
        if constexpr (M < N) {
            // For M == 0 we visit all N coefficients (k = N-1 .. 0);
            // for M >= 1 we skip the top M (their .lo() == 0 by contract).
            constexpr int lo_start = (M == 0) ? (N - 1) : (N - M - 1);
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int k = lo_start; k >= 0; --k)
            {
                corr = fma_rn(xh, corr, c[k].lo());
            }
        }

        // === Phase 2b: x.lo * p'(x.hi)  (full derivative, all N-1 terms) ===
        FpType dp = static_cast<FpType>(0);
    #if defined(__CUDA_ARCH__)
        #pragma unroll
    #endif
        for (int k = N - 1; k >= 1; --k)
        {
            dp = fma_rn(xh, dp, mul_rn(static_cast<FpType>(k), c[k].hi()));
        }
        corr = fma_rn(xl, dp, corr);

        // === Phase 3: combine into normalized ff ===
        FpType lo  = add_rn(err, corr);
        FpType rhi = add_rn(acc, lo);
        FpType rlo = sub_rn(lo, sub_rn(rhi, acc));
        return fpmp2_t<FpType, met>(rhi, rlo);
    } // poly_horner_comp

    /*********************************************************************
    * Polynomial-evaluation strategy selector for `poly_eval`.
    *
    * Listed kernels currently route to a Horner backend; future
    * additions (e.g. factorized / Estrin / Knuth-Eve evaluation)
    * are expected to slot in as new enumerators here without
    * changing the dispatcher signature.
    *********************************************************************/
    enum class poly_method
    {
        horner_mixed = 0,  // mixed-precision Horner (`poly_horner_mixed<M>`)
        horner_comp  = 1,  // compensated  Horner    (`poly_horner_comp`)
    };

    /*********************************************************************
    * Polynomial-evaluation dispatcher
    * (internal building block -- namespace `fpmp`)
    *
    * Thin compile-time switch between the polynomial-evaluation kernels:
    *
    *     fpmp::poly_eval<poly_method::horner_mixed, M>(x, c)
    *         -> fpmp::poly_horner_mixed<M>(x, c)
    *
    *     fpmp::poly_eval<poly_method::horner_comp,  M>(x, c)
    *         -> fpmp::poly_horner_comp <M>(x, c)
    *
    * Both backends share the same M-split semantics: the M HIGHEST-degree
    * coefficients are evaluated in plain FpType arithmetic (with the
    * convention that their .lo() == 0), and the remaining N-M coefficients
    * are evaluated in the precision-preserving regime of the selected
    * backend (ff-Horner for `horner_mixed`, error-tracking Horner for
    * `horner_comp`).  Switching between the two costs nothing more than
    * editing the strategy enumerator at the call site.
    *
    * Template params:
    *   strategy : poly_method::horner_mixed or poly_method::horner_comp
    *              (additional non-Horner methods may be added later).
    *   M        : split parameter forwarded to both backends.
    *              Defaults to 0 (= pure ff Horner for `horner_mixed`,
    *              = full compensated Horner for `horner_comp`).
    *   N        : number of coefficients (deduced).
    *   FpType   : float or double (deduced).
    *   met      : fpmp arithmetic accuracy level (deduced).
    *********************************************************************/
    template<poly_method strategy, int M = 0,
             int N, typename FpType, fpmp2_accuracy met>
    __FPMP_API_DECL__ fpmp2_t<FpType, met>
    poly_eval(const fpmp2_t<FpType, met>& x,
              const fpmp2_t<FpType, met> (&c)[N])
    {
        if constexpr (strategy == poly_method::horner_mixed) {
            return poly_horner_mixed<M>(x, c);
        } else /* poly_method::horner_comp */ {
            return poly_horner_comp <M>(x, c);
        }
    } // poly_eval
} // namespace fpmp

    /*
    * --------------------------------------------------------------------
    * Exponential function (fp32mp2)
    * --------------------------------------------------------------------
    * Compute exp(x) for float-float (fp32mp2) precision
    * 
    * Algorithm:
    *   1. Argument reduction: x = n*ln(2) + r, where |r| < ln(2)/2
    *   2. Compute exp(r) using 14-term Taylor series with Horner's method
    *   3. Scale result by 2^n using IEEE-754 bit manipulation
    *
    * Range reduction ensures that the Taylor series converges quickly since |r| < ln(2)/2 ~= 0.35.
    * With 14 terms and float-float arithmetic, this achieves approximately 10^-10 to 10^-11 relative accuracy.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_exp (const FpType x_hi, 
                                                const FpType x_lo, 
                                                FpType*      res_hi, 
                                                FpType*      res_lo)
    {
        using namespace fpmp;
        using ffloat = fp32mp2_low;
     
        // Constants as C99 hex floating-point literals - split via constexpr constructor
        constexpr float  inv_ln2(0x1.715476p+0f);     // 1/ln(2)
        constexpr float  shift_bias(12582912.0f+127.0f*2.0f); // 127.0f*2.0f is the bias for the exponent
        constexpr ffloat ln2(0x1.62e42fefa39efp-1);  // ln(2)
        
        /* Taylor series coefficients 1/k! for k = 1..13.
         *
         * The polynomial evaluated below is
         *   p(r) = c1 + c2*r + c3*r^2 + ... + c13*r^12
         * with p(r) = (exp(r) - 1) / r, so exp(r) = r*p(r) + 1.
         *
         * Layout for the mixed-precision Horner dispatcher:
         *   - c1, c2 are plain `float` and live at the LOW-degree end
         *     of p(r), which is the wrong end for the M-split.  We
         *     keep them as scalar constants and fold them in via two
         *     trailing Horner steps OUTSIDE `poly_eval`.
         *   - c3..c13 form an 11-coefficient table consumed by
         *     `poly_eval<horner_mixed, 6>`: 6 high-degree entries
         *     (c8..c13) are plain float literals, the remaining
         *     5 (c3..c7) carry an ff `.lo()` part.
         */
        constexpr float c1(0x1.0p+0);
        constexpr float c2(0x1.0p-1);

        constexpr ffloat exp_c[11] = {
            ffloat(0x1.5555555555555p-3),   // [ 0] (= c3,  constant of q)
            ffloat(0x1.5555555555555p-5),   // [ 1] (= c4)
            ffloat(0x1.1111111111111p-7),   // [ 2] (= c5)
            ffloat(0x1.6c16c16c16c17p-10),  // [ 3] (= c6)
            ffloat(0x1.a01a01a01a01ap-13),  // [ 4] (= c7,  last ff term)
            /* high-order M = 6 entries: .lo() == 0 by construction */
            ffloat(0x1.a01a0p-16f),         // [ 5] (= c8)
            ffloat(0x1.71de4p-19f),         // [ 6] (= c9)
            ffloat(0x1.27e50p-22f),         // [ 7] (= c10)
            ffloat(0x1.ae646p-26f),         // [ 8] (= c11)
            ffloat(0x1.1eedap-29f),         // [ 9] (= c12)
            ffloat(0x1.6125p-33f)           // [10] (= c13, leading)
        };
        
        // Overflow threshold for single precision: ln(FLT_MAX) ~= 88.7228 = 0x1.62e430p+6
        if (x_hi > 0x1.62e430p+6f) 
        {
            *res_hi = __builtin_huge_valf();
            *res_lo = 0.0f;
            return;
        }

        // Underflow threshold for single precision: ln(FLT_MIN) ~= -87.3365 = -0x1.5d589ep+6
        if (x_hi < -0x1.5d589ep+6f) 
        {
            *res_hi = 0.0f;
            *res_lo = 0.0f;
            return;
        }
        
        ffloat    x(x_hi, x_lo);

        // Step 1: Argument reduction: x = n*ln(2) + r, where |r| < ln(2)/2
        float t  = x_hi*inv_ln2 + shift_bias;

        // Shift the exponent by 23 bits to get the scale as fp32 value
        int32_t scale = internal_bit_cast<int32_t>(t);
        scale <<= 23;

        // Split the scale into high and low parts
        uint32_t scale_lo = scale >> 1;
        scale_lo &= 0x7F800000u;
        scale    -= scale_lo;

        // Cast the scales to fp32 values
        float fscale    = internal_bit_cast<float>(scale);
        float fscale_lo = internal_bit_cast<float>(scale_lo);

        // Compute the reduced argument r = x - n*ln(2)
        float tt = t - shift_bias;
        ffloat r = x - ffloat(static_cast<float>(tt)) * ln2;
        r        = renormalize(r);

        // Scale the reduced argument by the low part of the scale
        ffloat r_scale = r * fscale_lo;

        // Evaluate q(r) = c3 + c4*r + c5*r^2 + ... + c13*r^10 via the
        // mixed-precision dispatcher (6 high-order terms in plain float,
        // remaining 5 in ff).
        //
        // Note: the dispatcher's transition step uses float*float + ff
        // (matching the erfc-style layout of `poly_horner_mixed`),
        // whereas the previous hand-rolled chain used float*ff + ff at
        // the c8->c7 boundary. The numerical difference is below 1 ULP
        // at the polynomial value, well inside the Taylor truncation
        // noise floor.
        ffloat p = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 6>(r, exp_c);

        // Fold in the low-degree float coefficients c1, c2 outside the
        // dispatcher (they live at the wrong end of the polynomial for
        // the M-split optimisation).
        p = p * r + c2;
        p = p * r + c1;

        p = p * r_scale + fscale_lo;

        // Scale the result by the high part of the scale
        p = p * ffloat(fscale);
        ffloat result = renormalize(p);
        
        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_exp

    /*
    * --------------------------------------------------------------------
    * Natural logarithm log(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Natural logarithm, fully implemented in fp32mp2.
    * Range reduction: x = m * 2^e with m in [1, sqrt(2)].
    * Core: log(m) = 2*atanh((m-1)/(m+1)) via degree-8 minimax polynomial.
    * Reconstruction: log(x) = log(m) + e*ln(2).
    * All arithmetic in fp32mp2_low; no fp64 operations.
    * Handles denormals via pre-scaling by 2^24.
    * Does not handle NaN, +-0, or negative inputs.
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_log (const FpType x_hi,
                                                const FpType x_lo,
                                                FpType*      res_hi,
                                                FpType*      res_lo)
    {
        using namespace fpmp;
        using ffloat = fp32mp2_low;

        constexpr ffloat ln2(0x1.62e42fefa39efp-1);  // ln(2)

        /* Minimax polynomial coefficients for the atanh series:
         * log(m) = u * (1 + v*(c1 + v*(c2 + ... + v*c8)))
         * where u = 2*(m-1)/(m+1), v = u^2
         * Coefficients: ~= 1/((2k+1)*4^k) for k = 1..8
         *
         * Packed for the mixed-precision dispatcher in ascending degree
         * (atanh_c[0] = constant c1, atanh_c[7] = leading c8).  The
         * 5 highest-degree entries (atanh_c[3..7] = c4..c8) are plain
         * float literals (.lo() == 0 by construction), so
         * poly_eval<horner_mixed, 5> evaluates them in float and
         * transitions to ff arithmetic at atanh_c[2] = c3 -- exactly
         * matching the previous hand-rolled float*float + ff step.
         */
        constexpr ffloat atanh_c[8] = {
            ffloat(0x1.5555555555554p-4),   // [0] (= c1, constant of q)
            ffloat(0x1.999999999a3c4p-7),   // [1] (= c2)
            ffloat(0x1.24924923be72dp-9),   // [2] (= c3, last ff term)
            /* high-order M = 5 entries: .lo() == 0 by construction */
            ffloat(0x1.c71c72p-12f),        // [3] (= c4)
            ffloat(0x1.745cbap-14f),        // [4] (= c5)
            ffloat(0x1.3b266ap-16f),        // [5] (= c6)
            ffloat(0x1.0ee258p-18f),        // [6] (= c7)
            ffloat(0x1.1380b4p-20f)         // [7] (= c8, leading)
        };

        /* Range reduction: x = m * 2^e, m in [1, sqrt(2)] */
        float  a_hi  = x_hi;
        float  a_lo  = x_lo;
        int    e_adj = 0;

        /* Normalize denormals: scale by 2^24 to make the exponent field nonzero */
        uint32_t xbits = internal_bit_cast<uint32_t>(a_hi);
        if ((xbits & 0x7F800000u) == 0u) 
        {
            a_hi  = a_hi * 0x1.0p24f;
            a_lo  = a_lo * 0x1.0p24f;
            e_adj = -24;
            xbits = internal_bit_cast<uint32_t>(a_hi);
        }

        int e = static_cast<int>((xbits >> 23) & 0xFFu) - 127 + e_adj;

        /* m_hi in [1, 2) by replacing exponent field with bias 127 */
        float m_hi = internal_bit_cast<float>((xbits & 0x007FFFFFu) | 0x3F800000u);

        /* Scale a_lo by 2^(-e_orig) where e_orig = e - e_adj,
         * using split factors to stay in normal float range */
        int e_orig = e - e_adj;
        int e2     = e_orig / 2;
        float s1   = internal_bit_cast<float>(static_cast<uint32_t>(127 - e2) << 23);
        float s2   = internal_bit_cast<float>(static_cast<uint32_t>(127 - (e_orig - e2)) << 23);
        float m_lo = a_lo * s1 * s2;

        ffloat m = renormalize(ffloat(m_hi, m_lo));

        /* If m > sqrt(2), halve m and increment e */
        if (m.hi() > 0x1.6a09e6p+0f) 
        {
            m = m * 0.5f;
            e = e + 1;
        }

        /* u = 2*(m-1)/(m+1), v = u^2
         * Use accurate subtraction for (m - 1) to handle catastrophic
         * cancellation when m ~= 1 (x near a power of 2).
         */
        ffloat f = sub<fpmp2_accuracy::high>(m, 1.0f);
        ffloat g = m + 1.0f;
        ffloat u = f / g;
        u = u + u;
        u = renormalize(u);
        ffloat v = u * u;

        /* Horner evaluation: q(v) = c1 + c2*v + c3*v^2 + ... + c8*v^7
         * via the mixed-precision dispatcher (5 high-order terms in
         * plain float, remaining 3 in ff).  The dispatcher transition
         * `qf * v.hi() + c3` matches the previous hand-written step
         * bit-for-bit, so this refactor is numerically identical to
         * the previous implementation.
         */
        ffloat q = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 5>(v, atanh_c);

        /* log(m) = u + u*v*q(v) */
        q = q * v;
        ffloat log_m = q * u + u;

        /* log(x) = log(m) + e*ln(2) */
        ffloat result = renormalize(log_m + ffloat(static_cast<float>(e)) * ln2);

        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_log

    /*
    * --------------------------------------------------------------------
    * Natural logarithm of (1 + x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Strategy:
    *   - Small |x_hi| (< 1/16):  direct Taylor series in (x_hi, x_lo),
    *     keeping the full fp32mp2 input intact.
    *   - Otherwise:              forward to `__nv_fpmp2_log` at (1+x).
    *
    * Why a small-|x| branch is necessary:
    *   The forward-to-log path runs the input through (1+x) packing,
    *   accurate-sub `m - 1`, fast-div `f/g`, and fast-mul Horner.
    *   Each step introduces ~1 ulp of relative error in the *lo* limb
    *   of the intermediate.  These ulps are insignificant when the
    *   final |result| is order |x|, but as |x| -> 0 the absolute lo
    *   error stays roughly constant while |result| ~= |x| shrinks, so
    *   the relative error blows up: at |x| ~ 3e-8 the chain leaves
    *   ~24 bits of accuracy (rel_err ~ 5e-8) -- barely fp32 quality.
    *
    *   The Taylor series
    *       log1p(x) = x * (1 - x/2 + x^2/3 - x^3/4 + ...)
    *                = x + x^2 * T(x),  T(x) = -1/2 + x/3 - x^2/4 + ...
    *   never inflates rel error: x is preserved verbatim and the
    *   correction `x^2 * T` is order x^2, so its ulps cost rel_err of
    *   order x * ulp ~= negligible against |x|.  This restores full
    *   fp32mp2 precision (~46 bits) for arbitrarily small |x|.
    *
    * Branch point 1/16 = 2^-4 keeps the polynomial narrow (covers ~6%
    * of the typical work range) so most threads stay on the log path,
    * limiting warp divergence; at |x| = 1/16 the omitted x^12 term
    * contributes 0.0625^12/14 ~= 2.6*10^-16, well below fp32mp2 ulp at
    * log1p(1/16) ~= 0.061.
    *
    * Special-case handling (mirrors libm log1p semantics):
    *   - NaN propagation (any NaN component -> NaN result).
    *   - +inf      -> +inf.
    *   - x = -1    -> -inf (1 + x = 0 exactly).
    *   - x < -1    -> NaN  (1 + x < 0).
    *   - -inf      -> NaN  (1 + (-inf) = -inf, log of negative).
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_log1p (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_log1p is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* NaN propagation: any NaN component -> NaN result. */
        if (x_hi != x_hi || x_lo != x_lo)
        {
            const float nan_val = x_hi + x_lo;
            *res_hi = nan_val;
            *res_lo = nan_val;
            return;
        }

        /* +inf input -> +inf. */
        if (x_hi == __builtin_huge_valf())
        {
            *res_hi = __builtin_huge_valf();
            *res_lo = 0.0f;
            return;
        }

        /* -inf input -> log(-inf) = NaN. */
        if (x_hi == -__builtin_huge_valf())
        {
            *res_hi = __builtin_nanf("");
            *res_lo = __builtin_nanf("");
            return;
        }

        /* Small-|x| polynomial branch.  See header comment for the
         * rationale: bypasses the (1+x) -> log() pipeline whose
         * accumulated lo-ulp errors dominate as |x| -> 0.  Domain check
         * for x = -1 / x < -1 still applies (covered by the |x|<1/16
         * threshold trivially: any x in this range is well above -1). */
        const float abs_hi = (x_hi < 0.0f) ? -x_hi : x_hi;
        constexpr float LOG1P_BRANCH_POINT = 0.0625f;  /* 1/16 = 2^-4 */
        if (abs_hi < LOG1P_BRANCH_POINT)
        {
            /* T(x) = sum_{k>=0} (-1)^k * x^k / (k+2),
             *   T[0] = -1/2, T[1] = +1/3, ..., T[11] = +1/13.
             * Layout for poly_eval<horner_mixed, M=4>: bottom 8 entries
             * are full ff (their contributions stay above fp32mp2 ulp at
             * the branch point), top 4 entries are plain float (.lo == 0
             * by construction; their contributions sit below 0.5 ulp). */
            constexpr ffloat log1p_poly_c[12] = {
                ffloat(-5.0e-1),                  /* [ 0] -1/2 (constant) */
                ffloat( 3.3333333333333333e-1),   /* [ 1] +1/3 */
                ffloat(-2.5e-1),                  /* [ 2] -1/4 */
                ffloat( 2.0e-1),                  /* [ 3] +1/5 */
                ffloat(-1.6666666666666666e-1),   /* [ 4] -1/6 */
                ffloat( 1.4285714285714285e-1),   /* [ 5] +1/7 */
                ffloat(-1.25e-1),                 /* [ 6] -1/8 */
                ffloat( 1.1111111111111111e-1),   /* [ 7] +1/9  (last ff term) */
                /* high-order M = 4 entries: .lo() == 0 by construction */
                ffloat(-1.0e-1f),                 /* [ 8] -1/10 */
                ffloat( 9.0909094e-2f),           /* [ 9] +1/11 */
                ffloat(-8.3333336e-2f),           /* [10] -1/12 */
                ffloat( 7.6923080e-2f),           /* [11] +1/13 (leading) */
            };

            ffloat x      (x_hi, x_lo);
            ffloat x2     = x * x;
            ffloat T      = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 4>(x, log1p_poly_c);
            ffloat result = renormalize(x + x2 * T);
            *res_hi = result.hi();
            *res_lo = result.lo();
            return;
        }

        /* Compute (1 + x) in fp32mp2 with accurate add: this preserves
         * the residual to fp32mp2 precision even when 1 + x.hi cancels
         * to a small magnitude (i.e., x close to -1).  The lo of the
         * result captures the rounding loss that a plain fast 2-sum
         * folds into the leading term and then quantizes to float
         * precision in the subsequent operations. */
        ffloat sum = add<fpmp2_accuracy::high>(ffloat(1.0f), ffloat(x_hi, x_lo));

        /* (1 + x) < 0  -> NaN. */
        if (sum.hi() < 0.0f)
        {
            *res_hi = __builtin_nanf("");
            *res_lo = __builtin_nanf("");
            return;
        }

        /* (1 + x) == 0 (i.e., x == -1) -> log(0) = -inf. */
        if (sum.hi() == 0.0f && sum.lo() == 0.0f)
        {
            *res_hi = -__builtin_huge_valf();
            *res_lo = 0.0f;
            return;
        }

        /* Edge case: sum.hi == 0, sum.lo > 0 (x = -1 + tiny).  Promote
         * lo to hi so log() sees a normalized argument. */
        if (sum.hi() == 0.0f)
        {
            if (sum.lo() < 0.0f)
            {
                *res_hi = __builtin_nanf("");
                *res_lo = __builtin_nanf("");
                return;
            }
            sum = ffloat(sum.lo(), 0.0f);
        }

        /* Forward to dedicated fp32mp2 log. */
        __nv_fpmp2_log<float>(sum.hi(), sum.lo(), res_hi, res_lo);
    } // __nv_fpmp2_log1p

    /*
    * --------------------------------------------------------------------
    * Base-2 logarithm log2(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Composition over the dedicated fp32mp2 natural log:
    *     log2(x) = log(x) * (1/ln(2))
    * with (1/ln(2)) carried as an fp32mp2 constant (hi+lo).  The single
    * ff-multiply costs ~1 ulp on the lo limb; combined with the ~46-bit
    * precision of __nv_fpmp2_log this still leaves >44 bits of accuracy
    * across the whole representable input range, which matches the
    * fp32mp2 noise floor (cf. log/log1p reports in the test suite).
    *
    * All special cases (x<=0, NaN, +inf) are handled inside
    * __nv_fpmp2_log<float>; this wrapper only scales the result.
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_log2 (const FpType x_hi,
                                                 const FpType x_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_log2 is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* 1/ln(2) ~= 1.4426950408889634073599... */
        constexpr ffloat inv_ln2(0x1.71547652b82fep+0);

        float l_hi, l_lo;
        __nv_fpmp2_log<float>(x_hi, x_lo, &l_hi, &l_lo);

        /* Propagate non-finite outputs (NaN, +-inf) unchanged: a multiply
         * by a finite constant would still yield the same kind for +-inf,
         * but NaN composition is cleanest with an explicit short-circuit
         * (avoids an unnecessary mul that could quiet a signaling NaN
         * on some platforms). */
        if (l_hi != l_hi || l_hi == __builtin_huge_valf() || l_hi == -__builtin_huge_valf())
        {
            *res_hi = l_hi;
            *res_lo = (l_hi != l_hi) ? l_hi : 0.0f;
            return;
        }

        ffloat result = renormalize(ffloat(l_hi, l_lo) * inv_ln2);
        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_log2

    /*
    * --------------------------------------------------------------------
    * Base-10 logarithm log10(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Composition over the dedicated fp32mp2 natural log:
    *     log10(x) = log(x) * (1/ln(10))
    * with (1/ln(10)) carried as an fp32mp2 constant.  Same accuracy
    * trade-off as log2; see __nv_fpmp2_log2 header comment.
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_log10 (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_log10 is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* 1/ln(10) ~= 0.4342944819032518276511289... */
        constexpr ffloat inv_ln10(0x1.bcb7b1526e50ep-2);

        float l_hi, l_lo;
        __nv_fpmp2_log<float>(x_hi, x_lo, &l_hi, &l_lo);

        if (l_hi != l_hi || l_hi == __builtin_huge_valf() || l_hi == -__builtin_huge_valf())
        {
            *res_hi = l_hi;
            *res_lo = (l_hi != l_hi) ? l_hi : 0.0f;
            return;
        }

        ffloat result = renormalize(ffloat(l_hi, l_lo) * inv_ln10);
        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_log10

    /*
    * --------------------------------------------------------------------
    * Internal helper: 2^r kernel for fp32mp2, |r| <= 0.5
    * --------------------------------------------------------------------
    * Evaluates  2^r  directly with a 13-term Taylor polynomial whose
    * coefficients absorb the ln(2) scaling:
    *
    *     2^r = exp(r * ln 2) = Sum_{k>=0}  a_k * r^k,   a_k = (ln 2)^k / k!
    *
    * This is the software analogue of libdevice's MUFU.EX2 instruction:
    * the polynomial argument is the *base-2* reduced argument, so no
    * intermediate r * ln(2) multiplication sits between the reduction
    * and the polynomial.  Compared to evaluating exp(r * ln 2) via the
    * natural-log Taylor inside __nv_fpmp2_exp, this kernel saves:
    *   - the y = r * ln 2 fp32mp2 product (1 ULP),
    *   - the trivial-reduction housekeeping inside __nv_fpmp2_exp
    *     (the y/ln 2 round, the n*ln 2 subtraction, the 2^0 split-scale
    *     dance) -- none of which mathematically contribute when |y| is
    *     already inside the post-reduction window, but each adds 1-2
    *     ULP of rounding noise to the lo limb.
    *
    * Coefficient layout for the mixed-precision Horner dispatcher:
    *   - a1 is folded outside `poly_eval` (it's the wrong end of the
    *     polynomial for the M-split optimisation) and kept as ffloat:
    *     a single fp32-rounded a1 would lose ~2 ULPs absolute, which
    *     directly pollutes the result lo since the final fold step is
    *     `p * r + a1` with `p * r` already at the magnitude of a1.
    *   - a2..a7 are the 5 lowest-degree coeffs evaluated in ff
    *     arithmetic by `poly_eval`'s ff tail; their .lo() bits matter
    *     because a_k * r^k stays above the fp32mp2 noise floor through
    *     k <= 7 at |r| = 0.5.
    *   - a8..a13 are the M = 6 highest-degree coeffs evaluated in
    *     float-only Horner steps (the dispatcher's "horner_mixed"
    *     inner loop); their fp32 round-off contributes <2^-46 to the
    *     final polynomial value at |r| <= 0.5.
    *
    * Truncation error at |r| = 0.5:
    *   |a_13 * r^13| <= (ln 2 / 2)^13 / 13! ~= 1.7 * 10^-16, well below
    *   the fp32mp2 ulp floor.
    * --------------------------------------------------------------------
    */
    __FPMP_INTERNAL_DECL__ fp32mp2_low __nv_fp32mp2_exp2_kernel(fp32mp2_low r)
    {
        using ffloat = fp32mp2_low;

        /* a1..a13 = (ln 2)^k / k!,  ordered low -> high degree.
         * 7 low-degree ff entries (carry .lo()) + 6 high-degree float
         * entries (M = 6) consumed by poly_eval's float-only inner loop. */
        constexpr ffloat exp2_c[13] = {
            ffloat(0x1.62e42fefa39efp-1),     /* [ 0] a1 = ln 2  */
            ffloat(0x1.ebfbdff82c58ep-3),     /* [ 1] a2  */
            ffloat(0x1.c6b08d704a0bfp-5),     /* [ 2] a3  */
            ffloat(0x1.3b2ab6fba4e77p-7),     /* [ 3] a4  */
            ffloat(0x1.5d87fe78a6730p-10),    /* [ 4] a5  */
            ffloat(0x1.430912f86c786p-13),    /* [ 5] a6  */
            ffloat(0x1.ffcbfc588b0c5p-17),    /* [ 6] a7  */
            /* high-degree M = 6, zero .lo() by construction */
            ffloat(0x1.62c022p-20f),          /* [ 7] a8  */
            ffloat(0x1.b5253ep-24f),          /* [ 8] a9  */
            ffloat(0x1.e4cf52p-28f),          /* [ 9] a10 */
            ffloat(0x1.e8cac8p-32f),          /* [10] a11 */
            ffloat(0x1.c3bd66p-36f),          /* [11] a12 */
            ffloat(0x1.816194p-40f)           /* [12] a13 */
        };

        /* G(r) = a1 + a2*r + a3*r^2 + ... + a13*r^12 ~= (2^r - 1)/r */
        ffloat p = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 6>(r, exp2_c);

        /* Close with the implicit a0 = 1 constant:
         *   2^r = 1 + r * G(r) */
        p = p * r + ffloat(1.0f, 0.0f);
        return p;
    }

    /*
    * --------------------------------------------------------------------
    * Internal helper: 10^r kernel for fp32mp2, |r| <= log10(2)/2 ~= 0.1505
    * --------------------------------------------------------------------
    * Evaluates 10^r directly with a 13-term Taylor polynomial whose
    * coefficients absorb the ln(10) scaling:
    *
    *     10^r = exp(r * ln 10) = Sum_{k>=0}  b_k * r^k,   b_k = (ln 10)^k / k!
    *
    * This is the natural companion to __nv_fp32mp2_exp2_kernel for the
    * libdevice-style "exp10(x) = 2^n * 10^(x - n*log10(2))" reduction.
    * Because |r| <= log10(2)/2 ~= 0.151 (vs. 0.5 for the base-2 kernel),
    * the Horner chain accumulates noticeably less rounding noise even
    * though the b_k coefficients are larger (peak ratio |b_k r^k|
    * matches |a_k (0.5)^k| ~= exp_k since (ln 10 * 0.151)^k = (ln 2 *
    * 0.5)^k).
    *
    * Coefficient layout:
    *   - b1..b6 (the 6 lowest-degree terms) kept as ff: their
    *     fp32-rounding error at |r| = log10(2)/2 lifts the polynomial
    *     value above the fp32mp2 noise floor and must carry .lo() bits.
    *   - b7..b13 (the M = 7 highest) are plain float (zero .lo());
    *     their fp32 round-off contributes < 2^-46 to the polynomial.
    *   - Implicit b0 = 1 is folded via the final  p * r + 1  step.
    *
    * Truncation error at |r| = log10(2)/2:
    *   |b_13 * r^13| = (ln 10)^13 (log10(2)/2)^13 / 13! ~= 1.7 * 10^-16,
    *   below the fp32mp2 ulp floor.
    * --------------------------------------------------------------------
    */
    __FPMP_INTERNAL_DECL__ fp32mp2_low __nv_fp32mp2_exp10_kernel(fp32mp2_low r)
    {
        using ffloat = fp32mp2_low;

        /* b_k = (ln 10)^k / k!,  ordered low -> high degree.
         * 6 low-degree ff entries (carry .lo()) + 7 high-degree float
         * entries (M = 7) consumed by poly_eval's float-only inner loop. */
        constexpr ffloat exp10_c[13] = {
            ffloat(0x1.26bb1bbb55516p+1),     /* [ 0] b1 = ln 10  */
            ffloat(0x1.53524c73cea6ap+1),     /* [ 1] b2          */
            ffloat(0x1.0470591de2ca6p+1),     /* [ 2] b3          */
            ffloat(0x1.2bd7609fd98c6p+0),     /* [ 3] b4          */
            ffloat(0x1.1429ffd1d4d79p-1),     /* [ 4] b5          */
            ffloat(0x1.a7ed70847c8bap-3),     /* [ 5] b6          */
            /* high-degree M = 7, zero .lo() by construction */
            ffloat(0x1.16e4ep-4f),            /* [ 6] b7          */
            ffloat(0x1.4116bp-6f),            /* [ 7] b8          */
            ffloat(0x1.4897c4p-8f),           /* [ 8] b9          */
            ffloat(0x1.2ea52cp-10f),          /* [ 9] b10         */
            ffloat(0x1.facfd6p-13f),          /* [10] b11         */
            ffloat(0x1.84fe12p-15f),          /* [11] b12         */
            ffloat(0x1.1398aep-17f)           /* [12] b13         */
        };

        /* G(r) = b1 + b2*r + b3*r^2 + ... + b13*r^12 ~= (10^r - 1)/r */
        ffloat p = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 7>(r, exp10_c);

        /* Close with the implicit b0 = 1:  10^r = 1 + r * G(r) */
        p = p * r + ffloat(1.0f, 0.0f);
        return p;
    }

    /*
    * --------------------------------------------------------------------
    * Internal helper: 2^n scaling for fp32mp2 with split exponent
    * --------------------------------------------------------------------
    * Computes  result = p * 2^n  in fp32mp2 for any integer n that
    * keeps the final result in the fp32 representable range.  The 2^n
    * factor is split into two halves (2^(n/2) * 2^(n - n/2)) so neither
    * intermediate multiplier overflows or denormalizes; this matches
    * libdevice's `__internal_fast_ldexpf` and the existing scaling
    * trick used by __nv_fp32mp2_erfc.
    * --------------------------------------------------------------------
    */
    __FPMP_INTERNAL_DECL__ fp32mp2_low __nv_fp32mp2_ldexp2_internal(fp32mp2_low p, int n)
    {
        const int k    = n >> 1;        /* floor-div-by-2; signed shift on negative n */
        int       ek1  = 127 + k;
        int       ek2  = 127 + (n - k);
        /* Clamp split exponents into representable normal-float biased
         * range [1, 254].  When |n| is large, one half can saturate to
         * the denormal floor / overflow ceiling -- handled by the chained
         * multiply, which then sees a fully-collapsed factor at the
         * other end. */
        if (ek1 < 1)   ek1 = 1;
        if (ek2 < 1)   ek2 = 1;
        if (ek1 > 254) ek1 = 254;
        if (ek2 > 254) ek2 = 254;
        const float scale_a = fpmp::internal_bit_cast<float>(static_cast<unsigned>(ek1) << 23);
        const float scale_b = fpmp::internal_bit_cast<float>(static_cast<unsigned>(ek2) << 23);
        return p * scale_a * scale_b;
    }

    /*
    * --------------------------------------------------------------------
    * ldexp(x, n) = x * 2^n  (fp32mp2 dedicated)
    * --------------------------------------------------------------------
    * Replaces the legacy fp64 round-trip with an all-fp32 implementation:
    * the 2^n factor is built directly from its biased fp32 exponent via
    * bit-cast and multiplied into x.  No native double is ever touched,
    * so the routine runs at full fp32 throughput on GPUs where fp64 is
    * 1:32 or worse.
    *
    * Algorithm -- 3-piece split (vs. the 2-piece used by
    * __nv_fp32mp2_ldexp2_internal for the bounded-n exp2/exp10 callers):
    *
    *   2^n  =  2^k * 2^k * 2^(n - 2k),  k = n/3
    *
    * Each factor's biased exponent is clamped to [1, 254] (fp32 normal
    * range), so every scale is finite and strictly positive.  This
    * keeps the mul chain IEEE-clean for the special inputs that show up
    * in `ldexp` corner cases:
    *
    *   ldexp(0,   n) ->  +-0    (0 * finite positive = +-0, never NaN)
    *   ldexp(+-inf, n) ->  +-inf    (inf * finite positive = inf)
    *   ldexp(NaN, n) ->  NaN  (NaN propagates)
    *   ldexp(x,   n) -> +-inf    when |n| is huge enough to overflow fp32
    *   ldexp(x,   n) -> +-0    when |n| is huge enough to underflow fp32
    *
    * Saturation: n is pre-clamped to +-300.  Even a denormal x (|x| >=
    * 2^-149) scaled by 2^300 overflows to +inf, and any finite x scaled by
    * 2^-300 underflows to +-0 -- so further-out n values would produce the
    * same result and saturating is exact, not lossy.
    *
    * The fp64mp2 branch (compile-time-selected when FpType == double)
    * forwards to `::ldexp(double, int)`; fp64 hardware handles this in
    * one instruction and there is no fp64 cost concern on machines that
    * have fp64mp2 enabled.
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_ldexp(const FpType x_hi,
                                                  const FpType x_lo,
                                                  int          n,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        if constexpr (std::is_same<FpType, float>::value)
        {
            using namespace fpmp;
            using ffloat = fp32mp2_low;

            /* Saturate |n| to +-300.  Any |n| larger than this is provably
             * monotone in the final result (overflow or underflow) for
             * every finite fp32 input, so clamping does not lose info. */
            if (n >  300) n =  300;
            if (n < -300) n = -300;

            const int k    = n / 3;
            int       ek1  = 127 + k;
            int       ek2  = 127 + k;
            int       ek3  = 127 + (n - 2 * k);
            if (ek1 < 1)   ek1 = 1;
            if (ek2 < 1)   ek2 = 1;
            if (ek3 < 1)   ek3 = 1;
            if (ek1 > 254) ek1 = 254;
            if (ek2 > 254) ek2 = 254;
            if (ek3 > 254) ek3 = 254;
            const float s1 = fpmp::internal_bit_cast<float>(static_cast<unsigned>(ek1) << 23);
            const float s2 = fpmp::internal_bit_cast<float>(static_cast<unsigned>(ek2) << 23);
            const float s3 = fpmp::internal_bit_cast<float>(static_cast<unsigned>(ek3) << 23);

            const ffloat result = ffloat(x_hi, x_lo) * s1 * s2 * s3;

            *res_hi = result.hi();
            *res_lo = result.lo();
        }
        else
        {
            /* fp64mp2 path: forward to libm `::ldexp` via the existing
             * double round-trip.  An explicit __nv_fpmp2_ldexp<double>
             * specialization is also provided later for symmetry with
             * the rest of the fp64mp2 math surface; this else branch
             * exists so the primary template is well-formed for any
             * `FpType` and stays compilable in isolation. */
            using mp2_t = fpmp2_t<FpType>;
            const double r = ::ldexp(static_cast<double>(mp2_t(x_hi, x_lo)), n);
            mp2_t result(r);
            *res_hi = result.hi();
            *res_lo = result.lo();
        }
    }

    /*
    * --------------------------------------------------------------------
    * scalbn(x, n) = x * FLT_RADIX^n  (fp32mp2 dedicated)
    * --------------------------------------------------------------------
    * For IEEE 754 binary formats (FLT_RADIX == 2 -- true on every
    * platform we target) `scalbn(x, n)` is bit-identical to
    * `ldexp(x, n)`, so we simply forward to the dedicated fp32mp2
    * ldexp implementation above and avoid duplicating the 3-piece
    * bit-cast scaling logic.  No fp64 round-trip is required.
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_scalbn(const FpType x_hi,
                                                   const FpType x_lo,
                                                   int          n,
                                                   FpType*      res_hi,
                                                   FpType*      res_lo)
    {
        __nv_fpmp2_ldexp<FpType>(x_hi, x_lo, n, res_hi, res_lo);
    }

    /*
    * --------------------------------------------------------------------
    * fmod / remainder (fp32mp2) - dedicated implementations
    * --------------------------------------------------------------------
    * Mirrors libdevice's integer-mantissa long-division idea (see
    * __nv_fmod / __internal_fmodf_kernel in device_functions_impl.c):
    * decompose each operand into a 64-bit integer mantissa M and a
    * binary exponent E with  value = M * 2^E,  then reduce
    *   Mx * 2^(Ex-Ey)  (mod My)
    * with exact 64-bit integer arithmetic, chunking the left shift so
    * the running value never exceeds 2^63.  No native fp64 is touched,
    * so this runs at full fp32 throughput on GPUs where double is 1:32
    * or worse, and it is *more* accurate than the old
    * `::fmod(double, double)` round-trip for fp32mp2 values whose two
    * limbs straddle a wide exponent gap (the double cast collapsed lo
    * into hi).
    *
    * A renormalized fp32mp2 carries at most ~48 significant bits from
    * the top of hi down to the bottom of lo, so one uint64 mantissa
    * captures the whole value exactly in the common case.  When the
    * limbs are more than ~48 bits apart the far-below-lo bits are
    * dropped (fp2int_rn rounding of the scaled lo limb), contributing
    * only ~|lo| of absolute error -- negligible relative to a result of
    * magnitude < |y|.  Denormal results round once through the
    * power-of-two reconstruction; we do not chase sub-ulp denormal
    * accuracy here.
    * --------------------------------------------------------------------
    */

    /* Scale a float by 2^s using two power-of-two factors, so the
     * intermediate stays in range for the |s| we feed it here. */
    __FPMP_INTERNAL_DECL__ float __nv_fp32mp2_scale2_scalar(float v, int s)
    {
        const int   s1 = s >> 1;            /* floor(s/2) */
        const int   s2 = s - s1;
        const float f1 = fpmp::internal_bit_cast<float>(static_cast<uint32_t>(127 + s1) << 23);
        const float f2 = fpmp::internal_bit_cast<float>(static_cast<uint32_t>(127 + s2) << 23);
        return v * f1 * f2;
    }

    /* Decompose a strictly-positive renormalized fp32mp2 value
     * (hi > 0, finite; lo any sign, |lo| <= ulp(hi)/2) into M * 2^E with
     * M normalized to [2^52, 2^53).
     *
     * The 53-bit window matches IEEE double's significand: a renormalized
     * fp32mp2 carries 24 (hi) + 24 (lo) significant bits separated by a tiny
     * gap (the test generator places lo at exponent Eh-25), so its full span
     * is ~49 bits.  A 48-bit window would round away the bottom of lo whenever
     * lo sits below ulp(hi)/2 - that loss is invisible for most ops but is
     * catastrophically amplified by the cancellation inherent to fmod when the
     * result is far smaller than the inputs.  Capturing 53 bits keeps the value
     * exactly (equivalent to the fp64 fallback's double round-trip). */
    __FPMP_INTERNAL_DECL__ void __nv_fp32mp2_modf_decompose(float hi, float lo,
                                                            unsigned long long* M, int* E)
    {
        const uint32_t hb = fpmp::internal_bit_cast<uint32_t>(hi);
        int Eh;
        if ((hb & 0x7F800000u) == 0u)
        {
            /* denormal hi (hi > 0): value = mant * 2^-149 */
            const uint32_t mant = hb & 0x007FFFFFu;
            const float    fm   = static_cast<float>(mant);
            const uint32_t fmb  = fpmp::internal_bit_cast<uint32_t>(fm);
            Eh = static_cast<int>((fmb >> 23) & 0xFFu) - 127 - 149;
        }
        else
        {
            Eh = static_cast<int>((hb >> 23) & 0xFFu) - 127;
        }

        const int   s   = 52 - Eh;
        const float shi = __nv_fp32mp2_scale2_scalar(hi, s);   /* integer in [2^52, 2^53) */
        const float slo = __nv_fp32mp2_scale2_scalar(lo, s);   /* |slo| <= 2^28           */

        long long m = static_cast<long long>(static_cast<unsigned long long>(shi))
                    + static_cast<long long>(fpmp::fp2int_rn(slo));
        int e = Eh - 52;

        /* lo > 0 may push m just past 2^53; bring it back. */
        if ((static_cast<unsigned long long>(m) >> 53) != 0ULL)
        {
            m >>= 1;
            e  += 1;
        }
        /* Off-by-one fix: when hi is an exact power of two and lo < 0 the
         * true value sits just below 2^Eh, so m lands just under 2^52.
         * One left shift renormalizes it back into [2^52, 2^53). */
        if (m != 0 && (static_cast<unsigned long long>(m) >> 52) == 0ULL)
        {
            m <<= 1;
            e  -= 1;
        }

        *M = static_cast<unsigned long long>(m);
        *E = e;
    }

    /* Build a renormalized fp32mp2 from  (neg ? -1 : 1) * mag * 2^E.
     * mag may carry up to 53 significant bits; it is first rounded
     * (round-half-to-even) down to the 48 bits an fp32mp2 can hold, then split
     * into two <= 24-bit halves so each casts to float exactly. */
    __FPMP_INTERNAL_DECL__ void __nv_fp32mp2_modf_reconstruct(unsigned long long mag, int E, bool neg,
                                                              float* res_hi, float* res_lo)
    {
        if (mag == 0ULL)
        {
            *res_hi = neg ? -0.0f : 0.0f;
            *res_lo = 0.0f;
            return;
        }

        /* Round mag down to <= 48 significant bits. */
        int extra = 0;
        for (unsigned long long t = mag; t >= (1ULL << 48); t >>= 1) { ++extra; }
        if (extra > 0)
        {
            const unsigned long long half = 1ULL << (extra - 1);
            const unsigned long long frac = mag & ((1ULL << extra) - 1ULL);
            unsigned long long       q    = mag >> extra;
            if (frac > half || (frac == half && (q & 1ULL) != 0ULL)) { ++q; }
            mag = q;
            E  += extra;
            if ((mag >> 48) != 0ULL) { mag >>= 1; ++E; }   /* rounding carried out */
        }

        const unsigned hipart = static_cast<unsigned>(mag >> 24);           /* < 2^24 */
        const unsigned lopart = static_cast<unsigned>(mag & 0xFFFFFFULL);   /* < 2^24 */
        float rhi = __nv_fp32mp2_scale2_scalar(static_cast<float>(hipart), E + 24);
        float rlo = __nv_fp32mp2_scale2_scalar(static_cast<float>(lopart), E);
        if (neg) { rhi = -rhi; rlo = -rlo; }

        float lo;
        const float hi = fpmp::two_sum(rhi, rlo, &lo);   /* exact, no magnitude assumption */
        *res_hi = hi;
        *res_lo = lo;
    }

    /* Core reduction: assumes ax > ay > 0 (both finite, nonzero), inputs
     * given as positive renormalized (hi, lo) pairs.  Returns the fmod
     * remainder mantissa ia (< My), the divisor mantissa My, its
     * exponent Ey, and the low bits of the integer quotient
     * floor(ax/ay) in quo. */
    __FPMP_INTERNAL_DECL__ void __nv_fp32mp2_fmod_kernel(float ax_hi, float ax_lo,
                                                         float ay_hi, float ay_lo,
                                                         unsigned long long* ia_out,
                                                         unsigned long long* My_out,
                                                         int*                Ey_out,
                                                         unsigned long long* quo_out)
    {
        unsigned long long Mx, My;
        int Ex, Ey;
        __nv_fp32mp2_modf_decompose(ax_hi, ax_lo, &Mx, &Ex);
        __nv_fp32mp2_modf_decompose(ay_hi, ay_lo, &My, &Ey);

        int D = Ex - Ey;                 /* >= 0 since ax > ay and both M in [2^52,2^53) */
        if (D < 0) D = 0;                /* defensive */

        unsigned long long quo = Mx / My;
        unsigned long long ia  = Mx % My;
        int remaining = D;
        while (remaining > 0)
        {
            const int                s   = (remaining < 11) ? remaining : 11;
            const unsigned long long num = ia << s;          /* ia < My < 2^53, so num < 2^64 */
            quo = (quo << s) + (num / My);                   /* low bits of quotient (parity only) */
            ia  = num % My;
            remaining -= s;
        }

        *ia_out = ia;
        *My_out = My;
        *Ey_out = Ey;
        *quo_out = quo;
    }

    /*
    * fmod(x, y): result has the sign of x and magnitude in [0, |y|).
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_fmod(const FpType x_hi, const FpType x_lo,
                                                const FpType y_hi, const FpType y_lo,
                                                FpType* res_hi, FpType* res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_fmod is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;

        /* (hi + lo) != (hi + lo) also catches a degenerate (+inf, -inf) limb
         * pair, which the fp128 reference widens to inf + (-inf) = NaN. */
        const bool  x_nan = (x_hi != x_hi) || (x_lo != x_lo) || ((x_hi + x_lo) != (x_hi + x_lo));
        const bool  y_nan = (y_hi != y_hi) || (y_lo != y_lo) || ((y_hi + y_lo) != (y_hi + y_lo));
        const float axh   = (x_hi < 0.0f) ? -x_hi : x_hi;
        const float ayh   = (y_hi < 0.0f) ? -y_hi : y_hi;
        const bool  x_inf = (axh == __builtin_huge_valf());
        const bool  y_inf = (ayh == __builtin_huge_valf());
        const bool  y_zero = (y_hi == 0.0f);

        if (x_nan || y_nan || x_inf || y_zero)
        {
            *res_hi = __builtin_nanf(""); *res_lo = __builtin_nanf(""); return;
        }
        if (y_inf)                       /* fmod(finite, inf) = x */
        {
            *res_hi = x_hi; *res_lo = x_lo; return;
        }

        const float axl = (x_hi < 0.0f) ? -x_lo : x_lo;
        const float ayl = (y_hi < 0.0f) ? -y_lo : y_lo;

        int c;
        if      (axh != ayh) c = (axh < ayh) ? -1 : 1;
        else if (axl != ayl) c = (axl < ayl) ? -1 : 1;
        else                 c = 0;

        if (c < 0)  { *res_hi = x_hi; *res_lo = x_lo; return; }                 /* |x| < |y| -> x   */
        if (c == 0) { *res_hi = (x_hi < 0.0f) ? -0.0f : 0.0f; *res_lo = 0.0f; return; }

        unsigned long long ia, My, quo;
        int Ey;
        __nv_fp32mp2_fmod_kernel(axh, axl, ayh, ayl, &ia, &My, &Ey, &quo);
        __nv_fp32mp2_modf_reconstruct(ia, Ey, (x_hi < 0.0f), res_hi, res_lo);
    }

    /*
    * remainder(x, y): IEEE remainder, |result| <= |y|/2, round-to-nearest
    * with ties to even quotient.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_remainder(const FpType x_hi, const FpType x_lo,
                                                     const FpType y_hi, const FpType y_lo,
                                                     FpType* res_hi, FpType* res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_remainder is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* (hi + lo) != (hi + lo) also catches a degenerate (+inf, -inf) limb
         * pair, which the fp128 reference widens to inf + (-inf) = NaN. */
        const bool  x_nan = (x_hi != x_hi) || (x_lo != x_lo) || ((x_hi + x_lo) != (x_hi + x_lo));
        const bool  y_nan = (y_hi != y_hi) || (y_lo != y_lo) || ((y_hi + y_lo) != (y_hi + y_lo));
        const float axh   = (x_hi < 0.0f) ? -x_hi : x_hi;
        const float ayh   = (y_hi < 0.0f) ? -y_hi : y_hi;
        const bool  x_inf = (axh == __builtin_huge_valf());
        const bool  y_inf = (ayh == __builtin_huge_valf());
        const bool  y_zero = (y_hi == 0.0f);
        const bool  xneg  = (x_hi < 0.0f);

        if (x_nan || y_nan || x_inf || y_zero)
        {
            *res_hi = __builtin_nanf(""); *res_lo = __builtin_nanf(""); return;
        }
        if (y_inf)                       /* remainder(finite, inf) = x */
        {
            *res_hi = x_hi; *res_lo = x_lo; return;
        }

        const float axl = (x_hi < 0.0f) ? -x_lo : x_lo;
        const float ayl = (y_hi < 0.0f) ? -y_lo : y_lo;

        int c;
        if      (axh != ayh) c = (axh < ayh) ? -1 : 1;
        else if (axl != ayl) c = (axl < ayl) ? -1 : 1;
        else                 c = 0;

        if (c == 0)                      /* |x| == |y| -> remainder 0 (sign of x) */
        {
            *res_hi = xneg ? -0.0f : 0.0f; *res_lo = 0.0f; return;
        }

        if (c < 0)
        {
            /* |x| < |y|: quotient is 0 or +-1.  Compare 2|x| against |y|. */
            const float t_hi = 2.0f * axh, t_lo = 2.0f * axl;   /* 2|x| exact */
            int c2;
            if      (t_hi != ayh) c2 = (t_hi < ayh) ? -1 : 1;
            else if (t_lo != ayl) c2 = (t_lo < ayl) ? -1 : 1;
            else                  c2 = 0;                        /* tie -> quotient 0 (even) */

            if (c2 <= 0) { *res_hi = x_hi; *res_lo = x_lo; return; }   /* r = x */

            /* 2|x| > |y|: r = |x| - |y|  (negative in the |x| frame) */
            const ffloat r = sub<fpmp2_accuracy::high>(ffloat(axh, axl), ffloat(ayh, ayl));
            float rh = r.hi(), rl = r.lo();
            if (xneg) { rh = -rh; rl = -rl; }
            *res_hi = rh; *res_lo = rl; return;
        }

        /* |x| > |y|: full integer reduction, then round-to-nearest-even. */
        unsigned long long ia, My, quo;
        int Ey;
        __nv_fp32mp2_fmod_kernel(axh, axl, ayh, ayl, &ia, &My, &Ey, &quo);

        const unsigned long long two_ia = ia << 1;
        const bool round_up = (two_ia > My) || ((two_ia == My) && ((quo & 1ULL) != 0ULL));

        unsigned long long mag;
        bool neg_xframe;
        if (round_up) { mag = My - ia; neg_xframe = true;  }   /* r = (|x| mod |y|) - |y| < 0 */
        else          { mag = ia;      neg_xframe = false; }

        __nv_fp32mp2_modf_reconstruct(mag, Ey, static_cast<bool>(neg_xframe ^ xneg), res_hi, res_lo);
    }

    /*
    * --------------------------------------------------------------------
    * Base-2 exponential exp2(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Strategy mirrors libdevice's `__internal_accurate_expf_1p93ulp`:
    * do a single integer/fractional split in *base-2* units so the
    * integer power 2^n drops out exactly and the polynomial only sees a
    * small reduced argument.
    *
    *   n = round(x_hi)              [exact integer, ffloat sub is exact]
    *   r = x - n                    [|r_hi| <= 0.5, r_lo from x_lo preserved]
    *   y = r * ln(2)                [|y| <= ln(2)/2 ~= 0.347]
    *   exp(y)                       [via dedicated fp32mp2 exp; its
    *                                 internal reduction yields n_internal
    *                                 = 0 because |y| < ln(2)/2, so the
    *                                 call collapses to a clean Taylor
    *                                 evaluation with no further reduction
    *                                 loss]
    *   result = 2^n * exp(y)        [scaled via the split-exponent helper]
    *
    * Why this beats the previous `exp(x * ln 2)` composition:
    *   That path computed y_outer = x * ln 2 (large; |y_outer| ~= 62 for
    *   |x| ~= 90) and then re-derived n_internal = round(y_outer/ln 2)
    *   inside __nv_fpmp2_exp.  Two stacked reductions accumulate:
    *     (a) one ulp on the ffloat multiplication x * ln 2, then
    *     (b) the cancellation in y_outer - n_internal*ln 2 magnifies
    *         that ulp because both operands are O(|y_outer|).
    *   By doing the round() directly on x_hi we get an exact integer n
    *   in one shot, the residual r = x - n is exact in the ffloat
    *   sense, and only one (small) multiplication r * ln(2) remains
    *   before the polynomial.  Empirically the dedicated path gains
    *   2-3 bits on the `work` dataset versus the composed form.
    *
    * Overflow boundary: 2^x overflows float for x >= 128 (FLT_MAX has
    *   biased exponent 254, so 2^128 = +inf in fp32).
    * Underflow boundary: 2^-149 is the smallest positive denormal; for
    *   x <= -150 the result rounds to 0 in fp32.
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_exp2 (const FpType x_hi,
                                                 const FpType x_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_exp2 is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* NaN propagation: short-circuit so the lo limb propagates too. */
        if (x_hi != x_hi)
        {
            *res_hi = x_hi;
            *res_lo = x_hi;
            return;
        }

        /* Overflow / underflow shortcuts. */
        if (x_hi >= 128.0f)
        {
            *res_hi = __builtin_huge_valf();
            *res_lo = 0.0f;
            return;
        }
        if (x_hi <= -150.0f)
        {
            *res_hi = 0.0f;
            *res_lo = 0.0f;
            return;
        }

        /* Step 1: integer/fractional split directly in base-2 units. */
        const int    n   = fp2int_rn(x_hi);
        const ffloat n_f = int2fp_rn<float>(n);

        /* Step 2: r = x - n.  ffloat subtraction by an integer is exact
         * (n_f is representable in float for |n| <= 2^23, which our
         * overflow/underflow shortcuts guarantee). */
        const ffloat r = ffloat(x_hi, x_lo) - n_f;

        /* Step 3: 2^r via the dedicated base-2 Taylor kernel (no r * ln 2
         * detour, no internal natural-log reduction). */
        const ffloat u = __nv_fp32mp2_exp2_kernel(r);

        /* Step 4: multiply by 2^n via the split-exponent helper. */
        const ffloat result = __nv_fp32mp2_ldexp2_internal(u, n);

        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_exp2

    /*
    * --------------------------------------------------------------------
    * Base-10 exponential exp10(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Strategy:  base-2 integer split + base-10 fractional kernel, with
    *            Cody-Waite reduction for the residual.  Mirrors the
    *            libdevice idiom for 10^x (split log10(2) into 3 fp32
    *            pieces and accumulate via two_mult_fma + two_sum in a
    *            higher-precision afloat accumulator) so that the
    *            residual r' avoids the catastrophic-cancellation
    *            precision floor of the naive ff "compute t = x * log2 10,
    *            then r = t - n" path.
    *
    *   n  = round(x * log2 10)             [integer power of 2]
    *   r' = x - n * log10(2)               [Cody-Waite, |r'| <= log10 2 / 2]
    *   10^r' = base-10 Taylor kernel       [|r'| <= 0.151 -> small Horner accum.]
    *   result = 2^n * 10^r'                [split-exponent helper]
    *
    * Why this beats the earlier `2^n * 2^(t - n)` form:
    *   That path computed t = x * log2 10 in ff (~= 100 for x ~= 30),
    *   then r = t - n via 2-sum.  Although the 2-sum captures the exact
    *   difference, the ff representation of t has *absolute* precision
    *   bounded by ulp(t.hi)/2 * 2^-23 ~= 2^-40 (i.e. relative precision
    *   2^-46 times |t|).  Subtracting n exposes this absolute floor as
    *   the relative precision of r ~= 0.04 -- only ~34 bits.  After 2^r
    *   that becomes ~39 bits on the work range -- exactly what we
    *   measured.
    *   The Cody-Waite path computes r' = x - n * log10(2) where the
    *   cancellation is between two values of magnitude |x|, so the
    *   absolute precision of r' tracks ulp(x_hi) * 2^-23 rather than
    *   ulp(t_hi) * 2^-23 -- log2(10) ~= 3.32 worth of bits recovered.
    *   The smaller |r'| <= 0.151 (vs. |r| <= 0.5) also halves the
    *   Horner accumulation in the polynomial.
    *
    * Overflow boundary: 10^x overflows float at x ~= log10(FLT_MAX) =
    *   38.5318394... -- round generously to 39.
    * Underflow boundary: 10^-45.155 is the smallest positive denormal;
    *   round to -46.
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_exp10 (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_exp10 is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;
        using afloat = fp32mp2_high;

        if (x_hi != x_hi)
        {
            *res_hi = x_hi;
            *res_lo = x_hi;
            return;
        }
        if (x_hi >= 39.0f)
        {
            *res_hi = __builtin_huge_valf();
            *res_lo = 0.0f;
            return;
        }
        if (x_hi <= -46.0f)
        {
            *res_hi = 0.0f;
            *res_lo = 0.0f;
            return;
        }

        /* log2(10) ~= 3.32192809488736234787...  (fp32mp2 constant used
         * only for the *coarse* n estimate; precision floor of this
         * value is fine since we only need the integer part). */
        constexpr ffloat log2_10(0x1.a934f0979a371p+1);

        /* log10(2) ~= 0.30102999566398119521  split into 3 fp32 chunks
         * (Cody-Waite); the sum C1 + C2 + C3 reproduces log10(2)
         * exactly in double.  Layout mirrors the trig pi/2 split that
         * already lives in this file. */
        constexpr float C1 = 0x1.344136p-2f;   /* +0.30103001 */
        constexpr float C2 = -0x1.ec10c0p-27f; /* -1.432e-08 */
        constexpr float C3 = -0x1.000000p-54f; /*  ~-5.5e-17 */

        /* Step 1: coarse integer n = round(x * log2 10).
         * Uses an ordinary ff multiplication -- we only need the integer
         * part, so the lo limb of the product is discarded. */
        const ffloat t_approx = ffloat(x_hi, x_lo) * log2_10;
        const int    n        = fp2int_rn(t_approx.hi());
        const float  n_f      = int2fp_rn<float>(n);

        /* Step 2: Cody-Waite reduction  r' = x - n * log10(2)
         *   r' = (x_hi + x_lo) - n_f * (C1 + C2 + C3)
         * Computed via the same two_mult_fma + two_sum recipe used by
         * the trig kernel: every product / subtraction is captured as
         * an exact pair, then accumulated in fp32mp2_high so
         * the relative precision of r' is bounded by ulp(x_hi)*2^-23,
         * not by ulp(x*log2 10)*2^-23. */

        /* n_f * C1 = ph + pl  (exact pair) */
        float pl;
        const float ph = two_mult_fma(n_f, C1, &pl);

        /* x_hi - ph = s + e  (exact pair) */
        float e;
        const float s = two_sum(x_hi, -ph, &e);

        afloat r_acc(s, e);
        r_acc = r_acc + afloat(-pl);
        r_acc = r_acc + afloat(x_lo);

        /* n_f * C2 = nC2_hi + nC2_lo  (exact pair) */
        float nC2_lo;
        const float nC2_hi = two_mult_fma(n_f, C2, &nC2_lo);
        r_acc = r_acc - afloat(nC2_hi, nC2_lo);

        /* n_f * C3 is tiny (~10^-14 at the largest n we hit);
         * single-precision product is below the polynomial noise
         * floor but cheap to include for completeness. */
        r_acc = r_acc + afloat(mul_rn(n_f, -C3));

        /* Step 3: 10^r' via the dedicated base-10 Taylor kernel.
         * Hand off the accurate accumulator as fast ffloat -- the
         * polynomial cannot consume more than ff precision anyway. */
        const ffloat r = ffloat(r_acc.hi(), r_acc.lo());
        const ffloat u = __nv_fp32mp2_exp10_kernel(r);

        /* Step 4: scale by 2^n via the split-exponent helper. */
        const ffloat result = __nv_fp32mp2_ldexp2_internal(u, n);

        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_exp10

    /*
    * --------------------------------------------------------------------
    * exp(x) - 1, i.e. expm1(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Strategy:
    *   - Small |x_hi| (< 1/2):  direct Taylor series
    *         expm1(x) = x + x^2 * P(x),
    *         P(x) = 1/2 + x/6 + x^2/24 + ... + x^11/(13!)
    *     keeping the full fp32mp2 input intact.
    *   - Otherwise:              compute exp(x) and subtract 1 with
    *     fp32mp2 accurate sub.
    *
    * Why a small-|x| branch is necessary:
    *   The exp(x) - 1 path produces exp(x) ~= 1 + O(x) for tiny x, so
    *   the leading bits cancel in the subtraction -- the relative
    *   accuracy of the result drops to log2(1/|x|) ulps.  At |x| ~ 1
    *   the loss is negligible; below |x| ~ 1/2 we lose ~1 bit; below
    *   |x| ~ 2^-46 the result collapses to zero entirely.  The Taylor
    *   form has no cancellation: x is preserved verbatim and the x^2*P
    *   correction sits an order of magnitude below x, so its lo-ulp
    *   noise costs only ~ |x| * ulp ~= negligible relative error.
    *
    * Branch point 1/2 chosen so that the omitted x^13 term contributes
    *   0.5^13 / 13! ~= 1.96*10^-14, comfortably below fp32mp2 ulp at
    *   expm1(1/2) ~= 0.6487, while keeping the polynomial narrow enough
    *   that the warp divergence cost stays modest (the symmetric width
    *   covers ~25% of a normal-around-0 input distribution).
    *
    * Special cases:
    *   - NaN propagation (any NaN -> NaN).
    *   - +inf      -> +inf.
    *   - -inf      -> -1.
    *   - x = 0     -> 0 (Taylor branch returns it exactly).
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_expm1 (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_expm1 is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* NaN propagation: any NaN component -> NaN result. */
        if (x_hi != x_hi || x_lo != x_lo)
        {
            const float nan_val = x_hi + x_lo;
            *res_hi = nan_val;
            *res_lo = nan_val;
            return;
        }

        /* +inf input -> +inf. */
        if (x_hi == __builtin_huge_valf())
        {
            *res_hi = __builtin_huge_valf();
            *res_lo = 0.0f;
            return;
        }

        /* -inf input -> -1 exactly. */
        if (x_hi == -__builtin_huge_valf())
        {
            *res_hi = -1.0f;
            *res_lo = 0.0f;
            return;
        }

        const float abs_hi = (x_hi < 0.0f) ? -x_hi : x_hi;
        constexpr float EXPM1_BRANCH_POINT = 0.5f;
        if (abs_hi < EXPM1_BRANCH_POINT)
        {
            /* P(x) = sum_{k>=0} x^k / (k+2)!,
             *   P[0] = 1/2!, P[1] = 1/3!, ..., P[11] = 1/13!.
             * Layout for poly_eval<horner_mixed, M=4>: bottom 8 entries
             * are full ff (their contributions stay above fp32mp2 ulp at
             * the branch point), top 4 entries are plain float (.lo == 0
             * by construction; their contributions sit below 0.5 ulp). */
            constexpr ffloat expm1_poly_c[12] = {
                ffloat( 5.0000000000000000e-1),   /* [ 0] 1/2!  = 1/2 */
                ffloat( 1.6666666666666666e-1),   /* [ 1] 1/3!  = 1/6 */
                ffloat( 4.1666666666666664e-2),   /* [ 2] 1/4!  = 1/24 */
                ffloat( 8.3333333333333332e-3),   /* [ 3] 1/5!  = 1/120 */
                ffloat( 1.3888888888888889e-3),   /* [ 4] 1/6!  = 1/720 */
                ffloat( 1.9841269841269841e-4),   /* [ 5] 1/7!  = 1/5040 */
                ffloat( 2.4801587301587302e-5),   /* [ 6] 1/8!  = 1/40320 */
                ffloat( 2.7557319223985893e-6),   /* [ 7] 1/9!  (last ff term) */
                /* high-order M = 4 entries: .lo() == 0 by construction */
                ffloat( 2.7557320e-7f),           /* [ 8] 1/10! */
                ffloat( 2.5052108e-8f),           /* [ 9] 1/11! */
                ffloat( 2.0876756e-9f),           /* [10] 1/12! */
                ffloat( 1.6059044e-10f),          /* [11] 1/13! (leading) */
            };

            ffloat x      (x_hi, x_lo);
            ffloat x2     = x * x;
            ffloat P      = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 4>(x, expm1_poly_c);
            ffloat result = renormalize(x + x2 * P);
            *res_hi = result.hi();
            *res_lo = result.lo();
            return;
        }

        /* Large-|x| branch: compute exp(x) and subtract 1.0 with the
         * accurate sub variant so the lo-limb captures the cancellation
         * residual that a plain fast 2-sum would quantise away.  For
         * |x| >= 1/2 the leading term exp(x) is at least 0.6 away from 1
         * (positive side) or 0.6 below 1 (negative side), so the
         * subtraction never loses more than ~1 bit. */
        float e_hi, e_lo;
        __nv_fpmp2_exp<float>(x_hi, x_lo, &e_hi, &e_lo);

        /* exp() may already produce +inf for very large x; pass that
         * through without quietly turning it into NaN via inf - 1. */
        if (e_hi == __builtin_huge_valf())
        {
            *res_hi = __builtin_huge_valf();
            *res_lo = 0.0f;
            return;
        }

        ffloat result = sub<fpmp2_accuracy::high>(ffloat(e_hi, e_lo), ffloat(1.0f));
        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_expm1

    /*
    * --------------------------------------------------------------------
    * Power function pow(a, b) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Algorithm:
    *
    *    pow(a, b) = exp(b * log(|a|))
    *
    * with sign fixup for a < 0 with integer b.  All three primitives
    * (log, mul, exp) are dedicated fp32mp2 - no fp64 operations in the
    * main path.
    *
    * Structurally identical to libdevice's __nv_pow (special-case order
    * and IEEE 754-2008 corner-case semantics), but drops libdevice's
    * hi/lo bookkeeping around the exp call: libdevice has to fma-track
    * (t_hi, t_lo) around `b * log(a)` because its `__nv_exp` only takes
    * a single double and the lost low part has to be re-injected via
    * `tmp = fma(tmp, prod.x, tmp)`.  Our dedicated `__nv_fpmp2_exp`
    * consumes the full fp32mp2 pair natively, so the correction is
    * implicit and the main path is just three calls (log, mul, exp).
    *
    * Integer-b detection:
    *   b is integer iff b.lo == 0 AND truncf(b.hi) == b.hi.
    *   b is odd integer iff b is integer AND |b.hi| < 2^24 AND
    *   ((int32_t)b.hi & 1) != 0.  Above 2^24 every float-representable
    *   b.hi is automatically even (the LSB has weight >= 2).
    *
    * No b-clamping is needed (libdevice clamps |b| >= 2^126 to prevent
    * intermediate fma overflow).  For fp32mp2, |log(a)| <= ~88 for any
    * finite a > 0, so `b * loga` overflows to +-Inf only when the true
    * result truly overflows fp32 - and the dedicated `__nv_fpmp2_exp`
    * handles +-Inf input via its existing saturation paths.
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_pow (const FpType a_hi,
                                                const FpType a_lo,
                                                const FpType b_hi,
                                                const FpType b_lo,
                                                FpType*      res_hi,
                                                FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_pow is fp32mp2 only; fp64mp2 has its own specialization");

        /* ---- (1,2) pow(1,b) = pow(a,0) = 1, highest priority per IEEE 754-2008 ---- */
        if ((a_hi == 1.0f && a_lo == 0.0f) || (b_hi == 0.0f && b_lo == 0.0f))
        {
            *res_hi = 1.0f; *res_lo = 0.0f;
            return;
        }

        /* ---- (3) NaN propagation ---- */
        if ((a_hi != a_hi) || (b_hi != b_hi))
        {
            *res_hi = a_hi + b_hi; *res_lo = 0.0f;
            return;
        }

        /* ---- (4) integer / odd-integer b detection ---- */
        bool b_is_int     = false;
        bool b_is_odd_int = false;
        {
            const float b_trunc  = fpmp::internal_trunc<float>(b_hi);
            if (b_lo == 0.0f && b_trunc == b_hi)
            {
                b_is_int = true;
                const float abs_b_hi = b_hi < 0.0f ? -b_hi : b_hi;
                if (abs_b_hi < 0x1.0p+24f)   /* parity only meaningful below 2^24 */
                    b_is_odd_int = (static_cast<int32_t>(b_hi) & 1) != 0;
            }
        }

        const bool  a_is_neg = (a_hi < 0.0f) || (a_hi == 0.0f && a_lo < 0.0f);
        const float abs_a_hi = a_is_neg ? -a_hi : a_hi;
        const float abs_a_lo = a_is_neg ? -a_lo : a_lo;

        /* ---- (5) a == 0 ---- */
        if (abs_a_hi == 0.0f && abs_a_lo == 0.0f)
        {
            if (b_hi < 0.0f)
            {
                const float sign = (a_is_neg && b_is_odd_int) ? -1.0f : 1.0f;
                *res_hi = sign * __builtin_huge_valf();
            }
            else
            {
                *res_hi = (a_is_neg && b_is_odd_int) ? -0.0f : 0.0f;
            }
            *res_lo = 0.0f;
            return;
        }

        /* ---- (6) negative base with non-integer exponent ---- */
        if (a_is_neg && !b_is_int)
        {
            *res_hi = __builtin_nanf(""); *res_lo = 0.0f;
            return;
        }

        /* ---- (7) |a| = Inf ---- */
        if (abs_a_hi == __builtin_huge_valf())
        {
            const float sign = (a_is_neg && b_is_odd_int) ? -1.0f : 1.0f;
            *res_hi = (b_hi > 0.0f) ? sign * __builtin_huge_valf() : sign * 0.0f;
            *res_lo = 0.0f;
            return;
        }

        /* ---- (8) |b| = Inf ---- */
        if (b_hi == __builtin_huge_valf() || b_hi == -__builtin_huge_valf())
        {
            /* IEEE 754: pow(-1, +-Inf) = 1.  pow(+1, ...) already handled at (1). */
            if (abs_a_hi == 1.0f && abs_a_lo == 0.0f)
            {
                *res_hi = 1.0f; *res_lo = 0.0f;
                return;
            }
            const bool abs_a_gt_one = (abs_a_hi > 1.0f) ||
                                      (abs_a_hi == 1.0f && abs_a_lo > 0.0f);
            *res_hi = ((b_hi > 0.0f) == abs_a_gt_one) ? __builtin_huge_valf() : 0.0f;
            *res_lo = 0.0f;
            return;
        }

        /* ---- (9) main path: exp(b * log(|a|)) ---- */
        float loga_hi, loga_lo;
        __nv_fpmp2_log<float>(abs_a_hi, abs_a_lo, &loga_hi, &loga_lo);

        float prod_hi, prod_lo;
        __nv_fpmp2_mul<float>(b_hi, b_lo, loga_hi, loga_lo, &prod_hi, &prod_lo);

        float t_hi, t_lo;
        __nv_fpmp2_exp<float>(prod_hi, prod_lo, &t_hi, &t_lo);

        /* ---- sign fixup for a < 0 with odd integer b ---- */
        if (a_is_neg && b_is_odd_int) { t_hi = -t_hi; t_lo = -t_lo; }

        *res_hi = t_hi; *res_lo = t_lo;
    } // __nv_fpmp2_pow

    /*
    * --------------------------------------------------------------------
    * Cube root cbrt(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Algorithm (adapted from libdevice's double-precision __nv_cbrt):
    *
    *   1. Special cases: pass +-0, +-Inf, NaN through unchanged
    *      (cbrt(x) == x for these inputs).
    *   2. Operate on |x|; cbrt is odd, sign is restored at the end.
    *   3. Pre-scale denormal inputs by 2^24 (= 2^(3*8)) so the exponent
    *      extraction sees a normal float; the +8 bias in the result
    *      exponent is undone in the final scaling step.
    *   4. Argument reduction:  ax = r * 2^(3*nexpo)  with
    *      nexpo = round((expo - 126) / 3).  The reduced r sits in
    *      roughly [2^-1, 2^1), which gives the SFU lg2/ex2 pair a tight
    *      enough range that one Halley step recovers full fp32mp2
    *      precision.
    *   5. Initial single-precision approximation:
    *         s = fast_exp2(third * fast_log2(r_hi))            (~23 bits)
    *   6. One Halley iteration in fp32mp2 arithmetic
    *      (cubic convergence -> ~70 theoretical bits, capped by the
    *       ~46-bit fp32mp2 precision):
    *         t_new = t + t * (r - t^3) / (2 t^3 + r)
    *      The numerator (r - t^3) cancels catastrophically (t^3 ~= r),
    *      so it is evaluated with the accurate fp32mp2 fma:
    *         numer = fma<accurate>(-t^2, t, r)
    *      The denominator is well-conditioned (~3 r); a single-precision
    *      reciprocal of denom.hi() is enough -- the resulting correction
    *      contributes only at the 2^-46 level after multiplication by t.
    *   7. Multiply the result back by 2^(nexpo - denorm_div3) using
    *      a power-of-two scale factor (exact, no rounding).
    *   8. Restore the sign.
    *
    * No fp64 operations.  Negative inputs are supported (cbrt(-x) = -cbrt(x)).
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_cbrt (const FpType x_hi,
                                                 const FpType x_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        using namespace fpmp;
        using ffloat = fp32mp2_low;

        // 1/3 in single precision (round-to-nearest); the exact 1/3 is not
        // representable in any binary float, but ulp(1/3) is well below the
        // accuracy of the SFU lg2/ex2 pair so this is sufficient.
        constexpr float third_f = 0x1.555556p-2f;

        const uint32_t xbits   = internal_bit_cast<uint32_t>(x_hi);
        const uint32_t absbits = xbits & 0x7FFFFFFFu;
        const uint32_t signbit = xbits & 0x80000000u;

        /* Special inputs: +-0, +-Inf, NaN.  cbrt(x) returns x for these,
         * matching libdevice's behaviour (the libdevice routine fixes
         * them up via cmpsel against (a + a) at the end; an explicit
         * early-out is cleaner here). */
        if (absbits == 0u || absbits >= 0x7F800000u)
        {
            *res_hi = x_hi;
            *res_lo = (absbits >= 0x7F800000u) ? 0.0f : x_lo;
            return;
        }

        /* Operate on |x|; sign of x_lo follows the sign of x_hi. */
        float ax_hi = internal_bit_cast<float>(absbits);
        float ax_lo = (signbit != 0u) ? -x_lo : x_lo;

        /* Denormal pre-scaling: multiply by 2^24 (chosen so the offset is
         * divisible by 3 -> denorm_div3 = 8 unscales the result later). */
        int denorm_div3 = 0;
        uint32_t scaled_absbits = absbits;
        if ((absbits >> 23) == 0u)
        {
            constexpr float scale_up = 0x1.0p24f;
            ax_hi  *= scale_up;
            ax_lo  *= scale_up;
            denorm_div3    = 8;
            scaled_absbits = internal_bit_cast<uint32_t>(ax_hi);
        }

        /* Reduce: ax = r * 2^(3 * nexpo), with nexpo chosen so r ~= 1. */
        const int expo  = static_cast<int>(scaled_absbits >> 23);
        const int nexpo = fpmp::fp2int_rn(third_f * static_cast<float>(expo - 126));

        /* r_hi = ax_hi * 2^(-3*nexpo): exact, by exponent-field subtraction.
         * (The mantissa is untouched; only the biased exponent shifts.)
         * Use multiplication by 2^23 instead of left-shift to avoid UB
         * when (3 * nexpo) is negative. */
        constexpr int     EXP_SHIFT = 1 << 23;
        const     int     delta_exp = 3 * nexpo;
        const     int     new_bits  = static_cast<int>(scaled_absbits) - delta_exp * EXP_SHIFT;
        const     float   r_hi      = internal_bit_cast<float>(static_cast<uint32_t>(new_bits));

        /* r_lo: scale by the same power of two via float multiply.  Split
         * 2^(-3*nexpo) into two normal-range factors: for x near max float
         * |3*nexpo| can reach ~129, which would give an invalid biased
         * exponent of -2 if applied as a single bit-cast.  Splitting keeps
         * each factor's biased exponent in the normal range (about
         * [62, 190]); the product stays exact for all valid inputs. */
        const int half_pow  = -delta_exp / 2;
        const int rest_pow  = -delta_exp - half_pow;
        const float scale_a = internal_bit_cast<float>(static_cast<uint32_t>((127 + half_pow) * EXP_SHIFT));
        const float scale_b = internal_bit_cast<float>(static_cast<uint32_t>((127 + rest_pow) * EXP_SHIFT));
        const float r_lo    = (ax_lo * scale_a) * scale_b;

        /* Initial cbrt approximation via the SFU lg2/ex2 pair (~23 bits). */
        const float s = fpmp::fast_exp2(third_f * fpmp::fast_log2(r_hi));

        /* Halley refinement in fp32mp2:  t_new = t + t * (r - t^3) / (2 t^3 + r).
         *
         * The catastrophic cancellation in (r - t^3) is handled by an
         * accurate fma:  fma<accurate>(-t^2, t, r) computes r - t^2 * t
         * with a single rounding error followed by an exact correction,
         * preserving the small difference that drives the iteration.
         */
        const ffloat r(r_hi, r_lo);
        const ffloat t(s);

        const ffloat t2 = t * t;                        // t^2
        // numer = r - t^3 (computed as fma(-t^2, t, r) with accurate ff fma)
        const ffloat numer = fma<fpmp2_accuracy::high>(-t2, t, r);
        // denom = 2 t^3 + r ~= 3 r, well-conditioned so fast add suffices
        const ffloat t3    = t2 * t;
        const ffloat denom = (t3 + t3) + r;

        /* Single-precision reciprocal of denom.hi() is enough: the
         * correction u_corr ~ 2^-23 contributes t * u_corr ~ 2^-46 to
         * t_new -- exactly fp32mp2 precision. */
        const float  inv_denom = fpmp::rcp_rn(denom.hi());
        const ffloat u_corr    = numer * inv_denom;
        const ffloat t_new     = t + t * u_corr;

        /* Scale back by 2^(nexpo - denorm_div3) via an exact power-of-two
         * float multiply.  back_shift stays in a range that keeps the
         * scale factor a normal float for all valid float inputs
         * (biased exponent is always in [77, 170]). */
        const int   back_shift = nexpo - denorm_div3;
        const float scale_back = internal_bit_cast<float>(static_cast<uint32_t>((127 + back_shift) * EXP_SHIFT));
        float t_hi_back = t_new.hi() * scale_back;
        float t_lo_back = t_new.lo() * scale_back;

        /* Restore sign (cbrt is an odd function). */
        if (signbit != 0u)
        {
            t_hi_back = -t_hi_back;
            t_lo_back = -t_lo_back;
        }

        *res_hi = t_hi_back;
        *res_lo = t_lo_back;
    } // __nv_fpmp2_cbrt

    /*
    * --------------------------------------------------------------------
    * Reciprocal cube root rcbrt(x) = 1/cbrt(x) (fp32mp2) - dedicated impl
    * --------------------------------------------------------------------
    * Algorithm (adapted from libdevice's double-precision __nv_rcbrt):
    *
    *   1. Special cases:
    *        cbrt(+-0)   = +-Inf  (result inherits the sign of x)
    *        cbrt(+-Inf) = +-0
    *        cbrt(NaN)  = NaN
    *   2. Operate on |x|; rcbrt is odd, sign is restored at the end.
    *   3. Pre-scale denormal inputs by 2^24 (= 2^(3*8)) so the exponent
    *      extraction sees a normal float; the +8 offset ends up as a
    *      +8 shift on the back-scale (denorm_div3 = 8).
    *   4. Argument reduction:  ax = r * 2^(3*nexpo) with
    *      nexpo = round((expo - 126) / 3).  Reduced r sits in roughly
    *      [2^-2, 2^2), giving the SFU lg2/ex2 pair a tight enough range
    *      that one Halley step recovers full fp32mp2 precision.
    *   5. Initial single-precision approximation:
    *         s = fast_exp2(-third * fast_log2(r_hi))                 ~= 1/cbrt(r_hi) (~23 bits)
    *   6. One Halley iteration in fp32mp2 (cubic convergence ->
    *      ~70 theoretical bits, capped by ~46-bit fp32mp2 precision):
    *
    *          u     = 1 - r * t^3              (catastrophic cancellation)
    *          t_new = t * (1 + u/3 + (2/9) u^2)
    *
    *      The residual u is the dominant source of error and is computed
    *      with the accurate fp32mp2 fma:
    *         u = fma<accurate>(-r, t^3, 1)
    *      preserving the small difference that drives the iteration.
    *      The Halley quadratic (1/3 + (2/9) u) is well conditioned (no
    *      cancellation), so a single fast fma is sufficient there.
    *   7. Multiply the result back by 2^(-nexpo + denorm_div3) using a
    *      power-of-two scale factor (exact, no rounding).
    *   8. Restore the sign.
    *
    * No fp64 operations.  No final reciprocal: the algorithm targets 1/cbrt
    * directly, which is why it is faster than cbrt(x) followed by a divide.
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_rcbrt (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        using namespace fpmp;
        using ffloat = fp32mp2_low;

        // 1/3 and 2/9 in single precision (round-to-nearest); ulp at this
        // scale is far below the SFU lg2/ex2 estimate's accuracy.
        constexpr float third_f      = 0x1.555556p-2f;   // ~= 1/3
        constexpr float two_ninths_f = 0x1.c71c72p-3f;   // ~= 2/9

        const uint32_t xbits   = internal_bit_cast<uint32_t>(x_hi);
        const uint32_t absbits = xbits & 0x7FFFFFFFu;
        const uint32_t signbit = xbits & 0x80000000u;

        /* Special inputs:
         *   +-0   -> +-Inf
         *   +-Inf -> +-0
         *   NaN  -> NaN  (propagated via x_hi)
         */
        if (absbits == 0u || absbits >= 0x7F800000u)
        {
            if (absbits == 0u) {
                *res_hi = internal_bit_cast<float>(signbit | 0x7F800000u);
            } else if (absbits == 0x7F800000u) {
                *res_hi = internal_bit_cast<float>(signbit);
            } else {
                *res_hi = x_hi;     // NaN
            }
            *res_lo = 0.0f;
            return;
        }

        /* Operate on |x|; sign of x_lo follows the sign of x_hi. */
        float ax_hi = internal_bit_cast<float>(absbits);
        float ax_lo = (signbit != 0u) ? -x_lo : x_lo;

        /* Denormal pre-scaling: multiply by 2^24. */
        int denorm_div3 = 0;
        uint32_t scaled_absbits = absbits;
        if ((absbits >> 23) == 0u)
        {
            constexpr float scale_up = 0x1.0p24f;
            ax_hi  *= scale_up;
            ax_lo  *= scale_up;
            denorm_div3    = 8;
            scaled_absbits = internal_bit_cast<uint32_t>(ax_hi);
        }

        /* Reduce: ax = r * 2^(3 * nexpo), with nexpo chosen so r ~= 1. */
        const int expo  = static_cast<int>(scaled_absbits >> 23);
        const int nexpo = fpmp::fp2int_rn(third_f * static_cast<float>(expo - 126));

        /* r_hi = ax_hi * 2^(-3*nexpo): exact, by exponent-field subtraction.
         * Use multiplication by 2^23 instead of left-shift to avoid UB
         * when (3 * nexpo) is negative. */
        constexpr int     EXP_SHIFT = 1 << 23;
        const     int     delta_exp = 3 * nexpo;
        const     int     new_bits  = static_cast<int>(scaled_absbits) - delta_exp * EXP_SHIFT;
        const     float   r_hi      = internal_bit_cast<float>(static_cast<uint32_t>(new_bits));

        /* r_lo: scale by 2^(-3*nexpo) via float multiply.  Split into two
         * normal-range factors to keep each biased exponent in roughly
         * [62, 190] for all valid float inputs. */
        const int half_pow  = -delta_exp / 2;
        const int rest_pow  = -delta_exp - half_pow;
        const float scale_a = internal_bit_cast<float>(static_cast<uint32_t>((127 + half_pow) * EXP_SHIFT));
        const float scale_b = internal_bit_cast<float>(static_cast<uint32_t>((127 + rest_pow) * EXP_SHIFT));
        const float r_lo    = (ax_lo * scale_a) * scale_b;

        /* Initial 1/cbrt approximation via the SFU lg2/ex2 pair (~23 bits). */
        const float s = fpmp::fast_exp2(-third_f * fpmp::fast_log2(r_hi));

        /* Halley refinement in fp32mp2:  t_new = t * (1 + u/3 + (2/9) u^2)
         * with u = 1 - r * t^3.
         *
         * The cancellation in (1 - r * t^3) is the only sensitive step;
         * everything else (t^2, t^3, the Halley quadratic, the final
         * combination) is well conditioned in fast fp32mp2 arithmetic.
         */
        const ffloat r(r_hi, r_lo);
        const ffloat t(s);

        const ffloat t2 = t * t;                               // t^2
        const ffloat t3 = t2 * t;                              // t^3

        // u = 1 - r*t^3 (accurate fma to preserve catastrophic cancellation)
        const ffloat u  = fma<fpmp2_accuracy::high>(-r, t3, 1.0f);

        // Halley quadratic factor:  hf = 1/3 + (2/9) u   (no cancellation)
        const ffloat hf = fma<fpmp2_accuracy::def>(two_ninths_f, u, third_f);

        // delta = u * t * hf,  then  t_new = t + delta
        const ffloat ut    = u * t;
        const ffloat t_new = t + hf * ut;

        /* Scale back by 2^(-nexpo + denorm_div3) via an exact power-of-two
         * float multiply.  back_shift stays in [-43, +49] for all valid
         * float inputs, so the biased exponent is always in [84, 176]. */
        const int   back_shift = -nexpo + denorm_div3;
        const float scale_back = internal_bit_cast<float>(static_cast<uint32_t>((127 + back_shift) * EXP_SHIFT));
        float t_hi_back = t_new.hi() * scale_back;
        float t_lo_back = t_new.lo() * scale_back;

        /* Restore sign (rcbrt is an odd function). */
        if (signbit != 0u)
        {
            t_hi_back = -t_hi_back;
            t_lo_back = -t_lo_back;
        }

        *res_hi = t_hi_back;
        *res_lo = t_lo_back;
    } // __nv_fpmp2_rcbrt

    /*
    * ============================================================================
    * Trigonometric functions: sin, cos, sincos (fp32mp2) - dedicated
    * ============================================================================
    * Algorithm:
    *   1. Argument reduction: x = n*(pi/2) + r, |r| <= pi/4
    *      - Tiny (|x| < pi/4): no reduction
    *      - Fast (|x| < 2^20): Cody-Waite with exact error tracking
    *        via two_mult_fma + two_sum (3-piece pi/2, ~70 bits)
    *      - Large (|x| >= 2^20): controlled by __FPMP_LARGE_TRIG_FP64_FALLBACK__
    *        = 0 (default): Payne-Hanek using integer 2/pi table, combining x_hi
    *             and x_lo fractions in 64-bit fixed-point before
    *             converting to fp32mp2 (pure fp32 arithmetic, no fp64).
    *             Delivers ~46 bits in the reduced argument r.
    *        = 1: fall back to system fp64 sin/cos. Final precision capped by
    *             fp64; tan near singularities is further limited by tan'
    *             amplification of the fp32mp2 input quantization.
    *   2. Evaluate sin(r) and cos(r) via Taylor polynomials in fp32mp2
    *      sin: 8 terms (x through x^15), cos: 9 terms (1 through x^16)
    *   3. Map to correct quadrant using n mod 4
    *      sincos computes both kernels; sin/cos call sincos internally
    * ============================================================================
    */

#if (__FPMP_LARGE_TRIG_FP64_FALLBACK__ == 0)

    /*
    * Payne-Hanek stage 1: compute |a| * (2/pi) via integer arithmetic,
    * extract 2-bit quadrant and 62-bit unsigned fraction in [0, 1).
    * Does NOT apply the >0.5 adjustment -- caller handles that.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_fpmp2_ph_frac(
        FpType a_hi, unsigned* q_out, uint32_t* frac_hi, uint32_t* frac_lo)
    {
        using namespace fpmp;

        constexpr unsigned int i2opi[] = 
        {
            0x3c439041U, 0xdb629599U, 0xf534ddc0U,
            0xfc2757d1U, 0x4e441529U, 0xa2f9836eU,
        };

        uint32_t ia = internal_bit_cast<uint32_t>(a_hi);
        uint32_t result[7];
        uint32_t hi, lo;
        int iq;

        int e = (int)((ia >> 23U) & 0xFFU) - 128;
        ia = (ia << 8U) | 0x80000000U;
        hi = 0;

        for (iq = 0; iq < 6; iq++) 
        {
            uint64_t p = (uint64_t)i2opi[iq] * ia + hi;
            result[iq] = (uint32_t)p;
            hi = (uint32_t)(p >> 32);
        }
        result[iq] = hi;

        /* Extract the window containing quadrant + fraction bits.
         * For e >= 0 (|a| >= 2): standard extraction with left shift.
         * For e < 0 (|a| < 2): extraction from idx=4 with right shift
         *   to handle small inputs without extending the table.
         */
        uint32_t lo2;
        if (e >= 0) 
        {
            uint32_t ue = (uint32_t)e;
            uint32_t idx = 4U - (ue >> 5U);
            ue = ue & 31U;
            hi = result[idx + 2];
            lo = result[idx + 1];
            lo2 = (idx > 0) ? result[idx] : 0U;
            if (ue != 0U) 
            {
                uint32_t q = 32U - ue;
                hi = (hi << ue) + (lo >> q);
                lo = (lo << ue) + (lo2 >> q);
            }
        } 
        else 
        {
            int r = -e;
            hi = result[6];
            lo = result[5];
            lo2 = result[4];
            if (r < 32) 
            {
                uint32_t q = (uint32_t)(32 - r);
                uint32_t ur = (uint32_t)r;
                hi = hi >> ur;
                lo = (lo >> ur) | (result[6] << q);
            } 
            else 
            {
                hi = 0;
                lo = result[6];
            }
        }

        *q_out   = hi >> 30U;
        *frac_hi = (hi << 2U) + (lo >> 30U);
        *frac_lo = (lo << 2U);
    }

    /*
    * Payne-Hanek stage 2: convert a 64-bit unsigned fraction in [0, 0.5)
    * (after the >0.5 adjustment) to an fp32mp2 angle by multiplying
    * by pi/2 using 64 bits of pi/4.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_fpmp2_frac_to_angle(
        uint32_t hi, uint32_t lo, uint32_t s,
        FpType* r_hi, FpType* r_lo)
    {
        using namespace fpmp;

        /* Normalize: shift so MSB of hi is 1.
         * Handle hi == 0 separately to avoid shift-by-32 UB.
         */
        #ifdef __CUDA_ARCH__
        uint32_t lz = __clz((int)hi);
        #else
        uint32_t lz = (hi == 0U) ? 32U : (uint32_t)__builtin_clz(hi);
        #endif

        if (lz >= 32U) 
        {
            lz += (lo == 0U) ? 0U :
            #ifdef __CUDA_ARCH__
                (uint32_t)__clz((int)lo);
            #else
                (uint32_t)__builtin_clz(lo);
            #endif
            hi = lo; lo = 0U;
            uint32_t shift = lz - 32U;
            if (shift != 0U) { hi <<= shift; }
        } 
        else if (lz != 0U) 
        {
            hi = (hi << lz) | (lo >> (32U - lz));
            lo = lo << lz;
        }

        /* Multiply by pi/2 using 64 bits of pi/4.
         * pi/4 = 0x0.C90FDAA2_2168C234...
         * The *2 (pi/4 -> pi/2) is in biased_exp = 127 - lz.
         */
        constexpr uint32_t PIO4_HI32 = 0xC90FDAA2U;
        constexpr uint32_t PIO4_LO32 = 0x2168C234U;

        uint64_t p_hh = (uint64_t)hi * PIO4_HI32;
        uint64_t p_hl = (uint64_t)hi * PIO4_LO32;
        uint64_t p_lh = (uint64_t)lo * PIO4_HI32;

        uint64_t combined = p_hh + (p_hl >> 32) + (p_lh >> 32);
        uint32_t rhi = (uint32_t)(combined >> 32);
        uint32_t rlo = (uint32_t)combined;

        if ((int32_t)rhi > 0) 
        {
            rhi = (rhi << 1) | (rlo >> 31);
            rlo = rlo << 1;
            lz++;
        }

        /* Convert to fp32mp2 */
        uint32_t biased_exp = 127U - lz;
        uint32_t f1_bits = s | (biased_exp << 23) | ((rhi >> 8) & 0x7FFFFFU);

        uint32_t rem       = (rhi << 24) | (rlo >> 8);
        uint32_t rem_extra = rlo << 24;

        if (rem == 0U) 
        {
            *r_hi = internal_bit_cast<FpType>(f1_bits);
            *r_lo = FpType(0);
            return;
        }

        #ifdef __CUDA_ARCH__
        uint32_t rlz = __clz((int)rem);
        #else
        uint32_t rlz = (uint32_t)__builtin_clz(rem);
        #endif

        uint32_t rem_norm = (rlz > 0U)
            ? ((rem << rlz) | (rem_extra >> (32U - rlz)))
            : rem;

        int biased_exp2 = (int)biased_exp - 24 - (int)rlz;
        if (biased_exp2 < 1) 
        {
            *r_hi = internal_bit_cast<FpType>(f1_bits);
            *r_lo = FpType(0);
        } 
        else 
        {
            uint32_t f2_bits = s | ((uint32_t)biased_exp2 << 23)
                                 | ((rem_norm >> 8) & 0x7FFFFFU);
            *r_hi = internal_bit_cast<FpType>(f1_bits);
            *r_lo = internal_bit_cast<FpType>(f2_bits);
        }
    }

#endif /* __FPMP_LARGE_TRIG_FP64_FALLBACK__ == 0 */

    /*
    * Trigonometric argument reduction for fp32mp2.
    * Returns quadrant (mod 4) and reduced argument r  in  [-pi/4, pi/4].
    *
    * Three paths:
    *   Tiny:  |x| < pi/4 -> no reduction
    *   Fast:  |x| < 2^20 -> Cody-Waite with exact error tracking
    *   Large: |x| >= 2^20 -> Payne-Hanek (integer 2/pi table)
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_fpmp2_trig_reduction(
        FpType x_hi, FpType x_lo,
        int* quadrant, FpType* r_hi, FpType* r_lo)
    {
        using namespace fpmp;
        using afloat = fp32mp2_high;

        FpType abs_hi = (x_hi < FpType(0)) ? -x_hi : x_hi;
        uint32_t abs_bits = internal_bit_cast<uint32_t>(abs_hi);

        /* No reduction for |x| < pi/4 */
        if (abs_bits < 0x3F490FDBU) 
        {
            *quadrant = 0;
            *r_hi = x_hi;
            *r_lo = x_lo;
            return;
        }

        /* Inf / NaN -> return NaN, quadrant 0 */
        if (abs_bits >= 0x7F800000U) 
        {
            *quadrant = 0;
            *r_hi = x_hi - x_hi;
            *r_lo = FpType(0);
            return;
        }

        if (abs_bits < 0x49800000U) 
        {
            /* -- Fast path: Cody-Waite for |x_hi| < 2^20 --
             *
             * pi/2 split into 3 float pieces (~70 bits) from libdevice.
             * C1 has 2 trailing zero mantissa bits, making n*C1 exact
             * for |n| < 2^12 via two_mult_fma, and accurate for larger n.
             *
             * Error tracking: two_mult_fma gives exact n*C1 = ph + pl,
             * two_sum gives exact x_hi - ph = s + e.
             * Remaining corrections accumulated in fp32mp2_high
             * to preserve ~46 bits when s is near zero (catastrophic
             * cancellation near multiples of pi).
             */
            constexpr FpType C1 = FpType(1.5707962512969971e+000);
            constexpr FpType C2 = FpType(7.5497894158615964e-008);
            constexpr FpType C3 = FpType(5.3903029534742384e-015);

            int n = fp2int_rn(x_hi * FpType(0x1.45f306p-1f));
            FpType n_f = int2fp_rn<FpType>(n);

            /* Exact product n*C1 = ph + pl */
            FpType pl;
            FpType ph = two_mult_fma(n_f, C1, &pl);

            /* Exact subtraction x_hi - ph = s + e */
            FpType e;
            FpType s = two_sum(x_hi, -ph, &e);

            /* Build result as fp32mp2_high from exact (s, e),
             * then accumulate corrections with full precision.
             */
            afloat result(s, e);
            result = result + afloat(-pl);
            result = result + afloat(x_lo);

            /* Exact product n*C2 = nC2_hi + nC2_lo via two_mult_fma */
            FpType nC2_lo;
            FpType nC2_hi = two_mult_fma(n_f, C2, &nC2_lo);
            result = result - afloat(nC2_hi, nC2_lo);

            /* n*C3 is tiny (~10^-11), single-precision product suffices */
            result = result + afloat(mul_rn(n_f, -C3));

            *quadrant = n;
            *r_hi = result.hi();
            *r_lo = result.lo();
        } 
        else 
        {
            /* -- Slow path: |x_hi| >= 2^20 -- */

#if (__FPMP_LARGE_TRIG_FP64_FALLBACK__ == 0)
            /* Payne-Hanek: combine x_hi and x_lo 2/pi fractions in
             * 64-bit fixed-point BEFORE the pi/2 multiply to avoid
             * precision loss from floating-point cancellation.
             */
            uint32_t fhi, flo;
            unsigned q_hi;
            __internal_fpmp2_ph_frac(x_hi, &q_hi, &fhi, &flo);

            uint32_t x_hi_sign = internal_bit_cast<uint32_t>(x_hi) & 0x80000000U;
            int q = (int)q_hi;

            /* Add x_lo contribution in fixed-point.
             * |x_lo| <= |x_hi|*2^-24 can still span many quadrants,
             * and even small |x_lo| can dominate the fraction when
             * the result angle is near zero.  Handle ALL non-zero x_lo.
             */
            if (x_lo != FpType(0)) 
            {
                FpType abs_lo = (x_lo < FpType(0)) ? -x_lo : x_lo;
                uint32_t abs_lo_bits = internal_bit_cast<uint32_t>(abs_lo);
                bool same_sign = (x_lo > FpType(0)) == (x_hi > FpType(0));

                uint32_t fhi2 = 0, flo2 = 0;
                unsigned q_lo = 0;

                if (abs_lo_bits >= 0x00800000U) 
                {
                    __internal_fpmp2_ph_frac(abs_lo, &q_lo, &fhi2, &flo2);
                }

                uint64_t f1 = ((uint64_t)fhi << 32) | flo;
                uint64_t f2 = ((uint64_t)fhi2 << 32) | flo2;

                if (same_sign) 
                {
                    q += (int)q_lo;
                    uint64_t sum = f1 + f2;
                    if (sum < f1) q++;
                    fhi = (uint32_t)(sum >> 32);
                    flo = (uint32_t)sum;
                } 
                else 
                {
                    q -= (int)q_lo;
                    if (f1 >= f2) 
                    {
                        f1 -= f2;
                    } 
                    else 
                    {
                        f1 = 0ULL - (f2 - f1);
                        q--;
                    }
                    fhi = (uint32_t)(f1 >> 32);
                    flo = (uint32_t)f1;
                }
            }

            uint32_t top_bit = fhi >> 31U;
            q += top_bit;
            if (x_hi_sign != 0U) q = 0U - (unsigned)q;

            if (top_bit != 0U) 
            {
                fhi = ~fhi;
                flo = ~flo;
                x_hi_sign ^= 0x80000000U;
            }

            if (fhi == 0U && flo == 0U) 
            {
                *quadrant = (int)q;
                *r_hi = FpType(0);
                *r_lo = FpType(0);
                return;
            }

            *quadrant = (int)q;
            __internal_fpmp2_frac_to_angle(fhi, flo, x_hi_sign, r_hi, r_lo);
#endif /* __FPMP_LARGE_TRIG_FP64_FALLBACK__ */
        }
    }

    /*
    * Sin kernel: evaluate sin(x) for |x| <= pi/4 using fp32mp2 Taylor series.
    * sin(x) = x + x^3*Q(x^2), Q(u) = Sum Taylor coefficients from -1/3! to -1/15!.
    * Upper terms (s7..s4) in single precision, lower terms (s3..s1) in fp32mp2.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_fpmp2_sin_kernel(
        FpType x_hi, FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using ffloat = fp32mp2_low;

        constexpr ffloat s1(-1.6666666666666666e-01);
        constexpr ffloat s2( 8.3333333333333333e-03);
        constexpr ffloat s3(-1.9841269841269841e-04);
        constexpr float  s4( 2.7557319223985893e-06f);
        constexpr float  s5(-2.5052108385441719e-08f);
        constexpr float  s6( 1.6059043836821615e-10f);
        constexpr float  s7(-7.6471637318198165e-13f);

        ffloat x(x_hi, x_lo);
        ffloat x2 = x * x;
        float  x2f = x2.hi();

        float qf = s7;
        qf = fpmp::fma_rn(qf, x2f, s6);
        qf = fpmp::fma_rn(qf, x2f, s5);
        qf = fpmp::fma_rn(qf, x2f, s4);

        ffloat q = qf * x2 + s3;
        q = q * x2 + s2;
        q = q * x2 + s1;

        ffloat result = renormalize(q * x2 * x + x);
        *res_hi = result.hi();
        *res_lo = result.lo();
    }

    /*
    * Cos kernel: evaluate cos(x) for |x| <= pi/4 using fp32mp2 Taylor series.
    * cos(x) = 1 + x^2*Q(x^2), Q(u) = Sum Taylor coefficients from -1/2! to 1/16!.
    * Upper terms (c8..c4) in single precision, lower terms (c3..c1) in fp32mp2.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_fpmp2_cos_kernel(
        FpType x_hi, FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using ffloat = fp32mp2_low;

        constexpr ffloat c1(-5.0000000000000000e-01);
        constexpr ffloat c2( 4.1666666666666667e-02);
        constexpr ffloat c3(-1.3888888888888889e-03);
        constexpr float  c4( 2.4801587301587302e-05f);
        constexpr float  c5(-2.7557319223985893e-07f);
        constexpr float  c6( 2.0876756987868099e-09f);
        constexpr float  c7(-1.1470745597729725e-11f);
        constexpr float  c8( 4.7794773323873853e-14f);

        ffloat x(x_hi, x_lo);
        ffloat x2 = x * x;
        float  x2f = x2.hi();

        float qf = c8;
        qf = fpmp::fma_rn(qf, x2f, c7);
        qf = fpmp::fma_rn(qf, x2f, c6);
        qf = fpmp::fma_rn(qf, x2f, c5);
        qf = fpmp::fma_rn(qf, x2f, c4);

        ffloat q = qf * x2 + c3;
        q = q * x2 + c2;
        q = q * x2 + c1;

        ffloat result = renormalize(q * x2 + ffloat(FpType(1)));
        *res_hi = result.hi();
        *res_lo = result.lo();
    }

    /*
    * sincos for fp32mp2: compute sin(x) and cos(x) simultaneously.
    * Shared argument reduction, separate sin/cos kernels on [-pi/4, pi/4],
    * quadrant-based swap and sign adjustment (matching libdevice structure).
    *
    * When __FPMP_LARGE_TRIG_FP64_FALLBACK__ == 1, arguments with |x| >= 2^20
    * fall back to system fp64 sin/cos (avoids the Payne-Hanek code).
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_sincos(
        const FpType x_hi, const FpType x_lo,
        FpType* sin_hi, FpType* sin_lo,
        FpType* cos_hi, FpType* cos_lo)
    {
#if (__FPMP_LARGE_TRIG_FP64_FALLBACK__ == 1)
        FpType abs_hi = (x_hi < FpType(0)) ? -x_hi : x_hi;
        uint32_t abs_bits = fpmp::internal_bit_cast<uint32_t>(abs_hi);
        if (abs_bits >= 0x49800000U) 
        {
            using mp2_t = fpmp2_t<FpType>;
            double xd = static_cast<double>(mp2_t(x_hi, x_lo));
            double sd = ::sin(xd), cd = ::cos(xd);
            /* Split each fp64 result into (hi, lo) via the fp32mp2(double)
             * constructor -- casting to FpType first would drop the lo bits
             * and silently cap precision at ~24 bits instead of ~46.
             */
            mp2_t s_mp(sd);
            mp2_t c_mp(cd);
            *sin_hi = s_mp.hi(); *sin_lo = s_mp.lo();
            *cos_hi = c_mp.hi(); *cos_lo = c_mp.lo();
            return;
        }
#endif

        int quadrant;
        FpType r_hi, r_lo;
        __internal_fpmp2_trig_reduction(x_hi, x_lo, &quadrant, &r_hi, &r_lo);

        FpType s_hi, s_lo, c_hi, c_lo;
        __internal_fpmp2_sin_kernel(r_hi, r_lo, &s_hi, &s_lo);
        __internal_fpmp2_cos_kernel(r_hi, r_lo, &c_hi, &c_lo);

        int q = quadrant & 3;
        if (q < 0) q += 4;

        if (q & 1) 
        {
            FpType t;
            t = s_hi; s_hi = c_hi; c_hi = t;
            t = s_lo; s_lo = c_lo; c_lo = t;
        }
        if (q == 1 || q == 2) 
        {
            c_hi = -c_hi;
            c_lo = -c_lo;
        }
        if (q == 2 || q == 3) 
        {
            s_hi = -s_hi;
            s_lo = -s_lo;
        }

        *sin_hi = s_hi; *sin_lo = s_lo;
        *cos_hi = c_hi; *cos_lo = c_lo;
    }

    /*
    * sin for fp32mp2: calls sincos and returns only the sine.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_sin(
        const FpType x_hi, const FpType x_lo,
        FpType* res_hi, FpType* res_lo)
    {
        FpType c_hi, c_lo;
        __nv_fpmp2_sincos(x_hi, x_lo, res_hi, res_lo, &c_hi, &c_lo);
    }

    /*
    * cos for fp32mp2: calls sincos and returns only the cosine.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_cos(
        const FpType x_hi, const FpType x_lo,
        FpType* res_hi, FpType* res_lo)
    {
        FpType s_hi, s_lo;
        __nv_fpmp2_sincos(x_hi, x_lo, &s_hi, &s_lo, res_hi, res_lo);
    }

    /*
    * tan for fp32mp2: dedicated implementation modeled after libdevice
    * __nv_tan, but composed entirely from fp32mp2 primitives.
    *
    * Algorithm (no FP64 dependency on the hot path):
    *   1. Reduce x to r in [-pi/4, pi/4] via the shared
    *      __internal_fpmp2_trig_reduction; this also returns the quadrant
    *      index q (modulo 4).
    *   2. Evaluate sin(r) and cos(r) on the reduced interval via the
    *      shared __internal_fpmp2_sin_kernel / __internal_fpmp2_cos_kernel.
    *   3. tan has period pi, so only the LSB of q matters:
    *        q even  ->  tan(x) =  sin(r) / cos(r)
    *        q odd   ->  tan(x) = -cos(r) / sin(r)        (= -cot(r))
    *      The full quadrant-mod-4 sign dance used by sincos is unnecessary
    *      here because tan(x + pi) = tan(x) absorbs the q == 2,3 sign
    *      flips that sincos performs on its sin/cos outputs.
    *
    * Cost relative to libdevice __nv_tan: one shared reduction + one
    * sin kernel + one cos kernel + one fp32mp2 division.  Reusing the
    * already-tuned sin/cos kernels inherits their ~46-bit accuracy
    * envelope without having to fit a separate tan polynomial.
    *
    * Singularities at x === pi/2 (mod pi) produce +-inf through the q-odd
    * branch when sin(r) underflows to zero (matches the IEEE / libdevice
    * convention; signed-infinity direction follows the rounded reduced
    * argument).  Inf / NaN inputs propagate to NaN through the reduction.
    *
    * Optional large-|x| FP64 fallback (mirrors __nv_fpmp2_sincos): when
    * __FPMP_LARGE_TRIG_FP64_FALLBACK__ == 1 and |x_hi| >= 2^20, delegate to
    * the system ::tan to keep the dedicated path off the Payne-Hanek
    * reducer for extreme arguments.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_tan(
        const FpType x_hi, const FpType x_lo,
        FpType* res_hi, FpType* res_lo)
    {
#if (__FPMP_LARGE_TRIG_FP64_FALLBACK__ == 1)
        FpType abs_hi = (x_hi < FpType(0)) ? -x_hi : x_hi;
        uint32_t abs_bits = fpmp::internal_bit_cast<uint32_t>(abs_hi);
        if (abs_bits >= 0x49800000U)   /* |x_hi| >= 2^20 */
        {
            using mp2_t = fpmp2_t<FpType>;
            double xd = static_cast<double>(mp2_t(x_hi, x_lo));
            double td = ::tan(xd);
            /* Split the fp64 result into (hi, lo) via the fp32mp2(double)
             * constructor -- casting to FpType first would drop the lo bits
             * and silently cap precision at ~24 bits instead of ~46.
             */
            mp2_t r_mp(td);
            *res_hi = r_mp.hi(); *res_lo = r_mp.lo();
            return;
        }
#endif

        int quadrant;
        FpType r_hi, r_lo;
        __internal_fpmp2_trig_reduction(x_hi, x_lo, &quadrant, &r_hi, &r_lo);

        FpType s_hi, s_lo, c_hi, c_lo;
        __internal_fpmp2_sin_kernel(r_hi, r_lo, &s_hi, &s_lo);
        __internal_fpmp2_cos_kernel(r_hi, r_lo, &c_hi, &c_lo);

        using mp2_t = fpmp2_t<FpType>;
        mp2_t s(s_hi, s_lo);
        mp2_t c(c_hi, c_lo);

        mp2_t result = (quadrant & 1) ? mp2_t(-c / s) : mp2_t(s / c);

        *res_hi = result.hi();
        *res_lo = result.lo();
    }

    /*
    * ============================================================================
    * Inverse trigonometric functions: asin, acos, atan, atan2 (fp32mp2)
    *                                                            - dedicated
    * ============================================================================
    *
    * All four functions are built on two shared polynomial kernels evaluated
    * in fp32mp2 arithmetic.  Coefficients are the libdevice fp64 minimax fits
    * (see `__internal_atan_kernel`, `__internal_asin_kernel`,
    *  acos-large-branch poly in device_functions_impl.c); their truncation
    * noise is at fp64 ulp, well below the fp32mp2 ulp.  We evaluate them in
    * pure fp32mp2 Horner (M = 0): a coefficient as small as 2*10^-5
    * (atan c_18) still carries ~5*10^-13 of float-rounding noise when stored
    * as `float`, which is two decimals above the fp32mp2 ulp at the
    * |a| <= 1 boundary.  Mixed-precision Horner would require either a
    * finer reduction (|a| <= tan(pi/8) for atan, etc.) or refit coefficients;
    * pure-mp2 Horner buys us simplicity and full precision at the cost of
    * ~4* the float-only kernel ops.  These functions are not on the hottest
    * fp32mp2 paths, so the trade-off is favourable.
    *
    *   atan(x):   |x| > 1 -> atan(x) = sign(x)*(pi/2 - atan(1/|x|))
    *              |x| <= 1 -> polynomial Horner in x^2, 19 coefficients.
    *
    *   atan2(y,x):  octant analysis on (|y|,|x|), call atan_kernel on
    *              min/max ratio, reconstruct via pi and pi/2 anchors.
    *
    *   asin(x):   |x| < 0.575 -> polynomial in x^2, 13 coefficients;
    *                            asin(x) = x + x*(x^2*P(x^2))
    *              |x| >= 0.575 -> y = (1-|x|)/2;
    *                            asin(|x|) = pi/2 - 2*sqrty*(1 + y*P(y))
    *                            sign restored at the end.
    *
    *   acos(x):   |x| < 0.575 -> reuse asin polynomial,
    *                            acos(x) = pi/2 - asin(x)
    *              |x| >= 0.575 -> dedicated polynomial in y = 1 - |x|,
    *                            acos(|x|) = sqrt(2y)*(1 + y*P(y));
    *                            x < 0  -> acos(x) = pi - acos(|x|).
    *
    * Domain checks: NaN inputs propagate through arithmetic; |x| > 1
    * inputs to asin/acos return NaN via the sqrt of a negative y.
    * atan(+-inf) returns +-pi/2 via the 1/x reduction (1/+-inf -> +-0).
    * atan2 handles (0,0), (+-inf,+-inf) special cases explicitly.
    * ============================================================================
    */

    /* ---- (kernel 1) atan on |a| <= 1, returns atan(|a|) in fp32mp2 ---- */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_fpmp2_atan_kernel(
        const fp32mp2_low& a,
        fp32mp2_low* result)
    {
        using ffloat = fp32mp2_low;

        /* 19-coefficient libdevice fp64 minimax fit; ascending degree.
         * Polynomial P(a^2) such that atan(a) = a*(1 + a^2*P(a^2)). */
        constexpr ffloat atan_c[19] = {
            ffloat(-3.3333333333331860e-01), /* c0  */
            ffloat( 1.9999999999755019e-01), /* c1  */
            ffloat(-1.4285714271334815e-01), /* c2  */
            ffloat( 1.1111110678749424e-01), /* c3  */
            ffloat(-9.0909012354005225e-02), /* c4  */
            ffloat( 7.6922129305867837e-02), /* c5  */
            ffloat(-6.6658603633512573e-02), /* c6  */
            ffloat( 5.8773077721790849e-02), /* c7  */
            ffloat(-5.2392330054601317e-02), /* c8  */
            ffloat( 4.6739496199157994e-02), /* c9  */
            ffloat(-4.0926382420509971e-02), /* c10 */
            ffloat( 3.4067811082715123e-02), /* c11 */
            ffloat(-2.5826796814495994e-02), /* c12 */
            ffloat( 1.6978035834597331e-02), /* c13 */
            ffloat(-9.1845592187165485e-03), /* c14 */
            ffloat( 3.8559749383629918e-03), /* c15 */
            ffloat(-1.1640717779930576e-03), /* c16 */
            ffloat( 2.2302240345758510e-04), /* c17 */
            ffloat(-2.0258553044438358e-05), /* c18 */
        };

        ffloat a2 = a * a;
        ffloat q  = fpmp::poly_eval<fpmp::poly_method::horner_comp>(a2, atan_c);
        *result   = renormalize(a + a * (a2 * q));
    }

    /* ---- (kernel 2) asin polynomial P(y); used by both asin & acos ---- */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_fpmp2_asin_poly(
        const fpmp2_t<FpType>& y,
        fpmp2_t<FpType>* result)
    {
        using ffloat = fp32mp2_low;

        ffloat y_fast(y.hi(), y.lo());

        /* 13-coefficient libdevice fp64 minimax fit; ascending degree.
         * Polynomial P(y) such that asin(z)/z - 1 ~= z^2*P(z^2) for small z,
         * and pi/2 - asin(|x|) = 2*sqrty*(1 + y*P(y)) for y = (1-|x|)/2. */
        constexpr ffloat asin_c[13] = {
            ffloat( 1.666666666667375e-01), /* c0  */
            ffloat( 7.499999998342270e-02), /* c1  */
            ffloat( 4.464285849810986e-02), /* c2  */
            ffloat( 3.038188875134962e-02), /* c3  */
            ffloat( 2.237350511593569e-02), /* c4  */
            ffloat( 1.733194598980628e-02), /* c5  */
            ffloat( 1.418108777515123e-02), /* c6  */
            ffloat( 1.000422754245580e-02), /* c7  */
            ffloat( 1.745227928732326e-02), /* c8  */
            ffloat(-1.787828218369301e-02), /* c9  */
            ffloat( 6.686894879337643e-02), /* c10 */
            ffloat(-7.620591484676952e-02), /* c11 */
            ffloat( 6.259798167646803e-02), /* c12 */
        };

        ffloat q = fpmp::poly_eval<fpmp::poly_method::horner_comp>(y_fast, asin_c);
        fpmp2_t<FpType> res(q.hi(), q.lo());
        *result = res;
    }

    /* ---- (kernel 3) acos large-branch polynomial P(y); used by acos only ----
     *
     * Companion to `__internal_fpmp2_asin_poly` for the |x| >= 0.575 branch
     * of acos.  Evaluates the 13-coefficient libdevice fp64 minimax fit
     * P(y) such that, for y = 1 - |x|, acos(|x|) = sqrt(2y)*(1 + y*P(y)).
     * Same fp32mp2_low internal evaluation as the asin kernel -- no
     * per-op renormalisation, single conversion in/out around the call. */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __internal_fpmp2_acos_poly(
        const fpmp2_t<FpType>& y,
        fpmp2_t<FpType>* result)
    {
        using ffloat = fp32mp2_low;

        ffloat y_fast(y.hi(), y.lo());

        constexpr ffloat acos_c[13] = {
            ffloat( 8.3333333333333329e-02), /* c0  */
            ffloat( 1.8749999999999475e-02), /* c1  */
            ffloat( 5.5803571429249681e-03), /* c2  */
            ffloat( 1.8988715243469585e-03), /* c3  */
            ffloat( 6.9913006155254860e-04), /* c4  */
            ffloat( 2.7113554445344455e-04), /* c5  */
            ffloat( 1.0911426300865435e-04), /* c6  */
            ffloat( 4.5031965455307141e-05), /* c7  */
            ffloat( 1.9480663162164715e-05), /* c8  */
            ffloat( 6.9283438595562408e-06), /* c9  */
            ffloat( 6.1185294127269731e-06), /* c10 */
            ffloat(-1.5951212865388395e-06), /* c11 */
            ffloat( 2.7519189493111718e-06), /* c12 */
        };

        ffloat q = fpmp::poly_eval<fpmp::poly_method::horner_comp>(y_fast, acos_c);
        fpmp2_t<FpType> res(q.hi(), q.lo());
        *result = res;
    }

    /* ---- atan(x) ---- */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_atan(
        const FpType x_hi, const FpType x_lo,
        FpType* res_hi, FpType* res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_atan is fp32mp2 only; "
                      "fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        const bool is_neg = x_hi < FpType(0);
        ffloat x(x_hi, x_lo);
        ffloat absx = is_neg ? -x : x;

        /* |x| > 1: use atan(x) = pi/2 - atan(1/x).  This includes |x| = inf,
         * which gives 1/x = 0, atan(0) = 0, result = pi/2. */
        const bool large = absx.hi() > FpType(1);
        ffloat a = large ? (ffloat(FpType(1)) / absx) : absx;

        ffloat r;
        __internal_fpmp2_atan_kernel<FpType>(a, &r);

        if (large) {
            constexpr ffloat PIO2(1.5707963267948966); /* pi/2 split into hi+lo */
            r = PIO2 - r;
        }
        if (is_neg) r = -r;

        *res_hi = r.hi();
        *res_lo = r.lo();
    }

    /* ---- atan2(y, x) ---- */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_atan2(
        const FpType y_hi, const FpType y_lo,
        const FpType x_hi, const FpType x_lo,
        FpType* res_hi, FpType* res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_atan2 is fp32mp2 only; "
                      "fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* Signed-zero / signed-infinity safe sign probes via the sign bit
         * (a plain `x_hi < 0` test would return false for -0.0). */
        const uint32_t x_bits   = fpmp::internal_bit_cast<uint32_t>(x_hi);
        const uint32_t y_bits   = fpmp::internal_bit_cast<uint32_t>(y_hi);
        const bool     x_is_neg = (x_bits & 0x80000000U) != 0U;
        const bool     y_is_neg = (y_bits & 0x80000000U) != 0U;

        /* NaN propagation: any NaN component (in either hi or lo) forces
         * a NaN result.  Use self-inequality so the test doesn't falsely
         * fire on Inf + (-Inf) intermediates. */
        const bool x_has_nan = (x_hi != x_hi) || (x_lo != x_lo);
        const bool y_has_nan = (y_hi != y_hi) || (y_lo != y_lo);
        if (x_has_nan || y_has_nan) {
            const FpType nan_val = x_has_nan ? (x_hi + x_lo) : (y_hi + y_lo);
            *res_hi = nan_val;
            *res_lo = nan_val;
            return;
        }

        ffloat y(y_hi, y_lo);
        ffloat x(x_hi, x_lo);
        ffloat ay = y_is_neg ? -y : y;
        ffloat ax = x_is_neg ? -x : x;

        /* |a| == +inf  <->  bit-pattern 0x7f800000 (with sign bit already
         * stripped by the abs above). */
        const bool x_is_inf = (fpmp::internal_bit_cast<uint32_t>(ax.hi()) == 0x7f800000U);
        const bool y_is_inf = (fpmp::internal_bit_cast<uint32_t>(ay.hi()) == 0x7f800000U);

        /* Special cases.  IEEE-754 + C99 sectionF.10.1.4 atan2 semantics:
         *   atan2(+-0, +0)    = +-0           (preserves sign of y)
         *   atan2(+-0, -0)    = +-pi
         *   atan2(+-0, x>0)   = +-0
         *   atan2(+-0, x<0)   = +-pi
         *   atan2(+-inf, +-inf)    = +-pi/4, +-3pi/4  (signs decided per quadrant)
         *   atan2(+-inf, x finite)  = +-pi/2
         *   atan2(y finite, +inf)  = +-0
         *   atan2(y finite, -inf)  = +-pi
         *
         * NOTE on signed zero handling.  The test-framework reference is
         * computed as `atan2(double(y_hi+y_lo), double(x_hi+x_lo))`, which
         * collapses each fp32mp2 input to a single fp64 value before atan2.
         * The collapsed value's sign is the sign of the *sum*, which can
         * differ from the sign of `hi` alone (e.g. `(+0,-0)` collapses to
         * `+0`, but `(-0,-0)` collapses to `-0`).  To match the reference
         * we therefore probe the collapsed sign for any argument whose
         * `hi` is zero and route the sign decisions through that. */
        constexpr ffloat PI    (3.141592653589793);
        constexpr ffloat PIO2  (1.5707963267948966);
        constexpr ffloat PIO4  (0.7853981633974483);
        constexpr ffloat PI3O4 (2.356194490192345);   /* 3pi/4 */

        /* Effective (collapsed) sign of y, used whenever y_hi == 0.  When
         * y_hi != 0, IEEE `y_is_neg` is the right answer because the
         * collapsed sign matches `hi`'s sign for any normal value. */
        const FpType   y_sum     = y_hi + y_lo;
        const uint32_t y_sum_bits = fpmp::internal_bit_cast<uint32_t>(y_sum);
        const bool     y_eff_neg = (y_sum_bits & 0x80000000U) != 0U;

        ffloat r;
        if (ax.hi() == FpType(0) && ay.hi() == FpType(0)) {
            /* Both magnitudes "zero" at the high component.  The reference
             * still distinguishes +-0 by the collapsed sign of x, so honour
             * that here:  x_collapsed >= +0 -> r = +0;  x_collapsed = -0 -> r = pi
             * (the framework's `atan2(+-0, -0)` returns +-pi). */
            const FpType   x_sum      = x_hi + x_lo;
            const uint32_t x_sum_bits = fpmp::internal_bit_cast<uint32_t>(x_sum);
            const bool     x_eff_neg  = (x_sum_bits & 0x80000000U) != 0U;
            r = x_eff_neg ? PI : ffloat(FpType(0));
        } else if (x_is_inf && y_is_inf) {
            /* Both infinite: 45deg / 135deg depending on x sign. */
            r = x_is_neg ? PI3O4 : PIO4;
        } else if (y_is_inf) {
            /* |y| = inf, |x| finite:  result = +-pi/2 (sign from y). */
            r = PIO2;
        } else if (x_is_inf) {
            /* |x| = inf, |y| finite:  result = +-0 or +-pi depending on sign of x.
             * Skipping the division avoids NaN from `finite / Inf` in
             * fp32mp2's renormalisation step. */
            r = x_is_neg ? PI : ffloat(FpType(0));
        } else {
            /* Generic finite path: atan(num/den), then octant fixup. */
            const bool y_gt_x = ay.hi() > ax.hi();
            ffloat num = y_gt_x ? ax : ay;
            ffloat den = y_gt_x ? ay : ax;
            ffloat t = div<fpmp2_accuracy::def>(num, den);
            __internal_fpmp2_atan_kernel<FpType>(t, &t);

            if (y_gt_x) {
                /* |y| > |x|:  result = +-pi/2 -/+ atan(|x|/|y|) */
                r = x_is_neg ? (PIO2 + t) : (PIO2 - t);
            } else if (x_is_neg) {
                /* |y| <= |x|, x < 0:  result = pi - atan(|y|/|x|) */
                r = PI - t;
            } else {
                /* |y| <= |x|, x >= 0:  result =     atan(|y|/|x|) */
                r = t;
            }
        }

        /* Apply sign of y (mirror across x-axis).  When y_hi is exactly
         * zero, `y_is_neg` reflects only the sign bit of `hi`, but the
         * reference's `double(y_hi+y_lo)` collapse may yield a different
         * sign, so use `y_eff_neg` for the y_hi == 0 case. */
        const bool y_apply_neg = (y_hi == FpType(0)) ? y_eff_neg : y_is_neg;
        if (y_apply_neg) r = -r;

        *res_hi = r.hi();
        *res_lo = r.lo();
    }

    /* ---- asin(x) ---- */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_asin(
        const FpType x_hi, const FpType x_lo,
        FpType* res_hi, FpType* res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_asin is fp32mp2 only; "
                      "fp64mp2 has its own specialization");

        using ffloat = fpmp2_t<FpType>;

        const bool is_neg = x_hi < FpType(0);
        ffloat x(x_hi, x_lo);
        ffloat absx = is_neg ? -x : x;

        /* Crossover at |x| ~= 0.575 (libdevice fp64 choice; threshold is
         * the boundary above which the small-branch polynomial loses
         * conditioning and the large-branch sqrt reconstruction wins). */
        constexpr FpType BRANCH = FpType(0.575f);

        ffloat r;
        if (absx.hi() < BRANCH) {
            /* Small branch: asin(|x|) = |x| + |x|*(|x|^2*P(|x|^2)) */
            ffloat a2 = absx * absx;
            ffloat p;
            __internal_fpmp2_asin_poly<FpType>(a2, &p);
            r = renormalize(absx + absx * (a2 * p));
        } else {
            /* Large branch: y = (1 - |x|)/2,
             *   asin(|x|) = pi/2 - 2*sqrty*(1 + y*P(y))
             * sqrt(y) returns NaN for y < 0 (i.e., |x| > 1), so NaN
             * propagates through the rest of the chain naturally. */
            ffloat y = ffloat(FpType(0.5f)) - absx * ffloat(FpType(0.5f));
            FpType sy_hi, sy_lo;
            __nv_fpmp2_sqrt(y.hi(), y.lo(), &sy_hi, &sy_lo);
            ffloat sy(sy_hi, sy_lo);

            ffloat p;
            __internal_fpmp2_asin_poly<FpType>(y, &p);

            constexpr ffloat PIO2(1.5707963267948966);
            r = renormalize(PIO2 - ffloat(FpType(2)) * sy * (ffloat(FpType(1)) + y * p));
        }

        if (is_neg) r = -r;
        *res_hi = r.hi();
        *res_lo = r.lo();
    }

    /* ---- acos(x) ---- */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_acos(
        const FpType x_hi, const FpType x_lo,
        FpType* res_hi, FpType* res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_acos is fp32mp2 only; "
                      "fp64mp2 has its own specialization");

        using ffloat = fpmp2_t<FpType>;

        const bool is_neg = x_hi < FpType(0);
        ffloat x(x_hi, x_lo);
        ffloat absx = is_neg ? -x : x;

        constexpr FpType BRANCH = FpType(0.575f);
        constexpr ffloat PI  (3.141592653589793);
        constexpr ffloat PIO2(1.5707963267948966);

        ffloat r;
        if (absx.hi() < BRANCH) {
            /* Small branch: reuse asin polynomial.
             *   acos(x) = pi/2 - asin(x)   (sign of x already in asin) */
            ffloat a2 = absx * absx;
            ffloat p;
            __internal_fpmp2_asin_poly<FpType>(a2, &p);
            ffloat asin_abs = renormalize(absx + absx * (a2 * p));
            r = is_neg ? renormalize(PIO2 + asin_abs)
                       : renormalize(PIO2 - asin_abs);
        } else {
            /* Large branch (libdevice fp64 fit, 13 coefficients):
             *   y = 1 - |x|;   acos(|x|) = sqrt(2y)*(1 + y*P(y))
             *   x < 0  ->  acos(x) = pi - acos(|x|)
             * Polynomial P(y) is evaluated by `__internal_fpmp2_acos_poly`
             * (analogous to `__internal_fpmp2_asin_poly` used by the small
             *  branch and by asin). */
            ffloat y = ffloat(FpType(1)) - absx;
            ffloat two_y = ffloat(FpType(2)) * y;
            FpType s_hi, s_lo;
            __nv_fpmp2_sqrt(two_y.hi(), two_y.lo(), &s_hi, &s_lo);
            ffloat s(s_hi, s_lo);

            ffloat p;
            __internal_fpmp2_acos_poly<FpType>(y, &p);
            ffloat acos_abs = renormalize(s + s * (y * p));
            r = is_neg ? renormalize(PI - acos_abs) : acos_abs;
        }

        *res_hi = r.hi();
        *res_lo = r.lo();
    }

    /*
    * --------------------------------------------------------------------
    * Hyperbolic tangent tanh(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Adapted from libdevice's __nv_tanh, which itself uses an
    * odd-sigmoid skeleton structurally identical to our dedicated erf:
    *
    *   1. Saturation branch: |x| >= TANH_SAT  -> result = sign(x).
    *      Threshold chosen so that 1 - tanh(|x|) < 0.5 ulp at fp32mp2
    *      precision: with ulp_fp32mp2(1) ~ 2^-48 and
    *        1 - tanh(x) ~ 2*exp(-2x) for large x,
    *      we need 2*exp(-2x) <= 2^-49 ->  x >= 25*ln(2) ~ 17.33.
    *      We use 17.5 for a small safety margin.
    *
    *   2. Large-|x| branch (|x| >= 0.6554117):
    *        tanh(|x|) = 1 - 2/(exp(2|x|) + 1)
    *      Reuses the existing dedicated __nv_fpmp2_exp<float>; the
    *      branch point is the libdevice-optimal crossover (chosen so
    *      that the polynomial side stays within its 1.5-ulp envelope
    *      while keeping the exp-side argument bounded away from cancel-
    *      lation in exp(2x) - 1).
    *
    *   3. Small-|x| branch (|x| < 0.6554117): degree-22 minimax
    *      polynomial in x^2 (the same 11 coefficients libdevice fits
    *      for double-precision tanh, evaluated here in fp32mp2):
    *        tanh(x) = x + x * x^2 * Q(x^2)
    *      with Q(x^2) = d1 + d2*x^2 + ... + d11*x^20.
    *      The libdevice fit hits ~1.5 ulp at double precision, leaving
    *      ~16 ulp of headroom relative to the fp32mp2 target (~2^-48),
    *      so high-degree coefficient rounding fits in float without
    *      affecting accuracy (M-split below).
    *
    *   4. Apply sign(x) at the end. Both branches are computed on |x|
    *      using fp32mp2_low and the result is negated when x < 0,
    *      mirroring the erf code path.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_tanh (const FpType x_hi,
                                                 const FpType x_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_tanh is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* libdevice-optimal crossover between polynomial and exp paths. */
        constexpr float BRANCH_POINT = 0.6554117f;
        /* tanh(|x|) >= 1 - 0.5 ulp_fp32mp2 for |x| >= 17.33; use 17.5. */
        constexpr float TANH_SAT     = 17.5f;

        const bool   is_neg = x_hi < 0.f;
        const float  abs_hi = is_neg ? -x_hi : x_hi;

        /* ---- (1) saturation ------------------------------------------- */
        if (!(abs_hi < TANH_SAT))  /* also catches NaN -> falls through to poly */
        {
            if (abs_hi >= TANH_SAT) {
                *res_hi = is_neg ? -1.f : 1.f;
                *res_lo = 0.f;
                return;
            }
            /* NaN: propagate */
            *res_hi = x_hi + x_lo;
            *res_lo = *res_hi;
            return;
        }

        ffloat x   (x_hi, x_lo);
        ffloat absA = is_neg ? -x : x;

        if (abs_hi >= BRANCH_POINT)
        {
            /* ---- (2) large-|x| branch: 1 - 2/(exp(2|x|)+1) ------------ */
            ffloat two_abs = absA + absA;   /* exactly 2|x|: addition of equals */
            float  u_hi, u_lo;
            __nv_fpmp2_exp<float>(two_abs.hi(), two_abs.lo(), &u_hi, &u_lo);
            ffloat denom  = ffloat(u_hi, u_lo) + ffloat(1.f);
            ffloat r      = ffloat(2.f) / denom;
            ffloat result = ffloat(1.f) - r;
            if (is_neg) result = -result;
            *res_hi = result.hi();
            *res_lo = result.lo();
            return;
        }

        /* ---- (3) small-|x| branch: degree-22 polynomial in x^2 -------
         *
         * Q(x^2) = d1 + d2*x^2 + ... + d11*x^20, packed in ascending
         * degree.  Layout for the mixed-precision Horner dispatcher:
         *   - bottom 9 entries (d1..d9) are ff (full double precision):
         *     |d_n * x^{2n+1} * 2^-24| stays above the fp32mp2 ulp at
         *     these degrees, so float-rounding the coefficient would
         *     leak ~5e-9..3e-15 of absolute error into the result.
         *   - top 2 entries (d10, d11) are plain float literals (.lo == 0):
         *     |d_n * x^{2n+1} * 2^-24| <= 5e-16 (well below 0.5 ulp),
         *     so the high-degree Horner steps run in float for free.
         *
         * The 11 coefficients are the same libdevice minimax fit cited
         * by __nv_tanh; truncation noise of that polynomial is ~1.5 ulp
         * at double precision, ~16 ulp of headroom vs. the fp32mp2 target.
         */
        constexpr ffloat tanh_c[11] = {
            /* 9 low-degree ff entries (full double precision) */
            ffloat(-0.33333333333333304),    /* [0] = d1  = -1/3 */
            ffloat( 0.13333333333317149),    /* [1] = d2 */
            ffloat(-5.3968253953220913e-2),  /* [2] = d3 */
            ffloat( 2.1869487987893173e-2),  /* [3] = d4 */
            ffloat(-8.863225224458907e-3),   /* [4] = d5 */
            ffloat( 3.5920144108182715e-3),  /* [5] = d6 */
            ffloat(-1.4550475435045451e-3),  /* [6] = d7 */
            ffloat( 5.8648819462048805e-4),  /* [7] = d8 */
            ffloat(-2.2870121144856145e-4),  /* [8] = d9 (last ff term) */
            /* high-order M = 2 entries: .lo() == 0 by construction */
            ffloat( 7.709298e-5f),           /* [9] = d10 */
            ffloat(-1.596018e-5f),           /* [10] = d11 (leading) */
        };

        ffloat a2 = x * x;     /* x^2 (sign of x cancels) */
        ffloat q  = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 2>(a2, tanh_c);

        /* tanh(x) = x + x * x^2 * Q(x^2). Sign-preserving in x; no
         * separate sign fixup needed for the polynomial branch. */
        ffloat result = renormalize(x + x * (a2 * q));
        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_tanh

    /*
    * ============================================================================
    * Hyperbolic sine/cosine: sinh, cosh (fp32mp2)                  - dedicated
    * ============================================================================
    *
    * Both functions reuse the dedicated `__nv_fpmp2_exp<float>` once and
    * recombine its result.  All arithmetic in `fp32mp2_low`; one
    * `renormalize()` per output canonicalises the (hi, lo) form before
    * leaving the kernel -- same pattern as the rest of the dedicated
    * fp32mp2 math (sin/cos/exp/log/tanh/...).
    *
    *   cosh(x) = (e + 1/e) / 2,    e = exp(|x|)
    *
    *     The two terms `0.5*e` and `0.5/e` are both positive, so the
    *     addition incurs no cancellation at any |x|: at |x|=0 both are
    *     0.5 and the lo parts of `e` and `1/e` carry the cosh(x)-1
    *     ~= x^2/2 correction.  We compute it as a single branchless
    *     formula, with no need for a polynomial branch.
    *
    *   sinh(x) = sign(x) * (e - 1/e) / 2,    e = exp(|x|)
    *           ~= x + x * x^2 * P(x^2)         (small-|x| Taylor)
    *
    *     For |x| close to 0, e and 1/e differ only by ~2|x|, so the
    *     subtraction loses about log_2(2/(2|x|)) bits of precision.
    *     A polynomial branch covers the [0, BRANCH] interval.  The
    *     polynomial coefficients are the exact rational Taylor series
    *     P(y) = 1/3! + y/5! + y^2/7! + ... ; truncation noise of an
    *     11-coefficient table at |x| <= 0.6554 is well below fp32mp2 ulp.
    *
    * Saturation / NaN handling:
    *   - For |x| above the exp overflow boundary (~88.7), `__nv_fpmp2_exp`
    *     returns +inf; `0.5*inf +- 0.5/inf = inf`, which is the IEEE-correct
    *     +-inf for cosh/sinh.
    *   - NaN inputs: `__nv_fpmp2_exp` propagates the NaN through the
    *     subsequent fp32mp2 arithmetic (no explicit guard needed).
    * ============================================================================
    */

    /* sinh(x) on fp32mp2.
     *
     * Polynomial branch covers |x| <= 0.6554 (matches the tanh crossover);
     * exp branch covers everything above.  At the crossover point we have
     * sinh(0.6554)/cosh(0.6554) = tanh(0.6554) ~= 0.575, so the exp branch
     * loses < 1 bit of precision to cancellation -- well within fp32mp2 ulp. */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_sinh (const FpType x_hi,
                                                 const FpType x_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_sinh is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        constexpr float BRANCH_POINT = 0.6554117f;

        const bool   is_neg = x_hi < 0.f;
        const float  abs_hi = is_neg ? -x_hi : x_hi;

        /* NaN propagation: any NaN component pollutes the result. */
        if (x_hi != x_hi || x_lo != x_lo)
        {
            const float nan_val = x_hi + x_lo;
            *res_hi = nan_val;
            *res_lo = nan_val;
            return;
        }

        ffloat x   (x_hi, x_lo);
        ffloat absA = is_neg ? -x : x;

        if (abs_hi >= BRANCH_POINT)
        {
            /* ---- large-|x| branch:  sinh(|x|) = (e - 1/e) / 2 ---------- */
            float u_hi, u_lo;
            __nv_fpmp2_exp<float>(absA.hi(), absA.lo(), &u_hi, &u_lo);
            ffloat e(u_hi, u_lo);
            ffloat half_e     = e * ffloat(0.5f);
            ffloat half_inv_e = ffloat(0.5f) / e;
            ffloat result     = renormalize(half_e - half_inv_e);
            if (is_neg) result = -result;
            *res_hi = result.hi();
            *res_lo = result.lo();
            return;
        }

        /* ---- small-|x| branch: degree-23 polynomial in x^2 -------------
         *
         *   sinh(x) = x + x * x^2 * P(x^2),   P(y) = Sum_{k>=0} y^k / (2k+3)!
         *
         * Layout for the mixed-precision Horner dispatcher:
         *   - bottom 8 entries (1/3! .. 1/17!) are ff (full double precision):
         *     these contribute terms |x|^{2k+3} / (2k+3)! that, even at
         *     |x| = 0.6554, sit above ~3*10^-19 -- float-rounding the
         *     coefficient would leak ~10^-10 ... 10^-24 of absolute error,
         *     marginally near the fp32mp2 ulp at the bottom of the range.
         *   - top 3 entries (1/19!, 1/21!, 1/23!) are plain float (.lo == 0):
         *     contributions stay below 5*10^-24, so float-rounded constants
         *     are below 0.5 ulp -- same trade-off the tanh kernel makes.
         *
         * The exact rational Taylor coefficients have zero truncation
         * noise; the only source of error is fp32mp2 arithmetic.
         */
        constexpr ffloat sinh_c[11] = {
            /* 8 low-degree ff entries (full double precision) */
            ffloat( 1.6666666666666666e-1),  /* [0] = 1/3!  = 1/6 */
            ffloat( 8.3333333333333333e-3),  /* [1] = 1/5!  */
            ffloat( 1.9841269841269841e-4),  /* [2] = 1/7!  */
            ffloat( 2.7557319223985891e-6),  /* [3] = 1/9!  */
            ffloat( 2.5052108385441718e-8),  /* [4] = 1/11! */
            ffloat( 1.6059043836821614e-10), /* [5] = 1/13! */
            ffloat( 7.6471637318198164e-13), /* [6] = 1/15! */
            ffloat( 2.8114572543455207e-15), /* [7] = 1/17! (last ff term) */
            /* high-order M = 3 entries: .lo() == 0 by construction */
            ffloat( 8.220635246624329e-18f), /* [8] = 1/19! */
            ffloat( 1.957294106339126e-20f), /* [9] = 1/21! */
            ffloat( 3.866968596927381e-23f), /* [10] = 1/23! (leading) */
        };

        ffloat a2 = x * x;     /* x^2 (sign of x cancels) */
        ffloat q  = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 3>(a2, sinh_c);

        /* sinh(x) = x + x * x^2 * P(x^2).  Sign-preserving in x; no
         * separate sign fixup needed for the polynomial branch. */
        ffloat result = renormalize(x + x * (a2 * q));
        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_sinh

    /* cosh(x) on fp32mp2.
     *
     * Branchless: cosh is even (cosh(-x) = cosh(x)), and the formula
     * `(e + 1/e) / 2` is well-conditioned everywhere -- both terms are
     * positive, so addition never cancels.  At |x| = 0 the lo parts of
     * e and 1/e carry the x^2/2 correction exactly, so no separate
     * polynomial branch is needed for small |x|. */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_cosh (const FpType x_hi,
                                                 const FpType x_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_cosh is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* NaN propagation. */
        if (x_hi != x_hi || x_lo != x_lo)
        {
            const float nan_val = x_hi + x_lo;
            *res_hi = nan_val;
            *res_lo = nan_val;
            return;
        }

        const bool  is_neg = x_hi < 0.f;
        ffloat      x      (x_hi, x_lo);
        ffloat      absA   = is_neg ? -x : x;

        float u_hi, u_lo;
        __nv_fpmp2_exp<float>(absA.hi(), absA.lo(), &u_hi, &u_lo);
        ffloat e(u_hi, u_lo);

        /* cosh(|x|) = 0.5*e + 0.5/e  (both terms positive; no cancellation). */
        ffloat half_e     = e * ffloat(0.5f);
        ffloat half_inv_e = ffloat(0.5f) / e;
        ffloat result     = renormalize(half_e + half_inv_e);

        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_cosh

    /* Inverse hyperbolic functions on fp32mp2.
     *
     * All three implementations reduce to a single fp32mp2 log1p call,
     * with arithmetic forms chosen to avoid catastrophic cancellation
     * across the entire input domain:
     *
     *   asinh(x) = sign(x) * log1p(|x| + x^2 / (sqrt(x^2+1) + 1))
     *   acosh(x) = log1p((x-1) + sqrt((x-1)*(x+1)))   for x >= 1
     *   atanh(x) = 0.5 * sign(x) * log1p(2|x| / (1-|x|))  for |x| >= 0.25
     *   atanh(x) = x * (1 + y*P(y)),  y = x^2,             for |x| <  0.25
     *
     * Compared to the textbook formulas log(x + sqrt(x^2+1)) and
     * log((1+x)/(1-x)), the log1p forms preserve full fp32mp2 precision
     * around x = 0 (asinh, atanh) and x = 1 (acosh) by replacing the
     * cancellation-prone subtraction "(x + sqrt(...)) - 1" with an
     * algebraically equivalent expression whose terms have the same sign.
     *
     * For asinh, the rationalized form
     *      |x| + sqrt(x^2+1) - 1 = |x| + x^2 / (sqrt(x^2+1) + 1)
     * sidesteps the subtraction entirely (both summands are >= 0).
     *
     * For acosh, x^2-1 is computed as (x-1)*(x+1) -- both factors are
     * well-conditioned (x-1 >= 0, x+1 >= 2), so the product carries the
     * full fp32mp2 precision of the difference even at x ~= 1, matching
     * what libdevice's double-precision __nv_acosh achieves with a
     * single fma(x,x,-1) primitive.
     *
     * For atanh, the log1p form
     *      0.5 * log1p(2|x|/(1-|x|))
     * relies on a fast-method divide that costs ~1 ulp in the log1p
     * argument; that error becomes ~eps/|x| relative on the final result
     * and dominates as |x| -> 0.  A degree-23 Taylor polynomial in y=x^2
     * (i.e., 12 coefficients 1/3, 1/5, ..., 1/25) covers |x| < 0.25 with
     * truncation noise below fp32mp2 ulp, bypassing the divide and
     * delivering full precision near zero.  The branch range is
     * deliberately narrow to keep most threads on the log1p path and
     * limit warp divergence.  Above |x| = 1 the divisor -> 0+ and
     * log1p(+inf) saturates to +inf, so the IEEE-754 limits propagate
     * naturally without explicit guards.
     *
     * Large-argument paths (asinh, acosh) switch to log(2|x|) once |x|
     * exceeds 2^25 ~= 3.4e7.  The dropped 1/(4x^2) correction is below
     * fp32mp2 ulp from there on, and the early switch sidesteps the
     * lo-limb ulp accumulation that the `(x-1)(x+1)` / `x^2+1` chains
     * would suffer at very large |x|.
     */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_asinh (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_asinh is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* NaN propagation. */
        if (x_hi != x_hi || x_lo != x_lo)
        {
            const float nan_val = x_hi + x_lo;
            *res_hi = nan_val;
            *res_lo = nan_val;
            return;
        }

        /* asinh is odd: handle +-inf via the sign branch.
         * asinh(+-inf) = +-inf. */
        const bool   is_neg = x_hi < 0.0f;
        const float  abs_hi = is_neg ? -x_hi : x_hi;

        if (abs_hi == __builtin_huge_valf())
        {
            *res_hi = is_neg ? -__builtin_huge_valf() : __builtin_huge_valf();
            *res_lo = 0.0f;
            return;
        }

        /* asinh(+-0) = +-0. */
        if (abs_hi == 0.0f && x_lo == 0.0f)
        {
            *res_hi = x_hi;     /* preserves signed zero */
            *res_lo = 0.0f;
            return;
        }

        ffloat x   (x_hi, x_lo);
        ffloat absA = is_neg ? -x : x;

        /* Crossover threshold: above 2^25 we switch to the asymptotic
         * form
         *      asinh(x) = log(2|x|) + 1/(4x^2) - 3/(32x^4) + ...
         * and drop everything past the leading term.  At |x| = 2^25 the
         * dropped 1/(4x^2) is below 1.2*10^-17 relative -- comfortably
         * under fp32mp2 ulp.  Switching this early (rather than waiting
         * for x^2 to overflow at |x| ~= 1.8e19) avoids the precision loss
         * that the `|x| + x^2/(sqrt(x^2+1)+1)` chain accumulates at very
         * large |x|: each step (mul, sqrt, fast div, sum) bleeds 1-2
         * ulps into the lo limb and the absolute error survives the
         * subsequent log() because the chain produces a value of order
         * |x|^2 before the log compresses it back to log(2|x|), so the
         * lo errors don't shrink with the result.  Empirically, the
         * else branch loses ~10 bits at |x| ~ 2^60; the asymptotic form
         * is exact to fp32mp2 ulp throughout [2^25, FLT_MAX]. */
        constexpr float LARGE_ASINH = 0x1.0p+25f;

        ffloat result;
        if (abs_hi > LARGE_ASINH)
        {
            constexpr ffloat ln2(0x1.62e42fefa39efp-1);
            float l_hi, l_lo;
            __nv_fpmp2_log<float>(absA.hi(), absA.lo(), &l_hi, &l_lo);
            result = renormalize(ffloat(l_hi, l_lo) + ln2);
        }
        else
        {
            /* t = |x| + x^2 / (sqrt(x^2+1) + 1) ; result = log1p(t).
             * Both summands of t are non-negative -- no cancellation.
             * For |x| -> 0:  t ~= |x| + x^2/2,
             *   log1p(t) = |x| - |x|^3/6 + ... -> asinh series.
             * For |x| -> inf (within LARGE):  t ~= 2|x|,
             *   log1p(2|x|) = log(1 + 2|x|) ~= log(2|x|).
             *
             * Use accurate add for x^2+1: when |x| is small the +1
             * dominates and we want the lo to carry x^2 to full
             * fp32mp2 precision -- the same reasoning as in log1p. */
            ffloat a2     = absA * absA;
            ffloat a2p1   = add<fpmp2_accuracy::high>(a2, 1.0f);
            float  s_hi, s_lo;
            __nv_fpmp2_sqrt<float>(a2p1.hi(), a2p1.lo(), &s_hi, &s_lo);
            ffloat s      = ffloat(s_hi, s_lo);
            ffloat denom  = add<fpmp2_accuracy::high>(s, 1.0f);
            ffloat t      = renormalize(absA + a2 / denom);

            float r_hi, r_lo;
            __nv_fpmp2_log1p<float>(t.hi(), t.lo(), &r_hi, &r_lo);
            result = ffloat(r_hi, r_lo);
        }

        if (is_neg) result = -result;
        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_asinh

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_acosh (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_acosh is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* NaN propagation. */
        if (x_hi != x_hi || x_lo != x_lo)
        {
            const float nan_val = x_hi + x_lo;
            *res_hi = nan_val;
            *res_lo = nan_val;
            return;
        }

        /* Domain: x >= 1.  Anything strictly below produces NaN.
         * Use lexicographic compare on (hi, lo) to capture x = 1 with
         * a negative lo (i.e., x < 1 by a sub-ulp amount). */
        if (x_hi < 1.0f || (x_hi == 1.0f && x_lo < 0.0f))
        {
            *res_hi = __builtin_nanf("");
            *res_lo = __builtin_nanf("");
            return;
        }

        /* acosh(+inf) = +inf. */
        if (x_hi == __builtin_huge_valf())
        {
            *res_hi = __builtin_huge_valf();
            *res_lo = 0.0f;
            return;
        }

        /* acosh(1) = 0 exactly. */
        if (x_hi == 1.0f && x_lo == 0.0f)
        {
            *res_hi = 0.0f;
            *res_lo = 0.0f;
            return;
        }

        ffloat x(x_hi, x_lo);

        /* Crossover threshold: above 2^25 we switch to the asymptotic
         * form
         *      acosh(x) = log(2x) - 1/(4x^2) - 3/(32x^4) - ...
         * dropping everything past the leading log(2x).  At |x| = 2^25
         * the omitted 1/(4x^2) sits at 2.2*10^-16 absolute (~=1.2*10^-17
         * relative against log(2x) ~= 18) -- under fp32mp2 ulp.  The
         * (x-1)*(x+1) chain that the else branch uses bleeds ~1-2 ulps
         * per step into the lo limb at large |x|, and those absolute
         * errors don't shrink through the subsequent log compression,
         * so accuracy degrades to ~36 bits near |x| ~ 2^60.  Switching
         * this early restores fp32mp2 ulp throughout [2^25, FLT_MAX]. */
        constexpr float LARGE_ACOSH = 0x1.0p+25f;

        ffloat result;
        if (x_hi > LARGE_ACOSH)
        {
            /* Asymptotic form: acosh(x) ~= log(2x) = log(x) + ln(2).
             * O(1/x^2) correction is below fp32mp2 ulp at crossover. */
            constexpr ffloat ln2(0x1.62e42fefa39efp-1);
            float l_hi, l_lo;
            __nv_fpmp2_log<float>(x.hi(), x.lo(), &l_hi, &l_lo);
            result = renormalize(ffloat(l_hi, l_lo) + ln2);
        }
        else
        {
            /* t = (x - 1) + sqrt(x^2 - 1) ; result = log1p(t).
             * For x -> 1+:  x^2-1 ~= 2(x-1) -> 0,  sqrt(x^2-1) -> sqrt(2(x-1));
             *   t ~= (x-1) + sqrt(2(x-1)),  no cancellation.
             * For x large within LARGE:  t ~= 2x-1,  log1p(2x-1) = log(2x).
             *
             * Compute x^2-1 as (x-1)*(x+1) to sidestep the catastrophic
             * cancellation in `x*x - 1` when x is close to 1.  Both
             * factors are well-conditioned (x-1 >= 0, x+1 >= 2), so the
             * product carries the full fp32mp2 precision of the
             * difference -- which is exactly what sqrt() needs to deliver
             * the bits that drive the log1p argument near the branch
             * point.  The libdevice double-precision __nv_acosh uses
             * `fma(a,a,-1)` to achieve the same effect via a single
             * correctly-rounded primitive, but our fp32mp2 fast_t mul
             * does not give full mathematical precision in `x*x`, so
             * the (x-1)(x+1) factorization is the cleanest equivalent. */
            ffloat xm1   = sub<fpmp2_accuracy::high>(x, 1.0f);
            ffloat xp1   = add<fpmp2_accuracy::high>(x, 1.0f);
            ffloat x2m1  = xm1 * xp1;
            float  s_hi, s_lo;
            __nv_fpmp2_sqrt<float>(x2m1.hi(), x2m1.lo(), &s_hi, &s_lo);
            ffloat s     = ffloat(s_hi, s_lo);
            ffloat t     = renormalize(xm1 + s);

            float r_hi, r_lo;
            __nv_fpmp2_log1p<float>(t.hi(), t.lo(), &r_hi, &r_lo);
            result = ffloat(r_hi, r_lo);
        }

        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_acosh

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_atanh (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        static_assert(std::is_same<FpType, float>::value,
                      "dedicated __nv_fpmp2_atanh is fp32mp2 only; fp64mp2 has its own specialization");

        using namespace fpmp;
        using ffloat = fp32mp2_low;

        /* NaN propagation. */
        if (x_hi != x_hi || x_lo != x_lo)
        {
            const float nan_val = x_hi + x_lo;
            *res_hi = nan_val;
            *res_lo = nan_val;
            return;
        }

        const bool   is_neg = x_hi < 0.0f;
        const float  abs_hi = is_neg ? -x_hi : x_hi;

        /* atanh(+-0) = +-0. */
        if (abs_hi == 0.0f && x_lo == 0.0f)
        {
            *res_hi = x_hi;     /* preserves signed zero */
            *res_lo = 0.0f;
            return;
        }

        /* atanh(+-1) = +-inf.  Strict |x| > 1 -> NaN. */
        if (abs_hi >= 1.0f)
        {
            const float abs_lo = is_neg ? -x_lo : x_lo;
            if (abs_hi == 1.0f && abs_lo == 0.0f)
            {
                *res_hi = is_neg ? -__builtin_huge_valf() : __builtin_huge_valf();
                *res_lo = 0.0f;
                return;
            }
            /* |x| > 1 (including +inf): outside domain. */
            *res_hi = __builtin_nanf("");
            *res_lo = __builtin_nanf("");
            return;
        }

        ffloat x    (x_hi, x_lo);
        ffloat absA = is_neg ? -x : x;

        /* Small-|x| polynomial branch.
         *
         * Why split: the log1p form
         *     0.5 * log1p(2|x|/(1-|x|))
         * relies on a fast-method divide that introduces ~1 ulp of
         * relative error into the log1p argument; that error becomes
         * ~eps / |x| relative on the final result and dominates as
         * |x| -> 0.  The Taylor series
         *     atanh(x) = x * (1 + y*P(y)),  y = x^2,
         *     P(y) = 1/3 + y/5 + y^2/7 + ... + y^k/(2k+3) + ...
         * is sign-preserving in x (the leading factor carries the sign)
         * and avoids the divide entirely, so it delivers full fp32mp2
         * precision for small |x| with no setup error.
         *
         * Branch point 0.25 keeps the polynomial narrow (covers ~25 %
         * of the typical work range) so most threads stay on the log1p
         * path, limiting warp divergence; at |x| = 0.25 the y^11 term is
         * 0.04 * 0.0625^11 ~= 5*10^-16, below fp32mp2 ulp at atanh(0.25). */
        constexpr float ATANH_BRANCH_POINT = 0.25f;
        if (abs_hi < ATANH_BRANCH_POINT)
        {
            /* P(y) = sum_{k>=0} y^k / (2k+3), packed in ascending degree.
             *   atanh_poly_c[0] = 1/3 (constant of P),
             *   atanh_poly_c[k] = 1/(2k+3),
             *   atanh_poly_c[11] = 1/25 (leading).
             * Layout for poly_eval<horner_mixed, M=4>: bottom 8 entries
             * are full ff (their contributions stay above fp32mp2 ulp at
             * the branch point), top 4 entries are plain float (.lo == 0
             * by construction; their contributions sit below 0.5 ulp). */
            constexpr ffloat atanh_poly_c[12] = {
                ffloat( 3.3333333333333333e-1), /* [ 0] 1/3  */
                ffloat( 2.0e-1),                /* [ 1] 1/5  */
                ffloat( 1.4285714285714286e-1), /* [ 2] 1/7  */
                ffloat( 1.1111111111111111e-1), /* [ 3] 1/9  */
                ffloat( 9.0909090909090909e-2), /* [ 4] 1/11 */
                ffloat( 7.6923076923076923e-2), /* [ 5] 1/13 */
                ffloat( 6.6666666666666667e-2), /* [ 6] 1/15 */
                ffloat( 5.8823529411764706e-2), /* [ 7] 1/17  (last ff term) */
                /* high-order M = 4 entries: .lo() == 0 by construction */
                ffloat( 5.263158e-2f),          /* [ 8] 1/19 */
                ffloat( 4.761905e-2f),          /* [ 9] 1/21 */
                ffloat( 4.347826e-2f),          /* [10] 1/23 */
                ffloat( 4.0e-2f),               /* [11] 1/25  (leading) */
            };

            ffloat y      = x * x;     /* x^2 (sign of x cancels) */
            ffloat q      = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 4>(y, atanh_poly_c);
            ffloat result = renormalize(x + x * (y * q));
            *res_hi = result.hi();
            *res_lo = result.lo();
            return;
        }

        /* t = 2|x| / (1 - |x|) ; result = 0.5 * log1p(t).
         *
         * For |x| -> 1-:  1 - |x| -> 0+, t -> +inf, log1p(+inf) = +inf.
         * For |x| ~= 0.25 (lower edge of this branch):  t ~= 0.667,
         *   log1p(0.667) ~= 0.511, * 0.5 = 0.2554 ~= atanh(0.25).
         *
         * Use accurate sub for 1 - |x| to capture full precision when
         * |x| is close to 1. */
        ffloat one_minus = sub<fpmp2_accuracy::high>(ffloat(1.0f), absA);
        ffloat two_abs   = absA + absA;
        ffloat t         = two_abs / one_minus;

        float l_hi, l_lo;
        __nv_fpmp2_log1p<float>(t.hi(), t.lo(), &l_hi, &l_lo);

        ffloat result = ffloat(l_hi, l_lo) * ffloat(0.5f);
        if (is_neg) result = -result;

        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_atanh

    /*
    * --------------------------------------------------------------------
    * Error function erf(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Error function erf(x) = -expm1(-|x|*P(|x|)) where P is a Remez
    * polynomial in |x|, and expm1 is computed via argument reduction
    * and a mixed-precision polynomial, all in fp32mp2.
    *
    * Two polynomial variants are provided, selected at compile time
    * by the __FPMP_USE_FAST_ERF__ macro:
    *
    *   undefined           : uniform degree-23 Remez polynomial over
    *                         [0, 5.92], evaluated with full compensated
    *                         Horner.  Smallest SASS footprint.
    *   defined             : split-domain Remez at x* = 2.1134011
    *                         (LEFT degree 17 over [0, 2.1134011),
    *                          RIGHT degree 16 over [2.1134011, 5.92]),
    *                         each branch evaluated with compensated
    *                         Horner.  Both fits hit the same 46-bit
    *                         precision floor as the default; the
    *                         shorter polynomials trade ~+29% SASS
    *                         lines for ~+20% throughput / ~-8%
    *                         latency on coherent input distributions.
    *                         Warps straddling x* serialize both
    *                         branches, so scattered inputs reach the
    *                         default cost as an upper bound.
    * --------------------------------------------------------------------
    */
    #ifndef __FPMP_USE_FAST_ERF__
        #define __FPMP_USE_FAST_ERF__ 1
    #endif
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_erf(const FpType x_hi,
                                               const FpType x_lo,
                                               FpType*      res_hi,
                                               FpType*      res_lo)
    {
        using ffloat = fp32mp2_low;

        /* expm1(r) polynomial: u(r) = m2 + m3*r + m4*r^2 + ... + m11*r^9,
         * packed in ascending degree (m_c[0] = m2 = constant, m_c[9] =
         * m11 = leading).  The 4 highest-degree entries (m_c[6..9] =
         * original m8..m11) are plain float literals and run in the
         * dispatcher's FpType phase; the remaining 6 carry an ff `.lo()`.
         *
         * Note: the dispatcher transition uses `uf * r.hi() + m7`
         * (float*float + ff), whereas the previous hand-rolled step
         * was `uf * r + m7` (float * full-ff + ff).  This is the same
         * sub-ULP shift the exp refactor produced -- well below the
         * polynomial truncation noise floor. */
        constexpr ffloat m_c[10] = {
            ffloat(0.50000000000000056),    // [0] (= m2, constant)
            ffloat(0.16666666666666607),    // [1] (= m3)
            ffloat(4.1666666666573884e-2),  // [2] (= m4)
            ffloat(8.3333333333771645e-3),  // [3] (= m5)
            ffloat(1.3888888932264757e-3),  // [4] (= m6)
            ffloat(1.9841269746984988e-4),  // [5] (= m7, last ff term)
            /* high-order M = 4 entries: .lo() == 0 by construction */
            ffloat(2.4801505e-5f),          // [6] (= m8)
            ffloat(2.7557382e-6f),          // [7] (= m9)
            ffloat(2.7626265e-7f),          // [8] (= m10)
            ffloat(2.5062102e-8f)           // [9] (= m11, leading)
        };

        constexpr ffloat L2E   (1.4426950408889634);
        constexpr ffloat LN2_HI(0.6931471805599453);

        ffloat x     = renormalize(ffloat(x_hi, x_lo));
        bool is_neg  = x.hi() < 0.f;
        uint32_t xhi = fpmp::internal_bit_cast<uint32_t>(x.hi()) & 0x7fffffffU;
        ffloat absA  = is_neg ? -x : x;

        /* |x| >= saturation_bound (~5.92) or Inf -> erf = +-1 */
        if (xhi >= 0x40bd7da4U && xhi <= 0x7f800000U) 
        {
            *res_hi = is_neg ? -1.f : 1.f;
            *res_lo = 0.f;
            return;
        }

#if __FPMP_USE_FAST_ERF__ == 1
        /* Fast variant: split-domain Remez at x* = 2.1134011.
         *   LEFT  : degree 17 (18 coeffs) over [0, 2.1134011)
         *   RIGHT : degree 16 (17 coeffs) over [2.1134011, 5.92]
         * Both fits sit at the same precision floor as the default
         * degree-23 polynomial.  Each branch keeps its own clean ILP
         * dataflow (a branchless coefficient-select variant was
         * measured to put `selp.f32` on the critical path of every
         * Horner step, cancelling the latency win).  Trades ~+29%
         * SASS for ~+20% throughput / ~-8% latency on coherent
         * workloads.
         */
        constexpr float X_STAR = 2.1134011f;

        constexpr ffloat dc_left[18] = {
            ffloat( 1.2837916709551273e-01),  // [ 0] constant
            ffloat( 6.3661977236753761e-01),  // [ 1]
            ffloat( 1.0277260330382626e-01),  // [ 2]
            ffloat(-1.9128447038837399e-02),  // [ 3]
            ffloat(-2.0919443027514459e-04),  // [ 4]
            ffloat( 1.6962054283924491e-03),  // [ 5]
            ffloat(-5.9012551064862781e-04),  // [ 6]
            ffloat( 2.5894044204962638e-05),  // [ 7]
            ffloat( 6.4414111344269855e-05),  // [ 8]
            ffloat(-2.9502940222999094e-05),  // [ 9]
            ffloat( 2.9772044480981463e-06),  // [10]
            ffloat( 3.4470407727555699e-06),  // [11]
            ffloat(-2.3997080766216321e-06),  // [12]
            ffloat( 8.8126532430964285e-07),  // [13]
            ffloat(-2.1347246296037766e-07),  // [14]
            ffloat( 3.4395369235060941e-08),  // [15]
            ffloat(-3.3767065506818252e-09),  // [16]
            ffloat( 1.5374576174679341e-10)   // [17] leading
        };

        constexpr ffloat dc_right[17] = {
            ffloat( 1.2838182329753376e-01),  // [ 0] constant
            ffloat( 6.3664135493147287e-01),  // [ 1]
            ffloat( 1.0262001147255973e-01),  // [ 2]
            ffloat(-1.8718159485718171e-02),  // [ 3]
            ffloat(-8.6902967978178309e-04),  // [ 4]
            ffloat( 2.4246233937155400e-03),  // [ 5]
            ffloat(-1.1769573995400237e-03),  // [ 6]
            ffloat( 3.7914816346061311e-04),  // [ 7]
            ffloat(-9.2599432840590657e-05),  // [ 8]
            ffloat( 1.7765977912059822e-05),  // [ 9]
            ffloat(-2.6957054726382021e-06),  // [10]
            ffloat( 3.2096938307796375e-07),  // [11]
            ffloat(-2.9400887323643838e-08),  // [12]
            ffloat( 2.0010667763467461e-09),  // [13]
            ffloat(-9.5320838658187351e-11),  // [14]
            ffloat( 2.8357961123952052e-12),  // [15]
            ffloat(-3.9648740890296208e-14)   // [16] leading
        };

        ffloat poly;
        if (absA.hi() < X_STAR)
            poly = fpmp::poly_eval<fpmp::poly_method::horner_comp>(absA, dc_left);
        else
            poly = fpmp::poly_eval<fpmp::poly_method::horner_comp>(absA, dc_right);
#else // __FPMP_USE_FAST_ERF__ == 0
        /* Default: uniform degree-23 Remez polynomial over [0, 5.92],
         * evaluated with full compensated Horner (P(0) = d1).
         * Well-conditioned uniform Horner is the case where
         * compensated evaluation wins on both accuracy and SASS
         * footprint at this polynomial length. */
        constexpr ffloat d1 ( 0.12837916709551259);
        constexpr ffloat d2 ( 0.6366197723675876);
        constexpr ffloat d3 ( 0.10277260330144233);
        constexpr ffloat d4 (-1.9128446995328407e-2);
        constexpr ffloat d5 (-2.0919483164788562e-4);
        constexpr ffloat d6 ( 1.696207528729842e-3);
        constexpr ffloat d7 (-5.901318195328236e-4);
        constexpr ffloat d8 ( 2.5902605702646151e-5);
        constexpr ffloat d9 ( 6.4424832324704525e-5);
        constexpr ffloat d10(-2.9583306728241582e-5);
        constexpr ffloat d11( 3.1800461703546548e-6);
        constexpr ffloat d12( 3.1218939658311085e-6);
        constexpr ffloat d13(-2.0278249778025215e-6);
        constexpr ffloat d14( 5.643145203798444e-7);
        constexpr ffloat d15(-8.299332548682465e-9);
        constexpr ffloat d16(-6.7203270800518394e-8);
        constexpr ffloat d17( 3.5089011868220468e-8);
        constexpr ffloat d18(-1.0909760903049583e-8);
        constexpr ffloat d19( 2.389211325400646e-9);
        constexpr ffloat d20(-3.806599039253438e-10);
        constexpr ffloat d21( 4.3555974045566826e-11);
        constexpr ffloat d22(-3.4079297100747907e-12);
        constexpr ffloat d23( 1.6366247078834561e-13);
        constexpr ffloat d24(-3.642577040697121e-15);

        constexpr ffloat dc[24] = {
            d1,  d2,  d3,  d4,  d5,  d6,  d7,  d8,
            d9,  d10, d11, d12, d13, d14, d15, d16,
            d17, d18, d19, d20, d21, d22, d23, d24
        };
        ffloat poly = fpmp::poly_eval<fpmp::poly_method::horner_comp>(absA, dc);
#endif // __FPMP_USE_FAST_ERF__ == 0

        /* arg = |x| * P(|x|) + |x| (replaces polyHi/polyLo splitting) */
        ffloat arg = renormalize(poly * absA + absA);

        /* Compute -expm1(-arg): argument reduction */
        ffloat neg_arg     = -arg;
        float  neg_arg_l2e = (neg_arg * L2E).hi();
        int    n           = fpmp::fp2int_rn(neg_arg_l2e);
        ffloat fn          = fpmp::int2fp_rn<float>(n);
        ffloat r           = neg_arg - fn * LN2_HI;

        /* Evaluate u(r) = m2 + m3*r + ... + m11*r^9 via the mixed-precision
         * dispatcher (4 high-order float coeffs m8..m11, 6 low-order ff
         * coeffs m2..m7).  Cheaper than a unified compensated Horner
         * because the high terms contribute below the noise floor and
         * don't need error tracking.
         */
        ffloat u = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 4>(r, m_c);

        /* expm1(r) = u*r^2 + r (no separate alo needed, r carries full precision) */
        u = u * r;
        u = u * r;
        u = u + r;

        /* scale = 2^n, scalem1 = 1 - 2^n */
        int en           = 127 + n;
        if (en < 1)   en = 1;
        if (en > 254) en = 254;
        float  scale     = fpmp::internal_bit_cast<float>(static_cast<unsigned>(en) << 23);
        ffloat scalem1   = ffloat(1.f, 0.f) - ffloat(scale, 0.f);

        /* result = -expm1(-arg) = -u*scale + scalem1 */
        ffloat result = renormalize(-u * ffloat(scale, 0.f) + scalem1);

        /* Apply sign */
        if (is_neg)
            result = -result;

        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_erf

    /*
    * --------------------------------------------------------------------
    * Complementary error function erfc(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Computes erfc(x) = erfcx(|x|) * exp(-x^2), where erfcx is the
    * scaled complementary error function erfcx(a) = (1+2a)*exp(a^2)*erfc(a).
    *
    * Algorithm:
    *   1. Transform variable: t = (|x| - 4) / (|x| + 4)  (maps [0,inf) -> [-1,1))
    *   2. Evaluate degree-22 Chebyshev polynomial in t to approximate
    *      (1+2|x|)*exp(x^2)*erfc(|x|), then divide by (1+2|x|) to get erfcx.
    *   3. Evaluate exp(-x^2) via argument reduction x^2 = n*ln2 + r and a
    *      degree-11 polynomial for exp(r), with split exponent scaling
    *      for large |x^2| and a fma-style correction for rounding of x^2.
    *   4. Multiply erfcx * exp(-x^2) to obtain erfc(|x|).
    *   5. For negative x, apply erfc(-x) = 2 - erfc(x).
    *
    * Coefficient layout: lower-order Chebyshev/exp terms use single float
    * (negligible contribution), higher-order terms use fp32mp2 (ffloat).
    * Saturates to 0 or 2 for |x| > 27.5.
    * All arithmetic is in fp32mp2 (no double-precision operations).
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_erfc(const FpType x_hi, 
                                                const FpType x_lo, 
                                                FpType*      res_hi, 
                                                FpType*      res_lo)
    {
        /*
        * erfc(x) = erfcx(|x|) * exp(-x^2); erfcx = (1+2*x)*exp(x^2)*erfc(x) 
        * from Chebyshev approx.
        */
        using ffloat = fp32mp2_low;            
        
        /* erfcx polynomial (Chebyshev coefficients), ascending degree:
         *   cheb[0]  = constant term (= original c22)
         *   cheb[22] = leading coeff (= original c0)
         * The M = 7 highest-degree entries (cheb[16..22]) are encoded as
         * float literals -- their .lo() parts are zero by construction --
         * so `poly_eval<poly_method::horner_mixed, 7>` evaluates them
         * in plain float and transitions to ff arithmetic at cheb[15]. */
        constexpr ffloat cheb[23] = {
            ffloat( 1.2329951186255526E+000),  // [ 0] (= c22, constant)
            ffloat(-1.3962111684056291E-001),  // [ 1] (= c21)
            ffloat( 1.5379652102605428E-002),  // [ 2] (= c20)
            ffloat( 6.8097054254735140E-002),  // [ 3] (= c19)
            ffloat(-1.0103906603555676E-001),  // [ 4] (= c18)
            ffloat( 9.3732834997115544E-002),  // [ 5] (= c17)
            ffloat(-6.6330365827532434E-002),  // [ 6] (= c16)
            ffloat( 3.7167515553018733E-002),  // [ 7] (= c15)
            ffloat(-1.6197733895953217E-002),  // [ 8] (= c14)
            ffloat( 5.0319698792599572E-003),  // [ 9] (= c13)
            ffloat(-7.5777429182785833E-004),  // [10] (= c12)
            ffloat(-1.9925637684786154E-004),  // [11] (= c11)
            ffloat( 1.5062557169571788E-004),  // [12] (= c10)
            ffloat(-2.4399558857200190E-005),  // [13] (= c9)
            ffloat(-1.1231787437600085E-005),  // [14] (= c8)
            ffloat( 5.7087871844325649E-006),  // [15] (= c7, last ff term)
            /* high-order M = 7 entries: .lo() == 0 by construction */
            ffloat( 3.095641e-7f),             // [16] (= c6)
            ffloat(-8.214741e-7f),             // [17] (= c5)
            ffloat( 5.88067e-8f),              // [18] (= c4)
            ffloat( 1.0404431e-7f),            // [19] (= c3)
            ffloat(-8.935022e-9f),             // [20] (= c2)
            ffloat(-9.723912e-9f),             // [21] (= c1)
            ffloat(-3.5602695e-10f)            // [22] (= c0, leading)
        };

        /* exp polynomial coefficients, ascending degree:
         *   exp_c[0]  = constant term (= original ep11)
         *   exp_c[11] = leading coeff (= original ep0)
         * M = 5 highest-degree entries (exp_c[7..11]) run in float. */
        constexpr ffloat exp_c[12] = {
            ffloat(1.0E+000),                  // [ 0] (= ep11, constant)
            ffloat(1.0E+000),                  // [ 1] (= ep10)
            ffloat(5.0000000000000122E-001),   // [ 2] (= ep9)
            ffloat(1.6666666666666477E-001),   // [ 3] (= ep8)
            ffloat(4.1666666666519754E-002),   // [ 4] (= ep7)
            ffloat(8.3333333334550432E-003),   // [ 5] (= ep6)
            ffloat(1.3888888945916380E-003),   // [ 6] (= ep5, last ff term)
            /* high-order M = 5 entries: .lo() == 0 by construction */
            ffloat(1.984127e-4f),              // [ 7] (= ep4)
            ffloat(2.480149e-5f),              // [ 8] (= ep3)
            ffloat(2.7557515e-6f),             // [ 9] (= ep2)
            ffloat(2.76309e-7f),               // [10] (= ep1)
            ffloat(2.5022323e-8f)              // [11] (= ep0, leading)
        };

        constexpr ffloat L2E   (1.4426950408889634e+0);
        constexpr ffloat LN2_HI(6.9314718055994529e-1);
        constexpr ffloat LN2_LO(2.3190468138462996e-17);            

        ffloat x     = renormalize(ffloat(x_hi, x_lo));
        bool is_neg  = x.hi() < 0.f;
        uint32_t xhi = fpmp::internal_bit_cast<uint32_t>(x.hi()) & 0x7fffffffU;
        ffloat a = (is_neg) ? -x : x;

        // handle x > 27.5 && <= Inf
        if ((xhi > 0x41dc0000U) && (xhi <= 0x7f800000U))
        {
            *res_hi = (is_neg) ? 2.f : 0.f;
            *res_lo = 0.f;
            return;
        }

        /* erfcx kernel: (1+2*a)*exp(a^2)*erfc(a) on a = |x|, transform (a-4)/(a+4) */
        ffloat t1 = a - ffloat(4.0);
        ffloat t2 = a + ffloat(4.0);
        t2        = ffloat(1.0) / t2;
        ffloat t3 = (t1 * t2);
        ffloat t4 = t3 + ffloat(1.0);
        t1        = (ffloat(-4.0) * t4 + a);
        t1        = t1 - t3 * a;
        t2        = (t2 * t1 + t3);

        // Chebyshev polynomial: 7 high-order terms in float, remaining 16 in ff
        t1 = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 7>(t2, cheb);

        /* (1+2*a)*exp(a^2)*erfc(a) / (1+2*a) -> exp(a^2)*erfc(a) = erfcx */
        t2 = (ffloat(2.0) * a + ffloat(1.0));
        t2 = ffloat(1.0) / t2;
        t3 = t1 * t2;
        t4 = a * (ffloat(-2.0) * t3) + t1;
        t4 = (t4 - t3);
        t1 = (t4 * t2 + t3);

        /* erfc(x) = erfcx * exp(-x^2) */
        ffloat xx = renormalize(-a * a);

        /* i = round(xx * L2E); t = exp_mantissa(xx); t3 = accurate_scale(t, i) */
        float prod_hi = (xx * L2E).hi();
        int i         = fpmp::fp2int_rn(prod_hi);
        ffloat t_rint = fpmp::int2fp_rn<float>(i);
        ffloat z = renormalize(xx - t_rint * LN2_HI - t_rint * LN2_LO);

        // exp polynomial: 5 high-order terms in float, remaining 7 in ff
        ffloat t = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 5>(z, exp_c);

        /* accurate_scale(t, i): t * 2^i in fp32mp2 (split exponent for large |i|)*/
        int k   = i / 2;
        int ek  = 127 + k;
        int ek2 = 127 + (i - k);
        if (ek  < 1) ek  = 1;
        if (ek2 < 1) ek2 = 1;

        float  scale_lo   = fpmp::internal_bit_cast<float>(static_cast<unsigned>(ek)  << 23);
        float  scale_hi   = fpmp::internal_bit_cast<float>(static_cast<unsigned>(ek2) << 23);
        ffloat exp_scaled = ffloat(t.hi() * scale_lo * scale_hi, t.lo() * scale_lo * scale_hi);

        /* Correction: exp(-x^2) = exp_scaled * (1 + (-x^2 - xx)) same as double fma(t3, -x*x - xx, t3) */
        ffloat remainder = renormalize(-a * a - xx);
        ffloat exp_xx    = exp_scaled * remainder + exp_scaled;
        ffloat erfc_val  = renormalize(t1 * exp_xx);

        if (is_neg)
            erfc_val = renormalize(ffloat(2.0) - erfc_val);

        *res_hi = erfc_val.hi();
        *res_lo = erfc_val.lo();
    } // __nv_fpmp2_erfc

    /*
    * --------------------------------------------------------------------
    * Zeroth-order Boys function F_0(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    *   F_0(x) = 0.5 * sqrt(pi/x) * erf(sqrt(x))     for x > 0
    *   F_0(0) = 1
    *
    * Minimax polynomial approximation converted from double precision.
    * Four ranges with transformed-argument Horner polynomials:
    *   x > 34.38    : asymptotic  (sqrt(pi)/2) * rsqrt(x)
    *   x < 4        : 17-term minimax in (3 - x)
    *   4 <= x < 11.46: 20-term minimax in (6.92 - x)
    *   11.46 <= x <= 34.38: 19-term minimax in (rsqrt(x)^2 - 0.058)
    *
    * No erf, no sqrt.  Only rsqrt in ranges 1 and 4.
    * All arithmetic is in fp32mp2 (no double-precision operations).
    * --------------------------------------------------------------------
    */
    /*
    // Note: __FPMP_API_DECL__ (non-static) in inline mode for performance.
    // With -dc compilation, nvcc caps registers at ~37 for static
    // __forceinline__ functions, ignoring __launch_bounds__ on the caller.
    // This limits the boys kernel to 5.8x speedup instead of 12x.
    // Without static, the compiler respects __launch_bounds__ and allocates
    // 156 registers, enabling full ILP across the polynomial chains.
    //
    // In library build mode (__FPMP_BUILD_LIB__), static is required to
    // avoid ODR violations with the explicit float specialization in the
    // __FPMP_USE_LIB__ block, which would cause infinite recursion in LTO.
    */
    /*
    * Define __FPMP_USE_ACCURATE_BOYS_F0__ to use the accurate implementation
    * providing 43 bits of accuracy
    */
    // #define __FPMP_USE_ACCURATE_BOYS_F0__

    #if defined(__FPMP_BUILD_LIB__)
      #define __FPMP_INTERNAL_CUSTOM_DECL__ __FPMP_INTERNAL_DECL__
    #else
      #define __FPMP_INTERNAL_CUSTOM_DECL__ __FPMP_API_DECL__
    #endif

    #if !defined(__FPMP_USE_ACCURATE_BOYS_F0__)
      #define __FPMP_RENORMALIZE__(v) v = renormalize(v)
      #define __FPMP_SUB__(v, x) sub<fpmp2_accuracy::high>(v, x)
      #define __FPMP_METHOD__ fpmp2_accuracy::low
    #else
      #define __FPMP_RENORMALIZE__(v)
      #define __FPMP_SUB__(v, x) ((v) - (x))
      #define __FPMP_METHOD__ fpmp2_accuracy::def
    #endif

    template<typename FpType = float>
    __FPMP_INTERNAL_CUSTOM_DECL__ void __nv_fpmp2_boys_f0 (const FpType a_hi,
                                                           const FpType a_lo,
                                                           FpType*      res_hi,
                                                           FpType*      res_lo)
    {
        using ffloat = fpmp2_t<FpType, __FPMP_METHOD__>;

        ffloat a(a_hi, a_lo);
        ffloat r;

        if (a_hi >= 0x1.6ebc6ap3f) // a >= 11.46
            r = rsqrt(a);

        if (a_hi > 34.3816f) 
        {
            constexpr ffloat sqrt_pi_4(0x1.c5bf891b4ef6bp-1);
            ffloat result = sqrt_pi_4 * r;
            __FPMP_RENORMALIZE__(result);
            *res_hi = result.hi(); *res_lo = result.lo();
            return;
        }

        if (a_hi < 0x1p2f) 
        {
            /* a < 4: 17-term minimax in x = 3 - a (|x| <= 3, |x| <= 1 typical),
             * evaluated via compensated Horner with an M = 4 plain-FpType
             * head (ascending order: c[0] = constant, c[16] = leading).
             *
             * The top 4 coefficients are c[13]..c[16] with magnitudes
             * 2^-42 .. 2^-52 ~= 3.6e-13 .. 2.3e-16. After 4 plain Horner
             * steps the accumulator magnitude is ~= 7e-12 (worst case |x|=3),
             * so the rounding error those steps drop is < |acc| * 2^-25
             * ~= 1e-19; carried forward through the remaining 13 ff Horner
             * steps and amplified by x^13 ~= 1.6e6 at worst, this still
             * sits well below the ff precision floor (~1.4e-14).  Phase 2a's
             * skipped iterations drop c[13..16].lo() ~= 2^-72 .. 2^-76,
             * totally negligible.  Phase 1 compensation still covers the
             * 13 lower-degree iterations where |acc| grows from 7e-12 to
             * ~0.5 -- that's where the rounding error actually matters.
             *
             * Tried M = 5 (skip compensation on c[12] too, |c[12]| ~= 5.3e-12)
             * and lost 2 bits across all buckets (43->41) with warn % blowing
             * up by 4 orders of magnitude -- c[12]'s rounding error,
             * propagated through 12 ff steps and amplified by x^12 worst-case,
             * just barely surfaces above the precision floor.  M = 4 is the
             * sweet spot.
             */
            ffloat x = __FPMP_SUB__(ffloat(0x1.8p1), a);
            constexpr ffloat c[17] = {
                ffloat(0x1.023951b248d32p-1),
                ffloat(0x1.364f8131f82eap-4),
                ffloat(0x1.e4ab5374f7553p-7),
                ffloat(0x1.65408fedfe46fp-9),
                ffloat(0x1.d70cd2ae22daap-12),
                ffloat(0x1.133abad3c99dp-14),
                ffloat(0x1.1e134e84b9a2ap-17),
                ffloat(0x1.0a6cf9d0cf714p-20),
                ffloat(0x1.c039bccbce7dep-24),
                ffloat(0x1.572c0936d0dcp-27),
                ffloat(0x1.e19e5b3b8b31bp-31),
                ffloat(0x1.37ce8ea919fd3p-34),
                ffloat(0x1.76402f1b7e023p-38),
                ffloat(0x1.99cd5cbd06043p-42),
                ffloat(0x1.03356d73ab25fp-45),
                ffloat(0x1.887a5d0c86047p-52),
                ffloat(0x1.07f3442d6af1ep-52)
            };
            ffloat v = fpmp::poly_eval<fpmp::poly_method::horner_comp, 4>(x, c);
            *res_hi = v.hi(); *res_lo = v.lo();
            return;
        } // if (a_hi < 0x1p2f)

        if (a_hi < 0x1.6ebc6ap3f) 
        {
            /* 4 <= a < 11.46: 20-term minimax in x = 6.92 - a (degree 19).
             * Standard ff-Horner with periodic renormalization -- the
             * compensated variant loses accuracy here because the
             * coefficients span ~22 orders of magnitude. */
            ffloat x = __FPMP_SUB__(ffloat(0x1.baf1a8p1), a);
            ffloat v =  ffloat(0x1.95402da668f4fp-73);
            v = v * x + ffloat(0x1.43744ab1a0e5ap-66);
            v = v * x + ffloat(0x1.f70f3953813b1p-61);
            v = v * x + ffloat(0x1.00b2c5aae06a1p-55);
            v = v * x + ffloat(0x1.87ddc6a10f513p-51);
            v = v * x + ffloat(0x1.e450e0340da6fp-47);
            v = v * x + ffloat(0x1.ffc73283f2e3dp-43);
            v = v * x + ffloat(0x1.dff8a98149ce4p-39);
            v = v * x + ffloat(0x1.98aa56613b23p-35);
            __FPMP_RENORMALIZE__(v);
            v = v * x + ffloat(0x1.3f3d23359c3f4p-31);
            v = v * x + ffloat(0x1.ca89e4f410357p-28);
            v = v * x + ffloat(0x1.2ddf249b49215p-24);
            __FPMP_RENORMALIZE__(v);   
            v = v * x + ffloat(0x1.6a60fc5c32d39p-21);
            v = v * x + ffloat(0x1.8a0af8927f728p-18);
            v = v * x + ffloat(0x1.81949bbc35f76p-15);
            __FPMP_RENORMALIZE__(v);   
            v = v * x + ffloat(0x1.51d1e0119bf15p-12);
            v = v * x + ffloat(0x1.090a189fdb05bp-9);
            __FPMP_RENORMALIZE__(v);   
            v = v * x + ffloat(0x1.7a16985c09ba2p-7);
            v = v * x + ffloat(0x1.04f3fb31bb071p-4);
            v = v * x + ffloat(0x1.e3ae966b0f402p-2);
            __FPMP_RENORMALIZE__(v);   
            *res_hi = v.hi(); *res_lo = v.lo();
            return;
        } // if (a_hi < 0x1.6ebc6ap3f)

        /* 11.46 <= a <= 34.38: 19-term minimax in x = rsqrt(a)^2 - offset
         * (degree 18), evaluated via compensated Horner. Coefficients are
         * in ascending order (c[0] = constant, c[18] = leading). */
        ffloat x = __FPMP_SUB__(r * r, ffloat(0x1.dc88f0479694p-5));
        constexpr ffloat c[19] = {
            ffloat( 0x1.fffffed709646p-1),
            ffloat(-0x1.71471b65714a8p-20),
            ffloat(-0x1.85179c0504089p-13),
            ffloat(-0x1.d99f05bac9192p-7),
            ffloat(-0x1.681ebc0bfc87p-1),
            ffloat(-0x1.531388eeb3e37p4),
            ffloat(-0x1.56423d3c9aee8p8),
            ffloat(-0x1.55574adbabed4p9),
            ffloat( 0x1.fc17297038ab6p15),
            ffloat( 0x1.5d36617bab8fep18),
            ffloat(-0x1.c1e691926af02p23),
            ffloat(-0x1.d06f6451b9b99p24),
            ffloat( 0x1.baa02bef66d96p31),
            ffloat(-0x1.35636d415d49bp34),
            ffloat(-0x1.61bc3c687e6ffp39),
            ffloat( 0x1.39af9c72a5c92p43),
            ffloat( 0x1.3f351ae8d044ap46),
            ffloat(-0x1.ba4d5cfd521a5p50),
            ffloat(-0x1.6d64bf85e3416p50)
        };
        ffloat v = fpmp::poly_eval<fpmp::poly_method::horner_comp>(x, c);
        r = r * ffloat(0x1.c5bf8ap-1);
        ffloat result = v * r;
        __FPMP_RENORMALIZE__(result);

        *res_hi = result.hi(); *res_lo = result.lo();
    } // __nv_fpmp2_boys_f0

    /*
    * --------------------------------------------------------------------
    * Inverse normal CDF: normcdfinv(p) = sqrt(2) * erfinv(2p - 1)
    * --------------------------------------------------------------------
    * Rational approximation (Mike Giles coefficients),
    * fully converted to fp32mp2 arithmetic (no fp64 operations).
    * Three regions selected by w = -log(4p(1-p)), a = 2p - 1:
    *   Central (w < 6.125):      degree-22 polynomial in (w - 3.125)
    *   Tail 1  (6.125 <= w < 16): degree-18 polynomial in (sqrt(w) - 3.25)
    *   Tail 2  (w >= 16):        degree-24 polynomial in (sqrt(w) - 7.25)
    * The central path (~99.9% of inputs) is straight-line code; tail
    * regions are branched off so the common path skips sqrt and tail
    * polynomials entirely.  All regions produce full fp32mp2 precision.
    *
    * fp64mp2 specialization uses CUDA's erfcinv on device and falls
    * back to this polynomial on host (no standard erfcinv/normcdfinv).
    * --------------------------------------------------------------------
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_normcdfinv(const FpType x_hi,
                                                      const FpType x_lo,
                                                      FpType*      res_hi,
                                                      FpType*      res_lo)
    {
        using ffloat = fp32mp2_low;

        constexpr ffloat sqrt2(0x1.6a09e667f3bcdp+0);

        /* Central polynomial: rc(tc) = c22 + c21*tc + ... + c0*tc^22,
         * tc = w - 3.125  (>99.9% of inputs land here).
         * Ascending degree; M = 9 high-order entries (rc_c[14..22] =
         * original c0..c8) are plain float and run in the dispatcher's
         * FpType phase.  The transition step `rcf * tc.hi() + c9`
         * matches the previous hand-rolled float*float + ff step
         * bit-for-bit, so this refactor is numerically identical.
         */
        constexpr ffloat rc_c[23] = {
            ffloat( 1.6536545626831027e+00),   // [ 0] (= c22, constant)
            ffloat( 2.4015818242558962e-01),   // [ 1] (= c21)
            ffloat(-6.0336708714301491e-03),   // [ 2] (= c20)
            ffloat(-7.4070253416626698e-04),   // [ 3] (= c19)
            ffloat( 1.8673420803405714e-04),   // [ 4] (= c18)
            ffloat(-1.3882523362786469e-05),   // [ 5] (= c17)
            ffloat(-1.3654692000834679e-06),   // [ 6] (= c16)
            ffloat( 4.2347877827932404e-07),   // [ 7] (= c15)
            ffloat(-2.9070369957882005e-08),   // [ 8] (= c14)
            ffloat(-4.1126339803469837e-09),   // [ 9] (= c13)
            ffloat( 1.0512122733215323e-09),   // [10] (= c12)
            ffloat(-5.4154120542946279e-11),   // [11] (= c11)
            ffloat(-1.2975133253453532e-11),   // [12] (= c10)
            ffloat( 2.6335093153082323e-12),   // [13] (= c9, last ff term)
            /* high-order M = 9 entries: .lo() == 0 by construction */
            ffloat(-8.1519342e-14f),           // [14] (= c8)
            ffloat(-4.0545663e-14f),           // [15] (= c7)
            ffloat( 6.6376381e-15f),           // [16] (= c6)
            ffloat( 2.0972768e-17f),           // [17] (= c5)
            ffloat(-1.3331717e-16f),           // [18] (= c4)
            ffloat( 1.1157878e-17f),           // [19] (= c3)
            ffloat( 1.2858481e-18f),           // [20] (= c2)
            ffloat(-1.6850591e-19f),           // [21] (= c1)
            ffloat(-3.6444121e-21f)            // [22] (= c0, leading)
        };

        /* Tail 1 polynomial: rt(tt) = t18 + t17*tt + ... + t0*tt^18,
         * tt = sqrt(w) - 3.25 (w in [6.25, 16],  |z| ~ 2.5 to 5.5 sigma).
         * Ascending degree; M = 9 high-order entries (rt_c[10..18] =
         * original t0..t8) are plain float.  Transition is float*float
         * + ff at rt_c[9] = t9 -- bit-identical to the previous chain.
         */
        constexpr ffloat rt_c[19] = {
            ffloat( 3.0838856104922208e+00),   // [ 0] (= t18, constant)
            ffloat( 1.0052589676941592e+00),   // [ 1] (= t17)
            ffloat( 5.3709145535900636e-03),   // [ 2] (= t16)
            ffloat(-3.7512085075692412e-03),   // [ 3] (= t15)
            ffloat( 2.4914420961078508e-03),   // [ 4] (= t14)
            ffloat(-1.6882755560235047e-03),   // [ 5] (= t13)
            ffloat( 9.5328937973738050e-04),   // [ 6] (= t12)
            ffloat(-3.5503752036284748e-04),   // [ 7] (= t11)
            ffloat( 2.4031110387097894e-05),   // [ 8] (= t10)
            ffloat( 6.8284851459573175e-05),   // [ 9] (= t9, last ff term)
            /* high-order M = 9 entries: .lo() == 0 by construction */
            ffloat(-4.7318229e-05f),           // [10] (= t8)
            ffloat( 1.2475304e-05f),           // [11] (= t7)
            ffloat( 2.9234449e-06f),           // [12] (= t6)
            ffloat(-4.0138675e-06f),           // [13] (= t5)
            ffloat( 1.5027404e-06f),           // [14] (= t4)
            ffloat( 1.8239629e-08f),           // [15] (= t3)
            ffloat(-2.7517406e-07f),           // [16] (= t2)
            ffloat( 9.0756562e-08f),           // [17] (= t1)
            ffloat( 2.2137377e-09f)            // [18] (= t0, leading)
        };

        /* Tail 2 polynomial: rt2(tt2) = u24 + u23*tt2 + ... + u0*tt2^24,
         * tt2 = sqrt(w) - 7.25  (w >= 16,  |z| > 5.5 sigma).
         * Covers all representable float inputs including denormals;
         * Chebyshev interp at 100-digit precision, relative approx
         * error < 2^{-46} over the fitted range.
         * Ascending degree; M = 13 high-order entries (rt2_c[12..24] =
         * original u0..u12) are plain float.  The transition step
         * `rt2f * tt2.hi() + u13` is one ULP-level different from
         * the previous chain (which lifted to ff one step earlier and
         * thus included tt2.lo() in the transition product); the
         * change is well inside the polynomial truncation noise.
         */
        constexpr ffloat rt2_c[25] = {
            ffloat( 7.12113663660053842e+00),  // [ 0] (= u24, constant)
            ffloat( 1.00834082079167930e+00),  // [ 1] (= u23)
            ffloat(-5.05906408540271685e-04),  // [ 2] (= u22)
            ffloat( 1.14184074807230187e-05),  // [ 3] (= u21)
            ffloat( 4.29790660561751423e-06),  // [ 4] (= u20)
            ffloat(-1.21177482126504764e-06),  // [ 5] (= u19)
            ffloat( 2.33428873326838655e-07),  // [ 6] (= u18)
            ffloat(-3.92578613880982197e-08),  // [ 7] (= u17)
            ffloat( 6.14877480871698432e-09),  // [ 8] (= u16)
            ffloat(-9.24007580865063697e-10),  // [ 9] (= u15)
            ffloat( 1.34759296085592452e-10),  // [10] (= u14)
            ffloat(-1.76387252450593334e-11),  // [11] (= u13, last ff term)
            /* high-order M = 13 entries: .lo() == 0 by construction */
            ffloat( 2.09393731e-12f),          // [12] (= u12)
            ffloat(-6.91218317e-13f),          // [13] (= u11)
            ffloat( 1.95788733e-13f),          // [14] (= u10)
            ffloat( 4.98296865e-14f),          // [15] (= u9)
            ffloat(-2.64007334e-14f),          // [16] (= u8)
            ffloat(-5.68006053e-15f),          // [17] (= u7)
            ffloat( 3.21120849e-15f),          // [18] (= u6)
            ffloat( 2.6060760e-16f),           // [19] (= u5)
            ffloat(-2.2467865e-16f),           // [20] (= u4)
            ffloat( 1.9526573e-18f),           // [21] (= u3)
            ffloat( 7.8681698e-18f),           // [22] (= u2)
            ffloat(-6.7040324e-19f),           // [23] (= u1)
            ffloat(-2.2357236e-20f)            // [24] (= u0, leading)
        };

        ffloat p = renormalize(ffloat(x_hi, x_lo));

        /* Standard mathematical convention: normcdfinv(0) = -inf, normcdfinv(1) = +inf */
        if (p.hi() <= 0.0f) { *res_hi = fpmp::internal_bit_cast<float>(0xFF800000U); *res_lo = 0.0f; return; }
        if (p.hi() >= 1.0f) { *res_hi = fpmp::internal_bit_cast<float>(0x7F800000U); *res_lo = 0.0f; return; }

        /* a = 2p - 1, accurate subtraction for p ~= 0.5 */
        ffloat two_p = p + p;
        ffloat a = sub<fpmp2_accuracy::high>(two_p, 1.0f);

        /* w = -log(1 - a^2) = -log(4p(1-p))
         * Compute 1-p with accurate subtraction to handle p near 0 or 1
         */
        ffloat omp = sub<fpmp2_accuracy::high>(1.0f, p);
        ffloat arg = 4.0f * p * omp;

        if (arg.hi() <= 0.0f)
            arg = ffloat(0x1.0p-126f);

        float log_hi, log_lo;
        __nv_fpmp2_log(arg.hi(), arg.lo(), &log_hi, &log_lo);
        ffloat w = -ffloat(log_hi, log_lo);

        /* Central region (w < 6.125, |z| < ~3.3 sigma, >99.9% of inputs):
         * Horner in tc = w - 3.125 via the mixed-precision dispatcher
         * (9 high-order float coeffs c0..c8, 14 low-order ff coeffs c9..c22).
         */
        ffloat tc   = w - ffloat(3.125f);
        ffloat poly = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 9>(tc, rc_c);

        /* Tail regions (w >= 6.125): branched since <0.1% of inputs.
         * sqrt(w) is also deferred into this branch.
         */
        if (w.hi() >= 6.125f) {
            float sw_hi, sw_lo;
            __nv_fpmp2_sqrt(w.hi(), w.lo(), &sw_hi, &sw_lo);
            ffloat sw(sw_hi, sw_lo);

            /* Tail 1 (6.125 <= w < 16, |z| ~ 3.3 to 5.5 sigma):
             * Horner in tt = sqrt(w) - 3.25 via the dispatcher
             * (9 high-order float coeffs t0..t8, 10 low-order ff coeffs t9..t18).
             */
            ffloat tt = sw - ffloat(3.25f);
            poly      = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 9>(tt, rt_c);

            /* Tail 2 (w >= 16, |z| > 5.5 sigma):
             * Horner in tt2 = sqrt(w) - 7.25 via the dispatcher
             * (13 high-order float coeffs u0..u12, 12 low-order ff coeffs u13..u24).
             *
             * Note: the dispatcher's transition step uses tt2.hi() only
             * (float*float + ff), whereas the previous hand-rolled chain
             * promoted to ff one step earlier and thus included tt2.lo()
             * in the transition product.  The numerical change is sub-ULP
             * at the polynomial value and well inside the truncation noise.
             */
            if (w.hi() >= 16.0f) {
                ffloat tt2 = sw - ffloat(7.25f);
                poly       = fpmp::poly_eval<fpmp::poly_method::horner_mixed, 13>(tt2, rt2_c);
            }
        }

        /* Scale: result = poly * a * sqrt(2) */
        ffloat result = renormalize(poly * a * sqrt2);

        *res_hi = result.hi();
        *res_lo = result.lo();
    } // __nv_fpmp2_normcdfinv

    /*
    * ============================================================================
    * Inverse CDF for integer uniform RNG outputs
    *
    * Convert a uniform integer random value to a Gaussian fp32mp2 sample
    * via normcdfinv.  Mirrors the input around 0.5 to keep the probability
    * argument in (0, 0.5], preserving precision in the polynomial.
    *
    *   uint32_t version: p = (x + 0.5) / 2^32
    *   uint64_t version: p = (x + 0.5) / 2^48  (top 48 bits of 64)
    * ============================================================================
    */
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_icdf(uint32_t x, float* res_hi, float* res_lo)
    {
        float sign = 1.0f;
        if (x > 0x80000000u) {
            x = 0xFFFFFFFFu - x;
            sign = -1.0f;
        }
        /* p = (x + 0.5) / 2^32  in  (0, 0.5]
         * Split x into two 16-bit halves for exact fp32mp2 representation.
         */
        float hi = (float)(x >> 16) * 0x1.0p-16f;
        float lo = ((float)(x & 0xFFFFu) + 0.5f) * 0x1.0p-32f;
        float p_hi = hi + lo;
        float p_lo = lo - (p_hi - hi);

        __nv_fpmp2_normcdfinv(p_hi, p_lo, res_hi, res_lo);
        /* Clamp to +-FLT_MAX for safe Gaussian variate generation (no infinities) */
        if (*res_hi >= 0x1.fffffep+127f)  { *res_hi =  0x1.fffffep+127f; *res_lo = 0.0f; }
        if (*res_hi <= -0x1.fffffep+127f) { *res_hi = -0x1.fffffep+127f; *res_lo = 0.0f; }
        *res_hi *= sign;
        *res_lo *= sign;
    } // __nv_fpmp2_icdf

    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_icdf(uint64_t x, float* res_hi, float* res_lo)
    {
        float sign = 1.0f;
        x >>= 16;   /* keep top 48 bits (matches fp32mp2 precision) */
        if (x > 0x800000000000ULL) {
            x = 0xFFFFFFFFFFFFULL - x;
            sign = -1.0f;
        }
        /* p = (x + 0.5) / 2^48  in  (0, 0.5]
         * Split 48-bit x into two 24-bit halves for exact float representation.
         */
        float hi = (float)(uint32_t)(x >> 24) * 0x1.0p-24f;
        float lo = ((float)(uint32_t)(x & 0xFFFFFFu) + 0.5f) * 0x1.0p-48f;
        float p_hi = hi + lo;
        float p_lo = lo - (p_hi - hi);

        __nv_fpmp2_normcdfinv(p_hi, p_lo, res_hi, res_lo);
        /* Clamp to +-FLT_MAX for safe Gaussian variate generation (no infinities) */
        if (*res_hi >= 0x1.fffffep+127f)  { *res_hi =  0x1.fffffep+127f; *res_lo = 0.0f; }
        if (*res_hi <= -0x1.fffffep+127f) { *res_hi = -0x1.fffffep+127f; *res_lo = 0.0f; }
        *res_hi *= sign;
        *res_lo *= sign;
    } // __nv_fpmp2_icdf

    /*
    * ============================================================================
    * Floating-Point Manipulation and Min/Max
    * ============================================================================
    * Dedicated implementations of fabs / fmin / fmax / min / max for both fp32mp2 and
    * fp64mp2 that operate directly on the (hi, lo) pair without going through
    * a lossy intermediate `double` conversion.  The same template body is
    * correct for FpType == float and FpType == double, so no separate fp64
    * specialization is needed.
    *
    * Conventions follow C99/IEEE-754:
    *   - fabs preserves NaN payload and turns -0 into +0 (via ::fabs on hi).
    *   - fmin/fmax treat NaN as missing data: if exactly one operand is NaN,
    *     the non-NaN one is returned; if both are NaN, NaN is returned.
    *   - min/max keep std::min/std::max branch semantics:
    *     ties/unordered select the first argument.
    * ============================================================================
    */

    /*
    * fabs: |x|.  For a normalized (hi, lo) pair the value's sign is the sign
    * of `hi`, and `|hi| > |lo|`, so flipping both components when `hi` is
    * negative yields the absolute value while preserving the residual `lo`
    * exactly.  We use ::fabs on `hi` to get IEEE-correct handling of -0 and
    * NaN, and use the sign of the original `hi` to decide whether to flip
    * `lo`.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_fabs (const FpType x_hi,
                                                 const FpType x_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        *res_hi = ::fabs(x_hi);
        *res_lo = (x_hi < FpType(0)) ? -x_lo : x_lo;
    }

    /*
    * fmax: max(x, y).  Lexicographic comparison on (hi, lo) -- valid because
    * normalized fpmp2 inputs satisfy |lo| < ulp(hi)/2, so `x > y` iff
    * `x_hi > y_hi || (x_hi == y_hi && x_lo > y_lo)`.  NaN handling follows
    * C99/IEEE-754-2008: if exactly one operand is NaN, return the other.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_fmax (const FpType x_hi,
                                                 const FpType x_lo,
                                                 const FpType y_hi,
                                                 const FpType y_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        const bool x_is_nan = fpmp::internal_isnan(x_hi);
        const bool y_is_nan = fpmp::internal_isnan(y_hi);
        if (x_is_nan && !y_is_nan) { *res_hi = y_hi; *res_lo = y_lo; return; }
        if (y_is_nan && !x_is_nan) { *res_hi = x_hi; *res_lo = x_lo; return; }
        const bool x_greater = (x_hi > y_hi) || (x_hi == y_hi && x_lo > y_lo);
        if (x_greater) { *res_hi = x_hi; *res_lo = x_lo; }
        else           { *res_hi = y_hi; *res_lo = y_lo; }
    }

    /*
    * fmin: min(x, y).  Mirror image of fmax.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_fmin (const FpType x_hi,
                                                 const FpType x_lo,
                                                 const FpType y_hi,
                                                 const FpType y_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        const bool x_is_nan = fpmp::internal_isnan(x_hi);
        const bool y_is_nan = fpmp::internal_isnan(y_hi);
        if (x_is_nan && !y_is_nan) { *res_hi = y_hi; *res_lo = y_lo; return; }
        if (y_is_nan && !x_is_nan) { *res_hi = x_hi; *res_lo = x_lo; return; }
        const bool x_less = (x_hi < y_hi) || (x_hi == y_hi && x_lo < y_lo);
        if (x_less) { *res_hi = x_hi; *res_lo = x_lo; }
        else        { *res_hi = y_hi; *res_lo = y_lo; }
    }

    /*
    * max: std::max-like selection for fpmp2 values.  Uses the same
    * lexicographic ordering as fmax, but keeps std::max semantics:
    * return y only when x < y; otherwise return x (ties/unordered -> x).
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_max (const FpType x_hi,
                                                const FpType x_lo,
                                                const FpType y_hi,
                                                const FpType y_lo,
                                                FpType*      res_hi,
                                                FpType*      res_lo)
    {
        const bool x_less = (x_hi < y_hi) || (x_hi == y_hi && x_lo < y_lo);
        if (x_less) { *res_hi = y_hi; *res_lo = y_lo; }
        else        { *res_hi = x_hi; *res_lo = x_lo; }
    }

    /*
    * min: std::min-like selection for fpmp2 values.  Uses the same
    * lexicographic ordering as fmin, but keeps std::min semantics:
    * return y only when y < x; otherwise return x (ties/unordered -> x).
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_min (const FpType x_hi,
                                                const FpType x_lo,
                                                const FpType y_hi,
                                                const FpType y_lo,
                                                FpType*      res_hi,
                                                FpType*      res_lo)
    {
        const bool y_less = (y_hi < x_hi) || (y_hi == x_hi && y_lo < x_lo);
        if (y_less) { *res_hi = y_hi; *res_lo = y_lo; }
        else        { *res_hi = x_hi; *res_lo = x_lo; }
    }

    /*
    * floor/ceil/round: dedicated fpmp2 implementations that operate directly
    * on the (hi, lo) pair and avoid collapsing through an intermediate double.
    *
    * floor:
    *   Let n = floor(x_hi). If x >= n, result is n. Otherwise result is n - 1.
    *
    * ceil:
    *   Let n = ceil(x_hi). If x <= n, result is n. Otherwise result is n + 1.
    *
    * round:
    *   C semantics (halfway away from zero), implemented as
    *     round(x) = floor(x + 0.5)  for x >= 0
    *     round(x) = ceil (x - 0.5)  for x <  0
    * using fpmp2_acc to keep the adjustment in pair precision.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_floor (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        // NaN check
        if ((x_hi != x_hi) || (x_lo != x_lo)) 
        {
            FpType res = x_hi+x_lo;
            *res_hi = res; *res_lo = res;
            return;
        }

        const FpType abs_hi = fpmp::internal_fabs(x_hi);
        const FpType int_scale = std::is_same<FpType, float>::value ? FpType(0x1.0p23f) : FpType(0x1.0p52);
        if (abs_hi >= int_scale) 
        {
            // x_hi is already an integer at this scale; floor(x_hi + x_lo) = x_hi + floor(x_lo).
            const FpType lo_floor = fpmp::internal_floor<FpType>(x_lo);
            FpType t_hi = x_hi, t_lo = FpType(0);
            __nv_fpmp2_acc<FpType>(lo_floor, &t_hi, &t_lo);
            *res_hi = t_hi; *res_lo = t_lo;
            return;
        }

        const FpType n = fpmp::internal_floor<FpType>(x_hi);
        if (x_hi != n || x_lo >= FpType(0)) 
        {
            *res_hi = n; *res_lo = FpType(0);
            return;
        }

        FpType t_hi = n, t_lo = FpType(0);
        __nv_fpmp2_acc<FpType>(FpType(-1), &t_hi, &t_lo);
        *res_hi = t_hi; *res_lo = t_lo;
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_ceil (const FpType x_hi,
                                                 const FpType x_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        // NaN check
        if ((x_hi != x_hi) || (x_lo != x_lo)) 
        {
            FpType res = x_hi+x_lo;
            *res_hi = res; *res_lo = res;
            return;
        }

        const FpType abs_hi = fpmp::internal_fabs(x_hi);
        const FpType int_scale = std::is_same<FpType, float>::value ? FpType(0x1.0p23f) : FpType(0x1.0p52);
        if (abs_hi >= int_scale) 
        {
            // x_hi is already an integer at this scale; ceil(x_hi + x_lo) = x_hi + ceil(x_lo).
            const FpType lo_ceil = fpmp::internal_ceil<FpType>(x_lo);
            FpType t_hi = x_hi, t_lo = FpType(0);
            __nv_fpmp2_acc<FpType>(lo_ceil, &t_hi, &t_lo);
            *res_hi = t_hi; *res_lo = t_lo;
            return;
        }

        const FpType n = fpmp::internal_ceil<FpType>(x_hi);
        if (x_hi != n || x_lo <= FpType(0)) 
        {
            *res_hi = n; *res_lo = FpType(0);
            return;
        }

        FpType t_hi = n, t_lo = FpType(0);
        __nv_fpmp2_acc<FpType>(FpType(1), &t_hi, &t_lo);
        *res_hi = t_hi; *res_lo = t_lo;
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_round (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        // NaN check
        if ((x_hi != x_hi) || (x_lo != x_lo)) 
        {
            FpType res = x_hi+x_lo;
            *res_hi = res; *res_lo = res;
            return;
        }

        const bool x_neg = (x_hi < FpType(0)) || (x_hi == FpType(0) && x_lo < FpType(0));

        FpType t_hi = x_hi, t_lo = x_lo;
        __nv_fpmp2_acc<FpType>(x_neg ? FpType(-0.5) : FpType(0.5), &t_hi, &t_lo);

        if (x_neg) __nv_fpmp2_ceil (t_hi, t_lo, res_hi, res_lo);
        else       __nv_fpmp2_floor(t_hi, t_lo, res_hi, res_lo);
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_trunc (const FpType x_hi,
                                                  const FpType x_lo,
                                                  FpType*      res_hi,
                                                  FpType*      res_lo)
    {
        // NaN check
        if ((x_hi != x_hi) || (x_lo != x_lo)) 
        {
            FpType res = x_hi+x_lo;
            *res_hi = res; *res_lo = res;
            return;
        }

        const FpType abs_hi = fpmp::internal_fabs(x_hi);
        const FpType int_scale = std::is_same<FpType, float>::value ? FpType(0x1.0p23f) : FpType(0x1.0p52);
        if (abs_hi >= int_scale) 
        {
            // x_hi is integral at this scale and dominates sign, so trunc is:
            //   x_hi > 0 : floor(x_hi + x_lo) = x_hi + floor(x_lo)
            //   x_hi < 0 : ceil (x_hi + x_lo) = x_hi + ceil (x_lo)
            const FpType lo_trunc = (x_hi < FpType(0))
                                  ? fpmp::internal_ceil<FpType>(x_lo)
                                  : fpmp::internal_floor<FpType>(x_lo);
            FpType t_hi = x_hi, t_lo = FpType(0);
            __nv_fpmp2_acc<FpType>(lo_trunc, &t_hi, &t_lo);
            *res_hi = t_hi; *res_lo = t_lo;
            return;
        }

        // Fast small-magnitude path:
        // Start from trunc(x_hi), then apply at most a +/-1 correction only when
        // x_hi is already integral and x_lo nudges the exact value across that integer.
        const FpType n = fpmp::internal_trunc<FpType>(x_hi);
        if (x_hi != n) 
        {
            *res_hi = n; *res_lo = FpType(0);
            return;
        }

        const bool x_neg = (x_hi < FpType(0)) || (x_hi == FpType(0) && x_lo < FpType(0));
        const int delta = (!x_neg && x_lo < FpType(0)) ? -1 :
                          ( x_neg && x_lo > FpType(0)) ?  1 : 0;
        if (delta != 0) 
        {
            FpType t_hi = n, t_lo = FpType(0);
            __nv_fpmp2_acc<FpType>(static_cast<FpType>(delta), &t_hi, &t_lo);
            *res_hi = t_hi; *res_lo = t_lo;
            return;
        }

        *res_hi = n;
        *res_lo = FpType(0);
    }

    /*
    * ============================================================================
    * Transcendental Math Function Placeholders (fp32mp2)
    * ============================================================================
    * Placeholder implementations that delegate to standard double-precision
    * system functions with proper hi/lo splitting.
    * ============================================================================
    */

    /* Defensive #undef for all placeholder macros: protect against the
     * (unlikely) case that something earlier in the translation unit
     * already defined them.  Matching #undef block follows the last
     * placeholder definition below to avoid leaking these macros into
     * downstream includes. */
    #undef __FPMP_MATH_PLACEHOLDER_1A__
    #undef __FPMP_MATH_PLACEHOLDER_2A__
    #undef __FPMP_MATH_PLACEHOLDER_1A_RETINT__
    #undef __FPMP_MATH_PLACEHOLDER_1A_RETLL__
    #undef __FPMP_MATH_PLACEHOLDER_1A_RETL__
    #undef __FPMP_MATH_PLACEHOLDER_FP_INT__
    #undef __FPMP_MATH_PLACEHOLDER_FP_LINT__
    #undef __FPMP_MATH_PLACEHOLDER_INT_FP__

    // Helper macro: placeholder implementation that delegates to a standard double-precision
    // math function with proper hi/lo splitting via fpmp2_t conversions.
    // Uses explicit fpmp2_t construction/conversion to avoid NVCC name resolution issues.
    #define __FPMP_MATH_PLACEHOLDER_1A__(name) \
    template<typename FpType = float> \
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_##name (const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo) \
    { \
        using mp2_t = fpmp2_t<FpType>; \
        double r = ::name(static_cast<double>(mp2_t(x_hi, x_lo))); \
        mp2_t result(r); \
        *res_hi = result.hi(); *res_lo = result.lo(); \
    }

    #define __FPMP_MATH_PLACEHOLDER_2A__(name) \
    template<typename FpType = float> \
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_##name (const FpType x_hi, const FpType x_lo, const FpType y_hi, const FpType y_lo, FpType* res_hi, FpType* res_lo) \
    { \
        using mp2_t = fpmp2_t<FpType>; \
        double r = ::name(static_cast<double>(mp2_t(x_hi, x_lo)), static_cast<double>(mp2_t(y_hi, y_lo))); \
        mp2_t result(r); \
        *res_hi = result.hi(); *res_lo = result.lo(); \
    }

    #define __FPMP_MATH_PLACEHOLDER_1A_RETINT__(name) \
    template<typename FpType = float> \
    __FPMP_INTERNAL_DECL__ int __nv_fpmp2_##name (const FpType x_hi, const FpType x_lo) \
    { \
        using mp2_t = fpmp2_t<FpType>; \
        return ::name(static_cast<double>(mp2_t(x_hi, x_lo))); \
    }

    #define __FPMP_MATH_PLACEHOLDER_1A_RETLL__(name) \
    template<typename FpType = float> \
    __FPMP_INTERNAL_DECL__ long long int __nv_fpmp2_##name (const FpType x_hi, const FpType x_lo) \
    { \
        using mp2_t = fpmp2_t<FpType>; \
        return ::name(static_cast<double>(mp2_t(x_hi, x_lo))); \
    }

    #define __FPMP_MATH_PLACEHOLDER_1A_RETL__(name) \
    template<typename FpType = float> \
    __FPMP_INTERNAL_DECL__ long int __nv_fpmp2_##name (const FpType x_hi, const FpType x_lo) \
    { \
        using mp2_t = fpmp2_t<FpType>; \
        return ::name(static_cast<double>(mp2_t(x_hi, x_lo))); \
    }

    #define __FPMP_MATH_PLACEHOLDER_FP_INT__(name) \
    template<typename FpType = float> \
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_##name (const FpType x_hi, const FpType x_lo, int n, FpType* res_hi, FpType* res_lo) \
    { \
        using mp2_t = fpmp2_t<FpType>; \
        double r = ::name(static_cast<double>(mp2_t(x_hi, x_lo)), n); \
        mp2_t result(r); \
        *res_hi = result.hi(); *res_lo = result.lo(); \
    }

    #define __FPMP_MATH_PLACEHOLDER_FP_LINT__(name) \
    template<typename FpType = float> \
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_##name (const FpType x_hi, const FpType x_lo, long int n, FpType* res_hi, FpType* res_lo) \
    { \
        using mp2_t = fpmp2_t<FpType>; \
        double r = ::name(static_cast<double>(mp2_t(x_hi, x_lo)), n); \
        mp2_t result(r); \
        *res_hi = result.hi(); *res_lo = result.lo(); \
    }

    #define __FPMP_MATH_PLACEHOLDER_INT_FP__(name) \
    template<typename FpType = float> \
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_##name (int n, const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo) \
    { \
        using mp2_t = fpmp2_t<FpType>; \
        double r = ::name(n, static_cast<double>(mp2_t(x_hi, x_lo))); \
        mp2_t result(r); \
        *res_hi = result.hi(); *res_lo = result.lo(); \
    }

    /* log2, log10, exp2, exp10, expm1: dedicated fp32mp2 implementations
     * live in the dedicated math section above (composed over the
     * dedicated fp32mp2 log/exp with ln(2)/ln(10) constants in fp32mp2;
     * expm1 has a small-|x| Taylor branch).  fp64mp2 specializations are
     * declared below via __FPMP_CALL_FP64MP2_MATH__.  exp10 had a hand
     * fallback even on fp32mp2 prior to the dedicated implementation;
     * that version is superseded. */
    __FPMP_MATH_PLACEHOLDER_1A__(logb)
    __FPMP_MATH_PLACEHOLDER_1A__(lgamma)
    __FPMP_MATH_PLACEHOLDER_1A__(tgamma)
    /* fmod, remainder: dedicated fp32mp2 implementations (integer
     * mantissa long-division, libdevice-style) live in the dedicated
     * math section above.  fp64mp2 paths stay on the explicit double
     * specializations further below. */
    __FPMP_MATH_PLACEHOLDER_2A__(hypot)
    __FPMP_MATH_PLACEHOLDER_2A__(copysign)
    __FPMP_MATH_PLACEHOLDER_2A__(fdim)
    __FPMP_MATH_PLACEHOLDER_2A__(nextafter)
    __FPMP_MATH_PLACEHOLDER_1A_RETINT__(ilogb)
    __FPMP_MATH_PLACEHOLDER_1A_RETLL__(llrint)
    __FPMP_MATH_PLACEHOLDER_1A_RETLL__(llround)
    __FPMP_MATH_PLACEHOLDER_1A_RETL__(lrint)
    __FPMP_MATH_PLACEHOLDER_1A_RETL__(lround)
    /* ldexp, scalbn: dedicated fp32mp2 implementations live in the
     * dedicated math section above (bit-cast 3-piece base-2 scaling,
     * no fp64 round-trip).  scalbn forwards to ldexp since FLT_RADIX
     * is 2 on every IEEE 754 platform we support.  fp64mp2 paths stay
     * on the explicit double specializations further below for
     * symmetry. */
    __FPMP_MATH_PLACEHOLDER_FP_LINT__(scalbln)

    // Rounding functions rint, nearbyint
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_rint (const FpType x_hi,
                                                 const FpType x_lo,
                                                 FpType*      res_hi,
                                                 FpType*      res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        const double r = ::rint(static_cast<double>(mp2_t(x_hi, x_lo)));
        mp2_t result(r);
        *res_hi = result.hi();
        *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_nearbyint (const FpType x_hi,
                                                      const FpType x_lo,
                                                      FpType*      res_hi,
                                                      FpType*      res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        const double r = ::nearbyint(static_cast<double>(mp2_t(x_hi, x_lo)));
        mp2_t result(r);
        *res_hi = result.hi();
        *res_lo = result.lo();
    }

    // Bessel functions (CUDA device; assert+return 0 on host)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_j0(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::j0(static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)x_hi; (void)x_lo;
        assert(0 && "j0: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_j1(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::j1(static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)x_hi; (void)x_lo;
        assert(0 && "j1: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_y0(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::y0(static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)x_hi; (void)x_lo;
        assert(0 && "y0: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_y1(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::y1(static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)x_hi; (void)x_lo;
        assert(0 && "y1: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    // Modified Bessel functions of the first kind (CUDA device; assert+return 0 on host)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_cyl_bessel_i0(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::cyl_bessel_i0(static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)x_hi; (void)x_lo;
        assert(0 && "cyl_bessel_i0: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_cyl_bessel_i1(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::cyl_bessel_i1(static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)x_hi; (void)x_lo;
        assert(0 && "cyl_bessel_i1: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    // Bessel functions with (int, fpmp2) -> fpmp2 signature (CUDA device; assert+return 0 on host)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_jn(const int n, const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::jn(n, static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)n; (void)x_hi; (void)x_lo;
        assert(0 && "jn: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_yn(const int n, const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::yn(n, static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)n; (void)x_hi; (void)x_lo;
        assert(0 && "yn: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    // frexp: extract mantissa and exponent
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_frexp(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo, int* nptr)
    {
        using mp2_t = fpmp2_t<FpType>;
        double r = ::frexp(static_cast<double>(mp2_t(x_hi, x_lo)), nptr);
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    // modf: break into integer and fractional parts
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_modf(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo, FpType* iptr_hi, FpType* iptr_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double ipart;
        double r = ::modf(static_cast<double>(mp2_t(x_hi, x_lo)), &ipart);
        mp2_t result(r), iresult(ipart);
        *res_hi = result.hi(); *res_lo = result.lo();
        *iptr_hi = iresult.hi(); *iptr_lo = iresult.lo();
    }

    // remquo: compute remainder and part of quotient
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_remquo(const FpType x_hi, const FpType x_lo, const FpType y_hi, const FpType y_lo, FpType* res_hi, FpType* res_lo, int* quo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double r = ::remquo(static_cast<double>(mp2_t(x_hi, x_lo)), static_cast<double>(mp2_t(y_hi, y_lo)), quo);
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    // Classification and sign functions
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ int __nv_fpmp2_isfinite(const FpType x_hi, const FpType x_lo) { (void)x_lo; return (std::isfinite)(static_cast<double>(x_hi)); }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ int __nv_fpmp2_isinf(const FpType x_hi, const FpType x_lo) { (void)x_lo; return (std::isinf)(static_cast<double>(x_hi)); }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ int __nv_fpmp2_isnan(const FpType x_hi, const FpType x_lo) { (void)x_lo; return (std::isnan)(static_cast<double>(x_hi)); }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ int __nv_fpmp2_signbit(const FpType x_hi, const FpType x_lo) { (void)x_lo; return (std::signbit)(static_cast<double>(x_hi)); }

    /*
    * CUDA-specific functions with host fallbacks
    *
    * Note: __nv_fpmp2_exp10 used to live here as a `::exp10` (CUDA) /
    * `::pow(10, x)` (host) fallback.  The dedicated fp32mp2 version now
    * sits in the exponential/logarithmic family at the top of this
    * file; fp64mp2 routes through __FPMP_CALL_FP64MP2_MATH__ with the
    * new __FPMP_EXP10Q backend macro.
    */
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_sinpi(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double xd = static_cast<double>(mp2_t(x_hi, x_lo));
    #if defined(__CUDA_ARCH__)
        double r = ::sinpi(xd);
    #else
        double r = ::sin(xd * 3.14159265358979323846);
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_cospi(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double xd = static_cast<double>(mp2_t(x_hi, x_lo));
    #if defined(__CUDA_ARCH__)
        double r = ::cospi(xd);
    #else
        double r = ::cos(xd * 3.14159265358979323846);
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_sincospi(const FpType x_hi, const FpType x_lo, FpType* sin_hi, FpType* sin_lo, FpType* cos_hi, FpType* cos_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double xd = static_cast<double>(mp2_t(x_hi, x_lo));
        double sd, cd;
    #if defined(__CUDA_ARCH__)
        ::sincospi(xd, &sd, &cd);
    #else
        double xpi = xd * 3.14159265358979323846;
        sd = ::sin(xpi); cd = ::cos(xpi);
    #endif
        mp2_t s(sd), c(cd);
        *sin_hi = s.hi(); *sin_lo = s.lo();
        *cos_hi = c.hi(); *cos_lo = c.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_normcdf(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double xd = static_cast<double>(mp2_t(x_hi, x_lo));
    #if defined(__CUDA_ARCH__)
        double r = ::normcdf(xd);
    #else
        double r = 0.5 * ::erfc(-xd * 0.70710678118654752440);
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    // (rcbrt: dedicated fp32mp2 implementation defined above; see __nv_fpmp2_rcbrt.)

    // Inverse error functions and scaled complementary error function (CUDA device; assert+return 0 on host)
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_erfcinv(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::erfcinv(static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)x_hi; (void)x_lo;
        assert(0 && "erfcinv: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_erfinv(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::erfinv(static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)x_hi; (void)x_lo;
        assert(0 && "erfinv: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_erfcx(const FpType x_hi, const FpType x_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::erfcx(static_cast<double>(mp2_t(x_hi, x_lo)));
    #else
        (void)x_hi; (void)x_lo;
        assert(0 && "erfcx: no host fallback, returning 0");
        double r = 0.0;
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    // Vector norm functions
    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_norm3d(const FpType a_hi, const FpType a_lo, const FpType b_hi, const FpType b_lo, const FpType c_hi, const FpType c_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double ad = static_cast<double>(mp2_t(a_hi, a_lo));
        double bd = static_cast<double>(mp2_t(b_hi, b_lo));
        double cd = static_cast<double>(mp2_t(c_hi, c_lo));
    #if defined(__CUDA_ARCH__)
        double r = ::norm3d(ad, bd, cd);
    #else
        double r = ::sqrt(ad*ad + bd*bd + cd*cd);
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_norm4d(const FpType a_hi, const FpType a_lo, const FpType b_hi, const FpType b_lo, const FpType c_hi, const FpType c_lo, const FpType d_hi, const FpType d_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double ad = static_cast<double>(mp2_t(a_hi, a_lo));
        double bd = static_cast<double>(mp2_t(b_hi, b_lo));
        double cd = static_cast<double>(mp2_t(c_hi, c_lo));
        double dd = static_cast<double>(mp2_t(d_hi, d_lo));
    #if defined(__CUDA_ARCH__)
        double r = ::norm4d(ad, bd, cd, dd);
    #else
        double r = ::sqrt(ad*ad + bd*bd + cd*cd + dd*dd);
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_rnorm3d(const FpType a_hi, const FpType a_lo, const FpType b_hi, const FpType b_lo, const FpType c_hi, const FpType c_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double ad = static_cast<double>(mp2_t(a_hi, a_lo));
        double bd = static_cast<double>(mp2_t(b_hi, b_lo));
        double cd = static_cast<double>(mp2_t(c_hi, c_lo));
    #if defined(__CUDA_ARCH__)
        double r = ::rnorm3d(ad, bd, cd);
    #else
        double r = 1.0 / ::sqrt(ad*ad + bd*bd + cd*cd);
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_rnorm4d(const FpType a_hi, const FpType a_lo, const FpType b_hi, const FpType b_lo, const FpType c_hi, const FpType c_lo, const FpType d_hi, const FpType d_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
        double ad = static_cast<double>(mp2_t(a_hi, a_lo));
        double bd = static_cast<double>(mp2_t(b_hi, b_lo));
        double cd = static_cast<double>(mp2_t(c_hi, c_lo));
        double dd = static_cast<double>(mp2_t(d_hi, d_lo));
    #if defined(__CUDA_ARCH__)
        double r = ::rnorm4d(ad, bd, cd, dd);
    #else
        double r = 1.0 / ::sqrt(ad*ad + bd*bd + cd*cd + dd*dd);
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    template<typename FpType = float>
    __FPMP_INTERNAL_DECL__ void __nv_fpmp2_rhypot(const FpType x_hi, const FpType x_lo, const FpType y_hi, const FpType y_lo, FpType* res_hi, FpType* res_lo)
    {
        using mp2_t = fpmp2_t<FpType>;
    #if defined(__CUDA_ARCH__)
        double r = ::rhypot(static_cast<double>(mp2_t(x_hi, x_lo)), static_cast<double>(mp2_t(y_hi, y_lo)));
    #else
        double r = 1.0 / ::hypot(static_cast<double>(mp2_t(x_hi, x_lo)), static_cast<double>(mp2_t(y_hi, y_lo)));
    #endif
        mp2_t result(r);
        *res_hi = result.hi(); *res_lo = result.lo();
    }

    /* Cleanup: undefine the placeholder factory macros so they don't
     * leak into headers/translation units that include this file. */
    #undef __FPMP_MATH_PLACEHOLDER_1A__
    #undef __FPMP_MATH_PLACEHOLDER_2A__
    #undef __FPMP_MATH_PLACEHOLDER_1A_RETINT__
    #undef __FPMP_MATH_PLACEHOLDER_1A_RETLL__
    #undef __FPMP_MATH_PLACEHOLDER_1A_RETL__
    #undef __FPMP_MATH_PLACEHOLDER_FP_INT__
    #undef __FPMP_MATH_PLACEHOLDER_FP_LINT__
    #undef __FPMP_MATH_PLACEHOLDER_INT_FP__

/*
* ============================================================================
* Double precision (fp64mp2) template specializations
* ============================================================================
*/
#if (FPMP_FP64MP2_ENABLE == 1)

    #if (FPMP_FP128_MATH_FALLBACK == 1)

        #if defined(__CUDA_ARCH__) \
          && (defined(__aarch64__) \
          || defined(_M_ARM64)) \
          && defined(FPMP_CUDA_FP128_INTRINSICS) \
          && !(defined(__GNUC__) \
          && !defined(__clang__) \
          && !defined(__NVCOMPILER_MAJOR__) \
          && ((__GNUC__ > 13) \
          || (__GNUC__ == 13 && __GNUC_MINOR__ >= 1))) \
          && !defined(__FLOAT128_CPP_SPELLING_ENABLED__)
            #define __FLOAT128_CPP_SPELLING_ENABLED__
        #endif
        
} // namespace cuda::experimental
        #if defined(__CUDA_ARCH__)
            // CUDA device
            #include "crt/device_fp128_functions.h"
        #elif (FPMP_HOST_SUPPORTS_LIBQUADMATH == 1)
            // x86 host: libquadmath
            #include <quadmath.h>
        #elif (FPMP_HOST_SUPPORTS_LDOUBLE128 == 1)
            // ARM64/s390x host: long double is 128-bit IEEE
            #include <cmath>
        #endif
namespace cuda::experimental
{

        // ----------------------------------------------------------------------
        // Branch 1 -- CUDA DEVICE with the *extended* NVVM fp128 intrinsics
        //             (the primary GPU path).
        // Target: device compile built with fp128 spelling that
        // the crt header can declare the overloads under (__float128 on x86_64,
        // or _Float128 on AArch64 with GCC >= 13.1). Serves both x86_64 and
        // AArch64 devices: true binary128 via native __nv_fp128_* for every
        // function that has one; the five without a native intrinsic (cbrt,
        // atan2, erf, erfc, nearbyint) widen through double.
        // ----------------------------------------------------------------------
        #if defined(__CUDA_ARCH__) && defined(FPMP_CUDA_FP128_INTRINSICS) \
          && (defined(__FLOAT128_CPP_SPELLING_ENABLED__) ||  defined(__FLOAT128_C_SPELLING_ENABLED__))
            #define __FPMP_EXPQ(x)         __nv_fp128_exp(x)
            #define __FPMP_EXP2Q(x)        __nv_fp128_exp2(x)
            #define __FPMP_EXP10Q(x)       __nv_fp128_exp10(x)
            #define __FPMP_EXPM1Q(x)       __nv_fp128_expm1(x)
            #define __FPMP_LOGQ(x)         __nv_fp128_log(x)
            #define __FPMP_LOG2Q(x)        __nv_fp128_log2(x)
            #define __FPMP_LOG10Q(x)       __nv_fp128_log10(x)
            #define __FPMP_LOG1PQ(x)       __nv_fp128_log1p(x)
            #define __FPMP_SINQ(x)         __nv_fp128_sin(x)
            #define __FPMP_COSQ(x)         __nv_fp128_cos(x)
            #define __FPMP_TANQ(x)         __nv_fp128_tan(x)
            #define __FPMP_ASINQ(x)        __nv_fp128_asin(x)
            #define __FPMP_ACOSQ(x)        __nv_fp128_acos(x)
            #define __FPMP_ATANQ(x)        __nv_fp128_atan(x)
            #define __FPMP_SINHQ(x)        __nv_fp128_sinh(x)
            #define __FPMP_COSHQ(x)        __nv_fp128_cosh(x)
            #define __FPMP_TANHQ(x)        __nv_fp128_tanh(x)
            #define __FPMP_ASINHQ(x)       __nv_fp128_asinh(x)
            #define __FPMP_ACOSHQ(x)       __nv_fp128_acosh(x)
            #define __FPMP_ATANHQ(x)       __nv_fp128_atanh(x)
            #define __FPMP_SQRTQ(x)        __nv_fp128_sqrt(x)
            #define __FPMP_FABSQ(x)        __nv_fp128_fabs(x)
            #define __FPMP_POWQ(x,y)       __nv_fp128_pow((x),(y))
            #define __FPMP_FMODQ(x,y)      __nv_fp128_fmod((x),(y))
            #define __FPMP_REMAINDERQ(x,y) __nv_fp128_remainder((x),(y))
            #define __FPMP_FLOORQ(x)       __nv_fp128_floor(x)
            #define __FPMP_CEILQ(x)        __nv_fp128_ceil(x)
            #define __FPMP_TRUNCQ(x)       __nv_fp128_trunc(x)
            #define __FPMP_ROUNDQ(x)       __nv_fp128_round(x)
            #define __FPMP_RINTQ(x)        __nv_fp128_rint(x)
            #define __FPMP_NEARBYINTQ(x)   __nv_fp128_rint(x)
            #define __FPMP_CBRTQ(x)        __nv_fp128_copysign(__nv_fp128_pow(__nv_fp128_fabs(x), (__fpmp_fp128)1 / (__fpmp_fp128)3), (x))
            #define __FPMP_ATAN2Q(y,x)     ((__fpmp_fp128)atan2((double)(y), (double)(x)))
            #define __FPMP_ERFQ(x)         ((__fpmp_fp128)erf((double)(x)))
            #define __FPMP_ERFCQ(x)        ((__fpmp_fp128)erfc((double)(x)))
        // ----------------------------------------------------------------------
        // Branch 2 -- HOST with libquadmath (the primary x86_64 host path).
        // Target: host compile (no __CUDA_ARCH__) where libquadmath is present
        // (typically x86_64 GCC distributions). Reference math uses the true
        // binary128 libquadmath entry points (the `*q` suffix); __fpmp_fp128 is
        // __float128 here. The explicit !defined(__CUDA_ARCH__) guard keeps the
        // device pass on an x86_64 host (where FPMP_HOST_SUPPORTS_LIBQUADMATH is
        // also 1) from matching this host-only branch.
        // ----------------------------------------------------------------------
        #elif (FPMP_HOST_SUPPORTS_LIBQUADMATH == 1) && !defined(__CUDA_ARCH__)
            #define __FPMP_EXPQ(x)         expq(x)
            #define __FPMP_EXP2Q(x)        exp2q(x)
            #define __FPMP_EXPM1Q(x)       expm1q(x)
            #define __FPMP_LOGQ(x)         logq(x)
            #define __FPMP_LOG2Q(x)        log2q(x)
            #define __FPMP_LOG10Q(x)       log10q(x)
            #define __FPMP_LOG1PQ(x)       log1pq(x)
            #define __FPMP_SINQ(x)         sinq(x)
            #define __FPMP_COSQ(x)         cosq(x)
            #define __FPMP_TANQ(x)         tanq(x)
            #define __FPMP_ASINQ(x)        asinq(x)
            #define __FPMP_ACOSQ(x)        acosq(x)
            #define __FPMP_ATANQ(x)        atanq(x)
            #define __FPMP_SINHQ(x)        sinhq(x)
            #define __FPMP_COSHQ(x)        coshq(x)
            #define __FPMP_TANHQ(x)        tanhq(x)
            #define __FPMP_ASINHQ(x)       asinhq(x)
            #define __FPMP_ACOSHQ(x)       acoshq(x)
            #define __FPMP_ATANHQ(x)       atanhq(x)
            #define __FPMP_SQRTQ(x)        sqrtq(x)
            #define __FPMP_CBRTQ(x)        cbrtq(x)
            #define __FPMP_FABSQ(x)        fabsq(x)
            #define __FPMP_POWQ(x,y)       powq((x),(y))
            #define __FPMP_ATAN2Q(y,x)     atan2q((y),(x))
            #define __FPMP_FMODQ(x,y)      fmodq((x),(y))
            #define __FPMP_REMAINDERQ(x,y) remainderq((x),(y))
            #define __FPMP_ERFQ(x)         erfq(x)
            #define __FPMP_ERFCQ(x)        erfcq(x)
            #define __FPMP_FLOORQ(x)       floorq(x)
            #define __FPMP_CEILQ(x)        ceilq(x)
            #define __FPMP_TRUNCQ(x)       truncq(x)
            #define __FPMP_ROUNDQ(x)       roundq(x)
            #define __FPMP_RINTQ(x)        rintq(x)
            #define __FPMP_NEARBYINTQ(x)   nearbyintq(x)
            #define __FPMP_EXP10Q(x)       powq((__float128)10.0, (x))
        // ----------------------------------------------------------------------
        // Branch 3 -- HOST, no libquadmath, 128-bit `long double`
        //             (the primary AArch64 / non-x86 host path).
        // Target: host compile (no __CUDA_ARCH__) on platforms whose C
        // `long double` is a true 128-bit type (IEEE binary128 on AArch64 /
        // PPC64LE, or 80-bit x87 extended on x86 without libquadmath) AND where
        // libquadmath is unavailable. Reference math uses the standard C
        // `long double` libm entry points (the `*l` suffix).
        // ----------------------------------------------------------------------
        #elif (FPMP_HOST_SUPPORTS_LDOUBLE128 == 1) && !defined(__CUDA_ARCH__) \
         && (FPMP_HOST_SUPPORTS_LIBQUADMATH == 0)
            #define __FPMP_EXPQ(x)         expl(x)
            #define __FPMP_EXP2Q(x)        exp2l(x)
            #define __FPMP_EXPM1Q(x)       expm1l(x)
            #define __FPMP_LOGQ(x)         logl(x)
            #define __FPMP_LOG2Q(x)        log2l(x)
            #define __FPMP_LOG10Q(x)       log10l(x)
            #define __FPMP_LOG1PQ(x)       log1pl(x)
            #define __FPMP_SINQ(x)         sinl(x)
            #define __FPMP_COSQ(x)         cosl(x)
            #define __FPMP_TANQ(x)         tanl(x)
            #define __FPMP_ASINQ(x)        asinl(x)
            #define __FPMP_ACOSQ(x)        acosl(x)
            #define __FPMP_ATANQ(x)        atanl(x)
            #define __FPMP_SINHQ(x)        sinhl(x)
            #define __FPMP_COSHQ(x)        coshl(x)
            #define __FPMP_TANHQ(x)        tanhl(x)
            #define __FPMP_ASINHQ(x)       asinhl(x)
            #define __FPMP_ACOSHQ(x)       acoshl(x)
            #define __FPMP_ATANHQ(x)       atanhl(x)
            #define __FPMP_SQRTQ(x)        sqrtl(x)
            #define __FPMP_FABSQ(x)        fabsl(x)
            #define __FPMP_POWQ(x,y)       powl((x),(y))
            #define __FPMP_CBRTQ(x)        cbrtl(x)
            #define __FPMP_ATAN2Q(y,x)     atan2l((y),(x))
            #define __FPMP_FMODQ(x,y)      fmodl((x),(y))
            #define __FPMP_REMAINDERQ(x,y) remainderl((x),(y))
            #define __FPMP_ERFQ(x)         erfl(x)
            #define __FPMP_ERFCQ(x)        erfcl(x)
            #define __FPMP_FLOORQ(x)       floorl(x)
            #define __FPMP_CEILQ(x)        ceill(x)
            #define __FPMP_TRUNCQ(x)       truncl(x)
            #define __FPMP_ROUNDQ(x)       roundl(x)
            #define __FPMP_RINTQ(x)        rintl(x)
            #define __FPMP_NEARBYINTQ(x)   nearbyintl(x)
            #define __FPMP_EXP10Q(x)       ((long double)powl(10.0L, (x)))
        // ----------------------------------------------------------------------
        // Branch 4 -- CUDA DEVICE WITHOUT a usable native fp128 path
        //             (the fp64 fallback).
        // Target: any device build (x86_64 or AArch64) that did NOT enable the
        // extended NVVM intrinsics (no FPMP_CUDA_FP128_INTRINSICS), or where the
        // host toolchain cannot declare a usable __float128/_Float128 spelling
        // for the crt overloads (e.g. AArch64 with GCC < 13.1).
        //
        // There is no "base subset" of __nv_fp128_* that links without the
        // extended switch: on CUDA >= 12.8 
        // makes 128-bit FP unsupported in device code and rejects the ENTIRE
        // __nv_fp128_* family at declaration (verified on 12.8 and 13.0). So when
        // we land here, NO native fp128 is available and EVERY function degrades
        // to a double-precision computation widened back to __fpmp_fp128 (i.e. the
        // reference is effectively fp64-accurate on this target).
        // ----------------------------------------------------------------------
        #elif defined(__CUDA_ARCH__)
            #define __FPMP_EXPQ(x)         ((__fpmp_fp128)exp((double)(x)))
            #define __FPMP_EXP2Q(x)        ((__fpmp_fp128)exp2((double)(x)))
            #define __FPMP_EXP10Q(x)       ((__fpmp_fp128)exp10((double)(x)))
            #define __FPMP_EXPM1Q(x)       ((__fpmp_fp128)expm1((double)(x)))
            #define __FPMP_LOGQ(x)         ((__fpmp_fp128)log((double)(x)))
            #define __FPMP_LOG2Q(x)        ((__fpmp_fp128)log2((double)(x)))
            #define __FPMP_LOG10Q(x)       ((__fpmp_fp128)log10((double)(x)))
            #define __FPMP_LOG1PQ(x)       ((__fpmp_fp128)log1p((double)(x)))
            #define __FPMP_SINQ(x)         ((__fpmp_fp128)sin((double)(x)))
            #define __FPMP_COSQ(x)         ((__fpmp_fp128)cos((double)(x)))
            #define __FPMP_TANQ(x)         ((__fpmp_fp128)tan((double)(x)))
            #define __FPMP_ASINQ(x)        ((__fpmp_fp128)asin((double)(x)))
            #define __FPMP_ACOSQ(x)        ((__fpmp_fp128)acos((double)(x)))
            #define __FPMP_ATANQ(x)        ((__fpmp_fp128)atan((double)(x)))
            #define __FPMP_SINHQ(x)        ((__fpmp_fp128)sinh((double)(x)))
            #define __FPMP_COSHQ(x)        ((__fpmp_fp128)cosh((double)(x)))
            #define __FPMP_TANHQ(x)        ((__fpmp_fp128)tanh((double)(x)))
            #define __FPMP_ASINHQ(x)       ((__fpmp_fp128)asinh((double)(x)))
            #define __FPMP_ACOSHQ(x)       ((__fpmp_fp128)acosh((double)(x)))
            #define __FPMP_ATANHQ(x)       ((__fpmp_fp128)atanh((double)(x)))
            #define __FPMP_SQRTQ(x)        ((__fpmp_fp128)sqrt((double)(x)))
            #define __FPMP_FABSQ(x)        ((__fpmp_fp128)fabs((double)(x)))
            #define __FPMP_POWQ(x,y)       ((__fpmp_fp128)pow((double)(x), (double)(y)))
            #define __FPMP_CBRTQ(x)        ((__fpmp_fp128)cbrt((double)(x)))
            #define __FPMP_ATAN2Q(y,x)     ((__fpmp_fp128)atan2((double)(y), (double)(x)))
            #define __FPMP_FMODQ(x,y)      ((__fpmp_fp128)fmod((double)(x), (double)(y)))
            #define __FPMP_REMAINDERQ(x,y) ((__fpmp_fp128)remainder((double)(x), (double)(y)))
            #define __FPMP_ERFQ(x)         ((__fpmp_fp128)erf((double)(x)))
            #define __FPMP_ERFCQ(x)        ((__fpmp_fp128)erfc((double)(x)))
            #define __FPMP_FLOORQ(x)       ((__fpmp_fp128)floor((double)(x)))
            #define __FPMP_CEILQ(x)        ((__fpmp_fp128)ceil((double)(x)))
            #define __FPMP_TRUNCQ(x)       ((__fpmp_fp128)trunc((double)(x)))
            #define __FPMP_ROUNDQ(x)       ((__fpmp_fp128)round((double)(x)))
            #define __FPMP_RINTQ(x)        ((__fpmp_fp128)rint((double)(x)))
            #define __FPMP_NEARBYINTQ(x)   ((__fpmp_fp128)nearbyint((double)(x)))
        #endif // FPMP_FP128_MATH_FALLBACK == 1

        /*
         * Simplified dispatch macro: uses __FPMP_*Q wrapper macros which already
         * handle CUDA/libquadmath/long double dispatching internally.
         */
        #define __FPMP_CALL_FP64MP2_MATH__(dfunc,qfunc,xhi,xlo,reshi,reslo) __nv_fpmp2_from_quad(qfunc(__nv_fpmp2_to_quad(xhi,xlo)),reshi,reslo)
        #define __FPMP_CALL_FP64MP2_MATH_2A__(dfunc,qfunc,xhi,xlo,yhi,ylo,reshi,reslo) __nv_fpmp2_from_quad(qfunc(__nv_fpmp2_to_quad(xhi,xlo),__nv_fpmp2_to_quad(yhi,ylo)),reshi,reslo)
    #else
        #define __FPMP_CALL_FP64MP2_MATH__(dfunc,qfunc,xhi,xlo,reshi,reslo) __nv_fpmp2_from_double(::dfunc(__nv_fpmp2_to_double(xhi,xlo)),reshi,reslo)
        #define __FPMP_CALL_FP64MP2_MATH_2A__(dfunc,qfunc,xhi,xlo,yhi,ylo,reshi,reslo) __nv_fpmp2_from_double(::dfunc(__nv_fpmp2_to_double(xhi,xlo),__nv_fpmp2_to_double(yhi,ylo)),reshi,reslo)
    #endif // FPMP_FP128_MATH_FALLBACK == 1

    template<> __FPMP_API_DECL__ void __nv_fpmp2_exp<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(exp, __FPMP_EXPQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(log, __FPMP_LOGQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log2<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(log2, __FPMP_LOG2Q, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log10<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(log10, __FPMP_LOG10Q, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log1p<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(log1p, __FPMP_LOG1PQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sin<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(sin, __FPMP_SINQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cos<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(cos, __FPMP_COSQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_asin<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(asin, __FPMP_ASINQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_acos<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(acos, __FPMP_ACOSQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_atan<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(atan, __FPMP_ATANQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sinh<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(sinh, __FPMP_SINHQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cosh<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(cosh, __FPMP_COSHQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_tanh<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(tanh, __FPMP_TANHQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_pow<double>    (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH_2A__(pow, __FPMP_POWQ, x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sincos<double> (const double x_hi, const double x_lo, double* sin_hi, double* sin_lo, double* cos_hi, double* cos_lo) { __FPMP_CALL_FP64MP2_MATH__(sin, __FPMP_SINQ, x_hi, x_lo, sin_hi, sin_lo); __FPMP_CALL_FP64MP2_MATH__(cos, __FPMP_COSQ, x_hi, x_lo, cos_hi, cos_lo); }
    
    // Functions with no 128-bit support in CUDA
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erf<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::erf(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);}
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfc<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::erfc(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_boys_f0<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
        double x = __nv_fpmp2_to_double(x_hi, x_lo);
        double r;
        if (x < 1e-15) { r = 1.0; }
        else { r = 0.5 * ::sqrt(3.14159265358979323846 / x) * ::erf(::sqrt(x)); }
        __nv_fpmp2_from_double(r, res_hi, res_lo);
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_normcdfinv<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo)
    {
        double p = __nv_fpmp2_to_double(x_hi, x_lo);
    #if defined(__CUDA_ARCH__)
        // Hardcoded value since M_SQRT2 is not guaranteed to be defined on all platforms
        constexpr double sqrt2_v = 1.41421356237309504880;
        __nv_fpmp2_from_double(-sqrt2_v * ::erfcinv(2.0 * p), res_hi, res_lo);
    #else
        // Not implemented yet: double precision normcdfinv fallback to float precision
        float f_hi, f_lo;
        __nv_fpmp2_normcdfinv(static_cast<float>(p), 0.0f, &f_hi, &f_lo);
        *res_hi = static_cast<double>(f_hi) + static_cast<double>(f_lo);
        *res_lo = 0.0;
    #endif
    }

    template<> __FPMP_API_DECL__ void __nv_fpmp2_cbrt<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) 
    {
        #if (FPMP_FP128_MATH_FALLBACK == 1)
            __fpmp_fp128 res = __FPMP_CBRTQ(__nv_fpmp2_to_quad(x_hi, x_lo));
            __nv_fpmp2_from_quad(res, res_hi, res_lo);
        #else
            double res = ::cbrt(__nv_fpmp2_to_double(x_hi, x_lo));
            __nv_fpmp2_from_double(res, res_hi, res_lo);
        #endif
    }
// Note: On CUDA device, __FPMP_ATAN2Q widens through double atan2 (no fp128 intrinsic); __FPMP_CBRTQ is reconstructed from __nv_fp128_pow.
    template<> __FPMP_API_DECL__ void __nv_fpmp2_atan2<double>  (const double y_hi, const double y_lo, const double x_hi, const double x_lo, double* res_hi, double* res_lo) 
    {
        #if (FPMP_FP128_MATH_FALLBACK == 1)
            __fpmp_fp128 res = __FPMP_ATAN2Q(__nv_fpmp2_to_quad(y_hi, y_lo), __nv_fpmp2_to_quad(x_hi, x_lo));
            __nv_fpmp2_from_quad(res, res_hi, res_lo);
        #else
            double res = ::atan2(__nv_fpmp2_to_double(y_hi, y_lo), __nv_fpmp2_to_double(x_hi, x_lo));
            __nv_fpmp2_from_double(res, res_hi, res_lo);
        #endif
    } // __nv_fpmp2_atan2<double>

    // Additional fp64mp2 specializations (double-precision fallback for all)
    template<> __FPMP_API_DECL__ void __nv_fpmp2_acosh<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(acosh, __FPMP_ACOSHQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_asinh<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(asinh, __FPMP_ASINHQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_atanh<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(atanh, __FPMP_ATANHQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_tan<double>     (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(tan, __FPMP_TANQ, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_exp2<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(exp2,  __FPMP_EXP2Q,  x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_expm1<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(expm1, __FPMP_EXPM1Q, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_logb<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::logb(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo); }
    // Rounding family: fp64mp2 routes through higher precision (fp128) when
    // available; otherwise falls back to fp64 system rounding. This avoids
    // precision loss from collapsing the (hi, lo) pair into a single double.
    template<> __FPMP_API_DECL__ void __nv_fpmp2_ceil<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(ceil,      __FPMP_CEILQ,      x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_floor<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(floor,     __FPMP_FLOORQ,     x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_trunc<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(trunc,     __FPMP_TRUNCQ,     x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_round<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(round,     __FPMP_ROUNDQ,     x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rint<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(rint,      __FPMP_RINTQ,      x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_nearbyint<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH__(nearbyint, __FPMP_NEARBYINTQ, x_hi, x_lo, res_hi, res_lo); }
    // __nv_fpmp2_fabs<double>: handled by the type-agnostic primary template above (no double round-trip).
    template<> __FPMP_API_DECL__ void __nv_fpmp2_lgamma<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::lgamma(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_tgamma<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::tgamma(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_j0<double>      (const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::j0(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)x_hi; (void)x_lo; assert(0 && "j0: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_j1<double>      (const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::j1(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)x_hi; (void)x_lo; assert(0 && "j1: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_y0<double>      (const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::y0(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)x_hi; (void)x_lo; assert(0 && "y0: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_y1<double>      (const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::y1(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)x_hi; (void)x_lo; assert(0 && "y1: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cyl_bessel_i0<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::cyl_bessel_i0(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)x_hi; (void)x_lo; assert(0 && "cyl_bessel_i0: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cyl_bessel_i1<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::cyl_bessel_i1(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)x_hi; (void)x_lo; assert(0 && "cyl_bessel_i1: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    // __nv_fpmp2_fmax<double>, __nv_fpmp2_fmin<double>, __nv_fpmp2_max<double>, __nv_fpmp2_min<double>:
    // handled by the type-agnostic primary templates above (no double round-trip).
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fmod<double>    (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH_2A__(fmod,      __FPMP_FMODQ,      x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_remainder<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __FPMP_CALL_FP64MP2_MATH_2A__(remainder, __FPMP_REMAINDERQ, x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_hypot<double>   (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::hypot(__nv_fpmp2_to_double(x_hi, x_lo), __nv_fpmp2_to_double(y_hi, y_lo)), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_copysign<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::copysign(__nv_fpmp2_to_double(x_hi, x_lo), __nv_fpmp2_to_double(y_hi, y_lo)), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fdim<double>    (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::fdim(__nv_fpmp2_to_double(x_hi, x_lo), __nv_fpmp2_to_double(y_hi, y_lo)), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_nextafter<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::nextafter(__nv_fpmp2_to_double(x_hi, x_lo), __nv_fpmp2_to_double(y_hi, y_lo)), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rhypot<double>  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo)
    {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::rhypot(__nv_fpmp2_to_double(x_hi, x_lo), __nv_fpmp2_to_double(y_hi, y_lo)), res_hi, res_lo);
    #else
        __nv_fpmp2_from_double(1.0 / ::hypot(__nv_fpmp2_to_double(x_hi, x_lo), __nv_fpmp2_to_double(y_hi, y_lo)), res_hi, res_lo);
    #endif
    }
    template<> __FPMP_API_DECL__ int __nv_fpmp2_ilogb<double>    (const double x_hi, const double x_lo) { return ::ilogb(__nv_fpmp2_to_double(x_hi, x_lo)); }
    template<> __FPMP_API_DECL__ long long int __nv_fpmp2_llrint<double> (const double x_hi, const double x_lo) { return ::llrint(__nv_fpmp2_to_double(x_hi, x_lo)); }
    template<> __FPMP_API_DECL__ long long int __nv_fpmp2_llround<double>(const double x_hi, const double x_lo) { return ::llround(__nv_fpmp2_to_double(x_hi, x_lo)); }
    template<> __FPMP_API_DECL__ long int __nv_fpmp2_lrint<double>  (const double x_hi, const double x_lo) { return ::lrint(__nv_fpmp2_to_double(x_hi, x_lo)); }
    template<> __FPMP_API_DECL__ long int __nv_fpmp2_lround<double> (const double x_hi, const double x_lo) { return ::lround(__nv_fpmp2_to_double(x_hi, x_lo)); }
    template<> __FPMP_API_DECL__ int __nv_fpmp2_isfinite<double> (const double x_hi, const double x_lo) { (void)x_lo; return (std::isfinite)(x_hi); }
    template<> __FPMP_API_DECL__ int __nv_fpmp2_isinf<double>    (const double x_hi, const double x_lo) { (void)x_lo; return (std::isinf)(x_hi); }
    template<> __FPMP_API_DECL__ int __nv_fpmp2_isnan<double>    (const double x_hi, const double x_lo) { (void)x_lo; return (std::isnan)(x_hi); }
    template<> __FPMP_API_DECL__ int __nv_fpmp2_signbit<double>  (const double x_hi, const double x_lo) { (void)x_lo; return (std::signbit)(x_hi); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_ldexp<double>   (const double x_hi, const double x_lo, int n, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::ldexp(__nv_fpmp2_to_double(x_hi, x_lo), n), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_scalbn<double>  (const double x_hi, const double x_lo, int n, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::scalbn(__nv_fpmp2_to_double(x_hi, x_lo), n), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_scalbln<double> (const double x_hi, const double x_lo, long int n, double* res_hi, double* res_lo) { __nv_fpmp2_from_double(::scalbln(__nv_fpmp2_to_double(x_hi, x_lo), n), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_jn<double>      (int n, const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::jn(n, __nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)n; (void)x_hi; (void)x_lo; assert(0 && "jn: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_yn<double>      (int n, const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::yn(n, __nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)n; (void)x_hi; (void)x_lo; assert(0 && "yn: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_frexp<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo, int* nptr) { __nv_fpmp2_from_double(::frexp(__nv_fpmp2_to_double(x_hi, x_lo), nptr), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_modf<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo, double* iptr_hi, double* iptr_lo) { double ip; __nv_fpmp2_from_double(::modf(__nv_fpmp2_to_double(x_hi, x_lo), &ip), res_hi, res_lo); __nv_fpmp2_from_double(ip, iptr_hi, iptr_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_remquo<double>  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo, int* quo) { __nv_fpmp2_from_double(::remquo(__nv_fpmp2_to_double(x_hi, x_lo), __nv_fpmp2_to_double(y_hi, y_lo), quo), res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_exp10<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo)
    {
    #if (FPMP_FP128_MATH_FALLBACK == 1)
        /* fp128 path: __FPMP_EXP10Q handles every backend (libquadmath
         * powq, CUDA double widen, host long-double powl). */
        __FPMP_CALL_FP64MP2_MATH__(exp10, __FPMP_EXP10Q, x_hi, x_lo, res_hi, res_lo);
    #else
        /* fp64 fallback: libm has no portable `exp10`; synthesize via
         * pow(10, x).  CUDA device has the intrinsic, prefer it. */
        double xd = __nv_fpmp2_to_double(x_hi, x_lo);
        #if defined(__CUDA_ARCH__)
            __nv_fpmp2_from_double(::exp10(xd), res_hi, res_lo);
        #else
            __nv_fpmp2_from_double(::pow(10.0, xd), res_hi, res_lo);
        #endif
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sinpi<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo)
    {
        double xd = __nv_fpmp2_to_double(x_hi, x_lo);
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::sinpi(xd), res_hi, res_lo);
    #else
        __nv_fpmp2_from_double(::sin(xd * 3.14159265358979323846), res_hi, res_lo);
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cospi<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo)
    {
        double xd = __nv_fpmp2_to_double(x_hi, x_lo);
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::cospi(xd), res_hi, res_lo);
    #else
        __nv_fpmp2_from_double(::cos(xd * 3.14159265358979323846), res_hi, res_lo);
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sincospi<double>(const double x_hi, const double x_lo, double* sin_hi, double* sin_lo, double* cos_hi, double* cos_lo)
    {
        double xd = __nv_fpmp2_to_double(x_hi, x_lo);
    #if defined(__CUDA_ARCH__)
        double sd, cd; ::sincospi(xd, &sd, &cd);
        __nv_fpmp2_from_double(sd, sin_hi, sin_lo); __nv_fpmp2_from_double(cd, cos_hi, cos_lo);
    #else
        double xpi = xd * 3.14159265358979323846;
        __nv_fpmp2_from_double(::sin(xpi), sin_hi, sin_lo); __nv_fpmp2_from_double(::cos(xpi), cos_hi, cos_lo);
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_normcdf<double> (const double x_hi, const double x_lo, double* res_hi, double* res_lo)
    {
        double xd = __nv_fpmp2_to_double(x_hi, x_lo);
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::normcdf(xd), res_hi, res_lo);
    #else
        __nv_fpmp2_from_double(0.5 * ::erfc(-xd * 0.70710678118654752440), res_hi, res_lo);
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rcbrt<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo)
    {
        double xd = __nv_fpmp2_to_double(x_hi, x_lo);
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::rcbrt(xd), res_hi, res_lo);
    #else
        __nv_fpmp2_from_double(1.0 / ::cbrt(xd), res_hi, res_lo);
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfcinv<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::erfcinv(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)x_hi; (void)x_lo; assert(0 && "erfcinv: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfinv<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::erfinv(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)x_hi; (void)x_lo; assert(0 && "erfinv: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfcx<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) {
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::erfcx(__nv_fpmp2_to_double(x_hi, x_lo)), res_hi, res_lo);
    #else
        (void)x_hi; (void)x_lo; assert(0 && "erfcx: no host fallback, returning 0"); *res_hi = 0.0; *res_lo = 0.0;
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_norm3d<double>(const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, double* res_hi, double* res_lo) {
        double ad = __nv_fpmp2_to_double(a_hi, a_lo), bd = __nv_fpmp2_to_double(b_hi, b_lo), cd = __nv_fpmp2_to_double(c_hi, c_lo);
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::norm3d(ad, bd, cd), res_hi, res_lo);
    #else
        __nv_fpmp2_from_double(::sqrt(ad*ad + bd*bd + cd*cd), res_hi, res_lo);
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_norm4d<double>(const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, const double d_hi, const double d_lo, double* res_hi, double* res_lo) {
        double ad = __nv_fpmp2_to_double(a_hi, a_lo), bd = __nv_fpmp2_to_double(b_hi, b_lo), cd = __nv_fpmp2_to_double(c_hi, c_lo), dd = __nv_fpmp2_to_double(d_hi, d_lo);
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::norm4d(ad, bd, cd, dd), res_hi, res_lo);
    #else
        __nv_fpmp2_from_double(::sqrt(ad*ad + bd*bd + cd*cd + dd*dd), res_hi, res_lo);
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rnorm3d<double>(const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, double* res_hi, double* res_lo) {
        double ad = __nv_fpmp2_to_double(a_hi, a_lo), bd = __nv_fpmp2_to_double(b_hi, b_lo), cd = __nv_fpmp2_to_double(c_hi, c_lo);
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::rnorm3d(ad, bd, cd), res_hi, res_lo);
    #else
        __nv_fpmp2_from_double(1.0 / ::sqrt(ad*ad + bd*bd + cd*cd), res_hi, res_lo);
    #endif
    }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rnorm4d<double>(const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, const double d_hi, const double d_lo, double* res_hi, double* res_lo) {
        double ad = __nv_fpmp2_to_double(a_hi, a_lo), bd = __nv_fpmp2_to_double(b_hi, b_lo), cd = __nv_fpmp2_to_double(c_hi, c_lo), dd = __nv_fpmp2_to_double(d_hi, d_lo);
    #if defined(__CUDA_ARCH__)
        __nv_fpmp2_from_double(::rnorm4d(ad, bd, cd, dd), res_hi, res_lo);
    #else
        __nv_fpmp2_from_double(1.0 / ::sqrt(ad*ad + bd*bd + cd*cd + dd*dd), res_hi, res_lo);
    #endif
    }

#endif // FPMP_FP64MP2_ENABLE == 1

#else // __FPMP_USE_LIB__
    /*
    * ============================================================================
    * Library mode - fp32mp2 declarations
    * ============================================================================
    */
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_exp    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_log    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_log2   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_log10  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_log1p  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_pow    (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cbrt   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sin    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cos    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sincos (const float x_hi, const float x_lo, float* sin_hi, float* sin_lo, float* cos_hi, float* cos_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_asin   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_acos   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_atan   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_atan2  (const float y_hi, const float y_lo, const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sinh   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cosh   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_tanh   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erf    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erfc   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_normcdfinv (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_icdf32     (uint32_t x, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_icdf64     (uint64_t x, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_acosh  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_asinh  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_atanh  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_tan    (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_exp2   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_exp10  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_expm1  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_logb   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_ceil   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_floor  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_trunc  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_round  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rint   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_nearbyint(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fabs   (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_lgamma (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_tgamma (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_j0     (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_j1     (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_y0     (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_y1     (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cyl_bessel_i0(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cyl_bessel_i1(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sinpi  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_cospi  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_normcdf(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rcbrt  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erfcinv(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erfinv (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_erfcx  (const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_boys_f0(const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_norm3d (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_norm4d (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, const float d_hi, const float d_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rnorm3d(const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rnorm4d(const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, const float d_hi, const float d_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fmax   (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fmin   (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_max    (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_min    (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fmod   (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_remainder(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_hypot  (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_copysign(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_fdim   (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_nextafter(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_rhypot (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_remquo (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo, int* quo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_ilogb  (const float x_hi, const float x_lo);
    __FPMP_BUILTIN_DECL__ long long int __nv_fp32mp2_llrint (const float x_hi, const float x_lo);
    __FPMP_BUILTIN_DECL__ long long int __nv_fp32mp2_llround(const float x_hi, const float x_lo);
    __FPMP_BUILTIN_DECL__ long int __nv_fp32mp2_lrint  (const float x_hi, const float x_lo);
    __FPMP_BUILTIN_DECL__ long int __nv_fp32mp2_lround (const float x_hi, const float x_lo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_isfinite(const float x_hi, const float x_lo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_isinf   (const float x_hi, const float x_lo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_isnan   (const float x_hi, const float x_lo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp32mp2_signbit (const float x_hi, const float x_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_ldexp  (const float x_hi, const float x_lo, int n, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_scalbn (const float x_hi, const float x_lo, int n, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_scalbln(const float x_hi, const float x_lo, long int n, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_jn     (int n, const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_yn     (int n, const float x_hi, const float x_lo, float* res_hi, float* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_frexp  (const float x_hi, const float x_lo, float* res_hi, float* res_lo, int* nptr);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_modf   (const float x_hi, const float x_lo, float* res_hi, float* res_lo, float* iptr_hi, float* iptr_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp32mp2_sincospi(const float x_hi, const float x_lo, float* sin_hi, float* sin_lo, float* cos_hi, float* cos_lo);

    /*
    * ============================================================================
    * Library mode - fp64mp2 declarations
    * ============================================================================
    */
#if (FPMP_FP64MP2_ENABLE == 1)
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_exp    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_log    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_log2   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_log10  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_log1p  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_pow    (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cbrt   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sin    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cos    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sincos (const double x_hi, const double x_lo, double* sin_hi, double* sin_lo, double* cos_hi, double* cos_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_asin   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_acos   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_atan   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_atan2  (const double y_hi, const double y_lo, const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sinh   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cosh   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_tanh   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erf    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erfc   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_normcdfinv (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_acosh  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_asinh  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_atanh  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_tan    (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_exp2   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_exp10  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_expm1  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_logb   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_ceil   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_floor  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_trunc  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_round  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rint   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_nearbyint(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fabs   (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_lgamma (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_tgamma (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_j0     (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_j1     (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_y0     (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_y1     (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cyl_bessel_i0(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cyl_bessel_i1(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sinpi  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_cospi  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_normcdf(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rcbrt  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erfcinv(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erfinv (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_erfcx  (const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_boys_f0(const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_norm3d (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_norm4d (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, const double d_hi, const double d_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rnorm3d(const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rnorm4d(const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, const double d_hi, const double d_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fmax   (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fmin   (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_max    (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_min    (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fmod   (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_remainder(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_hypot  (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_copysign(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_fdim   (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_nextafter(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_rhypot (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_remquo (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo, int* quo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_ilogb  (const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ long long int __nv_fp64mp2_llrint (const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ long long int __nv_fp64mp2_llround(const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ long int __nv_fp64mp2_lrint  (const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ long int __nv_fp64mp2_lround (const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_isfinite(const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_isinf   (const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_isnan   (const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ int  __nv_fp64mp2_signbit (const double x_hi, const double x_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_ldexp  (const double x_hi, const double x_lo, int n, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_scalbn (const double x_hi, const double x_lo, int n, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_scalbln(const double x_hi, const double x_lo, long int n, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_jn     (int n, const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_yn     (int n, const double x_hi, const double x_lo, double* res_hi, double* res_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_frexp  (const double x_hi, const double x_lo, double* res_hi, double* res_lo, int* nptr);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_modf   (const double x_hi, const double x_lo, double* res_hi, double* res_lo, double* iptr_hi, double* iptr_lo);
    __FPMP_BUILTIN_DECL__ void __nv_fp64mp2_sincospi(const double x_hi, const double x_lo, double* sin_hi, double* sin_lo, double* cos_hi, double* cos_lo);
#endif // FPMP_FP64MP2_ENABLE == 1

    /*
    * ============================================================================
    * Template declarations and float specializations
    * ============================================================================
    */
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_exp    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_log    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_log2   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_log10  (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_log1p  (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_pow    (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_cbrt   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_sin    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_cos    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_sincos (const T x_hi, const T x_lo, T* sin_hi, T* sin_lo, T* cos_hi, T* cos_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_asin   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_acos   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_atan   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_atan2  (const T y_hi, const T y_lo, const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_sinh   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_cosh   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_tanh   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_erf    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_erfc   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_normcdfinv (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_acosh   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_asinh   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_atanh   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_tan     (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_exp2    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_exp10   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_expm1   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_logb    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_ceil    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_floor   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_trunc   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_round   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_rint    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_nearbyint(const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_fabs    (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_lgamma  (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_tgamma  (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_j0      (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_j1      (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_y0      (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_y1      (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_cyl_bessel_i0(const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_cyl_bessel_i1(const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_sinpi   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_cospi   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_normcdf (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_rcbrt   (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_erfcinv(const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_erfinv (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_erfcx  (const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_boys_f0(const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_norm3d (const T a_hi, const T a_lo, const T b_hi, const T b_lo, const T c_hi, const T c_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_norm4d (const T a_hi, const T a_lo, const T b_hi, const T b_lo, const T c_hi, const T c_lo, const T d_hi, const T d_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_rnorm3d(const T a_hi, const T a_lo, const T b_hi, const T b_lo, const T c_hi, const T c_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_rnorm4d(const T a_hi, const T a_lo, const T b_hi, const T b_lo, const T c_hi, const T c_lo, const T d_hi, const T d_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_fmax    (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_fmin    (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_max     (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_min     (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_fmod    (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_remainder(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_hypot   (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_copysign(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_fdim    (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_nextafter(const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_rhypot  (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_remquo  (const T x_hi, const T x_lo, const T y_hi, const T y_lo, T* res_hi, T* res_lo, int* quo);
    template<typename T> __FPMP_API_DECL__ int  __nv_fpmp2_ilogb   (const T x_hi, const T x_lo);
    template<typename T> __FPMP_API_DECL__ long long int __nv_fpmp2_llrint (const T x_hi, const T x_lo);
    template<typename T> __FPMP_API_DECL__ long long int __nv_fpmp2_llround(const T x_hi, const T x_lo);
    template<typename T> __FPMP_API_DECL__ long int __nv_fpmp2_lrint  (const T x_hi, const T x_lo);
    template<typename T> __FPMP_API_DECL__ long int __nv_fpmp2_lround (const T x_hi, const T x_lo);
    template<typename T> __FPMP_API_DECL__ int  __nv_fpmp2_isfinite(const T x_hi, const T x_lo);
    template<typename T> __FPMP_API_DECL__ int  __nv_fpmp2_isinf   (const T x_hi, const T x_lo);
    template<typename T> __FPMP_API_DECL__ int  __nv_fpmp2_isnan   (const T x_hi, const T x_lo);
    template<typename T> __FPMP_API_DECL__ int  __nv_fpmp2_signbit (const T x_hi, const T x_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_ldexp   (const T x_hi, const T x_lo, int n, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_scalbn  (const T x_hi, const T x_lo, int n, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_scalbln (const T x_hi, const T x_lo, long int n, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_jn      (int n, const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_yn      (int n, const T x_hi, const T x_lo, T* res_hi, T* res_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_frexp   (const T x_hi, const T x_lo, T* res_hi, T* res_lo, int* nptr);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_modf    (const T x_hi, const T x_lo, T* res_hi, T* res_lo, T* iptr_hi, T* iptr_lo);
    template<typename T> __FPMP_API_DECL__ void __nv_fpmp2_sincospi(const T x_hi, const T x_lo, T* sin_hi, T* sin_lo, T* cos_hi, T* cos_lo);

    __FPMP_API_DECL__ void __nv_fpmp2_icdf(uint32_t x, float* res_hi, float* res_lo) { __nv_fp32mp2_icdf32(x, res_hi, res_lo); }
    __FPMP_API_DECL__ void __nv_fpmp2_icdf(uint64_t x, float* res_hi, float* res_lo) { __nv_fp32mp2_icdf64(x, res_hi, res_lo); }

    // Float (fp32) template specializations
    template<> __FPMP_API_DECL__ void __nv_fpmp2_exp<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_exp(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_log(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log2<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_log2(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log10<float>  (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_log10(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log1p<float>  (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_log1p(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_pow<float>    (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_pow(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cbrt<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_cbrt(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sin<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_sin(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cos<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_cos(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sincos<float> (const float x_hi, const float x_lo, float* sin_hi, float* sin_lo, float* cos_hi, float* cos_lo) { __nv_fp32mp2_sincos(x_hi, x_lo, sin_hi, sin_lo, cos_hi, cos_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_asin<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_asin(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_acos<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_acos(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_atan<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_atan(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_atan2<float>  (const float y_hi, const float y_lo, const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_atan2(y_hi, y_lo, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sinh<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_sinh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cosh<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_cosh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_tanh<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_tanh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erf<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_erf(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfc<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_erfc(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_normcdfinv<float> (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_normcdfinv(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_acosh<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_acosh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_asinh<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_asinh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_atanh<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_atanh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_tan<float>      (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_tan(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_exp2<float>     (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_exp2(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_exp10<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_exp10(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_expm1<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_expm1(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_logb<float>     (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_logb(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_ceil<float>     (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_ceil(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_floor<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_floor(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_trunc<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_trunc(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_round<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_round(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rint<float>     (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_rint(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_nearbyint<float>(const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_nearbyint(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fabs<float>     (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_fabs(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_lgamma<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_lgamma(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_tgamma<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_tgamma(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_j0<float>       (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_j0(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_j1<float>       (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_j1(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_y0<float>       (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_y0(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_y1<float>       (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_y1(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cyl_bessel_i0<float>(const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_cyl_bessel_i0(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cyl_bessel_i1<float>(const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_cyl_bessel_i1(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sinpi<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_sinpi(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cospi<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_cospi(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_normcdf<float>  (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_normcdf(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rcbrt<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_rcbrt(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfcinv<float>  (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_erfcinv(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfinv<float>   (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_erfinv(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfcx<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_erfcx(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_boys_f0<float>  (const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_boys_f0(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_norm3d<float>   (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_norm3d(a_hi, a_lo, b_hi, b_lo, c_hi, c_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_norm4d<float>   (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, const float d_hi, const float d_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_norm4d(a_hi, a_lo, b_hi, b_lo, c_hi, c_lo, d_hi, d_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rnorm3d<float>  (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_rnorm3d(a_hi, a_lo, b_hi, b_lo, c_hi, c_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rnorm4d<float>  (const float a_hi, const float a_lo, const float b_hi, const float b_lo, const float c_hi, const float c_lo, const float d_hi, const float d_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_rnorm4d(a_hi, a_lo, b_hi, b_lo, c_hi, c_lo, d_hi, d_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fmax<float>     (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_fmax(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fmin<float>     (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_fmin(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_max<float>      (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_max(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_min<float>      (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_min(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fmod<float>     (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_fmod(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_remainder<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_remainder(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_hypot<float>    (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_hypot(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_copysign<float> (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_copysign(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fdim<float>     (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_fdim(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_nextafter<float>(const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_nextafter(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rhypot<float>   (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_rhypot(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_remquo<float>   (const float x_hi, const float x_lo, const float y_hi, const float y_lo, float* res_hi, float* res_lo, int* quo) { __nv_fp32mp2_remquo(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo, quo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_ilogb<float>    (const float x_hi, const float x_lo) { return __nv_fp32mp2_ilogb(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ long long int __nv_fpmp2_llrint<float> (const float x_hi, const float x_lo) { return __nv_fp32mp2_llrint(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ long long int __nv_fpmp2_llround<float>(const float x_hi, const float x_lo) { return __nv_fp32mp2_llround(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ long int __nv_fpmp2_lrint<float>  (const float x_hi, const float x_lo) { return __nv_fp32mp2_lrint(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ long int __nv_fpmp2_lround<float> (const float x_hi, const float x_lo) { return __nv_fp32mp2_lround(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_isfinite<float> (const float x_hi, const float x_lo) { return __nv_fp32mp2_isfinite(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_isinf<float>    (const float x_hi, const float x_lo) { return __nv_fp32mp2_isinf(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_isnan<float>    (const float x_hi, const float x_lo) { return __nv_fp32mp2_isnan(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_signbit<float>  (const float x_hi, const float x_lo) { return __nv_fp32mp2_signbit(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_ldexp<float>    (const float x_hi, const float x_lo, int n, float* res_hi, float* res_lo) { __nv_fp32mp2_ldexp(x_hi, x_lo, n, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_scalbn<float>   (const float x_hi, const float x_lo, int n, float* res_hi, float* res_lo) { __nv_fp32mp2_scalbn(x_hi, x_lo, n, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_scalbln<float>  (const float x_hi, const float x_lo, long int n, float* res_hi, float* res_lo) { __nv_fp32mp2_scalbln(x_hi, x_lo, n, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_jn<float>       (int n, const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_jn(n, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_yn<float>       (int n, const float x_hi, const float x_lo, float* res_hi, float* res_lo) { __nv_fp32mp2_yn(n, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_frexp<float>    (const float x_hi, const float x_lo, float* res_hi, float* res_lo, int* nptr) { __nv_fp32mp2_frexp(x_hi, x_lo, res_hi, res_lo, nptr); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_modf<float>     (const float x_hi, const float x_lo, float* res_hi, float* res_lo, float* iptr_hi, float* iptr_lo) { __nv_fp32mp2_modf(x_hi, x_lo, res_hi, res_lo, iptr_hi, iptr_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sincospi<float> (const float x_hi, const float x_lo, float* sin_hi, float* sin_lo, float* cos_hi, float* cos_lo) { __nv_fp32mp2_sincospi(x_hi, x_lo, sin_hi, sin_lo, cos_hi, cos_lo); }

    /*
    * ============================================================================
    * Double (fp64) template specializations
    * ============================================================================
    */
#if (FPMP_FP64MP2_ENABLE == 1)
    template<> __FPMP_API_DECL__ void __nv_fpmp2_exp<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_exp(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_log(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log2<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_log2(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log10<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_log10(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_log1p<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_log1p(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_pow<double>    (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_pow(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cbrt<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_cbrt(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sin<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_sin(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cos<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_cos(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sincos<double> (const double x_hi, const double x_lo, double* sin_hi, double* sin_lo, double* cos_hi, double* cos_lo) { __nv_fp64mp2_sincos(x_hi, x_lo, sin_hi, sin_lo, cos_hi, cos_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_asin<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_asin(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_acos<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_acos(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_atan<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_atan(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_atan2<double>  (const double y_hi, const double y_lo, const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_atan2(y_hi, y_lo, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sinh<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_sinh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cosh<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_cosh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_tanh<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_tanh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erf<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_erf(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfc<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_erfc(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_normcdfinv<double> (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_normcdfinv(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_acosh<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_acosh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_asinh<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_asinh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_atanh<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_atanh(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_tan<double>      (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_tan(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_exp2<double>     (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_exp2(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_exp10<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_exp10(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_expm1<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_expm1(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_logb<double>     (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_logb(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_ceil<double>     (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_ceil(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_floor<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_floor(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_trunc<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_trunc(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_round<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_round(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rint<double>     (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_rint(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_nearbyint<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_nearbyint(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fabs<double>     (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_fabs(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_lgamma<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_lgamma(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_tgamma<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_tgamma(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_j0<double>       (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_j0(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_j1<double>       (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_j1(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_y0<double>       (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_y0(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_y1<double>       (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_y1(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cyl_bessel_i0<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_cyl_bessel_i0(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cyl_bessel_i1<double>(const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_cyl_bessel_i1(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sinpi<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_sinpi(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_cospi<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_cospi(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_normcdf<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_normcdf(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rcbrt<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_rcbrt(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfcinv<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_erfcinv(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfinv<double>   (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_erfinv(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_erfcx<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_erfcx(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_boys_f0<double>  (const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_boys_f0(x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_norm3d<double>   (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_norm3d(a_hi, a_lo, b_hi, b_lo, c_hi, c_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_norm4d<double>   (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, const double d_hi, const double d_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_norm4d(a_hi, a_lo, b_hi, b_lo, c_hi, c_lo, d_hi, d_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rnorm3d<double>  (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_rnorm3d(a_hi, a_lo, b_hi, b_lo, c_hi, c_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rnorm4d<double>  (const double a_hi, const double a_lo, const double b_hi, const double b_lo, const double c_hi, const double c_lo, const double d_hi, const double d_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_rnorm4d(a_hi, a_lo, b_hi, b_lo, c_hi, c_lo, d_hi, d_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fmax<double>     (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_fmax(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fmin<double>     (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_fmin(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_max<double>      (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_max(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_min<double>      (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_min(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fmod<double>     (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_fmod(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_remainder<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_remainder(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_hypot<double>    (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_hypot(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_copysign<double> (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_copysign(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_fdim<double>     (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_fdim(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_nextafter<double>(const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_nextafter(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_rhypot<double>   (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_rhypot(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_remquo<double>   (const double x_hi, const double x_lo, const double y_hi, const double y_lo, double* res_hi, double* res_lo, int* quo) { __nv_fp64mp2_remquo(x_hi, x_lo, y_hi, y_lo, res_hi, res_lo, quo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_ilogb<double>    (const double x_hi, const double x_lo) { return __nv_fp64mp2_ilogb(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ long long int __nv_fpmp2_llrint<double> (const double x_hi, const double x_lo) { return __nv_fp64mp2_llrint(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ long long int __nv_fpmp2_llround<double>(const double x_hi, const double x_lo) { return __nv_fp64mp2_llround(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ long int __nv_fpmp2_lrint<double>  (const double x_hi, const double x_lo) { return __nv_fp64mp2_lrint(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ long int __nv_fpmp2_lround<double> (const double x_hi, const double x_lo) { return __nv_fp64mp2_lround(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_isfinite<double> (const double x_hi, const double x_lo) { return __nv_fp64mp2_isfinite(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_isinf<double>    (const double x_hi, const double x_lo) { return __nv_fp64mp2_isinf(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_isnan<double>    (const double x_hi, const double x_lo) { return __nv_fp64mp2_isnan(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ int  __nv_fpmp2_signbit<double>  (const double x_hi, const double x_lo) { return __nv_fp64mp2_signbit(x_hi, x_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_ldexp<double>    (const double x_hi, const double x_lo, int n, double* res_hi, double* res_lo) { __nv_fp64mp2_ldexp(x_hi, x_lo, n, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_scalbn<double>   (const double x_hi, const double x_lo, int n, double* res_hi, double* res_lo) { __nv_fp64mp2_scalbn(x_hi, x_lo, n, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_scalbln<double>  (const double x_hi, const double x_lo, long int n, double* res_hi, double* res_lo) { __nv_fp64mp2_scalbln(x_hi, x_lo, n, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_jn<double>       (int n, const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_jn(n, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_yn<double>       (int n, const double x_hi, const double x_lo, double* res_hi, double* res_lo) { __nv_fp64mp2_yn(n, x_hi, x_lo, res_hi, res_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_frexp<double>    (const double x_hi, const double x_lo, double* res_hi, double* res_lo, int* nptr) { __nv_fp64mp2_frexp(x_hi, x_lo, res_hi, res_lo, nptr); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_modf<double>     (const double x_hi, const double x_lo, double* res_hi, double* res_lo, double* iptr_hi, double* iptr_lo) { __nv_fp64mp2_modf(x_hi, x_lo, res_hi, res_lo, iptr_hi, iptr_lo); }
    template<> __FPMP_API_DECL__ void __nv_fpmp2_sincospi<double> (const double x_hi, const double x_lo, double* sin_hi, double* sin_lo, double* cos_hi, double* cos_lo) { __nv_fp64mp2_sincospi(x_hi, x_lo, sin_hi, sin_lo, cos_hi, cos_lo); }
#endif // FPMP_FP64MP2_ENABLE == 1

#endif // ! defined __FPMP_USE_LIB__

/*
* ============================================================================
* Freestanding API functions for fpmp2_t class
* ============================================================================
*/

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> exp (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_exp(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> log (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_log(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> log2 (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_log2(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> log10 (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_log10(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> log1p (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_log1p(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> pow (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y) 
{ FpType res_hi, res_lo; __nv_fpmp2_pow(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> cbrt (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_cbrt(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> sin (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_sin(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> cos (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_cos(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ void sincos (const fpmp2_t<FpType, met>& x, fpmp2_t<FpType, met>* s, fpmp2_t<FpType, met>* c) 
{ FpType sin_hi, sin_lo, cos_hi, cos_lo; __nv_fpmp2_sincos(x.hi(), x.lo(), &sin_hi, &sin_lo, &cos_hi, &cos_lo); *s = fpmp2_t<FpType, met>(sin_hi, sin_lo); *c = fpmp2_t<FpType, met>(cos_hi, cos_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> asin (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_asin(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> acos (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_acos(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> atan (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_atan(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> atan2 (const fpmp2_t<FpType, met>& y, const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_atan2(y.hi(), y.lo(), x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> sinh (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_sinh(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> cosh (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_cosh(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> tanh (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_tanh(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> erf (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_erf(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> erfc (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_erfc(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> boys_f0 (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_boys_f0(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> normcdfinv (const fpmp2_t<FpType, met>& x) 
{ FpType res_hi, res_lo; __nv_fpmp2_normcdfinv(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<float, met> icdf (uint32_t x) 
{ float res_hi, res_lo; __nv_fpmp2_icdf(x, &res_hi, &res_lo); return fpmp2_t<float, met>(res_hi, res_lo); }

template <fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<float, met> icdf (uint64_t x) 
{ float res_hi, res_lo; __nv_fpmp2_icdf(x, &res_hi, &res_lo); return fpmp2_t<float, met>(res_hi, res_lo); }

// Inverse hyperbolic functions
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> acosh (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_acosh(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> asinh (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_asinh(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> atanh (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_atanh(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// Tangent
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> tan (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_tan(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// Additional exponential/logarithmic functions
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> exp2 (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_exp2(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> exp10 (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_exp10(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> expm1 (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_expm1(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> logb (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_logb(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// Rounding functions
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> ceil (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_ceil(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> floor (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_floor(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> trunc (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_trunc(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> round (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_round(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> rint (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_rint(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> nearbyint (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_nearbyint(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// Absolute value
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> fabs (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_fabs(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// Gamma functions
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> lgamma (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_lgamma(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> tgamma (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_tgamma(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// Bessel functions
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> j0 (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_j0(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> j1 (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_j1(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> y0 (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_y0(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> y1 (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_y1(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> jn (int n, const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_jn(n, x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> yn (int n, const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_yn(n, x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> cyl_bessel_i0 (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_cyl_bessel_i0(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> cyl_bessel_i1 (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_cyl_bessel_i1(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// CUDA-specific trigonometric functions
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> sinpi (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_sinpi(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> cospi (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_cospi(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ void sincospi (const fpmp2_t<FpType, met>& x, fpmp2_t<FpType, met>* s, fpmp2_t<FpType, met>* c)
{ FpType sin_hi, sin_lo, cos_hi, cos_lo; __nv_fpmp2_sincospi(x.hi(), x.lo(), &sin_hi, &sin_lo, &cos_hi, &cos_lo); *s = fpmp2_t<FpType, met>(sin_hi, sin_lo); *c = fpmp2_t<FpType, met>(cos_hi, cos_lo); }

// Normal distribution CDF and reciprocal functions
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> normcdf (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_normcdf(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> rcbrt (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_rcbrt(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> erfcinv (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_erfcinv(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> erfinv (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_erfinv(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> erfcx (const fpmp2_t<FpType, met>& x)
{ FpType res_hi, res_lo; __nv_fpmp2_erfcx(x.hi(), x.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> norm3d (const fpmp2_t<FpType, met>& a, const fpmp2_t<FpType, met>& b, const fpmp2_t<FpType, met>& c)
{ FpType res_hi, res_lo; __nv_fpmp2_norm3d(a.hi(), a.lo(), b.hi(), b.lo(), c.hi(), c.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> norm4d (const fpmp2_t<FpType, met>& a, const fpmp2_t<FpType, met>& b, const fpmp2_t<FpType, met>& c, const fpmp2_t<FpType, met>& d)
{ FpType res_hi, res_lo; __nv_fpmp2_norm4d(a.hi(), a.lo(), b.hi(), b.lo(), c.hi(), c.lo(), d.hi(), d.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> rnorm3d (const fpmp2_t<FpType, met>& a, const fpmp2_t<FpType, met>& b, const fpmp2_t<FpType, met>& c)
{ FpType res_hi, res_lo; __nv_fpmp2_rnorm3d(a.hi(), a.lo(), b.hi(), b.lo(), c.hi(), c.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> rnorm4d (const fpmp2_t<FpType, met>& a, const fpmp2_t<FpType, met>& b, const fpmp2_t<FpType, met>& c, const fpmp2_t<FpType, met>& d)
{ FpType res_hi, res_lo; __nv_fpmp2_rnorm4d(a.hi(), a.lo(), b.hi(), b.lo(), c.hi(), c.lo(), d.hi(), d.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// Two-argument functions
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> fmax (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_fmax(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> fmin (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_fmin(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> max (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_max(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> min (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_min(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> fmod (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_fmod(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> remainder (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_remainder(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> hypot (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_hypot(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> copysign (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_copysign(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> fdim (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_fdim(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> nextafter (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_nextafter(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> rhypot (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y)
{ FpType res_hi, res_lo; __nv_fpmp2_rhypot(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// Functions with special signatures
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> remquo (const fpmp2_t<FpType, met>& x, const fpmp2_t<FpType, met>& y, int* quo)
{ FpType res_hi, res_lo; __nv_fpmp2_remquo(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo, quo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> ldexp (const fpmp2_t<FpType, met>& x, int n)
{ FpType res_hi, res_lo; __nv_fpmp2_ldexp(x.hi(), x.lo(), n, &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> scalbn (const fpmp2_t<FpType, met>& x, int n)
{ FpType res_hi, res_lo; __nv_fpmp2_scalbn(x.hi(), x.lo(), n, &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> scalbln (const fpmp2_t<FpType, met>& x, long int n)
{ FpType res_hi, res_lo; __nv_fpmp2_scalbln(x.hi(), x.lo(), n, &res_hi, &res_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> frexp (const fpmp2_t<FpType, met>& x, int* nptr)
{ FpType res_hi, res_lo; __nv_fpmp2_frexp(x.hi(), x.lo(), &res_hi, &res_lo, nptr); return fpmp2_t<FpType, met>(res_hi, res_lo); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ fpmp2_t<FpType, met> modf (const fpmp2_t<FpType, met>& x, fpmp2_t<FpType, met>* iptr)
{ FpType res_hi, res_lo, i_hi, i_lo; __nv_fpmp2_modf(x.hi(), x.lo(), &res_hi, &res_lo, &i_hi, &i_lo); *iptr = fpmp2_t<FpType, met>(i_hi, i_lo); return fpmp2_t<FpType, met>(res_hi, res_lo); }

// Functions returning integer types
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ int ilogb (const fpmp2_t<FpType, met>& x)
{ return __nv_fpmp2_ilogb(x.hi(), x.lo()); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ long long int llrint (const fpmp2_t<FpType, met>& x)
{ return __nv_fpmp2_llrint(x.hi(), x.lo()); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ long long int llround (const fpmp2_t<FpType, met>& x)
{ return __nv_fpmp2_llround(x.hi(), x.lo()); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ long int lrint (const fpmp2_t<FpType, met>& x)
{ return __nv_fpmp2_lrint(x.hi(), x.lo()); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ long int lround (const fpmp2_t<FpType, met>& x)
{ return __nv_fpmp2_lround(x.hi(), x.lo()); }

// Classification functions
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ int fpmp_isfinite (const fpmp2_t<FpType, met>& x)
__FPMP_NOEXCEPT__
{ return __nv_fpmp2_isfinite(x.hi(), x.lo()); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ int fpmp_isinf (const fpmp2_t<FpType, met>& x)
__FPMP_NOEXCEPT__
{ return __nv_fpmp2_isinf(x.hi(), x.lo()); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ int fpmp_isnan (const fpmp2_t<FpType, met>& x)
__FPMP_NOEXCEPT__
{ return __nv_fpmp2_isnan(x.hi(), x.lo()); }

template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ int fpmp_signbit (const fpmp2_t<FpType, met>& x)
__FPMP_NOEXCEPT__
{ return __nv_fpmp2_signbit(x.hi(), x.lo()); }

// Standard names are provided only when no conflicting macro is active.
#ifndef isfinite
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ int isfinite (const fpmp2_t<FpType, met>& x)
__FPMP_NOEXCEPT__
{ return fpmp_isfinite(x); }
#endif

#ifndef isinf
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ int isinf (const fpmp2_t<FpType, met>& x)
__FPMP_NOEXCEPT__
{ return fpmp_isinf(x); }
#endif

#ifndef isnan
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ int isnan (const fpmp2_t<FpType, met>& x)
__FPMP_NOEXCEPT__
{ return fpmp_isnan(x); }
#endif

#ifndef signbit
template <typename FpType = float, fpmp2_accuracy met = fpmp2_accuracy::def>
__FPMP_API_DECL__ int signbit (const fpmp2_t<FpType, met>& x)
__FPMP_NOEXCEPT__
{ return fpmp_signbit(x); }
#endif

/*
* Note: the fpmp2_t warp-shuffle overloads (__shfl_sync, __shfl_xor_sync,
* __shfl_down_sync, __shfl_up_sync) are thread-cooperation primitives, not math
* functions, so they live in the core header <cuda/__fp/fpmp.h> (available via
* <cuda/fpmp>) rather than here.
*/

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_H
