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

    Warp Shuffle (CUDA-only, modern __shfl_sync family): the fpmp2 overloads
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
      the ~46-bit precision of __fpmp2_log this still leaves >44
      bits of accuracy across the whole representable input range,
      matching the fp32mp2 noise floor.  All domain special cases
      (x<=0, +-inf, NaN) are handled inside __fpmp2_log.

    exp2(x), exp10(x) for fp32mp2:
      Dedicated implementations following libdevice's "single base-2
      split" strategy (`__internal_accurate_expf_1p93ulp` paired with
      MUFU.EX2 on the reduced argument).  Pseudocode:
        t      = x * log2(base)        [exp2: t = x; exp10: t = x * log2 10]
        n      = round(t.hi)           [exact integer = binary exponent]
        r      = t - n                 [|r.hi| <= 0.5]
        2^r    via the inlined base-2 Taylor kernel
               `__fp32mp2_exp2_kernel` with coefficients
               a_k = (ln 2)^k / k!     [no r * ln 2 detour, no
                                        natural-log reduction inside]
        result = 2^n * 2^r             [via split-exponent helper]
      The single integer split happens in *base-2* units, so the 2^n
      factor drops out exactly and the kernel never touches a value
      outside [-0.5, 0.5].  Compared to the earlier composition
      exp(x * ln base) -- which forced __fpmp2_exp to re-derive
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
    - When _CCCL_FPMP_FP128_MATH_FALLBACK=1, route through __float128 (libquadmath
      on host, CUDA fp128 intrinsics on device) for ~113-bit accuracy.
    - When _CCCL_FPMP_FP128_MATH_FALLBACK=0, fall back to double-precision math.
    - normcdfinv<double> uses CUDA erfcinv on device, the fp32mp2 polynomial
      on host (no standard erfcinv available).

    Configuration Macros:
    -------------------------------------------------------------------------
    - _CCCL_FPMP_FP128_ENABLE: Automatically detected from compiler/CUDA. When 1, enables
      __float128 type support for conversions. Can be set to 0 to disable.
    - _CCCL_FPMP_FP128_MATH_FALLBACK: Controls fp64mp2 transcendental function accuracy:
      * When 1: Uses quad-precision (__float128) via libquadmath (host) or CUDA fp128
        intrinsics (device). Provides ~113-bit accuracy but requires libquadmath linkage,
        slower compilation, and larger code size.
      * When 0 (default): Falls back to double-precision math. Faster builds, smaller
        code, no extra library dependencies, but accuracy limited to ~53-bit mantissa
        for transcendental functions.
    - _CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK: Controls fp32mp2 sin/cos/sincos/tan for large
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
#include <cuda/std/cassert>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

#if !(defined _CCCL_FPMP_USE_LIB)

/*
 * Polynomial evaluation helpers
 */

    /*********************************************************************
    * Mixed-precision (split-M) Horner polynomial evaluation
    * (internal building block -- namespace `fpmp`)
    *
    *   p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[N-1]*x^(N-1)
    *
    * with x given as fpmp2<FpType, met> and the coefficient
    * table `c[]` packed as fpmp2<FpType, met> in ascending order
    * of degree (c[0] = constant term, c[N-1] = leading coefficient).
    *
    * The template parameter M controls the precision split:
    *
    *   - The M HIGHEST-degree coefficients
    *         c[N-1], c[N-2], ..., c[N-M]
    *     are treated as plain FpType constants. Their `.lo()` parts
    *     are assumed to be zero (which is the natural state when the
    *     coefficient is built from a single FpType literal via the
    *     implicit `fpmp2(FpType)` ctor; using a ffloat literal
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
    *     mirrors the float*float + ff layout used in `__fpmp2_erfc`
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
    *     ffloat v = __fpmp_poly_horner_mixed<M>(x, c);  // mixed standard
    *     ffloat v = __fpmp_poly_horner_comp    (x, c);  // compensated
    * or dispatch via `__fpmp_poly_eval<strategy, M>(x, c)` below.
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
    template<int _Mp, int _Np, typename _FpType, fpmp2_accuracy _TypeAcc>
    _CCCL_API inline fpmp2<_FpType, _TypeAcc>
    __fpmp_poly_horner_mixed(const fpmp2<_FpType, _TypeAcc>& __x,
                      const fpmp2<_FpType, _TypeAcc> (&__c)[_Np]) noexcept
    {
        static_assert(_Np >= 2, "poly_horner_mixed requires at least 2 coefficients (degree >= 1)");
        static_assert(_Mp >= 0, "poly_horner_mixed: M must be non-negative");
        static_assert(_Mp <= _Np, "poly_horner_mixed: M must not exceed N");

        using ff_t = fpmp2<_FpType, _TypeAcc>;

        if constexpr (_Mp == 0)
        {
            // Pure ff Horner -- no FpType phase.
            ff_t __v = __c[_Np - 1];
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int __k = _Np - 2; __k >= 0; --__k) {
                __v = __v * __x + __c[__k];
            }
            return __v;
        }
        else
        {
            // FpType phase: M iterations consuming c[N-1] ... c[N-M].
            const _FpType __xh = __x.hi();
            _FpType __v_f      = __c[_Np - 1].hi();
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int __k = _Np - 2; __k >= _Np - _Mp; --__k) {
                __v_f = __v_f * __xh + __c[__k].hi();
            }

            if constexpr (_Mp == _Np)
            {
                // No ff phase at all -- promote the FpType result.
                return ff_t(__v_f);
            }
            else
            {
                // Transition step: (float * float) + ff -> ff
                // (the mixed-type operator+ promotes the FpType product
                // to ff_t with .lo() == 0 before adding c[N-M-1].)
                ff_t __v = __v_f * __xh + __c[_Np - _Mp - 1];
            #if defined(__CUDA_ARCH__)
                #pragma unroll
            #endif
                for (int __k = _Np - _Mp - 2; __k >= 0; --__k) {
                    __v = __v * __x + __c[__k];
                }
                return __v;
            }
        }
    } // poly_horner_mixed

    /*********************************************************************
    * Compensated Horner polynomial evaluation for fp32mp2 / fp64mp2
    * (internal building block -- namespace `fpmp`)
    *
    *   p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[N-1]*x^(N-1)
    *
    * with x and c[k] given as fpmp2<FpType, met> (ascending order
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
    *             FpType literals built via `fpmp2(FpType)` ctor).
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
    * FpType constant via the implicit `fpmp2(FpType)` ctor)
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
    template<int _Mp = 0, int _Np, typename _FpType, fpmp2_accuracy _TypeAcc>
    _CCCL_API inline fpmp2<_FpType, _TypeAcc>
    __fpmp_poly_horner_comp(const fpmp2<_FpType, _TypeAcc>& __x,
                     const fpmp2<_FpType, _TypeAcc> (&__c)[_Np]) noexcept
    {
        static_assert(_Np >= 2, "poly_horner_comp requires at least 2 coefficients (degree >= 1)");
        static_assert(_Mp >= 0, "poly_horner_comp: M must be non-negative");
        static_assert(_Mp <= _Np, "poly_horner_comp: M must not exceed N");

        const _FpType __xh = __x.hi();
        const _FpType __xl = __x.lo();

        // === Phase 0: M-1 plain FpType Horner steps (no error tracking) ===
        _FpType __acc = __c[_Np - 1].hi();
        if constexpr (_Mp >= 2) {
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int __k = _Np - 2; __k >= _Np - _Mp; --__k) {
                __acc = __acc * __xh + __c[__k].hi();
            }
        }

        // === Phase 1: N-M compensated Horner steps ===
        _FpType __err = static_cast<_FpType>(0);
        if constexpr (_Mp < _Np) {
            // For M == 0 the init handled c[N-1], so compensated loop
            // starts at c[N-2]; for M >= 1 Phase 0 handled c[N-1]..c[N-M],
            // so compensated loop starts at c[N-M-1].
            constexpr int __comp_start = (_Mp == 0) ? (_Np - 2) : (_Np - _Mp - 1);
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int __k = __comp_start; __k >= 0; --__k)
            {
                const _FpType __ckh = __c[__k].hi();

                // two_mult_fma: P + pi == xh * acc  (exact)
                _FpType __pval  = __fpmp_mul_rn(__xh, __acc);
                _FpType __pi = __fpmp_fma_rn(__xh, __acc, -__pval);

                // two_sum: S + sg == P + ckh  (exact, no magnitude assumption)
                _FpType __S  = __fpmp_add_rn(__pval, __ckh);
                _FpType __bb = __fpmp_sub_rn(__S, __pval);
                _FpType __t  = __fpmp_sub_rn(__S, __bb);
                _FpType __u  = __fpmp_sub_rn(__pval, __t);
                _FpType __v  = __fpmp_sub_rn(__ckh, __bb);
                _FpType __sg = __fpmp_add_rn(__u, __v);

                __err = __fpmp_fma_rn(__xh, __err, __fpmp_add_rn(__pi, __sg));
                __acc = __S;
            }
        }

        // === Phase 2a: contribution of c[k].lo (top M iterations skipped) ===
        _FpType __corr = static_cast<_FpType>(0);
        if constexpr (_Mp < _Np) {
            // For M == 0 we visit all N coefficients (k = N-1 .. 0);
            // for M >= 1 we skip the top M (their .lo() == 0 by contract).
            constexpr int __lo_start = (_Mp == 0) ? (_Np - 1) : (_Np - _Mp - 1);
        #if defined(__CUDA_ARCH__)
            #pragma unroll
        #endif
            for (int __k = __lo_start; __k >= 0; --__k)
            {
                __corr = __fpmp_fma_rn(__xh, __corr, __c[__k].lo());
            }
        }

        // === Phase 2b: x.lo * p'(x.hi)  (full derivative, all N-1 terms) ===
        _FpType __dp = static_cast<_FpType>(0);
    #if defined(__CUDA_ARCH__)
        #pragma unroll
    #endif
        for (int __k = _Np - 1; __k >= 1; --__k)
        {
            __dp = __fpmp_fma_rn(__xh, __dp, __fpmp_mul_rn(static_cast<_FpType>(__k), __c[__k].hi()));
        }
        __corr = __fpmp_fma_rn(__xl, __dp, __corr);

        // === Phase 3: combine into normalized ff ===
        _FpType __lo  = __fpmp_add_rn(__err, __corr);
        _FpType __rhi = __fpmp_add_rn(__acc, __lo);
        _FpType __rlo = __fpmp_sub_rn(__lo, __fpmp_sub_rn(__rhi, __acc));
        return fpmp2<_FpType, _TypeAcc>(__rhi, __rlo);
    } // poly_horner_comp

    /*********************************************************************
    * Polynomial-evaluation strategy selector for `poly_eval`.
    *
    * Listed kernels currently route to a Horner backend; future
    * additions (e.g. factorized / Estrin / Knuth-Eve evaluation)
    * are expected to slot in as new enumerators here without
    * changing the dispatcher signature.
    *********************************************************************/
    enum class __fpmp_poly_method
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
    *     __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, M>(x, c)
    *         -> __fpmp_poly_horner_mixed<M>(x, c)
    *
    *     __fpmp_poly_eval<__fpmp_poly_method::horner_comp,  M>(x, c)
    *         -> __fpmp_poly_horner_comp <M>(x, c)
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
    template<__fpmp_poly_method _Strategy, int _Mp = 0,
             int _Np, typename _FpType, fpmp2_accuracy _TypeAcc>
    _CCCL_API inline fpmp2<_FpType, _TypeAcc>
    __fpmp_poly_eval(const fpmp2<_FpType, _TypeAcc>& __x,
              const fpmp2<_FpType, _TypeAcc> (&__c)[_Np]) noexcept
    {
        if constexpr (_Strategy == __fpmp_poly_method::horner_mixed) {
            return __fpmp_poly_horner_mixed<_Mp>(__x, __c);
        } else /* poly_method::horner_comp */ {
            return __fpmp_poly_horner_comp <_Mp>(__x, __c);
        }
    } // poly_eval


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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_exp (const _FpType __x_hi, 
                                                const _FpType __x_lo, 
                                                _FpType*      __res_hi, 
                                                _FpType*      __res_lo) noexcept
    {
        using ffloat = fp32mp2_low;
     
        // Constants as C99 hex floating-point literals - split via constexpr constructor
        constexpr float  __inv_ln2(0x1.715476p+0f);     // 1/ln(2)
        constexpr float  __shift_bias(12582912.0f+127.0f*2.0f); // 127.0f*2.0f is the bias for the exponent
        constexpr ffloat __ln2(0x1.62e42fefa39efp-1);  // ln(2)
        
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
        constexpr float __c1(0x1.0p+0);
        constexpr float __c2(0x1.0p-1);

        constexpr ffloat __exp_c[11] = {
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
        if (__x_hi > 0x1.62e430p+6f) 
        {
            *__res_hi = __builtin_huge_valf();
            *__res_lo = 0.0f;
            return;
        }

        // Underflow threshold for single precision: ln(FLT_MIN) ~= -87.3365 = -0x1.5d589ep+6
        if (__x_hi < -0x1.5d589ep+6f) 
        {
            *__res_hi = 0.0f;
            *__res_lo = 0.0f;
            return;
        }
        
        ffloat    __x(__x_hi, __x_lo);

        // Step 1: Argument reduction: x = n*ln(2) + r, where |r| < ln(2)/2
        float __t  = __x_hi*__inv_ln2 + __shift_bias;

        // Shift the exponent by 23 bits to get the scale as fp32 value
        int32_t __scale = __fpmp_internal_bit_cast<int32_t>(__t);
        __scale <<= 23;

        // Split the scale into high and low parts
        uint32_t __scale_lo = __scale >> 1;
        __scale_lo &= 0x7F800000u;
        __scale    -= __scale_lo;

        // Cast the scales to fp32 values
        float __fscale    = __fpmp_internal_bit_cast<float>(__scale);
        float __fscale_lo = __fpmp_internal_bit_cast<float>(__scale_lo);

        // Compute the reduced argument r = x - n*ln(2)
        float __tt = __t - __shift_bias;
        ffloat __r = __x - ffloat(static_cast<float>(__tt)) * __ln2;
        __r        = renormalize(__r);

        // Scale the reduced argument by the low part of the scale
        ffloat __r_scale = __r * __fscale_lo;

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
        ffloat __p = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 6>(__r, __exp_c);

        // Fold in the low-degree float coefficients c1, c2 outside the
        // dispatcher (they live at the wrong end of the polynomial for
        // the M-split optimisation).
        __p = __p * __r + __c2;
        __p = __p * __r + __c1;

        __p = __p * __r_scale + __fscale_lo;

        // Scale the result by the high part of the scale
        __p = __p * ffloat(__fscale);
        ffloat __result = renormalize(__p);
        
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_exp

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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_log (const _FpType __x_hi,
                                                const _FpType __x_lo,
                                                _FpType*      __res_hi,
                                                _FpType*      __res_lo) noexcept
    {
        using ffloat = fp32mp2_low;

        constexpr ffloat __ln2(0x1.62e42fefa39efp-1);  // ln(2)

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
        constexpr ffloat __atanh_c[8] = {
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
        float  __a_hi  = __x_hi;
        float  __a_lo  = __x_lo;
        int    __e_adj = 0;

        /* Normalize denormals: scale by 2^24 to make the exponent field nonzero */
        uint32_t __xbits = __fpmp_internal_bit_cast<uint32_t>(__a_hi);
        if ((__xbits & 0x7F800000u) == 0u) 
        {
            __a_hi  = __a_hi * 0x1.0p24f;
            __a_lo  = __a_lo * 0x1.0p24f;
            __e_adj = -24;
            __xbits = __fpmp_internal_bit_cast<uint32_t>(__a_hi);
        }

        int __e = static_cast<int>((__xbits >> 23) & 0xFFu) - 127 + __e_adj;

        /* m_hi in [1, 2) by replacing exponent field with bias 127 */
        float __m_hi = __fpmp_internal_bit_cast<float>((__xbits & 0x007FFFFFu) | 0x3F800000u);

        /* Scale a_lo by 2^(-e_orig) where e_orig = e - e_adj,
         * using split factors to stay in normal float range */
        int __e_orig = __e - __e_adj;
        int __e2     = __e_orig / 2;
        float __s1   = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>(127 - __e2) << 23);
        float __s2   = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>(127 - (__e_orig - __e2)) << 23);
        float __m_lo = __a_lo * __s1 * __s2;

        ffloat __m = renormalize(ffloat(__m_hi, __m_lo));

        /* If m > sqrt(2), halve m and increment e */
        if (__m.hi() > 0x1.6a09e6p+0f) 
        {
            __m = __m * 0.5f;
            __e = __e + 1;
        }

        /* u = 2*(m-1)/(m+1), v = u^2
         * Use accurate subtraction for (m - 1) to handle catastrophic
         * cancellation when m ~= 1 (x near a power of 2).
         */
        ffloat __f = sub<fpmp2_accuracy::high>(__m, 1.0f);
        ffloat __g = __m + 1.0f;
        ffloat __u = __f / __g;
        __u = __u + __u;
        __u = renormalize(__u);
        ffloat __v = __u * __u;

        /* Horner evaluation: q(v) = c1 + c2*v + c3*v^2 + ... + c8*v^7
         * via the mixed-precision dispatcher (5 high-order terms in
         * plain float, remaining 3 in ff).  The dispatcher transition
         * `qf * v.hi() + c3` matches the previous hand-written step
         * bit-for-bit, so this refactor is numerically identical to
         * the previous implementation.
         */
        ffloat __q = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 5>(__v, __atanh_c);

        /* log(m) = u + u*v*q(v) */
        __q = __q * __v;
        ffloat __log_m = __q * __u + __u;

        /* log(x) = log(m) + e*ln(2) */
        ffloat __result = renormalize(__log_m + ffloat(static_cast<float>(__e)) * __ln2);

        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_log

    /*
    * --------------------------------------------------------------------
    * Natural logarithm of (1 + x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Strategy:
    *   - Small |x_hi| (< 1/16):  direct Taylor series in (x_hi, x_lo),
    *     keeping the full fp32mp2 input intact.
    *   - Otherwise:              forward to `__fpmp2_log` at (1+x).
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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_log1p (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_log1p is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* NaN propagation: any NaN component -> NaN result. */
        if (__x_hi != __x_hi || __x_lo != __x_lo)
        {
            const float __nan_val = __x_hi + __x_lo;
            *__res_hi = __nan_val;
            *__res_lo = __nan_val;
            return;
        }

        /* +inf input -> +inf. */
        if (__x_hi == __builtin_huge_valf())
        {
            *__res_hi = __builtin_huge_valf();
            *__res_lo = 0.0f;
            return;
        }

        /* -inf input -> log(-inf) = NaN. */
        if (__x_hi == -__builtin_huge_valf())
        {
            *__res_hi = __builtin_nanf("");
            *__res_lo = __builtin_nanf("");
            return;
        }

        /* Small-|x| polynomial branch.  See header comment for the
         * rationale: bypasses the (1+x) -> log() pipeline whose
         * accumulated lo-ulp errors dominate as |x| -> 0.  Domain check
         * for x = -1 / x < -1 still applies (covered by the |x|<1/16
         * threshold trivially: any x in this range is well above -1). */
        const float __abs_hi = (__x_hi < 0.0f) ? -__x_hi : __x_hi;
        constexpr float __LOG1P_BRANCH_POINT = 0.0625f;  /* 1/16 = 2^-4 */
        if (__abs_hi < __LOG1P_BRANCH_POINT)
        {
            /* T(x) = sum_{k>=0} (-1)^k * x^k / (k+2),
             *   T[0] = -1/2, T[1] = +1/3, ..., T[11] = +1/13.
             * Layout for poly_eval<horner_mixed, M=4>: bottom 8 entries
             * are full ff (their contributions stay above fp32mp2 ulp at
             * the branch point), top 4 entries are plain float (.lo == 0
             * by construction; their contributions sit below 0.5 ulp). */
            constexpr ffloat __log1p_poly_c[12] = {
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

            ffloat __x      (__x_hi, __x_lo);
            ffloat __x2     = __x * __x;
            ffloat __T      = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 4>(__x, __log1p_poly_c);
            ffloat __result = renormalize(__x + __x2 * __T);
            *__res_hi = __result.hi();
            *__res_lo = __result.lo();
            return;
        }

        /* Compute (1 + x) in fp32mp2 with accurate add: this preserves
         * the residual to fp32mp2 precision even when 1 + x.hi cancels
         * to a small magnitude (i.e., x close to -1).  The lo of the
         * result captures the rounding loss that a plain fast 2-sum
         * folds into the leading term and then quantizes to float
         * precision in the subsequent operations. */
        ffloat __sum = add<fpmp2_accuracy::high>(ffloat(1.0f), ffloat(__x_hi, __x_lo));

        /* (1 + x) < 0  -> NaN. */
        if (__sum.hi() < 0.0f)
        {
            *__res_hi = __builtin_nanf("");
            *__res_lo = __builtin_nanf("");
            return;
        }

        /* (1 + x) == 0 (i.e., x == -1) -> log(0) = -inf. */
        if (__sum.hi() == 0.0f && __sum.lo() == 0.0f)
        {
            *__res_hi = -__builtin_huge_valf();
            *__res_lo = 0.0f;
            return;
        }

        /* Edge case: sum.hi == 0, sum.lo > 0 (x = -1 + tiny).  Promote
         * lo to hi so log() sees a normalized argument. */
        if (__sum.hi() == 0.0f)
        {
            if (__sum.lo() < 0.0f)
            {
                *__res_hi = __builtin_nanf("");
                *__res_lo = __builtin_nanf("");
                return;
            }
            __sum = ffloat(__sum.lo(), 0.0f);
        }

        /* Forward to dedicated fp32mp2 log. */
        __fpmp2_log<float>(__sum.hi(), __sum.lo(), __res_hi, __res_lo);
    } // __fpmp2_log1p

    /*
    * --------------------------------------------------------------------
    * Base-2 logarithm log2(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Composition over the dedicated fp32mp2 natural log:
    *     log2(x) = log(x) * (1/ln(2))
    * with (1/ln(2)) carried as an fp32mp2 constant (hi+lo).  The single
    * ff-multiply costs ~1 ulp on the lo limb; combined with the ~46-bit
    * precision of __fpmp2_log this still leaves >44 bits of accuracy
    * across the whole representable input range, which matches the
    * fp32mp2 noise floor (cf. log/log1p reports in the test suite).
    *
    * All special cases (x<=0, NaN, +inf) are handled inside
    * __fpmp2_log<float>; this wrapper only scales the result.
    * --------------------------------------------------------------------
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_log2 (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_log2 is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* 1/ln(2) ~= 1.4426950408889634073599... */
        constexpr ffloat __inv_ln2(0x1.71547652b82fep+0);

        float __l_hi, __l_lo;
        __fpmp2_log<float>(__x_hi, __x_lo, &__l_hi, &__l_lo);

        /* Propagate non-finite outputs (NaN, +-inf) unchanged: a multiply
         * by a finite constant would still yield the same kind for +-inf,
         * but NaN composition is cleanest with an explicit short-circuit
         * (avoids an unnecessary mul that could quiet a signaling NaN
         * on some platforms). */
        if (__l_hi != __l_hi || __l_hi == __builtin_huge_valf() || __l_hi == -__builtin_huge_valf())
        {
            *__res_hi = __l_hi;
            *__res_lo = (__l_hi != __l_hi) ? __l_hi : 0.0f;
            return;
        }

        ffloat __result = renormalize(ffloat(__l_hi, __l_lo) * __inv_ln2);
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_log2

    /*
    * --------------------------------------------------------------------
    * Base-10 logarithm log10(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Composition over the dedicated fp32mp2 natural log:
    *     log10(x) = log(x) * (1/ln(10))
    * with (1/ln(10)) carried as an fp32mp2 constant.  Same accuracy
    * trade-off as log2; see __fpmp2_log2 header comment.
    * --------------------------------------------------------------------
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_log10 (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_log10 is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* 1/ln(10) ~= 0.4342944819032518276511289... */
        constexpr ffloat __inv_ln10(0x1.bcb7b1526e50ep-2);

        float __l_hi, __l_lo;
        __fpmp2_log<float>(__x_hi, __x_lo, &__l_hi, &__l_lo);

        if (__l_hi != __l_hi || __l_hi == __builtin_huge_valf() || __l_hi == -__builtin_huge_valf())
        {
            *__res_hi = __l_hi;
            *__res_lo = (__l_hi != __l_hi) ? __l_hi : 0.0f;
            return;
        }

        ffloat __result = renormalize(ffloat(__l_hi, __l_lo) * __inv_ln10);
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_log10

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
    * natural-log Taylor inside __fpmp2_exp, this kernel saves:
    *   - the y = r * ln 2 fp32mp2 product (1 ULP),
    *   - the trivial-reduction housekeeping inside __fpmp2_exp
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
    _CCCL_TRIVIAL_API fp32mp2_low __fp32mp2_exp2_kernel(fp32mp2_low __r) noexcept
    {
        using ffloat = fp32mp2_low;

        /* a1..a13 = (ln 2)^k / k!,  ordered low -> high degree.
         * 7 low-degree ff entries (carry .lo()) + 6 high-degree float
         * entries (M = 6) consumed by poly_eval's float-only inner loop. */
        constexpr ffloat __exp2_c[13] = {
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
        ffloat __p = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 6>(__r, __exp2_c);

        /* Close with the implicit a0 = 1 constant:
         *   2^r = 1 + r * G(r) */
        __p = __p * __r + ffloat(1.0f, 0.0f);
        return __p;
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
    * This is the natural companion to __fp32mp2_exp2_kernel for the
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
    _CCCL_TRIVIAL_API fp32mp2_low __fp32mp2_exp10_kernel(fp32mp2_low __r) noexcept
    {
        using ffloat = fp32mp2_low;

        /* b_k = (ln 10)^k / k!,  ordered low -> high degree.
         * 6 low-degree ff entries (carry .lo()) + 7 high-degree float
         * entries (M = 7) consumed by poly_eval's float-only inner loop. */
        constexpr ffloat __exp10_c[13] = {
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
        ffloat __p = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 7>(__r, __exp10_c);

        /* Close with the implicit b0 = 1:  10^r = 1 + r * G(r) */
        __p = __p * __r + ffloat(1.0f, 0.0f);
        return __p;
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
    * trick used by __fp32mp2_erfc.
    * --------------------------------------------------------------------
    */
    _CCCL_TRIVIAL_API fp32mp2_low __fp32mp2_ldexp2_internal(fp32mp2_low __p, int __n) noexcept
    {
        const int __k    = __n >> 1;        /* floor-div-by-2; signed shift on negative n */
        int       __ek1  = 127 + __k;
        int       __ek2  = 127 + (__n - __k);
        /* Clamp split exponents into representable normal-float biased
         * range [1, 254].  When |n| is large, one half can saturate to
         * the denormal floor / overflow ceiling -- handled by the chained
         * multiply, which then sees a fully-collapsed factor at the
         * other end. */
        if (__ek1 < 1)   __ek1 = 1;
        if (__ek2 < 1)   __ek2 = 1;
        if (__ek1 > 254) __ek1 = 254;
        if (__ek2 > 254) __ek2 = 254;
        const float __scale_a = __fpmp_internal_bit_cast<float>(static_cast<unsigned>(__ek1) << 23);
        const float __scale_b = __fpmp_internal_bit_cast<float>(static_cast<unsigned>(__ek2) << 23);
        return __p * __scale_a * __scale_b;
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
    * __fp32mp2_ldexp2_internal for the bounded-n exp2/exp10 callers):
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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_ldexp(const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  int          __n,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        if constexpr (::cuda::std::is_same_v<_FpType, float>)
        {
            using ffloat = fp32mp2_low;

            /* Saturate |n| to +-300.  Any |n| larger than this is provably
             * monotone in the final result (overflow or underflow) for
             * every finite fp32 input, so clamping does not lose info. */
            if (__n >  300) __n =  300;
            if (__n < -300) __n = -300;

            const int __k    = __n / 3;
            int       __ek1  = 127 + __k;
            int       __ek2  = 127 + __k;
            int       __ek3  = 127 + (__n - 2 * __k);
            if (__ek1 < 1)   __ek1 = 1;
            if (__ek2 < 1)   __ek2 = 1;
            if (__ek3 < 1)   __ek3 = 1;
            if (__ek1 > 254) __ek1 = 254;
            if (__ek2 > 254) __ek2 = 254;
            if (__ek3 > 254) __ek3 = 254;
            const float __s1 = __fpmp_internal_bit_cast<float>(static_cast<unsigned>(__ek1) << 23);
            const float __s2 = __fpmp_internal_bit_cast<float>(static_cast<unsigned>(__ek2) << 23);
            const float __s3 = __fpmp_internal_bit_cast<float>(static_cast<unsigned>(__ek3) << 23);

            const ffloat __result = ffloat(__x_hi, __x_lo) * __s1 * __s2 * __s3;

            *__res_hi = __result.hi();
            *__res_lo = __result.lo();
        }
        else
        {
            /* fp64mp2 path: forward to libm `::ldexp` via the existing
             * double round-trip.  An explicit __fpmp2_ldexp<double>
             * specialization is also provided later for symmetry with
             * the rest of the fp64mp2 math surface; this else branch
             * exists so the primary template is well-formed for any
             * `FpType` and stays compilable in isolation. */
            using mp2_t = fpmp2<_FpType>;
            const double __r = ::ldexp(static_cast<double>(mp2_t(__x_hi, __x_lo)), __n);
            mp2_t __result(__r);
            *__res_hi = __result.hi();
            *__res_lo = __result.lo();
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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_scalbn(const _FpType __x_hi,
                                                   const _FpType __x_lo,
                                                   int          __n,
                                                   _FpType*      __res_hi,
                                                   _FpType*      __res_lo) noexcept
    {
        __fpmp2_ldexp<_FpType>(__x_hi, __x_lo, __n, __res_hi, __res_lo);
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
    _CCCL_TRIVIAL_API float __fp32mp2_scale2_scalar(float __v, int __s) noexcept
    {
        const int   __s1 = __s >> 1;            /* floor(s/2) */
        const int   __s2 = __s - __s1;
        const float __f1 = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>(127 + __s1) << 23);
        const float __f2 = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>(127 + __s2) << 23);
        return __v * __f1 * __f2;
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
    _CCCL_TRIVIAL_API void __fp32mp2_modf_decompose(float __hi, float __lo,
                                                            unsigned long long* __M, int* __E) noexcept
    {
        const uint32_t __hb = __fpmp_internal_bit_cast<uint32_t>(__hi);
        int __Eh;
        if ((__hb & 0x7F800000u) == 0u)
        {
            /* denormal hi (hi > 0): value = mant * 2^-149 */
            const uint32_t __mant = __hb & 0x007FFFFFu;
            const float    __fm   = static_cast<float>(__mant);
            const uint32_t __fmb  = __fpmp_internal_bit_cast<uint32_t>(__fm);
            __Eh = static_cast<int>((__fmb >> 23) & 0xFFu) - 127 - 149;
        }
        else
        {
            __Eh = static_cast<int>((__hb >> 23) & 0xFFu) - 127;
        }

        const int   __s   = 52 - __Eh;
        const float __shi = __fp32mp2_scale2_scalar(__hi, __s);   /* integer in [2^52, 2^53) */
        const float __slo = __fp32mp2_scale2_scalar(__lo, __s);   /* |slo| <= 2^28           */

        long long __m = static_cast<long long>(static_cast<unsigned long long>(__shi))
                    + static_cast<long long>(__fpmp_fp2int_rn(__slo));
        int __e = __Eh - 52;

        /* lo > 0 may push m just past 2^53; bring it back. */
        if ((static_cast<unsigned long long>(__m) >> 53) != 0ULL)
        {
            __m >>= 1;
            __e  += 1;
        }
        /* Off-by-one fix: when hi is an exact power of two and lo < 0 the
         * true value sits just below 2^Eh, so m lands just under 2^52.
         * One left shift renormalizes it back into [2^52, 2^53). */
        if (__m != 0 && (static_cast<unsigned long long>(__m) >> 52) == 0ULL)
        {
            __m <<= 1;
            __e  -= 1;
        }

        *__M = static_cast<unsigned long long>(__m);
        *__E = __e;
    }

    /* Build a renormalized fp32mp2 from  (neg ? -1 : 1) * mag * 2^E.
     * mag may carry up to 53 significant bits; it is first rounded
     * (round-half-to-even) down to the 48 bits an fp32mp2 can hold, then split
     * into two <= 24-bit halves so each casts to float exactly. */
    _CCCL_TRIVIAL_API void __fp32mp2_modf_reconstruct(unsigned long long __mag, int __E, bool __neg,
                                                              float* __res_hi, float* __res_lo) noexcept
    {
        if (__mag == 0ULL)
        {
            *__res_hi = __neg ? -0.0f : 0.0f;
            *__res_lo = 0.0f;
            return;
        }

        /* Round mag down to <= 48 significant bits. */
        int __extra = 0;
        for (unsigned long long __t = __mag; __t >= (1ULL << 48); __t >>= 1) { ++__extra; }
        if (__extra > 0)
        {
            const unsigned long long __half = 1ULL << (__extra - 1);
            const unsigned long long __frac = __mag & ((1ULL << __extra) - 1ULL);
            unsigned long long       __q    = __mag >> __extra;
            if (__frac > __half || (__frac == __half && (__q & 1ULL) != 0ULL)) { ++__q; }
            __mag = __q;
            __E  += __extra;
            if ((__mag >> 48) != 0ULL) { __mag >>= 1; ++__E; }   /* rounding carried out */
        }

        const unsigned __hipart = static_cast<unsigned>(__mag >> 24);           /* < 2^24 */
        const unsigned __lopart = static_cast<unsigned>(__mag & 0xFFFFFFULL);   /* < 2^24 */
        float __rhi = __fp32mp2_scale2_scalar(static_cast<float>(__hipart), __E + 24);
        float __rlo = __fp32mp2_scale2_scalar(static_cast<float>(__lopart), __E);
        if (__neg) { __rhi = -__rhi; __rlo = -__rlo; }

        float __lo;
        const float __hi = __fpmp_two_sum(__rhi, __rlo, &__lo);   /* exact, no magnitude assumption */
        *__res_hi = __hi;
        *__res_lo = __lo;
    }

    /* Core reduction: assumes ax > ay > 0 (both finite, nonzero), inputs
     * given as positive renormalized (hi, lo) pairs.  Returns the fmod
     * remainder mantissa ia (< My), the divisor mantissa My, its
     * exponent Ey, and the low bits of the integer quotient
     * floor(ax/ay) in quo. */
    _CCCL_TRIVIAL_API void __fp32mp2_fmod_kernel(float __ax_hi, float __ax_lo,
                                                         float __ay_hi, float __ay_lo,
                                                         unsigned long long* __ia_out,
                                                         unsigned long long* __My_out,
                                                         int*                __Ey_out,
                                                         unsigned long long* __quo_out) noexcept
    {
        unsigned long long __Mx, __My;
        int __Ex, __Ey;
        __fp32mp2_modf_decompose(__ax_hi, __ax_lo, &__Mx, &__Ex);
        __fp32mp2_modf_decompose(__ay_hi, __ay_lo, &__My, &__Ey);

        int __D = __Ex - __Ey;                 /* >= 0 since ax > ay and both M in [2^52,2^53) */
        if (__D < 0) __D = 0;                /* defensive */

        unsigned long long __quo = __Mx / __My;
        unsigned long long __ia  = __Mx % __My;
        int __remaining = __D;
        while (__remaining > 0)
        {
            const int                __s   = (__remaining < 11) ? __remaining : 11;
            const unsigned long long __num = __ia << __s;          /* ia < My < 2^53, so num < 2^64 */
            __quo = (__quo << __s) + (__num / __My);                   /* low bits of quotient (parity only) */
            __ia  = __num % __My;
            __remaining -= __s;
        }

        *__ia_out = __ia;
        *__My_out = __My;
        *__Ey_out = __Ey;
        *__quo_out = __quo;
    }

    /*
    * fmod(x, y): result has the sign of x and magnitude in [0, |y|).
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_fmod(const _FpType __x_hi, const _FpType __x_lo,
                                                const _FpType __y_hi, const _FpType __y_lo,
                                                _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_fmod is fp32mp2 only; fp64mp2 has its own specialization");


        /* (hi + lo) != (hi + lo) also catches a degenerate (+inf, -inf) limb
         * pair, which the fp128 reference widens to inf + (-inf) = NaN. */
        const bool  __x_nan = (__x_hi != __x_hi) || (__x_lo != __x_lo) || ((__x_hi + __x_lo) != (__x_hi + __x_lo));
        const bool  __y_nan = (__y_hi != __y_hi) || (__y_lo != __y_lo) || ((__y_hi + __y_lo) != (__y_hi + __y_lo));
        const float __axh   = (__x_hi < 0.0f) ? -__x_hi : __x_hi;
        const float __ayh   = (__y_hi < 0.0f) ? -__y_hi : __y_hi;
        const bool  __x_inf = (__axh == __builtin_huge_valf());
        const bool  __y_inf = (__ayh == __builtin_huge_valf());
        const bool  __y_zero = (__y_hi == 0.0f);

        if (__x_nan || __y_nan || __x_inf || __y_zero)
        {
            *__res_hi = __builtin_nanf(""); *__res_lo = __builtin_nanf(""); return;
        }
        if (__y_inf)                       /* fmod(finite, inf) = x */
        {
            *__res_hi = __x_hi; *__res_lo = __x_lo; return;
        }

        const float __axl = (__x_hi < 0.0f) ? -__x_lo : __x_lo;
        const float __ayl = (__y_hi < 0.0f) ? -__y_lo : __y_lo;

        int __c;
        if      (__axh != __ayh) __c = (__axh < __ayh) ? -1 : 1;
        else if (__axl != __ayl) __c = (__axl < __ayl) ? -1 : 1;
        else                 __c = 0;

        if (__c < 0)  { *__res_hi = __x_hi; *__res_lo = __x_lo; return; }                 /* |x| < |y| -> x   */
        if (__c == 0) { *__res_hi = (__x_hi < 0.0f) ? -0.0f : 0.0f; *__res_lo = 0.0f; return; }

        unsigned long long __ia, __My, __quo;
        int __Ey;
        __fp32mp2_fmod_kernel(__axh, __axl, __ayh, __ayl, &__ia, &__My, &__Ey, &__quo);
        __fp32mp2_modf_reconstruct(__ia, __Ey, (__x_hi < 0.0f), __res_hi, __res_lo);
    }

    /*
    * remainder(x, y): IEEE remainder, |result| <= |y|/2, round-to-nearest
    * with ties to even quotient.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_remainder(const _FpType __x_hi, const _FpType __x_lo,
                                                     const _FpType __y_hi, const _FpType __y_lo,
                                                     _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_remainder is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* (hi + lo) != (hi + lo) also catches a degenerate (+inf, -inf) limb
         * pair, which the fp128 reference widens to inf + (-inf) = NaN. */
        const bool  __x_nan = (__x_hi != __x_hi) || (__x_lo != __x_lo) || ((__x_hi + __x_lo) != (__x_hi + __x_lo));
        const bool  __y_nan = (__y_hi != __y_hi) || (__y_lo != __y_lo) || ((__y_hi + __y_lo) != (__y_hi + __y_lo));
        const float __axh   = (__x_hi < 0.0f) ? -__x_hi : __x_hi;
        const float __ayh   = (__y_hi < 0.0f) ? -__y_hi : __y_hi;
        const bool  __x_inf = (__axh == __builtin_huge_valf());
        const bool  __y_inf = (__ayh == __builtin_huge_valf());
        const bool  __y_zero = (__y_hi == 0.0f);
        const bool  __xneg  = (__x_hi < 0.0f);

        if (__x_nan || __y_nan || __x_inf || __y_zero)
        {
            *__res_hi = __builtin_nanf(""); *__res_lo = __builtin_nanf(""); return;
        }
        if (__y_inf)                       /* remainder(finite, inf) = x */
        {
            *__res_hi = __x_hi; *__res_lo = __x_lo; return;
        }

        const float __axl = (__x_hi < 0.0f) ? -__x_lo : __x_lo;
        const float __ayl = (__y_hi < 0.0f) ? -__y_lo : __y_lo;

        int __c;
        if      (__axh != __ayh) __c = (__axh < __ayh) ? -1 : 1;
        else if (__axl != __ayl) __c = (__axl < __ayl) ? -1 : 1;
        else                 __c = 0;

        if (__c == 0)                      /* |x| == |y| -> remainder 0 (sign of x) */
        {
            *__res_hi = __xneg ? -0.0f : 0.0f; *__res_lo = 0.0f; return;
        }

        if (__c < 0)
        {
            /* |x| < |y|: quotient is 0 or +-1.  Compare 2|x| against |y|. */
            const float __t_hi = 2.0f * __axh, __t_lo = 2.0f * __axl;   /* 2|x| exact */
            int __c2;
            if      (__t_hi != __ayh) __c2 = (__t_hi < __ayh) ? -1 : 1;
            else if (__t_lo != __ayl) __c2 = (__t_lo < __ayl) ? -1 : 1;
            else                  __c2 = 0;                        /* tie -> quotient 0 (even) */

            if (__c2 <= 0) { *__res_hi = __x_hi; *__res_lo = __x_lo; return; }   /* r = x */

            /* 2|x| > |y|: r = |x| - |y|  (negative in the |x| frame) */
            const ffloat __r = sub<fpmp2_accuracy::high>(ffloat(__axh, __axl), ffloat(__ayh, __ayl));
            float __rh = __r.hi(), __rl = __r.lo();
            if (__xneg) { __rh = -__rh; __rl = -__rl; }
            *__res_hi = __rh; *__res_lo = __rl; return;
        }

        /* |x| > |y|: full integer reduction, then round-to-nearest-even. */
        unsigned long long __ia, __My, __quo;
        int __Ey;
        __fp32mp2_fmod_kernel(__axh, __axl, __ayh, __ayl, &__ia, &__My, &__Ey, &__quo);

        const unsigned long long __two_ia = __ia << 1;
        const bool __round_up = (__two_ia > __My) || ((__two_ia == __My) && ((__quo & 1ULL) != 0ULL));

        unsigned long long __mag;
        bool __neg_xframe;
        if (__round_up) { __mag = __My - __ia; __neg_xframe = true;  }   /* r = (|x| mod |y|) - |y| < 0 */
        else          { __mag = __ia;      __neg_xframe = false; }

        __fp32mp2_modf_reconstruct(__mag, __Ey, static_cast<bool>(__neg_xframe ^ __xneg), __res_hi, __res_lo);
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
    *   inside __fpmp2_exp.  Two stacked reductions accumulate:
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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_exp2 (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_exp2 is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* NaN propagation: short-circuit so the lo limb propagates too. */
        if (__x_hi != __x_hi)
        {
            *__res_hi = __x_hi;
            *__res_lo = __x_hi;
            return;
        }

        /* Overflow / underflow shortcuts. */
        if (__x_hi >= 128.0f)
        {
            *__res_hi = __builtin_huge_valf();
            *__res_lo = 0.0f;
            return;
        }
        if (__x_hi <= -150.0f)
        {
            *__res_hi = 0.0f;
            *__res_lo = 0.0f;
            return;
        }

        /* Step 1: integer/fractional split directly in base-2 units. */
        const int    __n   = __fpmp_fp2int_rn(__x_hi);
        const ffloat __n_f = __fpmp_int2fp_rn<float>(__n);

        /* Step 2: r = x - n.  ffloat subtraction by an integer is exact
         * (n_f is representable in float for |n| <= 2^23, which our
         * overflow/underflow shortcuts guarantee). */
        const ffloat __r = ffloat(__x_hi, __x_lo) - __n_f;

        /* Step 3: 2^r via the dedicated base-2 Taylor kernel (no r * ln 2
         * detour, no internal natural-log reduction). */
        const ffloat __u = __fp32mp2_exp2_kernel(__r);

        /* Step 4: multiply by 2^n via the split-exponent helper. */
        const ffloat __result = __fp32mp2_ldexp2_internal(__u, __n);

        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_exp2

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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_exp10 (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_exp10 is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;
        using afloat = fp32mp2_high;

        if (__x_hi != __x_hi)
        {
            *__res_hi = __x_hi;
            *__res_lo = __x_hi;
            return;
        }
        if (__x_hi >= 39.0f)
        {
            *__res_hi = __builtin_huge_valf();
            *__res_lo = 0.0f;
            return;
        }
        if (__x_hi <= -46.0f)
        {
            *__res_hi = 0.0f;
            *__res_lo = 0.0f;
            return;
        }

        /* log2(10) ~= 3.32192809488736234787...  (fp32mp2 constant used
         * only for the *coarse* n estimate; precision floor of this
         * value is fine since we only need the integer part). */
        constexpr ffloat __log2_10(0x1.a934f0979a371p+1);

        /* log10(2) ~= 0.30102999566398119521  split into 3 fp32 chunks
         * (Cody-Waite); the sum C1 + C2 + C3 reproduces log10(2)
         * exactly in double.  Layout mirrors the trig pi/2 split that
         * already lives in this file. */
        constexpr float __C1 = 0x1.344136p-2f;   /* +0.30103001 */
        constexpr float __C2 = -0x1.ec10c0p-27f; /* -1.432e-08 */
        constexpr float __C3 = -0x1.000000p-54f; /*  ~-5.5e-17 */

        /* Step 1: coarse integer n = round(x * log2 10).
         * Uses an ordinary ff multiplication -- we only need the integer
         * part, so the lo limb of the product is discarded. */
        const ffloat __t_approx = ffloat(__x_hi, __x_lo) * __log2_10;
        const int    __n        = __fpmp_fp2int_rn(__t_approx.hi());
        const float  __n_f      = __fpmp_int2fp_rn<float>(__n);

        /* Step 2: Cody-Waite reduction  r' = x - n * log10(2)
         *   r' = (x_hi + x_lo) - n_f * (C1 + C2 + C3)
         * Computed via the same two_mult_fma + two_sum recipe used by
         * the trig kernel: every product / subtraction is captured as
         * an exact pair, then accumulated in fp32mp2_high so
         * the relative precision of r' is bounded by ulp(x_hi)*2^-23,
         * not by ulp(x*log2 10)*2^-23. */

        /* n_f * C1 = ph + pl  (exact pair) */
        float __pl;
        const float __ph = __fpmp_two_mult_fma(__n_f, __C1, &__pl);

        /* x_hi - ph = s + e  (exact pair) */
        float __e;
        const float __s = __fpmp_two_sum(__x_hi, -__ph, &__e);

        afloat __r_acc(__s, __e);
        __r_acc = __r_acc + afloat(-__pl);
        __r_acc = __r_acc + afloat(__x_lo);

        /* n_f * C2 = nC2_hi + nC2_lo  (exact pair) */
        float __nC2_lo;
        const float __nC2_hi = __fpmp_two_mult_fma(__n_f, __C2, &__nC2_lo);
        __r_acc = __r_acc - afloat(__nC2_hi, __nC2_lo);

        /* n_f * C3 is tiny (~10^-14 at the largest n we hit);
         * single-precision product is below the polynomial noise
         * floor but cheap to include for completeness. */
        __r_acc = __r_acc + afloat(__fpmp_mul_rn(__n_f, -__C3));

        /* Step 3: 10^r' via the dedicated base-10 Taylor kernel.
         * Hand off the accurate accumulator as fast ffloat -- the
         * polynomial cannot consume more than ff precision anyway. */
        const ffloat __r = ffloat(__r_acc.hi(), __r_acc.lo());
        const ffloat __u = __fp32mp2_exp10_kernel(__r);

        /* Step 4: scale by 2^n via the split-exponent helper. */
        const ffloat __result = __fp32mp2_ldexp2_internal(__u, __n);

        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_exp10

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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_expm1 (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_expm1 is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* NaN propagation: any NaN component -> NaN result. */
        if (__x_hi != __x_hi || __x_lo != __x_lo)
        {
            const float __nan_val = __x_hi + __x_lo;
            *__res_hi = __nan_val;
            *__res_lo = __nan_val;
            return;
        }

        /* +inf input -> +inf. */
        if (__x_hi == __builtin_huge_valf())
        {
            *__res_hi = __builtin_huge_valf();
            *__res_lo = 0.0f;
            return;
        }

        /* -inf input -> -1 exactly. */
        if (__x_hi == -__builtin_huge_valf())
        {
            *__res_hi = -1.0f;
            *__res_lo = 0.0f;
            return;
        }

        const float __abs_hi = (__x_hi < 0.0f) ? -__x_hi : __x_hi;
        constexpr float __EXPM1_BRANCH_POINT = 0.5f;
        if (__abs_hi < __EXPM1_BRANCH_POINT)
        {
            /* P(x) = sum_{k>=0} x^k / (k+2)!,
             *   P[0] = 1/2!, P[1] = 1/3!, ..., P[11] = 1/13!.
             * Layout for poly_eval<horner_mixed, M=4>: bottom 8 entries
             * are full ff (their contributions stay above fp32mp2 ulp at
             * the branch point), top 4 entries are plain float (.lo == 0
             * by construction; their contributions sit below 0.5 ulp). */
            constexpr ffloat __expm1_poly_c[12] = {
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

            ffloat __x      (__x_hi, __x_lo);
            ffloat __x2     = __x * __x;
            ffloat __pval      = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 4>(__x, __expm1_poly_c);
            ffloat __result = renormalize(__x + __x2 * __pval);
            *__res_hi = __result.hi();
            *__res_lo = __result.lo();
            return;
        }

        /* Large-|x| branch: compute exp(x) and subtract 1.0 with the
         * accurate sub variant so the lo-limb captures the cancellation
         * residual that a plain fast 2-sum would quantise away.  For
         * |x| >= 1/2 the leading term exp(x) is at least 0.6 away from 1
         * (positive side) or 0.6 below 1 (negative side), so the
         * subtraction never loses more than ~1 bit. */
        float __e_hi, __e_lo;
        __fpmp2_exp<float>(__x_hi, __x_lo, &__e_hi, &__e_lo);

        /* exp() may already produce +inf for very large x; pass that
         * through without quietly turning it into NaN via inf - 1. */
        if (__e_hi == __builtin_huge_valf())
        {
            *__res_hi = __builtin_huge_valf();
            *__res_lo = 0.0f;
            return;
        }

        ffloat __result = sub<fpmp2_accuracy::high>(ffloat(__e_hi, __e_lo), ffloat(1.0f));
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_expm1

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
    * `tmp = fma(tmp, prod.x, tmp)`.  Our dedicated `__fpmp2_exp`
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
    * result truly overflows fp32 - and the dedicated `__fpmp2_exp`
    * handles +-Inf input via its existing saturation paths.
    * --------------------------------------------------------------------
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_pow (const _FpType __a_hi,
                                                const _FpType __a_lo,
                                                const _FpType __b_hi,
                                                const _FpType __b_lo,
                                                _FpType*      __res_hi,
                                                _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_pow is fp32mp2 only; fp64mp2 has its own specialization");

        /* ---- (1,2) pow(1,b) = pow(a,0) = 1, highest priority per IEEE 754-2008 ---- */
        if ((__a_hi == 1.0f && __a_lo == 0.0f) || (__b_hi == 0.0f && __b_lo == 0.0f))
        {
            *__res_hi = 1.0f; *__res_lo = 0.0f;
            return;
        }

        /* ---- (3) NaN propagation ---- */
        if ((__a_hi != __a_hi) || (__b_hi != __b_hi))
        {
            *__res_hi = __a_hi + __b_hi; *__res_lo = 0.0f;
            return;
        }

        /* ---- (4) integer / odd-integer b detection ---- */
        bool __b_is_int     = false;
        bool __b_is_odd_int = false;
        {
            const float __b_trunc  = __fpmp_internal_trunc<float>(__b_hi);
            if (__b_lo == 0.0f && __b_trunc == __b_hi)
            {
                __b_is_int = true;
                const float __abs_b_hi = __b_hi < 0.0f ? -__b_hi : __b_hi;
                if (__abs_b_hi < 0x1.0p+24f)   /* parity only meaningful below 2^24 */
                    __b_is_odd_int = (static_cast<int32_t>(__b_hi) & 1) != 0;
            }
        }

        const bool  __a_is_neg = (__a_hi < 0.0f) || (__a_hi == 0.0f && __a_lo < 0.0f);
        const float __abs_a_hi = __a_is_neg ? -__a_hi : __a_hi;
        const float __abs_a_lo = __a_is_neg ? -__a_lo : __a_lo;

        /* ---- (5) a == 0 ---- */
        if (__abs_a_hi == 0.0f && __abs_a_lo == 0.0f)
        {
            if (__b_hi < 0.0f)
            {
                const float __sign = (__a_is_neg && __b_is_odd_int) ? -1.0f : 1.0f;
                *__res_hi = __sign * __builtin_huge_valf();
            }
            else
            {
                *__res_hi = (__a_is_neg && __b_is_odd_int) ? -0.0f : 0.0f;
            }
            *__res_lo = 0.0f;
            return;
        }

        /* ---- (6) negative base with non-integer exponent ---- */
        if (__a_is_neg && !__b_is_int)
        {
            *__res_hi = __builtin_nanf(""); *__res_lo = 0.0f;
            return;
        }

        /* ---- (7) |a| = Inf ---- */
        if (__abs_a_hi == __builtin_huge_valf())
        {
            const float __sign = (__a_is_neg && __b_is_odd_int) ? -1.0f : 1.0f;
            *__res_hi = (__b_hi > 0.0f) ? __sign * __builtin_huge_valf() : __sign * 0.0f;
            *__res_lo = 0.0f;
            return;
        }

        /* ---- (8) |b| = Inf ---- */
        if (__b_hi == __builtin_huge_valf() || __b_hi == -__builtin_huge_valf())
        {
            /* IEEE 754: pow(-1, +-Inf) = 1.  pow(+1, ...) already handled at (1). */
            if (__abs_a_hi == 1.0f && __abs_a_lo == 0.0f)
            {
                *__res_hi = 1.0f; *__res_lo = 0.0f;
                return;
            }
            const bool __abs_a_gt_one = (__abs_a_hi > 1.0f) ||
                                      (__abs_a_hi == 1.0f && __abs_a_lo > 0.0f);
            *__res_hi = ((__b_hi > 0.0f) == __abs_a_gt_one) ? __builtin_huge_valf() : 0.0f;
            *__res_lo = 0.0f;
            return;
        }

        /* ---- (9) main path: exp(b * log(|a|)) ---- */
        float __loga_hi, __loga_lo;
        __fpmp2_log<float>(__abs_a_hi, __abs_a_lo, &__loga_hi, &__loga_lo);

        float __prod_hi, __prod_lo;
        __fpmp2_mul<float>(__b_hi, __b_lo, __loga_hi, __loga_lo, &__prod_hi, &__prod_lo);

        float __t_hi, __t_lo;
        __fpmp2_exp<float>(__prod_hi, __prod_lo, &__t_hi, &__t_lo);

        /* ---- sign fixup for a < 0 with odd integer b ---- */
        if (__a_is_neg && __b_is_odd_int) { __t_hi = -__t_hi; __t_lo = -__t_lo; }

        *__res_hi = __t_hi; *__res_lo = __t_lo;
    } // __fpmp2_pow

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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_cbrt (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        using ffloat = fp32mp2_low;

        // 1/3 in single precision (round-to-nearest); the exact 1/3 is not
        // representable in any binary float, but ulp(1/3) is well below the
        // accuracy of the SFU lg2/ex2 pair so this is sufficient.
        constexpr float __third_f = 0x1.555556p-2f;

        const uint32_t __xbits   = __fpmp_internal_bit_cast<uint32_t>(__x_hi);
        const uint32_t __absbits = __xbits & 0x7FFFFFFFu;
        const uint32_t __signbit = __xbits & 0x80000000u;

        /* Special inputs: +-0, +-Inf, NaN.  cbrt(x) returns x for these,
         * matching libdevice's behaviour (the libdevice routine fixes
         * them up via cmpsel against (a + a) at the end; an explicit
         * early-out is cleaner here). */
        if (__absbits == 0u || __absbits >= 0x7F800000u)
        {
            *__res_hi = __x_hi;
            *__res_lo = (__absbits >= 0x7F800000u) ? 0.0f : __x_lo;
            return;
        }

        /* Operate on |x|; sign of x_lo follows the sign of x_hi. */
        float __ax_hi = __fpmp_internal_bit_cast<float>(__absbits);
        float __ax_lo = (__signbit != 0u) ? -__x_lo : __x_lo;

        /* Denormal pre-scaling: multiply by 2^24 (chosen so the offset is
         * divisible by 3 -> denorm_div3 = 8 unscales the result later). */
        int __denorm_div3 = 0;
        uint32_t __scaled_absbits = __absbits;
        if ((__absbits >> 23) == 0u)
        {
            constexpr float __scale_up = 0x1.0p24f;
            __ax_hi  *= __scale_up;
            __ax_lo  *= __scale_up;
            __denorm_div3    = 8;
            __scaled_absbits = __fpmp_internal_bit_cast<uint32_t>(__ax_hi);
        }

        /* Reduce: ax = r * 2^(3 * nexpo), with nexpo chosen so r ~= 1. */
        const int __expo  = static_cast<int>(__scaled_absbits >> 23);
        const int __nexpo = __fpmp_fp2int_rn(__third_f * static_cast<float>(__expo - 126));

        /* r_hi = ax_hi * 2^(-3*nexpo): exact, by exponent-field subtraction.
         * (The mantissa is untouched; only the biased exponent shifts.)
         * Use multiplication by 2^23 instead of left-shift to avoid UB
         * when (3 * nexpo) is negative. */
        constexpr int     __EXP_SHIFT = 1 << 23;
        const     int     __delta_exp = 3 * __nexpo;
        const     int     __new_bits  = static_cast<int>(__scaled_absbits) - __delta_exp * __EXP_SHIFT;
        const     float   __r_hi      = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>(__new_bits));

        /* r_lo: scale by the same power of two via float multiply.  Split
         * 2^(-3*nexpo) into two normal-range factors: for x near max float
         * |3*nexpo| can reach ~129, which would give an invalid biased
         * exponent of -2 if applied as a single bit-cast.  Splitting keeps
         * each factor's biased exponent in the normal range (about
         * [62, 190]); the product stays exact for all valid inputs. */
        const int __half_pow  = -__delta_exp / 2;
        const int __rest_pow  = -__delta_exp - __half_pow;
        const float __scale_a = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>((127 + __half_pow) * __EXP_SHIFT));
        const float __scale_b = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>((127 + __rest_pow) * __EXP_SHIFT));
        const float __r_lo    = (__ax_lo * __scale_a) * __scale_b;

        /* Initial cbrt approximation via the SFU lg2/ex2 pair (~23 bits). */
        const float __s = __fpmp_fast_exp2(__third_f * __fpmp_fast_log2(__r_hi));

        /* Halley refinement in fp32mp2:  t_new = t + t * (r - t^3) / (2 t^3 + r).
         *
         * The catastrophic cancellation in (r - t^3) is handled by an
         * accurate fma:  fma<accurate>(-t^2, t, r) computes r - t^2 * t
         * with a single rounding error followed by an exact correction,
         * preserving the small difference that drives the iteration.
         */
        const ffloat __r(__r_hi, __r_lo);
        const ffloat __t(__s);

        const ffloat __t2 = __t * __t;                        // t^2
        // numer = r - t^3 (computed as fma(-t^2, t, r) with accurate ff fma)
        const ffloat __numer = fma<fpmp2_accuracy::high>(-__t2, __t, __r);
        // denom = 2 t^3 + r ~= 3 r, well-conditioned so fast add suffices
        const ffloat __t3    = __t2 * __t;
        const ffloat __denom = (__t3 + __t3) + __r;

        /* Single-precision reciprocal of denom.hi() is enough: the
         * correction u_corr ~ 2^-23 contributes t * u_corr ~ 2^-46 to
         * t_new -- exactly fp32mp2 precision. */
        const float  __inv_denom = __fpmp_rcp_rn(__denom.hi());
        const ffloat __u_corr    = __numer * __inv_denom;
        const ffloat __t_new     = __t + __t * __u_corr;

        /* Scale back by 2^(nexpo - denorm_div3) via an exact power-of-two
         * float multiply.  back_shift stays in a range that keeps the
         * scale factor a normal float for all valid float inputs
         * (biased exponent is always in [77, 170]). */
        const int   __back_shift = __nexpo - __denorm_div3;
        const float __scale_back = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>((127 + __back_shift) * __EXP_SHIFT));
        float __t_hi_back = __t_new.hi() * __scale_back;
        float __t_lo_back = __t_new.lo() * __scale_back;

        /* Restore sign (cbrt is an odd function). */
        if (__signbit != 0u)
        {
            __t_hi_back = -__t_hi_back;
            __t_lo_back = -__t_lo_back;
        }

        *__res_hi = __t_hi_back;
        *__res_lo = __t_lo_back;
    } // __fpmp2_cbrt

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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_rcbrt (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        using ffloat = fp32mp2_low;

        // 1/3 and 2/9 in single precision (round-to-nearest); ulp at this
        // scale is far below the SFU lg2/ex2 estimate's accuracy.
        constexpr float __third_f      = 0x1.555556p-2f;   // ~= 1/3
        constexpr float __two_ninths_f = 0x1.c71c72p-3f;   // ~= 2/9

        const uint32_t __xbits   = __fpmp_internal_bit_cast<uint32_t>(__x_hi);
        const uint32_t __absbits = __xbits & 0x7FFFFFFFu;
        const uint32_t __signbit = __xbits & 0x80000000u;

        /* Special inputs:
         *   +-0   -> +-Inf
         *   +-Inf -> +-0
         *   NaN  -> NaN  (propagated via x_hi)
         */
        if (__absbits == 0u || __absbits >= 0x7F800000u)
        {
            if (__absbits == 0u) {
                *__res_hi = __fpmp_internal_bit_cast<float>(__signbit | 0x7F800000u);
            } else if (__absbits == 0x7F800000u) {
                *__res_hi = __fpmp_internal_bit_cast<float>(__signbit);
            } else {
                *__res_hi = __x_hi;     // NaN
            }
            *__res_lo = 0.0f;
            return;
        }

        /* Operate on |x|; sign of x_lo follows the sign of x_hi. */
        float __ax_hi = __fpmp_internal_bit_cast<float>(__absbits);
        float __ax_lo = (__signbit != 0u) ? -__x_lo : __x_lo;

        /* Denormal pre-scaling: multiply by 2^24. */
        int __denorm_div3 = 0;
        uint32_t __scaled_absbits = __absbits;
        if ((__absbits >> 23) == 0u)
        {
            constexpr float __scale_up = 0x1.0p24f;
            __ax_hi  *= __scale_up;
            __ax_lo  *= __scale_up;
            __denorm_div3    = 8;
            __scaled_absbits = __fpmp_internal_bit_cast<uint32_t>(__ax_hi);
        }

        /* Reduce: ax = r * 2^(3 * nexpo), with nexpo chosen so r ~= 1. */
        const int __expo  = static_cast<int>(__scaled_absbits >> 23);
        const int __nexpo = __fpmp_fp2int_rn(__third_f * static_cast<float>(__expo - 126));

        /* r_hi = ax_hi * 2^(-3*nexpo): exact, by exponent-field subtraction.
         * Use multiplication by 2^23 instead of left-shift to avoid UB
         * when (3 * nexpo) is negative. */
        constexpr int     __EXP_SHIFT = 1 << 23;
        const     int     __delta_exp = 3 * __nexpo;
        const     int     __new_bits  = static_cast<int>(__scaled_absbits) - __delta_exp * __EXP_SHIFT;
        const     float   __r_hi      = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>(__new_bits));

        /* r_lo: scale by 2^(-3*nexpo) via float multiply.  Split into two
         * normal-range factors to keep each biased exponent in roughly
         * [62, 190] for all valid float inputs. */
        const int __half_pow  = -__delta_exp / 2;
        const int __rest_pow  = -__delta_exp - __half_pow;
        const float __scale_a = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>((127 + __half_pow) * __EXP_SHIFT));
        const float __scale_b = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>((127 + __rest_pow) * __EXP_SHIFT));
        const float __r_lo    = (__ax_lo * __scale_a) * __scale_b;

        /* Initial 1/cbrt approximation via the SFU lg2/ex2 pair (~23 bits). */
        const float __s = __fpmp_fast_exp2(-__third_f * __fpmp_fast_log2(__r_hi));

        /* Halley refinement in fp32mp2:  t_new = t * (1 + u/3 + (2/9) u^2)
         * with u = 1 - r * t^3.
         *
         * The cancellation in (1 - r * t^3) is the only sensitive step;
         * everything else (t^2, t^3, the Halley quadratic, the final
         * combination) is well conditioned in fast fp32mp2 arithmetic.
         */
        const ffloat __r(__r_hi, __r_lo);
        const ffloat __t(__s);

        const ffloat __t2 = __t * __t;                               // t^2
        const ffloat __t3 = __t2 * __t;                              // t^3

        // u = 1 - r*t^3 (accurate fma to preserve catastrophic cancellation)
        const ffloat __u  = fma<fpmp2_accuracy::high>(-__r, __t3, 1.0f);

        // Halley quadratic factor:  hf = 1/3 + (2/9) u   (no cancellation)
        const ffloat __hf = fma<fpmp2_accuracy::def>(__two_ninths_f, __u, __third_f);

        // delta = u * t * hf,  then  t_new = t + delta
        const ffloat __ut    = __u * __t;
        const ffloat __t_new = __t + __hf * __ut;

        /* Scale back by 2^(-nexpo + denorm_div3) via an exact power-of-two
         * float multiply.  back_shift stays in [-43, +49] for all valid
         * float inputs, so the biased exponent is always in [84, 176]. */
        const int   __back_shift = -__nexpo + __denorm_div3;
        const float __scale_back = __fpmp_internal_bit_cast<float>(static_cast<uint32_t>((127 + __back_shift) * __EXP_SHIFT));
        float __t_hi_back = __t_new.hi() * __scale_back;
        float __t_lo_back = __t_new.lo() * __scale_back;

        /* Restore sign (rcbrt is an odd function). */
        if (__signbit != 0u)
        {
            __t_hi_back = -__t_hi_back;
            __t_lo_back = -__t_lo_back;
        }

        *__res_hi = __t_hi_back;
        *__res_lo = __t_lo_back;
    } // __fpmp2_rcbrt

    /*
    * ============================================================================
    * Trigonometric functions: sin, cos, sincos (fp32mp2) - dedicated
    * ============================================================================
    * Algorithm:
    *   1. Argument reduction: x = n*(pi/2) + r, |r| <= pi/4
    *      - Tiny (|x| < pi/4): no reduction
    *      - Fast (|x| < 2^20): Cody-Waite with exact error tracking
    *        via two_mult_fma + two_sum (3-piece pi/2, ~70 bits)
    *      - Large (|x| >= 2^20): controlled by _CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK
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

#if (_CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK == 0)

    /*
    * Payne-Hanek stage 1: compute |a| * (2/pi) via integer arithmetic,
    * extract 2-bit quadrant and 62-bit unsigned fraction in [0, 1).
    * Does NOT apply the >0.5 adjustment -- caller handles that.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_fpmp2_ph_frac(
        _FpType __a_hi, unsigned* __q_out, uint32_t* __frac_hi, uint32_t* __frac_lo) noexcept
    {

        constexpr unsigned int __i2opi[] = 
        {
            0x3c439041U, 0xdb629599U, 0xf534ddc0U,
            0xfc2757d1U, 0x4e441529U, 0xa2f9836eU,
        };

        uint32_t __ia = __fpmp_internal_bit_cast<uint32_t>(__a_hi);
        uint32_t __result[7];
        uint32_t __hi, __lo;
        int __iq;

        int __e = (int)((__ia >> 23U) & 0xFFU) - 128;
        __ia = (__ia << 8U) | 0x80000000U;
        __hi = 0;

        for (__iq = 0; __iq < 6; __iq++) 
        {
            uint64_t __p = (uint64_t)__i2opi[__iq] * __ia + __hi;
            __result[__iq] = (uint32_t)__p;
            __hi = (uint32_t)(__p >> 32);
        }
        __result[__iq] = __hi;

        /* Extract the window containing quadrant + fraction bits.
         * For e >= 0 (|a| >= 2): standard extraction with left shift.
         * For e < 0 (|a| < 2): extraction from idx=4 with right shift
         *   to handle small inputs without extending the table.
         */
        uint32_t __lo2;
        if (__e >= 0) 
        {
            uint32_t __ue = (uint32_t)__e;
            uint32_t __idx = 4U - (__ue >> 5U);
            __ue = __ue & 31U;
            __hi = __result[__idx + 2];
            __lo = __result[__idx + 1];
            __lo2 = (__idx > 0) ? __result[__idx] : 0U;
            if (__ue != 0U) 
            {
                uint32_t __q = 32U - __ue;
                __hi = (__hi << __ue) + (__lo >> __q);
                __lo = (__lo << __ue) + (__lo2 >> __q);
            }
        } 
        else 
        {
            int __r = -__e;
            __hi = __result[6];
            __lo = __result[5];
            __lo2 = __result[4];
            if (__r < 32) 
            {
                uint32_t __q = (uint32_t)(32 - __r);
                uint32_t __ur = (uint32_t)__r;
                __hi = __hi >> __ur;
                __lo = (__lo >> __ur) | (__result[6] << __q);
            } 
            else 
            {
                __hi = 0;
                __lo = __result[6];
            }
        }

        *__q_out   = __hi >> 30U;
        *__frac_hi = (__hi << 2U) + (__lo >> 30U);
        *__frac_lo = (__lo << 2U);
    }

    /*
    * Payne-Hanek stage 2: convert a 64-bit unsigned fraction in [0, 0.5)
    * (after the >0.5 adjustment) to an fp32mp2 angle by multiplying
    * by pi/2 using 64 bits of pi/4.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_fpmp2_frac_to_angle(
        uint32_t __hi, uint32_t __lo, uint32_t __s,
        _FpType* __r_hi, _FpType* __r_lo) noexcept
    {

        /* Normalize: shift so MSB of hi is 1.
         * Handle hi == 0 separately to avoid shift-by-32 UB.
         */
        #ifdef __CUDA_ARCH__
        uint32_t __lz = __clz((int)__hi);
        #else
        uint32_t __lz = (__hi == 0U) ? 32U : (uint32_t)__builtin_clz(__hi);
        #endif

        if (__lz >= 32U) 
        {
            __lz += (__lo == 0U) ? 0U :
            #ifdef __CUDA_ARCH__
                (uint32_t)__clz((int)__lo);
            #else
                (uint32_t)__builtin_clz(__lo);
            #endif
            __hi = __lo; __lo = 0U;
            uint32_t __shift = __lz - 32U;
            if (__shift != 0U) { __hi <<= __shift; }
        } 
        else if (__lz != 0U) 
        {
            __hi = (__hi << __lz) | (__lo >> (32U - __lz));
            __lo = __lo << __lz;
        }

        /* Multiply by pi/2 using 64 bits of pi/4.
         * pi/4 = 0x0.C90FDAA2_2168C234...
         * The *2 (pi/4 -> pi/2) is in biased_exp = 127 - lz.
         */
        constexpr uint32_t __PIO4_HI32 = 0xC90FDAA2U;
        constexpr uint32_t __PIO4_LO32 = 0x2168C234U;

        uint64_t __p_hh = (uint64_t)__hi * __PIO4_HI32;
        uint64_t __p_hl = (uint64_t)__hi * __PIO4_LO32;
        uint64_t __p_lh = (uint64_t)__lo * __PIO4_HI32;

        uint64_t __combined = __p_hh + (__p_hl >> 32) + (__p_lh >> 32);
        uint32_t __rhi = (uint32_t)(__combined >> 32);
        uint32_t __rlo = (uint32_t)__combined;

        if ((int32_t)__rhi > 0) 
        {
            __rhi = (__rhi << 1) | (__rlo >> 31);
            __rlo = __rlo << 1;
            __lz++;
        }

        /* Convert to fp32mp2 */
        uint32_t __biased_exp = 127U - __lz;
        uint32_t __f1_bits = __s | (__biased_exp << 23) | ((__rhi >> 8) & 0x7FFFFFU);

        uint32_t __rem       = (__rhi << 24) | (__rlo >> 8);
        uint32_t __rem_extra = __rlo << 24;

        if (__rem == 0U) 
        {
            *__r_hi = __fpmp_internal_bit_cast<_FpType>(__f1_bits);
            *__r_lo = _FpType(0);
            return;
        }

        #ifdef __CUDA_ARCH__
        uint32_t __rlz = __clz((int)__rem);
        #else
        uint32_t __rlz = (uint32_t)__builtin_clz(__rem);
        #endif

        uint32_t __rem_norm = (__rlz > 0U)
            ? ((__rem << __rlz) | (__rem_extra >> (32U - __rlz)))
            : __rem;

        int __biased_exp2 = (int)__biased_exp - 24 - (int)__rlz;
        if (__biased_exp2 < 1) 
        {
            *__r_hi = __fpmp_internal_bit_cast<_FpType>(__f1_bits);
            *__r_lo = _FpType(0);
        } 
        else 
        {
            uint32_t __f2_bits = __s | ((uint32_t)__biased_exp2 << 23)
                                 | ((__rem_norm >> 8) & 0x7FFFFFU);
            *__r_hi = __fpmp_internal_bit_cast<_FpType>(__f1_bits);
            *__r_lo = __fpmp_internal_bit_cast<_FpType>(__f2_bits);
        }
    }

#endif /* _CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK == 0 */

    /*
    * Trigonometric argument reduction for fp32mp2.
    * Returns quadrant (mod 4) and reduced argument r  in  [-pi/4, pi/4].
    *
    * Three paths:
    *   Tiny:  |x| < pi/4 -> no reduction
    *   Fast:  |x| < 2^20 -> Cody-Waite with exact error tracking
    *   Large: |x| >= 2^20 -> Payne-Hanek (integer 2/pi table)
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_fpmp2_trig_reduction(
        _FpType __x_hi, _FpType __x_lo,
        int* __quadrant, _FpType* __r_hi, _FpType* __r_lo) noexcept
    {
        using afloat = fp32mp2_high;

        _FpType __abs_hi = (__x_hi < _FpType(0)) ? -__x_hi : __x_hi;
        uint32_t __abs_bits = __fpmp_internal_bit_cast<uint32_t>(__abs_hi);

        /* No reduction for |x| < pi/4 */
        if (__abs_bits < 0x3F490FDBU) 
        {
            *__quadrant = 0;
            *__r_hi = __x_hi;
            *__r_lo = __x_lo;
            return;
        }

        /* Inf / NaN -> return NaN, quadrant 0 */
        if (__abs_bits >= 0x7F800000U) 
        {
            *__quadrant = 0;
            *__r_hi = __x_hi - __x_hi;
            *__r_lo = _FpType(0);
            return;
        }

        if (__abs_bits < 0x49800000U) 
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
            constexpr _FpType __C1 = _FpType(1.5707962512969971e+000);
            constexpr _FpType __C2 = _FpType(7.5497894158615964e-008);
            constexpr _FpType __C3 = _FpType(5.3903029534742384e-015);

            int __n = __fpmp_fp2int_rn(__x_hi * _FpType(0x1.45f306p-1f));
            _FpType __n_f = __fpmp_int2fp_rn<_FpType>(__n);

            /* Exact product n*C1 = ph + pl */
            _FpType __pl;
            _FpType __ph = __fpmp_two_mult_fma(__n_f, __C1, &__pl);

            /* Exact subtraction x_hi - ph = s + e */
            _FpType __e;
            _FpType __s = __fpmp_two_sum(__x_hi, -__ph, &__e);

            /* Build result as fp32mp2_high from exact (s, e),
             * then accumulate corrections with full precision.
             */
            afloat __result(__s, __e);
            __result = __result + afloat(-__pl);
            __result = __result + afloat(__x_lo);

            /* Exact product n*C2 = nC2_hi + nC2_lo via two_mult_fma */
            _FpType __nC2_lo;
            _FpType __nC2_hi = __fpmp_two_mult_fma(__n_f, __C2, &__nC2_lo);
            __result = __result - afloat(__nC2_hi, __nC2_lo);

            /* n*C3 is tiny (~10^-11), single-precision product suffices */
            __result = __result + afloat(__fpmp_mul_rn(__n_f, -__C3));

            *__quadrant = __n;
            *__r_hi = __result.hi();
            *__r_lo = __result.lo();
        } 
        else 
        {
            /* -- Slow path: |x_hi| >= 2^20 -- */

#if (_CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK == 0)
            /* Payne-Hanek: combine x_hi and x_lo 2/pi fractions in
             * 64-bit fixed-point BEFORE the pi/2 multiply to avoid
             * precision loss from floating-point cancellation.
             */
            uint32_t __fhi, __flo;
            unsigned __q_hi;
            __internal_fpmp2_ph_frac(__x_hi, &__q_hi, &__fhi, &__flo);

            uint32_t __x_hi_sign = __fpmp_internal_bit_cast<uint32_t>(__x_hi) & 0x80000000U;
            int __q = (int)__q_hi;

            /* Add x_lo contribution in fixed-point.
             * |x_lo| <= |x_hi|*2^-24 can still span many quadrants,
             * and even small |x_lo| can dominate the fraction when
             * the result angle is near zero.  Handle ALL non-zero x_lo.
             */
            if (__x_lo != _FpType(0)) 
            {
                _FpType __abs_lo = (__x_lo < _FpType(0)) ? -__x_lo : __x_lo;
                uint32_t __abs_lo_bits = __fpmp_internal_bit_cast<uint32_t>(__abs_lo);
                bool __same_sign = (__x_lo > _FpType(0)) == (__x_hi > _FpType(0));

                uint32_t __fhi2 = 0, __flo2 = 0;
                unsigned __q_lo = 0;

                if (__abs_lo_bits >= 0x00800000U) 
                {
                    __internal_fpmp2_ph_frac(__abs_lo, &__q_lo, &__fhi2, &__flo2);
                }

                uint64_t __f1 = ((uint64_t)__fhi << 32) | __flo;
                uint64_t __f2 = ((uint64_t)__fhi2 << 32) | __flo2;

                if (__same_sign) 
                {
                    __q += (int)__q_lo;
                    uint64_t __sum = __f1 + __f2;
                    if (__sum < __f1) __q++;
                    __fhi = (uint32_t)(__sum >> 32);
                    __flo = (uint32_t)__sum;
                } 
                else 
                {
                    __q -= (int)__q_lo;
                    if (__f1 >= __f2) 
                    {
                        __f1 -= __f2;
                    } 
                    else 
                    {
                        __f1 = 0ULL - (__f2 - __f1);
                        __q--;
                    }
                    __fhi = (uint32_t)(__f1 >> 32);
                    __flo = (uint32_t)__f1;
                }
            }

            uint32_t __top_bit = __fhi >> 31U;
            __q += __top_bit;
            if (__x_hi_sign != 0U) __q = 0U - (unsigned)__q;

            if (__top_bit != 0U) 
            {
                __fhi = ~__fhi;
                __flo = ~__flo;
                __x_hi_sign ^= 0x80000000U;
            }

            if (__fhi == 0U && __flo == 0U) 
            {
                *__quadrant = (int)__q;
                *__r_hi = _FpType(0);
                *__r_lo = _FpType(0);
                return;
            }

            *__quadrant = (int)__q;
            __internal_fpmp2_frac_to_angle(__fhi, __flo, __x_hi_sign, __r_hi, __r_lo);
#endif /* _CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK */
        }
    }

    /*
    * Sin kernel: evaluate sin(x) for |x| <= pi/4 using fp32mp2 Taylor series.
    * sin(x) = x + x^3*Q(x^2), Q(u) = Sum Taylor coefficients from -1/3! to -1/15!.
    * Upper terms (s7..s4) in single precision, lower terms (s3..s1) in fp32mp2.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_fpmp2_sin_kernel(
        _FpType __x_hi, _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using ffloat = fp32mp2_low;

        constexpr ffloat __s1(-1.6666666666666666e-01);
        constexpr ffloat __s2( 8.3333333333333333e-03);
        constexpr ffloat __s3(-1.9841269841269841e-04);
        constexpr float  __s4( 2.7557319223985893e-06f);
        constexpr float  __s5(-2.5052108385441719e-08f);
        constexpr float  __s6( 1.6059043836821615e-10f);
        constexpr float  __s7(-7.6471637318198165e-13f);

        ffloat __x(__x_hi, __x_lo);
        ffloat __x2 = __x * __x;
        float  __x2f = __x2.hi();

        float __qf = __s7;
        __qf = __fpmp_fma_rn(__qf, __x2f, __s6);
        __qf = __fpmp_fma_rn(__qf, __x2f, __s5);
        __qf = __fpmp_fma_rn(__qf, __x2f, __s4);

        ffloat __q = __qf * __x2 + __s3;
        __q = __q * __x2 + __s2;
        __q = __q * __x2 + __s1;

        ffloat __result = renormalize(__q * __x2 * __x + __x);
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    }

    /*
    * Cos kernel: evaluate cos(x) for |x| <= pi/4 using fp32mp2 Taylor series.
    * cos(x) = 1 + x^2*Q(x^2), Q(u) = Sum Taylor coefficients from -1/2! to 1/16!.
    * Upper terms (c8..c4) in single precision, lower terms (c3..c1) in fp32mp2.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_fpmp2_cos_kernel(
        _FpType __x_hi, _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using ffloat = fp32mp2_low;

        constexpr ffloat __c1(-5.0000000000000000e-01);
        constexpr ffloat __c2( 4.1666666666666667e-02);
        constexpr ffloat __c3(-1.3888888888888889e-03);
        constexpr float  __c4( 2.4801587301587302e-05f);
        constexpr float  __c5(-2.7557319223985893e-07f);
        constexpr float  __c6( 2.0876756987868099e-09f);
        constexpr float  __c7(-1.1470745597729725e-11f);
        constexpr float  __c8( 4.7794773323873853e-14f);

        ffloat __x(__x_hi, __x_lo);
        ffloat __x2 = __x * __x;
        float  __x2f = __x2.hi();

        float __qf = __c8;
        __qf = __fpmp_fma_rn(__qf, __x2f, __c7);
        __qf = __fpmp_fma_rn(__qf, __x2f, __c6);
        __qf = __fpmp_fma_rn(__qf, __x2f, __c5);
        __qf = __fpmp_fma_rn(__qf, __x2f, __c4);

        ffloat __q = __qf * __x2 + __c3;
        __q = __q * __x2 + __c2;
        __q = __q * __x2 + __c1;

        ffloat __result = renormalize(__q * __x2 + ffloat(_FpType(1)));
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    }

    /*
    * sincos for fp32mp2: compute sin(x) and cos(x) simultaneously.
    * Shared argument reduction, separate sin/cos kernels on [-pi/4, pi/4],
    * quadrant-based swap and sign adjustment (matching libdevice structure).
    *
    * When _CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK == 1, arguments with |x| >= 2^20
    * fall back to system fp64 sin/cos (avoids the Payne-Hanek code).
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_sincos(
        const _FpType __x_hi, const _FpType __x_lo,
        _FpType* __sin_hi, _FpType* __sin_lo,
        _FpType* __cos_hi, _FpType* __cos_lo) noexcept
    {
#if (_CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK == 1)
        _FpType __abs_hi = (__x_hi < _FpType(0)) ? -__x_hi : __x_hi;
        uint32_t __abs_bits = __fpmp_internal_bit_cast<uint32_t>(__abs_hi);
        if (__abs_bits >= 0x49800000U) 
        {
            using mp2_t = fpmp2<_FpType>;
            double __xd = static_cast<double>(mp2_t(__x_hi, __x_lo));
            double __sd = ::sin(__xd), __cd = ::cos(__xd);
            /* Split each fp64 result into (hi, lo) via the fp32mp2(double)
             * constructor -- casting to FpType first would drop the lo bits
             * and silently cap precision at ~24 bits instead of ~46.
             */
            mp2_t s_mp(__sd);
            mp2_t c_mp(__cd);
            *__sin_hi = s_mp.hi(); *__sin_lo = s_mp.lo();
            *__cos_hi = c_mp.hi(); *__cos_lo = c_mp.lo();
            return;
        }
#endif

        int __quadrant;
        _FpType __r_hi, __r_lo;
        __internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo);

        _FpType __s_hi, __s_lo, __c_hi, __c_lo;
        __internal_fpmp2_sin_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo);
        __internal_fpmp2_cos_kernel(__r_hi, __r_lo, &__c_hi, &__c_lo);

        int __q = __quadrant & 3;
        if (__q < 0) __q += 4;

        if (__q & 1) 
        {
            _FpType __t;
            __t = __s_hi; __s_hi = __c_hi; __c_hi = __t;
            __t = __s_lo; __s_lo = __c_lo; __c_lo = __t;
        }
        if (__q == 1 || __q == 2) 
        {
            __c_hi = -__c_hi;
            __c_lo = -__c_lo;
        }
        if (__q == 2 || __q == 3) 
        {
            __s_hi = -__s_hi;
            __s_lo = -__s_lo;
        }

        *__sin_hi = __s_hi; *__sin_lo = __s_lo;
        *__cos_hi = __c_hi; *__cos_lo = __c_lo;
    }

    /*
    * sin for fp32mp2: calls sincos and returns only the sine.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_sin(
        const _FpType __x_hi, const _FpType __x_lo,
        _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        _FpType __c_hi, __c_lo;
        __fpmp2_sincos(__x_hi, __x_lo, __res_hi, __res_lo, &__c_hi, &__c_lo);
    }

    /*
    * cos for fp32mp2: calls sincos and returns only the cosine.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_cos(
        const _FpType __x_hi, const _FpType __x_lo,
        _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        _FpType __s_hi, __s_lo;
        __fpmp2_sincos(__x_hi, __x_lo, &__s_hi, &__s_lo, __res_hi, __res_lo);
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
    * Optional large-|x| FP64 fallback (mirrors __fpmp2_sincos): when
    * _CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK == 1 and |x_hi| >= 2^20, delegate to
    * the system ::tan to keep the dedicated path off the Payne-Hanek
    * reducer for extreme arguments.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_tan(
        const _FpType __x_hi, const _FpType __x_lo,
        _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
#if (_CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK == 1)
        _FpType __abs_hi = (__x_hi < _FpType(0)) ? -__x_hi : __x_hi;
        uint32_t __abs_bits = __fpmp_internal_bit_cast<uint32_t>(__abs_hi);
        if (__abs_bits >= 0x49800000U)   /* |x_hi| >= 2^20 */
        {
            using mp2_t = fpmp2<_FpType>;
            double __xd = static_cast<double>(mp2_t(__x_hi, __x_lo));
            double td = ::tan(__xd);
            /* Split the fp64 result into (hi, lo) via the fp32mp2(double)
             * constructor -- casting to FpType first would drop the lo bits
             * and silently cap precision at ~24 bits instead of ~46.
             */
            mp2_t r_mp(td);
            *__res_hi = r_mp.hi(); *__res_lo = r_mp.lo();
            return;
        }
#endif

        int __quadrant;
        _FpType __r_hi, __r_lo;
        __internal_fpmp2_trig_reduction(__x_hi, __x_lo, &__quadrant, &__r_hi, &__r_lo);

        _FpType __s_hi, __s_lo, __c_hi, __c_lo;
        __internal_fpmp2_sin_kernel(__r_hi, __r_lo, &__s_hi, &__s_lo);
        __internal_fpmp2_cos_kernel(__r_hi, __r_lo, &__c_hi, &__c_lo);

        using mp2_t = fpmp2<_FpType>;
        mp2_t __s(__s_hi, __s_lo);
        mp2_t __c(__c_hi, __c_lo);

        mp2_t __result = (__quadrant & 1) ? mp2_t(-__c / __s) : mp2_t(__s / __c);

        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_fpmp2_atan_kernel(
        const fp32mp2_low& __a,
        fp32mp2_low* __result) noexcept
    {
        using ffloat = fp32mp2_low;

        /* 19-coefficient libdevice fp64 minimax fit; ascending degree.
         * Polynomial P(a^2) such that atan(a) = a*(1 + a^2*P(a^2)). */
        constexpr ffloat __atan_c[19] = {
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

        ffloat __a2 = __a * __a;
        ffloat __q  = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__a2, __atan_c);
        *__result   = renormalize(__a + __a * (__a2 * __q));
    }

    /* ---- (kernel 2) asin polynomial P(y); used by both asin & acos ---- */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_fpmp2_asin_poly(
        const fpmp2<_FpType>& __y,
        fpmp2<_FpType>* __result) noexcept
    {
        using ffloat = fp32mp2_low;

        ffloat __y_fast(__y.hi(), __y.lo());

        /* 13-coefficient libdevice fp64 minimax fit; ascending degree.
         * Polynomial P(y) such that asin(z)/z - 1 ~= z^2*P(z^2) for small z,
         * and pi/2 - asin(|x|) = 2*sqrty*(1 + y*P(y)) for y = (1-|x|)/2. */
        constexpr ffloat __asin_c[13] = {
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

        ffloat __q = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__y_fast, __asin_c);
        fpmp2<_FpType> __res(__q.hi(), __q.lo());
        *__result = __res;
    }

    /* ---- (kernel 3) acos large-branch polynomial P(y); used by acos only ----
     *
     * Companion to `__internal_fpmp2_asin_poly` for the |x| >= 0.575 branch
     * of acos.  Evaluates the 13-coefficient libdevice fp64 minimax fit
     * P(y) such that, for y = 1 - |x|, acos(|x|) = sqrt(2y)*(1 + y*P(y)).
     * Same fp32mp2_low internal evaluation as the asin kernel -- no
     * per-op renormalisation, single conversion in/out around the call. */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __internal_fpmp2_acos_poly(
        const fpmp2<_FpType>& __y,
        fpmp2<_FpType>* __result) noexcept
    {
        using ffloat = fp32mp2_low;

        ffloat __y_fast(__y.hi(), __y.lo());

        constexpr ffloat __acos_c[13] = {
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

        ffloat __q = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__y_fast, __acos_c);
        fpmp2<_FpType> __res(__q.hi(), __q.lo());
        *__result = __res;
    }

    /* ---- atan(x) ---- */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_atan(
        const _FpType __x_hi, const _FpType __x_lo,
        _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_atan is fp32mp2 only; "
                      "fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        const bool __is_neg = __x_hi < _FpType(0);
        ffloat __x(__x_hi, __x_lo);
        ffloat __absx = __is_neg ? -__x : __x;

        /* |x| > 1: use atan(x) = pi/2 - atan(1/x).  This includes |x| = inf,
         * which gives 1/x = 0, atan(0) = 0, result = pi/2. */
        const bool __large = __absx.hi() > _FpType(1);
        ffloat __a = __large ? (ffloat(_FpType(1)) / __absx) : __absx;

        ffloat __r;
        __internal_fpmp2_atan_kernel<_FpType>(__a, &__r);

        if (__large) {
            constexpr ffloat __PIO2(1.5707963267948966); /* pi/2 split into hi+lo */
            __r = __PIO2 - __r;
        }
        if (__is_neg) __r = -__r;

        *__res_hi = __r.hi();
        *__res_lo = __r.lo();
    }

    /* ---- atan2(y, x) ---- */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_atan2(
        const _FpType __y_hi, const _FpType __y_lo,
        const _FpType __x_hi, const _FpType __x_lo,
        _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_atan2 is fp32mp2 only; "
                      "fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* Signed-zero / signed-infinity safe sign probes via the sign bit
         * (a plain `x_hi < 0` test would return false for -0.0). */
        const uint32_t __x_bits   = __fpmp_internal_bit_cast<uint32_t>(__x_hi);
        const uint32_t __y_bits   = __fpmp_internal_bit_cast<uint32_t>(__y_hi);
        const bool     __x_is_neg = (__x_bits & 0x80000000U) != 0U;
        const bool     __y_is_neg = (__y_bits & 0x80000000U) != 0U;

        /* NaN propagation: any NaN component (in either hi or lo) forces
         * a NaN result.  Use self-inequality so the test doesn't falsely
         * fire on Inf + (-Inf) intermediates. */
        const bool __x_has_nan = (__x_hi != __x_hi) || (__x_lo != __x_lo);
        const bool __y_has_nan = (__y_hi != __y_hi) || (__y_lo != __y_lo);
        if (__x_has_nan || __y_has_nan) {
            const _FpType __nan_val = __x_has_nan ? (__x_hi + __x_lo) : (__y_hi + __y_lo);
            *__res_hi = __nan_val;
            *__res_lo = __nan_val;
            return;
        }

        ffloat __y(__y_hi, __y_lo);
        ffloat __x(__x_hi, __x_lo);
        ffloat __ay = __y_is_neg ? -__y : __y;
        ffloat __ax = __x_is_neg ? -__x : __x;

        /* |a| == +inf  <->  bit-pattern 0x7f800000 (with sign bit already
         * stripped by the abs above). */
        const bool __x_is_inf = (__fpmp_internal_bit_cast<uint32_t>(__ax.hi()) == 0x7f800000U);
        const bool __y_is_inf = (__fpmp_internal_bit_cast<uint32_t>(__ay.hi()) == 0x7f800000U);

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
        constexpr ffloat __PI    (3.141592653589793);
        constexpr ffloat __PIO2  (1.5707963267948966);
        constexpr ffloat __PIO4  (0.7853981633974483);
        constexpr ffloat __PI3O4 (2.356194490192345);   /* 3pi/4 */

        /* Effective (collapsed) sign of y, used whenever y_hi == 0.  When
         * y_hi != 0, IEEE `y_is_neg` is the right answer because the
         * collapsed sign matches `hi`'s sign for any normal value. */
        const _FpType   __y_sum     = __y_hi + __y_lo;
        const uint32_t __y_sum_bits = __fpmp_internal_bit_cast<uint32_t>(__y_sum);
        const bool     __y_eff_neg = (__y_sum_bits & 0x80000000U) != 0U;

        ffloat __r;
        if (__ax.hi() == _FpType(0) && __ay.hi() == _FpType(0)) {
            /* Both magnitudes "zero" at the high component.  The reference
             * still distinguishes +-0 by the collapsed sign of x, so honour
             * that here:  x_collapsed >= +0 -> r = +0;  x_collapsed = -0 -> r = pi
             * (the framework's `atan2(+-0, -0)` returns +-pi). */
            const _FpType   __x_sum      = __x_hi + __x_lo;
            const uint32_t __x_sum_bits = __fpmp_internal_bit_cast<uint32_t>(__x_sum);
            const bool     __x_eff_neg  = (__x_sum_bits & 0x80000000U) != 0U;
            __r = __x_eff_neg ? __PI : ffloat(_FpType(0));
        } else if (__x_is_inf && __y_is_inf) {
            /* Both infinite: 45deg / 135deg depending on x sign. */
            __r = __x_is_neg ? __PI3O4 : __PIO4;
        } else if (__y_is_inf) {
            /* |y| = inf, |x| finite:  result = +-pi/2 (sign from y). */
            __r = __PIO2;
        } else if (__x_is_inf) {
            /* |x| = inf, |y| finite:  result = +-0 or +-pi depending on sign of x.
             * Skipping the division avoids NaN from `finite / Inf` in
             * fp32mp2's renormalisation step. */
            __r = __x_is_neg ? __PI : ffloat(_FpType(0));
        } else {
            /* Generic finite path: atan(num/den), then octant fixup. */
            const bool __y_gt_x = __ay.hi() > __ax.hi();
            ffloat __num = __y_gt_x ? __ax : __ay;
            ffloat __den = __y_gt_x ? __ay : __ax;
            ffloat __t = div<fpmp2_accuracy::def>(__num, __den);
            __internal_fpmp2_atan_kernel<_FpType>(__t, &__t);

            if (__y_gt_x) {
                /* |y| > |x|:  result = +-pi/2 -/+ atan(|x|/|y|) */
                __r = __x_is_neg ? (__PIO2 + __t) : (__PIO2 - __t);
            } else if (__x_is_neg) {
                /* |y| <= |x|, x < 0:  result = pi - atan(|y|/|x|) */
                __r = __PI - __t;
            } else {
                /* |y| <= |x|, x >= 0:  result =     atan(|y|/|x|) */
                __r = __t;
            }
        }

        /* Apply sign of y (mirror across x-axis).  When y_hi is exactly
         * zero, `y_is_neg` reflects only the sign bit of `hi`, but the
         * reference's `double(y_hi+y_lo)` collapse may yield a different
         * sign, so use `y_eff_neg` for the y_hi == 0 case. */
        const bool __y_apply_neg = (__y_hi == _FpType(0)) ? __y_eff_neg : __y_is_neg;
        if (__y_apply_neg) __r = -__r;

        *__res_hi = __r.hi();
        *__res_lo = __r.lo();
    }

    /* ---- asin(x) ---- */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_asin(
        const _FpType __x_hi, const _FpType __x_lo,
        _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_asin is fp32mp2 only; "
                      "fp64mp2 has its own specialization");

        using ffloat = fpmp2<_FpType>;

        const bool __is_neg = __x_hi < _FpType(0);
        ffloat __x(__x_hi, __x_lo);
        ffloat __absx = __is_neg ? -__x : __x;

        /* Crossover at |x| ~= 0.575 (libdevice fp64 choice; threshold is
         * the boundary above which the small-branch polynomial loses
         * conditioning and the large-branch sqrt reconstruction wins). */
        constexpr _FpType __BRANCH = _FpType(0.575f);

        ffloat __r;
        if (__absx.hi() < __BRANCH) {
            /* Small branch: asin(|x|) = |x| + |x|*(|x|^2*P(|x|^2)) */
            ffloat __a2 = __absx * __absx;
            ffloat __p;
            __internal_fpmp2_asin_poly<_FpType>(__a2, &__p);
            __r = renormalize(__absx + __absx * (__a2 * __p));
        } else {
            /* Large branch: y = (1 - |x|)/2,
             *   asin(|x|) = pi/2 - 2*sqrty*(1 + y*P(y))
             * sqrt(y) returns NaN for y < 0 (i.e., |x| > 1), so NaN
             * propagates through the rest of the chain naturally. */
            ffloat __y = ffloat(_FpType(0.5f)) - __absx * ffloat(_FpType(0.5f));
            _FpType __sy_hi, __sy_lo;
            __fpmp2_sqrt(__y.hi(), __y.lo(), &__sy_hi, &__sy_lo);
            ffloat __sy(__sy_hi, __sy_lo);

            ffloat __p;
            __internal_fpmp2_asin_poly<_FpType>(__y, &__p);

            constexpr ffloat __PIO2(1.5707963267948966);
            __r = renormalize(__PIO2 - ffloat(_FpType(2)) * __sy * (ffloat(_FpType(1)) + __y * __p));
        }

        if (__is_neg) __r = -__r;
        *__res_hi = __r.hi();
        *__res_lo = __r.lo();
    }

    /* ---- acos(x) ---- */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_acos(
        const _FpType __x_hi, const _FpType __x_lo,
        _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_acos is fp32mp2 only; "
                      "fp64mp2 has its own specialization");

        using ffloat = fpmp2<_FpType>;

        const bool __is_neg = __x_hi < _FpType(0);
        ffloat __x(__x_hi, __x_lo);
        ffloat __absx = __is_neg ? -__x : __x;

        constexpr _FpType __BRANCH = _FpType(0.575f);
        constexpr ffloat __PI  (3.141592653589793);
        constexpr ffloat __PIO2(1.5707963267948966);

        ffloat __r;
        if (__absx.hi() < __BRANCH) {
            /* Small branch: reuse asin polynomial.
             *   acos(x) = pi/2 - asin(x)   (sign of x already in asin) */
            ffloat __a2 = __absx * __absx;
            ffloat __p;
            __internal_fpmp2_asin_poly<_FpType>(__a2, &__p);
            ffloat __asin_abs = renormalize(__absx + __absx * (__a2 * __p));
            __r = __is_neg ? renormalize(__PIO2 + __asin_abs)
                       : renormalize(__PIO2 - __asin_abs);
        } else {
            /* Large branch (libdevice fp64 fit, 13 coefficients):
             *   y = 1 - |x|;   acos(|x|) = sqrt(2y)*(1 + y*P(y))
             *   x < 0  ->  acos(x) = pi - acos(|x|)
             * Polynomial P(y) is evaluated by `__internal_fpmp2_acos_poly`
             * (analogous to `__internal_fpmp2_asin_poly` used by the small
             *  branch and by asin). */
            ffloat __y = ffloat(_FpType(1)) - __absx;
            ffloat __two_y = ffloat(_FpType(2)) * __y;
            _FpType __s_hi, __s_lo;
            __fpmp2_sqrt(__two_y.hi(), __two_y.lo(), &__s_hi, &__s_lo);
            ffloat __s(__s_hi, __s_lo);

            ffloat __p;
            __internal_fpmp2_acos_poly<_FpType>(__y, &__p);
            ffloat __acos_abs = renormalize(__s + __s * (__y * __p));
            __r = __is_neg ? renormalize(__PI - __acos_abs) : __acos_abs;
        }

        *__res_hi = __r.hi();
        *__res_lo = __r.lo();
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
    *      Reuses the existing dedicated __fpmp2_exp<float>; the
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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_tanh (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_tanh is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* libdevice-optimal crossover between polynomial and exp paths. */
        constexpr float __BRANCH_POINT = 0.6554117f;
        /* tanh(|x|) >= 1 - 0.5 ulp_fp32mp2 for |x| >= 17.33; use 17.5. */
        constexpr float __TANH_SAT     = 17.5f;

        const bool   __is_neg = __x_hi < 0.f;
        const float  __abs_hi = __is_neg ? -__x_hi : __x_hi;

        /* ---- (1) saturation ------------------------------------------- */
        if (!(__abs_hi < __TANH_SAT))  /* also catches NaN -> falls through to poly */
        {
            if (__abs_hi >= __TANH_SAT) {
                *__res_hi = __is_neg ? -1.f : 1.f;
                *__res_lo = 0.f;
                return;
            }
            /* NaN: propagate */
            *__res_hi = __x_hi + __x_lo;
            *__res_lo = *__res_hi;
            return;
        }

        ffloat __x   (__x_hi, __x_lo);
        ffloat __absA = __is_neg ? -__x : __x;

        if (__abs_hi >= __BRANCH_POINT)
        {
            /* ---- (2) large-|x| branch: 1 - 2/(exp(2|x|)+1) ------------ */
            ffloat __two_abs = __absA + __absA;   /* exactly 2|x|: addition of equals */
            float  __u_hi, __u_lo;
            __fpmp2_exp<float>(__two_abs.hi(), __two_abs.lo(), &__u_hi, &__u_lo);
            ffloat __denom  = ffloat(__u_hi, __u_lo) + ffloat(1.f);
            ffloat __r      = ffloat(2.f) / __denom;
            ffloat __result = ffloat(1.f) - __r;
            if (__is_neg) __result = -__result;
            *__res_hi = __result.hi();
            *__res_lo = __result.lo();
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
        constexpr ffloat __tanh_c[11] = {
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

        ffloat __a2 = __x * __x;     /* x^2 (sign of x cancels) */
        ffloat __q  = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 2>(__a2, __tanh_c);

        /* tanh(x) = x + x * x^2 * Q(x^2). Sign-preserving in x; no
         * separate sign fixup needed for the polynomial branch. */
        ffloat __result = renormalize(__x + __x * (__a2 * __q));
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_tanh

    /*
    * ============================================================================
    * Hyperbolic sine/cosine: sinh, cosh (fp32mp2)                  - dedicated
    * ============================================================================
    *
    * Both functions reuse the dedicated `__fpmp2_exp<float>` once and
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
    *   - For |x| above the exp overflow boundary (~88.7), `__fpmp2_exp`
    *     returns +inf; `0.5*inf +- 0.5/inf = inf`, which is the IEEE-correct
    *     +-inf for cosh/sinh.
    *   - NaN inputs: `__fpmp2_exp` propagates the NaN through the
    *     subsequent fp32mp2 arithmetic (no explicit guard needed).
    * ============================================================================
    */

    /* sinh(x) on fp32mp2.
     *
     * Polynomial branch covers |x| <= 0.6554 (matches the tanh crossover);
     * exp branch covers everything above.  At the crossover point we have
     * sinh(0.6554)/cosh(0.6554) = tanh(0.6554) ~= 0.575, so the exp branch
     * loses < 1 bit of precision to cancellation -- well within fp32mp2 ulp. */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_sinh (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_sinh is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        constexpr float __BRANCH_POINT = 0.6554117f;

        const bool   __is_neg = __x_hi < 0.f;
        const float  __abs_hi = __is_neg ? -__x_hi : __x_hi;

        /* NaN propagation: any NaN component pollutes the result. */
        if (__x_hi != __x_hi || __x_lo != __x_lo)
        {
            const float __nan_val = __x_hi + __x_lo;
            *__res_hi = __nan_val;
            *__res_lo = __nan_val;
            return;
        }

        ffloat __x   (__x_hi, __x_lo);
        ffloat __absA = __is_neg ? -__x : __x;

        if (__abs_hi >= __BRANCH_POINT)
        {
            /* ---- large-|x| branch:  sinh(|x|) = (e - 1/e) / 2 ---------- */
            float __u_hi, __u_lo;
            __fpmp2_exp<float>(__absA.hi(), __absA.lo(), &__u_hi, &__u_lo);
            ffloat __e(__u_hi, __u_lo);
            ffloat __half_e     = __e * ffloat(0.5f);
            ffloat __half_inv_e = ffloat(0.5f) / __e;
            ffloat __result     = renormalize(__half_e - __half_inv_e);
            if (__is_neg) __result = -__result;
            *__res_hi = __result.hi();
            *__res_lo = __result.lo();
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
        constexpr ffloat __sinh_c[11] = {
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

        ffloat __a2 = __x * __x;     /* x^2 (sign of x cancels) */
        ffloat __q  = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 3>(__a2, __sinh_c);

        /* sinh(x) = x + x * x^2 * P(x^2).  Sign-preserving in x; no
         * separate sign fixup needed for the polynomial branch. */
        ffloat __result = renormalize(__x + __x * (__a2 * __q));
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_sinh

    /* cosh(x) on fp32mp2.
     *
     * Branchless: cosh is even (cosh(-x) = cosh(x)), and the formula
     * `(e + 1/e) / 2` is well-conditioned everywhere -- both terms are
     * positive, so addition never cancels.  At |x| = 0 the lo parts of
     * e and 1/e carry the x^2/2 correction exactly, so no separate
     * polynomial branch is needed for small |x|. */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_cosh (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_cosh is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* NaN propagation. */
        if (__x_hi != __x_hi || __x_lo != __x_lo)
        {
            const float __nan_val = __x_hi + __x_lo;
            *__res_hi = __nan_val;
            *__res_lo = __nan_val;
            return;
        }

        const bool  __is_neg = __x_hi < 0.f;
        ffloat      __x      (__x_hi, __x_lo);
        ffloat      __absA   = __is_neg ? -__x : __x;

        float __u_hi, __u_lo;
        __fpmp2_exp<float>(__absA.hi(), __absA.lo(), &__u_hi, &__u_lo);
        ffloat __e(__u_hi, __u_lo);

        /* cosh(|x|) = 0.5*e + 0.5/e  (both terms positive; no cancellation). */
        ffloat __half_e     = __e * ffloat(0.5f);
        ffloat __half_inv_e = ffloat(0.5f) / __e;
        ffloat __result     = renormalize(__half_e + __half_inv_e);

        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_cosh

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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_asinh (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_asinh is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* NaN propagation. */
        if (__x_hi != __x_hi || __x_lo != __x_lo)
        {
            const float __nan_val = __x_hi + __x_lo;
            *__res_hi = __nan_val;
            *__res_lo = __nan_val;
            return;
        }

        /* asinh is odd: handle +-inf via the sign branch.
         * asinh(+-inf) = +-inf. */
        const bool   __is_neg = __x_hi < 0.0f;
        const float  __abs_hi = __is_neg ? -__x_hi : __x_hi;

        if (__abs_hi == __builtin_huge_valf())
        {
            *__res_hi = __is_neg ? -__builtin_huge_valf() : __builtin_huge_valf();
            *__res_lo = 0.0f;
            return;
        }

        /* asinh(+-0) = +-0. */
        if (__abs_hi == 0.0f && __x_lo == 0.0f)
        {
            *__res_hi = __x_hi;     /* preserves signed zero */
            *__res_lo = 0.0f;
            return;
        }

        ffloat __x   (__x_hi, __x_lo);
        ffloat __absA = __is_neg ? -__x : __x;

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
        constexpr float __LARGE_ASINH = 0x1.0p+25f;

        ffloat __result;
        if (__abs_hi > __LARGE_ASINH)
        {
            constexpr ffloat __ln2(0x1.62e42fefa39efp-1);
            float __l_hi, __l_lo;
            __fpmp2_log<float>(__absA.hi(), __absA.lo(), &__l_hi, &__l_lo);
            __result = renormalize(ffloat(__l_hi, __l_lo) + __ln2);
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
            ffloat __a2     = __absA * __absA;
            ffloat __a2p1   = add<fpmp2_accuracy::high>(__a2, 1.0f);
            float  __s_hi, __s_lo;
            __fpmp2_sqrt<float>(__a2p1.hi(), __a2p1.lo(), &__s_hi, &__s_lo);
            ffloat __s      = ffloat(__s_hi, __s_lo);
            ffloat __denom  = add<fpmp2_accuracy::high>(__s, 1.0f);
            ffloat __t      = renormalize(__absA + __a2 / __denom);

            float __r_hi, __r_lo;
            __fpmp2_log1p<float>(__t.hi(), __t.lo(), &__r_hi, &__r_lo);
            __result = ffloat(__r_hi, __r_lo);
        }

        if (__is_neg) __result = -__result;
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_asinh

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_acosh (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_acosh is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* NaN propagation. */
        if (__x_hi != __x_hi || __x_lo != __x_lo)
        {
            const float __nan_val = __x_hi + __x_lo;
            *__res_hi = __nan_val;
            *__res_lo = __nan_val;
            return;
        }

        /* Domain: x >= 1.  Anything strictly below produces NaN.
         * Use lexicographic compare on (hi, lo) to capture x = 1 with
         * a negative lo (i.e., x < 1 by a sub-ulp amount). */
        if (__x_hi < 1.0f || (__x_hi == 1.0f && __x_lo < 0.0f))
        {
            *__res_hi = __builtin_nanf("");
            *__res_lo = __builtin_nanf("");
            return;
        }

        /* acosh(+inf) = +inf. */
        if (__x_hi == __builtin_huge_valf())
        {
            *__res_hi = __builtin_huge_valf();
            *__res_lo = 0.0f;
            return;
        }

        /* acosh(1) = 0 exactly. */
        if (__x_hi == 1.0f && __x_lo == 0.0f)
        {
            *__res_hi = 0.0f;
            *__res_lo = 0.0f;
            return;
        }

        ffloat __x(__x_hi, __x_lo);

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
        constexpr float __LARGE_ACOSH = 0x1.0p+25f;

        ffloat __result;
        if (__x_hi > __LARGE_ACOSH)
        {
            /* Asymptotic form: acosh(x) ~= log(2x) = log(x) + ln(2).
             * O(1/x^2) correction is below fp32mp2 ulp at crossover. */
            constexpr ffloat __ln2(0x1.62e42fefa39efp-1);
            float __l_hi, __l_lo;
            __fpmp2_log<float>(__x.hi(), __x.lo(), &__l_hi, &__l_lo);
            __result = renormalize(ffloat(__l_hi, __l_lo) + __ln2);
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
            ffloat __xm1   = sub<fpmp2_accuracy::high>(__x, 1.0f);
            ffloat __xp1   = add<fpmp2_accuracy::high>(__x, 1.0f);
            ffloat __x2m1  = __xm1 * __xp1;
            float  __s_hi, __s_lo;
            __fpmp2_sqrt<float>(__x2m1.hi(), __x2m1.lo(), &__s_hi, &__s_lo);
            ffloat __s     = ffloat(__s_hi, __s_lo);
            ffloat __t     = renormalize(__xm1 + __s);

            float __r_hi, __r_lo;
            __fpmp2_log1p<float>(__t.hi(), __t.lo(), &__r_hi, &__r_lo);
            __result = ffloat(__r_hi, __r_lo);
        }

        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_acosh

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_atanh (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        static_assert(::cuda::std::is_same_v<_FpType, float>,
                      "dedicated __fpmp2_atanh is fp32mp2 only; fp64mp2 has its own specialization");

        using ffloat = fp32mp2_low;

        /* NaN propagation. */
        if (__x_hi != __x_hi || __x_lo != __x_lo)
        {
            const float __nan_val = __x_hi + __x_lo;
            *__res_hi = __nan_val;
            *__res_lo = __nan_val;
            return;
        }

        const bool   __is_neg = __x_hi < 0.0f;
        const float  __abs_hi = __is_neg ? -__x_hi : __x_hi;

        /* atanh(+-0) = +-0. */
        if (__abs_hi == 0.0f && __x_lo == 0.0f)
        {
            *__res_hi = __x_hi;     /* preserves signed zero */
            *__res_lo = 0.0f;
            return;
        }

        /* atanh(+-1) = +-inf.  Strict |x| > 1 -> NaN. */
        if (__abs_hi >= 1.0f)
        {
            const float __abs_lo = __is_neg ? -__x_lo : __x_lo;
            if (__abs_hi == 1.0f && __abs_lo == 0.0f)
            {
                *__res_hi = __is_neg ? -__builtin_huge_valf() : __builtin_huge_valf();
                *__res_lo = 0.0f;
                return;
            }
            /* |x| > 1 (including +inf): outside domain. */
            *__res_hi = __builtin_nanf("");
            *__res_lo = __builtin_nanf("");
            return;
        }

        ffloat __x    (__x_hi, __x_lo);
        ffloat __absA = __is_neg ? -__x : __x;

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
        constexpr float __ATANH_BRANCH_POINT = 0.25f;
        if (__abs_hi < __ATANH_BRANCH_POINT)
        {
            /* P(y) = sum_{k>=0} y^k / (2k+3), packed in ascending degree.
             *   atanh_poly_c[0] = 1/3 (constant of P),
             *   atanh_poly_c[k] = 1/(2k+3),
             *   atanh_poly_c[11] = 1/25 (leading).
             * Layout for poly_eval<horner_mixed, M=4>: bottom 8 entries
             * are full ff (their contributions stay above fp32mp2 ulp at
             * the branch point), top 4 entries are plain float (.lo == 0
             * by construction; their contributions sit below 0.5 ulp). */
            constexpr ffloat __atanh_poly_c[12] = {
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

            ffloat __y      = __x * __x;     /* x^2 (sign of x cancels) */
            ffloat __q      = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 4>(__y, __atanh_poly_c);
            ffloat __result = renormalize(__x + __x * (__y * __q));
            *__res_hi = __result.hi();
            *__res_lo = __result.lo();
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
        ffloat __one_minus = sub<fpmp2_accuracy::high>(ffloat(1.0f), __absA);
        ffloat __two_abs   = __absA + __absA;
        ffloat __t         = __two_abs / __one_minus;

        float __l_hi, __l_lo;
        __fpmp2_log1p<float>(__t.hi(), __t.lo(), &__l_hi, &__l_lo);

        ffloat __result = ffloat(__l_hi, __l_lo) * ffloat(0.5f);
        if (__is_neg) __result = -__result;

        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_atanh

    /*
    * --------------------------------------------------------------------
    * Error function erf(x) (fp32mp2) - dedicated implementation
    * --------------------------------------------------------------------
    * Error function erf(x) = -expm1(-|x|*P(|x|)) where P is a Remez
    * polynomial in |x|, and expm1 is computed via argument reduction
    * and a mixed-precision polynomial, all in fp32mp2.
    *
    * Two polynomial variants are provided, selected at compile time
    * by the _CCCL_FPMP_USE_FAST_ERF macro:
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
    #ifndef _CCCL_FPMP_USE_FAST_ERF
        #define _CCCL_FPMP_USE_FAST_ERF 1
    #endif
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_erf(const _FpType __x_hi,
                                               const _FpType __x_lo,
                                               _FpType*      __res_hi,
                                               _FpType*      __res_lo) noexcept
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
        constexpr ffloat __m_c[10] = {
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

        constexpr ffloat __L2E   (1.4426950408889634);
        constexpr ffloat __LN2_HI(0.6931471805599453);

        ffloat __x     = renormalize(ffloat(__x_hi, __x_lo));
        bool __is_neg  = __x.hi() < 0.f;
        uint32_t __xhi = __fpmp_internal_bit_cast<uint32_t>(__x.hi()) & 0x7fffffffU;
        ffloat __absA  = __is_neg ? -__x : __x;

        /* |x| >= saturation_bound (~5.92) or Inf -> erf = +-1 */
        if (__xhi >= 0x40bd7da4U && __xhi <= 0x7f800000U) 
        {
            *__res_hi = __is_neg ? -1.f : 1.f;
            *__res_lo = 0.f;
            return;
        }

#if _CCCL_FPMP_USE_FAST_ERF == 1
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
        constexpr float __X_STAR = 2.1134011f;

        constexpr ffloat __dc_left[18] = {
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

        constexpr ffloat __dc_right[17] = {
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

        ffloat __poly;
        if (__absA.hi() < __X_STAR)
            __poly = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__absA, __dc_left);
        else
            __poly = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__absA, __dc_right);
#else // _CCCL_FPMP_USE_FAST_ERF == 0
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
        ffloat __poly = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__absA, dc);
#endif // _CCCL_FPMP_USE_FAST_ERF == 0

        /* arg = |x| * P(|x|) + |x| (replaces polyHi/polyLo splitting) */
        ffloat __arg = renormalize(__poly * __absA + __absA);

        /* Compute -expm1(-arg): argument reduction */
        ffloat __neg_arg     = -__arg;
        float  __neg_arg_l2e = (__neg_arg * __L2E).hi();
        int    __n           = __fpmp_fp2int_rn(__neg_arg_l2e);
        ffloat __fn          = __fpmp_int2fp_rn<float>(__n);
        ffloat __r           = __neg_arg - __fn * __LN2_HI;

        /* Evaluate u(r) = m2 + m3*r + ... + m11*r^9 via the mixed-precision
         * dispatcher (4 high-order float coeffs m8..m11, 6 low-order ff
         * coeffs m2..m7).  Cheaper than a unified compensated Horner
         * because the high terms contribute below the noise floor and
         * don't need error tracking.
         */
        ffloat __u = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 4>(__r, __m_c);

        /* expm1(r) = u*r^2 + r (no separate alo needed, r carries full precision) */
        __u = __u * __r;
        __u = __u * __r;
        __u = __u + __r;

        /* scale = 2^n, scalem1 = 1 - 2^n */
        int __en           = 127 + __n;
        if (__en < 1)   __en = 1;
        if (__en > 254) __en = 254;
        float  __scale     = __fpmp_internal_bit_cast<float>(static_cast<unsigned>(__en) << 23);
        ffloat __scalem1   = ffloat(1.f, 0.f) - ffloat(__scale, 0.f);

        /* result = -expm1(-arg) = -u*scale + scalem1 */
        ffloat __result = renormalize(-__u * ffloat(__scale, 0.f) + __scalem1);

        /* Apply sign */
        if (__is_neg)
            __result = -__result;

        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_erf

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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_erfc(const _FpType __x_hi, 
                                                const _FpType __x_lo, 
                                                _FpType*      __res_hi, 
                                                _FpType*      __res_lo) noexcept
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
        constexpr ffloat __cheb[23] = {
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
        constexpr ffloat __exp_c[12] = {
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

        constexpr ffloat __L2E   (1.4426950408889634e+0);
        constexpr ffloat __LN2_HI(6.9314718055994529e-1);
        constexpr ffloat __LN2_LO(2.3190468138462996e-17);            

        ffloat __x     = renormalize(ffloat(__x_hi, __x_lo));
        bool __is_neg  = __x.hi() < 0.f;
        uint32_t __xhi = __fpmp_internal_bit_cast<uint32_t>(__x.hi()) & 0x7fffffffU;
        ffloat __a = (__is_neg) ? -__x : __x;

        // handle x > 27.5 && <= Inf
        if ((__xhi > 0x41dc0000U) && (__xhi <= 0x7f800000U))
        {
            *__res_hi = (__is_neg) ? 2.f : 0.f;
            *__res_lo = 0.f;
            return;
        }

        /* erfcx kernel: (1+2*a)*exp(a^2)*erfc(a) on a = |x|, transform (a-4)/(a+4) */
        ffloat __t1 = __a - ffloat(4.0);
        ffloat __t2 = __a + ffloat(4.0);
        __t2        = ffloat(1.0) / __t2;
        ffloat __t3 = (__t1 * __t2);
        ffloat __t4 = __t3 + ffloat(1.0);
        __t1        = (ffloat(-4.0) * __t4 + __a);
        __t1        = __t1 - __t3 * __a;
        __t2        = (__t2 * __t1 + __t3);

        // Chebyshev polynomial: 7 high-order terms in float, remaining 16 in ff
        __t1 = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 7>(__t2, __cheb);

        /* (1+2*a)*exp(a^2)*erfc(a) / (1+2*a) -> exp(a^2)*erfc(a) = erfcx */
        __t2 = (ffloat(2.0) * __a + ffloat(1.0));
        __t2 = ffloat(1.0) / __t2;
        __t3 = __t1 * __t2;
        __t4 = __a * (ffloat(-2.0) * __t3) + __t1;
        __t4 = (__t4 - __t3);
        __t1 = (__t4 * __t2 + __t3);

        /* erfc(x) = erfcx * exp(-x^2) */
        ffloat __xx = renormalize(-__a * __a);

        /* i = round(xx * L2E); t = exp_mantissa(xx); t3 = accurate_scale(t, i) */
        float __prod_hi = (__xx * __L2E).hi();
        int __i         = __fpmp_fp2int_rn(__prod_hi);
        ffloat __t_rint = __fpmp_int2fp_rn<float>(__i);
        ffloat __z = renormalize(__xx - __t_rint * __LN2_HI - __t_rint * __LN2_LO);

        // exp polynomial: 5 high-order terms in float, remaining 7 in ff
        ffloat __t = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 5>(__z, __exp_c);

        /* accurate_scale(t, i): t * 2^i in fp32mp2 (split exponent for large |i|)*/
        int __k   = __i / 2;
        int __ek  = 127 + __k;
        int __ek2 = 127 + (__i - __k);
        if (__ek  < 1) __ek  = 1;
        if (__ek2 < 1) __ek2 = 1;

        float  __scale_lo   = __fpmp_internal_bit_cast<float>(static_cast<unsigned>(__ek)  << 23);
        float  __scale_hi   = __fpmp_internal_bit_cast<float>(static_cast<unsigned>(__ek2) << 23);
        ffloat __exp_scaled = ffloat(__t.hi() * __scale_lo * __scale_hi, __t.lo() * __scale_lo * __scale_hi);

        /* Correction: exp(-x^2) = exp_scaled * (1 + (-x^2 - xx)) same as double fma(t3, -x*x - xx, t3) */
        ffloat __remainder = renormalize(-__a * __a - __xx);
        ffloat __exp_xx    = __exp_scaled * __remainder + __exp_scaled;
        ffloat __erfc_val  = renormalize(__t1 * __exp_xx);

        if (__is_neg)
            __erfc_val = renormalize(ffloat(2.0) - __erfc_val);

        *__res_hi = __erfc_val.hi();
        *__res_lo = __erfc_val.lo();
    } // __fpmp2_erfc

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
    // Note: _CCCL_API inline (non-static) in inline mode for performance.
    // With -dc compilation, nvcc caps registers at ~37 for static
    // __forceinline__ functions, ignoring __launch_bounds__ on the caller.
    // This limits the boys kernel to 5.8x speedup instead of 12x.
    // Without static, the compiler respects __launch_bounds__ and allocates
    // 156 registers, enabling full ILP across the polynomial chains.
    //
    // In library build mode (_CCCL_FPMP_BUILD_LIB), static is required to
    // avoid ODR violations with the explicit float specialization in the
    // _CCCL_FPMP_USE_LIB block, which would cause infinite recursion in LTO.
    */
    /*
    * Define _CCCL_FPMP_USE_ACCURATE_BOYS_F0 to use the accurate implementation
    * providing 43 bits of accuracy
    */
    // #define _CCCL_FPMP_USE_ACCURATE_BOYS_F0

    #if defined(_CCCL_FPMP_BUILD_LIB)
      #define _CCCL_FPMP_INTERNAL_CUSTOM_DECL _CCCL_TRIVIAL_API
    #else
      #define _CCCL_FPMP_INTERNAL_CUSTOM_DECL _CCCL_API inline
    #endif

    #if !defined(_CCCL_FPMP_USE_ACCURATE_BOYS_F0)
      #define _CCCL_FPMP_RENORMALIZE(v) v = renormalize(v)
      #define _CCCL_FPMP_SUB(v, x) sub<fpmp2_accuracy::high>(v, x)
      #define _CCCL_FPMP_METHOD fpmp2_accuracy::low
    #else
      #define _CCCL_FPMP_RENORMALIZE(v)
      #define _CCCL_FPMP_SUB(v, x) ((v) - (x))
      #define _CCCL_FPMP_METHOD fpmp2_accuracy::def
    #endif

    template<typename _FpType = float>
    _CCCL_FPMP_INTERNAL_CUSTOM_DECL void __fpmp2_boys_f0 (const _FpType __a_hi,
                                                           const _FpType __a_lo,
                                                           _FpType*      __res_hi,
                                                           _FpType*      __res_lo)
    {
        using ffloat = fpmp2<_FpType, _CCCL_FPMP_METHOD>;

        ffloat __a(__a_hi, __a_lo);
        ffloat __r;

        if (__a_hi >= 0x1.6ebc6ap3f) // a >= 11.46
            __r = rsqrt(__a);

        if (__a_hi > 34.3816f) 
        {
            constexpr ffloat __sqrt_pi_4(0x1.c5bf891b4ef6bp-1);
            ffloat __result = __sqrt_pi_4 * __r;
            _CCCL_FPMP_RENORMALIZE(__result);
            *__res_hi = __result.hi(); *__res_lo = __result.lo();
            return;
        }

        if (__a_hi < 0x1p2f) 
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
            ffloat __x = _CCCL_FPMP_SUB(ffloat(0x1.8p1), __a);
            constexpr ffloat __c[17] = {
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
            ffloat __v = __fpmp_poly_eval<__fpmp_poly_method::horner_comp, 4>(__x, __c);
            *__res_hi = __v.hi(); *__res_lo = __v.lo();
            return;
        } // if (a_hi < 0x1p2f)

        if (__a_hi < 0x1.6ebc6ap3f) 
        {
            /* 4 <= a < 11.46: 20-term minimax in x = 6.92 - a (degree 19).
             * Standard ff-Horner with periodic renormalization -- the
             * compensated variant loses accuracy here because the
             * coefficients span ~22 orders of magnitude. */
            ffloat __x = _CCCL_FPMP_SUB(ffloat(0x1.baf1a8p1), __a);
            ffloat __v =  ffloat(0x1.95402da668f4fp-73);
            __v = __v * __x + ffloat(0x1.43744ab1a0e5ap-66);
            __v = __v * __x + ffloat(0x1.f70f3953813b1p-61);
            __v = __v * __x + ffloat(0x1.00b2c5aae06a1p-55);
            __v = __v * __x + ffloat(0x1.87ddc6a10f513p-51);
            __v = __v * __x + ffloat(0x1.e450e0340da6fp-47);
            __v = __v * __x + ffloat(0x1.ffc73283f2e3dp-43);
            __v = __v * __x + ffloat(0x1.dff8a98149ce4p-39);
            __v = __v * __x + ffloat(0x1.98aa56613b23p-35);
            _CCCL_FPMP_RENORMALIZE(__v);
            __v = __v * __x + ffloat(0x1.3f3d23359c3f4p-31);
            __v = __v * __x + ffloat(0x1.ca89e4f410357p-28);
            __v = __v * __x + ffloat(0x1.2ddf249b49215p-24);
            _CCCL_FPMP_RENORMALIZE(__v);   
            __v = __v * __x + ffloat(0x1.6a60fc5c32d39p-21);
            __v = __v * __x + ffloat(0x1.8a0af8927f728p-18);
            __v = __v * __x + ffloat(0x1.81949bbc35f76p-15);
            _CCCL_FPMP_RENORMALIZE(__v);   
            __v = __v * __x + ffloat(0x1.51d1e0119bf15p-12);
            __v = __v * __x + ffloat(0x1.090a189fdb05bp-9);
            _CCCL_FPMP_RENORMALIZE(__v);   
            __v = __v * __x + ffloat(0x1.7a16985c09ba2p-7);
            __v = __v * __x + ffloat(0x1.04f3fb31bb071p-4);
            __v = __v * __x + ffloat(0x1.e3ae966b0f402p-2);
            _CCCL_FPMP_RENORMALIZE(__v);   
            *__res_hi = __v.hi(); *__res_lo = __v.lo();
            return;
        } // if (a_hi < 0x1.6ebc6ap3f)

        /* 11.46 <= a <= 34.38: 19-term minimax in x = rsqrt(a)^2 - offset
         * (degree 18), evaluated via compensated Horner. Coefficients are
         * in ascending order (c[0] = constant, c[18] = leading). */
        ffloat __x = _CCCL_FPMP_SUB(__r * __r, ffloat(0x1.dc88f0479694p-5));
        constexpr ffloat __c[19] = {
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
        ffloat __v = __fpmp_poly_eval<__fpmp_poly_method::horner_comp>(__x, __c);
        __r = __r * ffloat(0x1.c5bf8ap-1);
        ffloat __result = __v * __r;
        _CCCL_FPMP_RENORMALIZE(__result);

        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    } // __fpmp2_boys_f0

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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_normcdfinv(const _FpType __x_hi,
                                                      const _FpType __x_lo,
                                                      _FpType*      __res_hi,
                                                      _FpType*      __res_lo) noexcept
    {
        using ffloat = fp32mp2_low;

        constexpr ffloat __sqrt2(0x1.6a09e667f3bcdp+0);

        /* Central polynomial: rc(tc) = c22 + c21*tc + ... + c0*tc^22,
         * tc = w - 3.125  (>99.9% of inputs land here).
         * Ascending degree; M = 9 high-order entries (rc_c[14..22] =
         * original c0..c8) are plain float and run in the dispatcher's
         * FpType phase.  The transition step `rcf * tc.hi() + c9`
         * matches the previous hand-rolled float*float + ff step
         * bit-for-bit, so this refactor is numerically identical.
         */
        constexpr ffloat __rc_c[23] = {
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
        constexpr ffloat __rt_c[19] = {
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
        constexpr ffloat __rt2_c[25] = {
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

        ffloat __p = renormalize(ffloat(__x_hi, __x_lo));

        /* Standard mathematical convention: normcdfinv(0) = -inf, normcdfinv(1) = +inf */
        if (__p.hi() <= 0.0f) { *__res_hi = __fpmp_internal_bit_cast<float>(0xFF800000U); *__res_lo = 0.0f; return; }
        if (__p.hi() >= 1.0f) { *__res_hi = __fpmp_internal_bit_cast<float>(0x7F800000U); *__res_lo = 0.0f; return; }

        /* a = 2p - 1, accurate subtraction for p ~= 0.5 */
        ffloat __two_p = __p + __p;
        ffloat __a = sub<fpmp2_accuracy::high>(__two_p, 1.0f);

        /* w = -log(1 - a^2) = -log(4p(1-p))
         * Compute 1-p with accurate subtraction to handle p near 0 or 1
         */
        ffloat __omp = sub<fpmp2_accuracy::high>(1.0f, __p);
        ffloat __arg = 4.0f * __p * __omp;

        if (__arg.hi() <= 0.0f)
            __arg = ffloat(0x1.0p-126f);

        float __log_hi, __log_lo;
        __fpmp2_log(__arg.hi(), __arg.lo(), &__log_hi, &__log_lo);
        ffloat __w = -ffloat(__log_hi, __log_lo);

        /* Central region (w < 6.125, |z| < ~3.3 sigma, >99.9% of inputs):
         * Horner in tc = w - 3.125 via the mixed-precision dispatcher
         * (9 high-order float coeffs c0..c8, 14 low-order ff coeffs c9..c22).
         */
        ffloat __tc   = __w - ffloat(3.125f);
        ffloat __poly = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 9>(__tc, __rc_c);

        /* Tail regions (w >= 6.125): branched since <0.1% of inputs.
         * sqrt(w) is also deferred into this branch.
         */
        if (__w.hi() >= 6.125f) {
            float __sw_hi, __sw_lo;
            __fpmp2_sqrt(__w.hi(), __w.lo(), &__sw_hi, &__sw_lo);
            ffloat __sw(__sw_hi, __sw_lo);

            /* Tail 1 (6.125 <= w < 16, |z| ~ 3.3 to 5.5 sigma):
             * Horner in tt = sqrt(w) - 3.25 via the dispatcher
             * (9 high-order float coeffs t0..t8, 10 low-order ff coeffs t9..t18).
             */
            ffloat __tt = __sw - ffloat(3.25f);
            __poly      = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 9>(__tt, __rt_c);

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
            if (__w.hi() >= 16.0f) {
                ffloat __tt2 = __sw - ffloat(7.25f);
                __poly       = __fpmp_poly_eval<__fpmp_poly_method::horner_mixed, 13>(__tt2, __rt2_c);
            }
        }

        /* Scale: result = poly * a * sqrt(2) */
        ffloat __result = renormalize(__poly * __a * __sqrt2);

        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    } // __fpmp2_normcdfinv

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
    _CCCL_TRIVIAL_API void __fpmp2_icdf(uint32_t __x, float* __res_hi, float* __res_lo) noexcept
    {
        float __sign = 1.0f;
        if (__x > 0x80000000u) {
            __x = 0xFFFFFFFFu - __x;
            __sign = -1.0f;
        }
        /* p = (x + 0.5) / 2^32  in  (0, 0.5]
         * Split x into two 16-bit halves for exact fp32mp2 representation.
         */
        float __hi = (float)(__x >> 16) * 0x1.0p-16f;
        float __lo = ((float)(__x & 0xFFFFu) + 0.5f) * 0x1.0p-32f;
        float __p_hi = __hi + __lo;
        float __p_lo = __lo - (__p_hi - __hi);

        __fpmp2_normcdfinv(__p_hi, __p_lo, __res_hi, __res_lo);
        /* Clamp to +-FLT_MAX for safe Gaussian variate generation (no infinities) */
        if (*__res_hi >= 0x1.fffffep+127f)  { *__res_hi =  0x1.fffffep+127f; *__res_lo = 0.0f; }
        if (*__res_hi <= -0x1.fffffep+127f) { *__res_hi = -0x1.fffffep+127f; *__res_lo = 0.0f; }
        *__res_hi *= __sign;
        *__res_lo *= __sign;
    } // __fpmp2_icdf

    _CCCL_TRIVIAL_API void __fpmp2_icdf(uint64_t __x, float* __res_hi, float* __res_lo) noexcept
    {
        float __sign = 1.0f;
        __x >>= 16;   /* keep top 48 bits (matches fp32mp2 precision) */
        if (__x > 0x800000000000ULL) {
            __x = 0xFFFFFFFFFFFFULL - __x;
            __sign = -1.0f;
        }
        /* p = (x + 0.5) / 2^48  in  (0, 0.5]
         * Split 48-bit x into two 24-bit halves for exact float representation.
         */
        float __hi = (float)(uint32_t)(__x >> 24) * 0x1.0p-24f;
        float __lo = ((float)(uint32_t)(__x & 0xFFFFFFu) + 0.5f) * 0x1.0p-48f;
        float __p_hi = __hi + __lo;
        float __p_lo = __lo - (__p_hi - __hi);

        __fpmp2_normcdfinv(__p_hi, __p_lo, __res_hi, __res_lo);
        /* Clamp to +-FLT_MAX for safe Gaussian variate generation (no infinities) */
        if (*__res_hi >= 0x1.fffffep+127f)  { *__res_hi =  0x1.fffffep+127f; *__res_lo = 0.0f; }
        if (*__res_hi <= -0x1.fffffep+127f) { *__res_hi = -0x1.fffffep+127f; *__res_lo = 0.0f; }
        *__res_hi *= __sign;
        *__res_lo *= __sign;
    } // __fpmp2_icdf

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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_fabs (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        *__res_hi = ::fabs(__x_hi);
        *__res_lo = (__x_hi < _FpType(0)) ? -__x_lo : __x_lo;
    }

    /*
    * fmax: max(x, y).  Lexicographic comparison on (hi, lo) -- valid because
    * normalized fpmp2 inputs satisfy |lo| < ulp(hi)/2, so `x > y` iff
    * `x_hi > y_hi || (x_hi == y_hi && x_lo > y_lo)`.  NaN handling follows
    * C99/IEEE-754-2008: if exactly one operand is NaN, return the other.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_fmax (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 const _FpType __y_hi,
                                                 const _FpType __y_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        const bool __x_is_nan = __fpmp_internal_isnan(__x_hi);
        const bool __y_is_nan = __fpmp_internal_isnan(__y_hi);
        if (__x_is_nan && !__y_is_nan) { *__res_hi = __y_hi; *__res_lo = __y_lo; return; }
        if (__y_is_nan && !__x_is_nan) { *__res_hi = __x_hi; *__res_lo = __x_lo; return; }
        const bool __x_greater = (__x_hi > __y_hi) || (__x_hi == __y_hi && __x_lo > __y_lo);
        if (__x_greater) { *__res_hi = __x_hi; *__res_lo = __x_lo; }
        else           { *__res_hi = __y_hi; *__res_lo = __y_lo; }
    }

    /*
    * fmin: min(x, y).  Mirror image of fmax.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_fmin (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 const _FpType __y_hi,
                                                 const _FpType __y_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        const bool __x_is_nan = __fpmp_internal_isnan(__x_hi);
        const bool __y_is_nan = __fpmp_internal_isnan(__y_hi);
        if (__x_is_nan && !__y_is_nan) { *__res_hi = __y_hi; *__res_lo = __y_lo; return; }
        if (__y_is_nan && !__x_is_nan) { *__res_hi = __x_hi; *__res_lo = __x_lo; return; }
        const bool __x_less = (__x_hi < __y_hi) || (__x_hi == __y_hi && __x_lo < __y_lo);
        if (__x_less) { *__res_hi = __x_hi; *__res_lo = __x_lo; }
        else        { *__res_hi = __y_hi; *__res_lo = __y_lo; }
    }

    /*
    * max: std::max-like selection for fpmp2 values.  Uses the same
    * lexicographic ordering as fmax, but keeps std::max semantics:
    * return y only when x < y; otherwise return x (ties/unordered -> x).
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_max (const _FpType __x_hi,
                                                const _FpType __x_lo,
                                                const _FpType __y_hi,
                                                const _FpType __y_lo,
                                                _FpType*      __res_hi,
                                                _FpType*      __res_lo) noexcept
    {
        const bool __x_less = (__x_hi < __y_hi) || (__x_hi == __y_hi && __x_lo < __y_lo);
        if (__x_less) { *__res_hi = __y_hi; *__res_lo = __y_lo; }
        else        { *__res_hi = __x_hi; *__res_lo = __x_lo; }
    }

    /*
    * min: std::min-like selection for fpmp2 values.  Uses the same
    * lexicographic ordering as fmin, but keeps std::min semantics:
    * return y only when y < x; otherwise return x (ties/unordered -> x).
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_min (const _FpType __x_hi,
                                                const _FpType __x_lo,
                                                const _FpType __y_hi,
                                                const _FpType __y_lo,
                                                _FpType*      __res_hi,
                                                _FpType*      __res_lo) noexcept
    {
        const bool __y_less = (__y_hi < __x_hi) || (__y_hi == __x_hi && __y_lo < __x_lo);
        if (__y_less) { *__res_hi = __y_hi; *__res_lo = __y_lo; }
        else        { *__res_hi = __x_hi; *__res_lo = __x_lo; }
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
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_floor (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        // NaN check
        if ((__x_hi != __x_hi) || (__x_lo != __x_lo)) 
        {
            _FpType __res = __x_hi+__x_lo;
            *__res_hi = __res; *__res_lo = __res;
            return;
        }

        const _FpType __abs_hi = __fpmp_internal_fabs(__x_hi);
        const _FpType __int_scale = ::cuda::std::is_same_v<_FpType, float> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);
        if (__abs_hi >= __int_scale) 
        {
            // x_hi is already an integer at this scale; floor(x_hi + x_lo) = x_hi + floor(x_lo).
            const _FpType __lo_floor = __fpmp_internal_floor<_FpType>(__x_lo);
            _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
            __fpmp2_acc<_FpType>(__lo_floor, &__t_hi, &__t_lo);
            *__res_hi = __t_hi; *__res_lo = __t_lo;
            return;
        }

        const _FpType __n = __fpmp_internal_floor<_FpType>(__x_hi);
        if (__x_hi != __n || __x_lo >= _FpType(0)) 
        {
            *__res_hi = __n; *__res_lo = _FpType(0);
            return;
        }

        _FpType __t_hi = __n, __t_lo = _FpType(0);
        __fpmp2_acc<_FpType>(_FpType(-1), &__t_hi, &__t_lo);
        *__res_hi = __t_hi; *__res_lo = __t_lo;
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_ceil (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        // NaN check
        if ((__x_hi != __x_hi) || (__x_lo != __x_lo)) 
        {
            _FpType __res = __x_hi+__x_lo;
            *__res_hi = __res; *__res_lo = __res;
            return;
        }

        const _FpType __abs_hi = __fpmp_internal_fabs(__x_hi);
        const _FpType __int_scale = ::cuda::std::is_same_v<_FpType, float> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);
        if (__abs_hi >= __int_scale) 
        {
            // x_hi is already an integer at this scale; ceil(x_hi + x_lo) = x_hi + ceil(x_lo).
            const _FpType __lo_ceil = __fpmp_internal_ceil<_FpType>(__x_lo);
            _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
            __fpmp2_acc<_FpType>(__lo_ceil, &__t_hi, &__t_lo);
            *__res_hi = __t_hi; *__res_lo = __t_lo;
            return;
        }

        const _FpType __n = __fpmp_internal_ceil<_FpType>(__x_hi);
        if (__x_hi != __n || __x_lo <= _FpType(0)) 
        {
            *__res_hi = __n; *__res_lo = _FpType(0);
            return;
        }

        _FpType __t_hi = __n, __t_lo = _FpType(0);
        __fpmp2_acc<_FpType>(_FpType(1), &__t_hi, &__t_lo);
        *__res_hi = __t_hi; *__res_lo = __t_lo;
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_round (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        // NaN check
        if ((__x_hi != __x_hi) || (__x_lo != __x_lo)) 
        {
            _FpType __res = __x_hi+__x_lo;
            *__res_hi = __res; *__res_lo = __res;
            return;
        }

        const bool __x_neg = (__x_hi < _FpType(0)) || (__x_hi == _FpType(0) && __x_lo < _FpType(0));

        _FpType __t_hi = __x_hi, __t_lo = __x_lo;
        __fpmp2_acc<_FpType>(__x_neg ? _FpType(-0.5) : _FpType(0.5), &__t_hi, &__t_lo);

        if (__x_neg) __fpmp2_ceil (__t_hi, __t_lo, __res_hi, __res_lo);
        else       __fpmp2_floor(__t_hi, __t_lo, __res_hi, __res_lo);
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_trunc (const _FpType __x_hi,
                                                  const _FpType __x_lo,
                                                  _FpType*      __res_hi,
                                                  _FpType*      __res_lo) noexcept
    {
        // NaN check
        if ((__x_hi != __x_hi) || (__x_lo != __x_lo)) 
        {
            _FpType __res = __x_hi+__x_lo;
            *__res_hi = __res; *__res_lo = __res;
            return;
        }

        const _FpType __abs_hi = __fpmp_internal_fabs(__x_hi);
        const _FpType __int_scale = ::cuda::std::is_same_v<_FpType, float> ? _FpType(0x1.0p23f) : _FpType(0x1.0p52);
        if (__abs_hi >= __int_scale) 
        {
            // x_hi is integral at this scale and dominates sign, so trunc is:
            //   x_hi > 0 : floor(x_hi + x_lo) = x_hi + floor(x_lo)
            //   x_hi < 0 : ceil (x_hi + x_lo) = x_hi + ceil (x_lo)
            const _FpType __lo_trunc = (__x_hi < _FpType(0))
                                  ? __fpmp_internal_ceil<_FpType>(__x_lo)
                                  : __fpmp_internal_floor<_FpType>(__x_lo);
            _FpType __t_hi = __x_hi, __t_lo = _FpType(0);
            __fpmp2_acc<_FpType>(__lo_trunc, &__t_hi, &__t_lo);
            *__res_hi = __t_hi; *__res_lo = __t_lo;
            return;
        }

        // Fast small-magnitude path:
        // Start from trunc(x_hi), then apply at most a +/-1 correction only when
        // x_hi is already integral and x_lo nudges the exact value across that integer.
        const _FpType __n = __fpmp_internal_trunc<_FpType>(__x_hi);
        if (__x_hi != __n) 
        {
            *__res_hi = __n; *__res_lo = _FpType(0);
            return;
        }

        const bool __x_neg = (__x_hi < _FpType(0)) || (__x_hi == _FpType(0) && __x_lo < _FpType(0));
        const int __delta = (!__x_neg && __x_lo < _FpType(0)) ? -1 :
                          ( __x_neg && __x_lo > _FpType(0)) ?  1 : 0;
        if (__delta != 0) 
        {
            _FpType __t_hi = __n, __t_lo = _FpType(0);
            __fpmp2_acc<_FpType>(static_cast<_FpType>(__delta), &__t_hi, &__t_lo);
            *__res_hi = __t_hi; *__res_lo = __t_lo;
            return;
        }

        *__res_hi = __n;
        *__res_lo = _FpType(0);
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
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_1A
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_2A
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETINT
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETLL
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETL
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_FP_INT
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_FP_LINT
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_INT_FP

    // Helper macro: placeholder implementation that delegates to a standard double-precision
    // math function with proper hi/lo splitting via fpmp2 conversions.
    // Uses explicit fpmp2 construction/conversion to avoid NVCC name resolution issues.
    #define _CCCL_FPMP_MATH_PLACEHOLDER_1A(name) \
    template<typename _FpType = float> \
    _CCCL_TRIVIAL_API void __fpmp2_##name (const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) \
    { \
        using mp2_t = fpmp2<_FpType>; \
        double __r = ::name(static_cast<double>(mp2_t(__x_hi, __x_lo))); \
        mp2_t __result(__r); \
        *__res_hi = __result.hi(); *__res_lo = __result.lo(); \
    }

    #define _CCCL_FPMP_MATH_PLACEHOLDER_2A(name) \
    template<typename _FpType = float> \
    _CCCL_TRIVIAL_API void __fpmp2_##name (const _FpType __x_hi, const _FpType __x_lo, const _FpType __y_hi, const _FpType __y_lo, _FpType* __res_hi, _FpType* __res_lo) \
    { \
        using mp2_t = fpmp2<_FpType>; \
        double __r = ::name(static_cast<double>(mp2_t(__x_hi, __x_lo)), static_cast<double>(mp2_t(__y_hi, __y_lo))); \
        mp2_t __result(__r); \
        *__res_hi = __result.hi(); *__res_lo = __result.lo(); \
    }

    #define _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETINT(name) \
    template<typename _FpType = float> \
    _CCCL_TRIVIAL_API int __fpmp2_##name (const _FpType __x_hi, const _FpType __x_lo) \
    { \
        using mp2_t = fpmp2<_FpType>; \
        return ::name(static_cast<double>(mp2_t(__x_hi, __x_lo))); \
    }

    #define _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETLL(name) \
    template<typename _FpType = float> \
    _CCCL_TRIVIAL_API long long int __fpmp2_##name (const _FpType __x_hi, const _FpType __x_lo) \
    { \
        using mp2_t = fpmp2<_FpType>; \
        return ::name(static_cast<double>(mp2_t(__x_hi, __x_lo))); \
    }

    #define _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETL(name) \
    template<typename _FpType = float> \
    _CCCL_TRIVIAL_API long int __fpmp2_##name (const _FpType __x_hi, const _FpType __x_lo) \
    { \
        using mp2_t = fpmp2<_FpType>; \
        return ::name(static_cast<double>(mp2_t(__x_hi, __x_lo))); \
    }

    #define _CCCL_FPMP_MATH_PLACEHOLDER_FP_INT(name) \
    template<typename _FpType = float> \
    _CCCL_TRIVIAL_API void __fpmp2_##name (const _FpType __x_hi, const _FpType __x_lo, int __n, _FpType* __res_hi, _FpType* __res_lo) \
    { \
        using mp2_t = fpmp2<_FpType>; \
        double __r = ::name(static_cast<double>(mp2_t(__x_hi, __x_lo)), __n); \
        mp2_t __result(__r); \
        *__res_hi = __result.hi(); *__res_lo = __result.lo(); \
    }

    #define _CCCL_FPMP_MATH_PLACEHOLDER_FP_LINT(name) \
    template<typename _FpType = float> \
    _CCCL_TRIVIAL_API void __fpmp2_##name (const _FpType __x_hi, const _FpType __x_lo, long int __n, _FpType* __res_hi, _FpType* __res_lo) \
    { \
        using mp2_t = fpmp2<_FpType>; \
        double __r = ::name(static_cast<double>(mp2_t(__x_hi, __x_lo)), __n); \
        mp2_t __result(__r); \
        *__res_hi = __result.hi(); *__res_lo = __result.lo(); \
    }

    #define _CCCL_FPMP_MATH_PLACEHOLDER_INT_FP(name) \
    template<typename _FpType = float> \
    _CCCL_TRIVIAL_API void __fpmp2_##name (int __n, const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) \
    { \
        using mp2_t = fpmp2<_FpType>; \
        double __r = ::name(__n, static_cast<double>(mp2_t(__x_hi, __x_lo))); \
        mp2_t __result(__r); \
        *__res_hi = __result.hi(); *__res_lo = __result.lo(); \
    }

    /* log2, log10, exp2, exp10, expm1: dedicated fp32mp2 implementations
     * live in the dedicated math section above (composed over the
     * dedicated fp32mp2 log/exp with ln(2)/ln(10) constants in fp32mp2;
     * expm1 has a small-|x| Taylor branch).  fp64mp2 specializations are
     * declared below via _CCCL_FPMP_CALL_FP64MP2_MATH.  exp10 had a hand
     * fallback even on fp32mp2 prior to the dedicated implementation;
     * that version is superseded. */
    _CCCL_FPMP_MATH_PLACEHOLDER_1A(logb)
    _CCCL_FPMP_MATH_PLACEHOLDER_1A(lgamma)
    _CCCL_FPMP_MATH_PLACEHOLDER_1A(tgamma)
    /* fmod, remainder: dedicated fp32mp2 implementations (integer
     * mantissa long-division, libdevice-style) live in the dedicated
     * math section above.  fp64mp2 paths stay on the explicit double
     * specializations further below. */
    _CCCL_FPMP_MATH_PLACEHOLDER_2A(hypot)
    _CCCL_FPMP_MATH_PLACEHOLDER_2A(copysign)
    _CCCL_FPMP_MATH_PLACEHOLDER_2A(fdim)
    _CCCL_FPMP_MATH_PLACEHOLDER_2A(nextafter)
    _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETINT(ilogb)
    _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETLL(llrint)
    _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETLL(llround)
    _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETL(lrint)
    _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETL(lround)
    /* ldexp, scalbn: dedicated fp32mp2 implementations live in the
     * dedicated math section above (bit-cast 3-piece base-2 scaling,
     * no fp64 round-trip).  scalbn forwards to ldexp since FLT_RADIX
     * is 2 on every IEEE 754 platform we support.  fp64mp2 paths stay
     * on the explicit double specializations further below for
     * symmetry. */
    _CCCL_FPMP_MATH_PLACEHOLDER_FP_LINT(scalbln)

    // Rounding functions rint, nearbyint
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_rint (const _FpType __x_hi,
                                                 const _FpType __x_lo,
                                                 _FpType*      __res_hi,
                                                 _FpType*      __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        const double __r = ::rint(static_cast<double>(mp2_t(__x_hi, __x_lo)));
        mp2_t __result(__r);
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_nearbyint (const _FpType __x_hi,
                                                      const _FpType __x_lo,
                                                      _FpType*      __res_hi,
                                                      _FpType*      __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        const double __r = ::nearbyint(static_cast<double>(mp2_t(__x_hi, __x_lo)));
        mp2_t __result(__r);
        *__res_hi = __result.hi();
        *__res_lo = __result.lo();
    }

    // Bessel functions (CUDA device; assert+return 0 on host)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_j0(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::j0(static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__x_hi; (void)__x_lo;
        assert(0 && "j0: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_j1(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::j1(static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__x_hi; (void)__x_lo;
        assert(0 && "j1: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_y0(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::y0(static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__x_hi; (void)__x_lo;
        assert(0 && "y0: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_y1(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::y1(static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__x_hi; (void)__x_lo;
        assert(0 && "y1: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    // Modified Bessel functions of the first kind (CUDA device; assert+return 0 on host)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_cyl_bessel_i0(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::cyl_bessel_i0(static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__x_hi; (void)__x_lo;
        assert(0 && "cyl_bessel_i0: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_cyl_bessel_i1(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::cyl_bessel_i1(static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__x_hi; (void)__x_lo;
        assert(0 && "cyl_bessel_i1: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    // Bessel functions with (int, fpmp2) -> fpmp2 signature (CUDA device; assert+return 0 on host)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_jn(const int __n, const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::jn(__n, static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__n; (void)__x_hi; (void)__x_lo;
        assert(0 && "jn: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_yn(const int __n, const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::yn(__n, static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__n; (void)__x_hi; (void)__x_lo;
        assert(0 && "yn: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    // frexp: extract mantissa and exponent
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_frexp(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo, int* __nptr) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __r = ::frexp(static_cast<double>(mp2_t(__x_hi, __x_lo)), __nptr);
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    // modf: break into integer and fractional parts
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_modf(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo, _FpType* __iptr_hi, _FpType* __iptr_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __ipart;
        double __r = ::modf(static_cast<double>(mp2_t(__x_hi, __x_lo)), &__ipart);
        mp2_t __result(__r), __iresult(__ipart);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
        *__iptr_hi = __iresult.hi(); *__iptr_lo = __iresult.lo();
    }

    // remquo: compute remainder and part of quotient
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_remquo(const _FpType __x_hi, const _FpType __x_lo, const _FpType __y_hi, const _FpType __y_lo, _FpType* __res_hi, _FpType* __res_lo, int* __quo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __r = ::remquo(static_cast<double>(mp2_t(__x_hi, __x_lo)), static_cast<double>(mp2_t(__y_hi, __y_lo)), __quo);
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    // Classification and sign functions
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API int __fpmp2_isfinite(const _FpType __x_hi, const _FpType __x_lo) noexcept { (void)__x_lo; return (std::isfinite)(static_cast<double>(__x_hi)); }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API int __fpmp2_isinf(const _FpType __x_hi, const _FpType __x_lo) noexcept { (void)__x_lo; return (std::isinf)(static_cast<double>(__x_hi)); }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API int __fpmp2_isnan(const _FpType __x_hi, const _FpType __x_lo) noexcept { (void)__x_lo; return (std::isnan)(static_cast<double>(__x_hi)); }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API int __fpmp2_signbit(const _FpType __x_hi, const _FpType __x_lo) noexcept { (void)__x_lo; return (std::signbit)(static_cast<double>(__x_hi)); }

    /*
    * CUDA-specific functions with host fallbacks
    *
    * Note: __fpmp2_exp10 used to live here as a `::exp10` (CUDA) /
    * `::pow(10, x)` (host) fallback.  The dedicated fp32mp2 version now
    * sits in the exponential/logarithmic family at the top of this
    * file; fp64mp2 routes through _CCCL_FPMP_CALL_FP64MP2_MATH with the
    * new _CCCL_FPMP_EXP10Q backend macro.
    */
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_sinpi(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __xd = static_cast<double>(mp2_t(__x_hi, __x_lo));
    #if defined(__CUDA_ARCH__)
        double __r = ::sinpi(__xd);
    #else
        double __r = ::sin(__xd * 3.14159265358979323846);
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_cospi(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __xd = static_cast<double>(mp2_t(__x_hi, __x_lo));
    #if defined(__CUDA_ARCH__)
        double __r = ::cospi(__xd);
    #else
        double __r = ::cos(__xd * 3.14159265358979323846);
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_sincospi(const _FpType __x_hi, const _FpType __x_lo, _FpType* __sin_hi, _FpType* __sin_lo, _FpType* __cos_hi, _FpType* __cos_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __xd = static_cast<double>(mp2_t(__x_hi, __x_lo));
        double __sd, __cd;
    #if defined(__CUDA_ARCH__)
        ::sincospi(__xd, &__sd, &__cd);
    #else
        double __xpi = __xd * 3.14159265358979323846;
        __sd = ::sin(__xpi); __cd = ::cos(__xpi);
    #endif
        mp2_t __s(__sd), __c(__cd);
        *__sin_hi = __s.hi(); *__sin_lo = __s.lo();
        *__cos_hi = __c.hi(); *__cos_lo = __c.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_normcdf(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __xd = static_cast<double>(mp2_t(__x_hi, __x_lo));
    #if defined(__CUDA_ARCH__)
        double __r = ::normcdf(__xd);
    #else
        double __r = 0.5 * ::erfc(-__xd * 0.70710678118654752440);
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    // (rcbrt: dedicated fp32mp2 implementation defined above; see __fpmp2_rcbrt.)

    // Inverse error functions and scaled complementary error function (CUDA device; assert+return 0 on host)
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_erfcinv(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::erfcinv(static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__x_hi; (void)__x_lo;
        assert(0 && "erfcinv: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_erfinv(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::erfinv(static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__x_hi; (void)__x_lo;
        assert(0 && "erfinv: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_erfcx(const _FpType __x_hi, const _FpType __x_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::erfcx(static_cast<double>(mp2_t(__x_hi, __x_lo)));
    #else
        (void)__x_hi; (void)__x_lo;
        assert(0 && "erfcx: no host fallback, returning 0");
        double __r = 0.0;
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    // Vector norm functions
    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_norm3d(const _FpType __a_hi, const _FpType __a_lo, const _FpType __b_hi, const _FpType __b_lo, const _FpType __c_hi, const _FpType __c_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __ad = static_cast<double>(mp2_t(__a_hi, __a_lo));
        double __bd = static_cast<double>(mp2_t(__b_hi, __b_lo));
        double __cd = static_cast<double>(mp2_t(__c_hi, __c_lo));
    #if defined(__CUDA_ARCH__)
        double __r = ::norm3d(__ad, __bd, __cd);
    #else
        double __r = ::sqrt(__ad*__ad + __bd*__bd + __cd*__cd);
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_norm4d(const _FpType __a_hi, const _FpType __a_lo, const _FpType __b_hi, const _FpType __b_lo, const _FpType __c_hi, const _FpType __c_lo, const _FpType __d_hi, const _FpType __d_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __ad = static_cast<double>(mp2_t(__a_hi, __a_lo));
        double __bd = static_cast<double>(mp2_t(__b_hi, __b_lo));
        double __cd = static_cast<double>(mp2_t(__c_hi, __c_lo));
        double __dd = static_cast<double>(mp2_t(__d_hi, __d_lo));
    #if defined(__CUDA_ARCH__)
        double __r = ::norm4d(__ad, __bd, __cd, __dd);
    #else
        double __r = ::sqrt(__ad*__ad + __bd*__bd + __cd*__cd + __dd*__dd);
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_rnorm3d(const _FpType __a_hi, const _FpType __a_lo, const _FpType __b_hi, const _FpType __b_lo, const _FpType __c_hi, const _FpType __c_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __ad = static_cast<double>(mp2_t(__a_hi, __a_lo));
        double __bd = static_cast<double>(mp2_t(__b_hi, __b_lo));
        double __cd = static_cast<double>(mp2_t(__c_hi, __c_lo));
    #if defined(__CUDA_ARCH__)
        double __r = ::rnorm3d(__ad, __bd, __cd);
    #else
        double __r = 1.0 / ::sqrt(__ad*__ad + __bd*__bd + __cd*__cd);
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_rnorm4d(const _FpType __a_hi, const _FpType __a_lo, const _FpType __b_hi, const _FpType __b_lo, const _FpType __c_hi, const _FpType __c_lo, const _FpType __d_hi, const _FpType __d_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
        double __ad = static_cast<double>(mp2_t(__a_hi, __a_lo));
        double __bd = static_cast<double>(mp2_t(__b_hi, __b_lo));
        double __cd = static_cast<double>(mp2_t(__c_hi, __c_lo));
        double __dd = static_cast<double>(mp2_t(__d_hi, __d_lo));
    #if defined(__CUDA_ARCH__)
        double __r = ::rnorm4d(__ad, __bd, __cd, __dd);
    #else
        double __r = 1.0 / ::sqrt(__ad*__ad + __bd*__bd + __cd*__cd + __dd*__dd);
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    template<typename _FpType = float>
    _CCCL_TRIVIAL_API void __fpmp2_rhypot(const _FpType __x_hi, const _FpType __x_lo, const _FpType __y_hi, const _FpType __y_lo, _FpType* __res_hi, _FpType* __res_lo) noexcept
    {
        using mp2_t = fpmp2<_FpType>;
    #if defined(__CUDA_ARCH__)
        double __r = ::rhypot(static_cast<double>(mp2_t(__x_hi, __x_lo)), static_cast<double>(mp2_t(__y_hi, __y_lo)));
    #else
        double __r = 1.0 / ::hypot(static_cast<double>(mp2_t(__x_hi, __x_lo)), static_cast<double>(mp2_t(__y_hi, __y_lo)));
    #endif
        mp2_t __result(__r);
        *__res_hi = __result.hi(); *__res_lo = __result.lo();
    }

    /* Cleanup: undefine the placeholder factory macros so they don't
     * leak into headers/translation units that include this file. */
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_1A
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_2A
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETINT
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETLL
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_1A_RETL
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_FP_INT
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_FP_LINT
    #undef _CCCL_FPMP_MATH_PLACEHOLDER_INT_FP

/*
* ============================================================================
* Double precision (fp64mp2) template specializations
* ============================================================================
*/

    #if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)

        #if defined(__CUDA_ARCH__) \
          && (defined(__aarch64__) \
          || defined(_M_ARM64)) \
          && defined(_CCCL_FPMP_CUDA_FP128_INTRINSICS) \
          && !(defined(__GNUC__) \
          && !defined(__clang__) \
          && !defined(__NVCOMPILER_MAJOR__) \
          && ((__GNUC__ > 13) \
          || (__GNUC__ == 13 && __GNUC_MINOR__ >= 1))) \
          && !defined(_CCCL_FLOAT128_CPP_SPELLING_ENABLED)
            #define _CCCL_FLOAT128_CPP_SPELLING_ENABLED
        #endif
        
} // namespace cuda::experimental
        #if defined(__CUDA_ARCH__)
            // CUDA device
            #include "crt/device_fp128_functions.h"
        #elif (_CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH == 1)
            // x86 host: libquadmath
            #include <quadmath.h>
        #elif (_CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 == 1)
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
        #if defined(__CUDA_ARCH__) && defined(_CCCL_FPMP_CUDA_FP128_INTRINSICS) \
          && (defined(_CCCL_FLOAT128_CPP_SPELLING_ENABLED) ||  defined(__FLOAT128_C_SPELLING_ENABLED__))
            #define _CCCL_FPMP_EXPQ(x)         __nv_fp128_exp(x)
            #define _CCCL_FPMP_EXP2Q(x)        __nv_fp128_exp2(x)
            #define _CCCL_FPMP_EXP10Q(x)       __nv_fp128_exp10(x)
            #define _CCCL_FPMP_EXPM1Q(x)       __nv_fp128_expm1(x)
            #define _CCCL_FPMP_LOGQ(x)         __nv_fp128_log(x)
            #define _CCCL_FPMP_LOG2Q(x)        __nv_fp128_log2(x)
            #define _CCCL_FPMP_LOG10Q(x)       __nv_fp128_log10(x)
            #define _CCCL_FPMP_LOG1PQ(x)       __nv_fp128_log1p(x)
            #define _CCCL_FPMP_SINQ(x)         __nv_fp128_sin(x)
            #define _CCCL_FPMP_COSQ(x)         __nv_fp128_cos(x)
            #define _CCCL_FPMP_TANQ(x)         __nv_fp128_tan(x)
            #define _CCCL_FPMP_ASINQ(x)        __nv_fp128_asin(x)
            #define _CCCL_FPMP_ACOSQ(x)        __nv_fp128_acos(x)
            #define _CCCL_FPMP_ATANQ(x)        __nv_fp128_atan(x)
            #define _CCCL_FPMP_SINHQ(x)        __nv_fp128_sinh(x)
            #define _CCCL_FPMP_COSHQ(x)        __nv_fp128_cosh(x)
            #define _CCCL_FPMP_TANHQ(x)        __nv_fp128_tanh(x)
            #define _CCCL_FPMP_ASINHQ(x)       __nv_fp128_asinh(x)
            #define _CCCL_FPMP_ACOSHQ(x)       __nv_fp128_acosh(x)
            #define _CCCL_FPMP_ATANHQ(x)       __nv_fp128_atanh(x)
            #define _CCCL_FPMP_SQRTQ(x)        __nv_fp128_sqrt(x)
            #define _CCCL_FPMP_FABSQ(x)        __nv_fp128_fabs(x)
            #define _CCCL_FPMP_POWQ(x,y)       __nv_fp128_pow((x),(y))
            #define _CCCL_FPMP_FMODQ(x,y)      __nv_fp128_fmod((x),(y))
            #define _CCCL_FPMP_REMAINDERQ(x,y) __nv_fp128_remainder((x),(y))
            #define _CCCL_FPMP_FLOORQ(x)       __nv_fp128_floor(x)
            #define _CCCL_FPMP_CEILQ(x)        __nv_fp128_ceil(x)
            #define _CCCL_FPMP_TRUNCQ(x)       __nv_fp128_trunc(x)
            #define _CCCL_FPMP_ROUNDQ(x)       __nv_fp128_round(x)
            #define _CCCL_FPMP_RINTQ(x)        __nv_fp128_rint(x)
            #define _CCCL_FPMP_NEARBYINTQ(x)   __nv_fp128_rint(x)
            #define _CCCL_FPMP_CBRTQ(x)        __nv_fp128_copysign(__nv_fp128_pow(__nv_fp128_fabs(x), (__fpmp_fp128)1 / (__fpmp_fp128)3), (x))
            #define _CCCL_FPMP_ATAN2Q(y,x)     ((__fpmp_fp128)atan2((double)(y), (double)(x)))
            #define _CCCL_FPMP_ERFQ(x)         ((__fpmp_fp128)erf((double)(x)))
            #define _CCCL_FPMP_ERFCQ(x)        ((__fpmp_fp128)erfc((double)(x)))
        // ----------------------------------------------------------------------
        // Branch 2 -- HOST with libquadmath (the primary x86_64 host path).
        // Target: host compile (no __CUDA_ARCH__) where libquadmath is present
        // (typically x86_64 GCC distributions). Reference math uses the true
        // binary128 libquadmath entry points (the `*q` suffix); __fpmp_fp128 is
        // __float128 here. The explicit !defined(__CUDA_ARCH__) guard keeps the
        // device pass on an x86_64 host (where _CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH is
        // also 1) from matching this host-only branch.
        // ----------------------------------------------------------------------
        #elif (_CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH == 1) && !defined(__CUDA_ARCH__)
            #define _CCCL_FPMP_EXPQ(x)         expq(x)
            #define _CCCL_FPMP_EXP2Q(x)        exp2q(x)
            #define _CCCL_FPMP_EXPM1Q(x)       expm1q(x)
            #define _CCCL_FPMP_LOGQ(x)         logq(x)
            #define _CCCL_FPMP_LOG2Q(x)        log2q(x)
            #define _CCCL_FPMP_LOG10Q(x)       log10q(x)
            #define _CCCL_FPMP_LOG1PQ(x)       log1pq(x)
            #define _CCCL_FPMP_SINQ(x)         sinq(x)
            #define _CCCL_FPMP_COSQ(x)         cosq(x)
            #define _CCCL_FPMP_TANQ(x)         tanq(x)
            #define _CCCL_FPMP_ASINQ(x)        asinq(x)
            #define _CCCL_FPMP_ACOSQ(x)        acosq(x)
            #define _CCCL_FPMP_ATANQ(x)        atanq(x)
            #define _CCCL_FPMP_SINHQ(x)        sinhq(x)
            #define _CCCL_FPMP_COSHQ(x)        coshq(x)
            #define _CCCL_FPMP_TANHQ(x)        tanhq(x)
            #define _CCCL_FPMP_ASINHQ(x)       asinhq(x)
            #define _CCCL_FPMP_ACOSHQ(x)       acoshq(x)
            #define _CCCL_FPMP_ATANHQ(x)       atanhq(x)
            #define _CCCL_FPMP_SQRTQ(x)        sqrtq(x)
            #define _CCCL_FPMP_CBRTQ(x)        cbrtq(x)
            #define _CCCL_FPMP_FABSQ(x)        fabsq(x)
            #define _CCCL_FPMP_POWQ(x,y)       powq((x),(y))
            #define _CCCL_FPMP_ATAN2Q(y,x)     atan2q((y),(x))
            #define _CCCL_FPMP_FMODQ(x,y)      fmodq((x),(y))
            #define _CCCL_FPMP_REMAINDERQ(x,y) remainderq((x),(y))
            #define _CCCL_FPMP_ERFQ(x)         erfq(x)
            #define _CCCL_FPMP_ERFCQ(x)        erfcq(x)
            #define _CCCL_FPMP_FLOORQ(x)       floorq(x)
            #define _CCCL_FPMP_CEILQ(x)        ceilq(x)
            #define _CCCL_FPMP_TRUNCQ(x)       truncq(x)
            #define _CCCL_FPMP_ROUNDQ(x)       roundq(x)
            #define _CCCL_FPMP_RINTQ(x)        rintq(x)
            #define _CCCL_FPMP_NEARBYINTQ(x)   nearbyintq(x)
            #define _CCCL_FPMP_EXP10Q(x)       powq((__float128)10.0, (x))
        // ----------------------------------------------------------------------
        // Branch 3 -- HOST, no libquadmath, 128-bit `long double`
        //             (the primary AArch64 / non-x86 host path).
        // Target: host compile (no __CUDA_ARCH__) on platforms whose C
        // `long double` is a true 128-bit type (IEEE binary128 on AArch64 /
        // PPC64LE, or 80-bit x87 extended on x86 without libquadmath) AND where
        // libquadmath is unavailable. Reference math uses the standard C
        // `long double` libm entry points (the `*l` suffix).
        // ----------------------------------------------------------------------
        #elif (_CCCL_FPMP_HOST_SUPPORTS_LDOUBLE128 == 1) && !defined(__CUDA_ARCH__) \
         && (_CCCL_FPMP_HOST_SUPPORTS_LIBQUADMATH == 0)
            #define _CCCL_FPMP_EXPQ(x)         expl(x)
            #define _CCCL_FPMP_EXP2Q(x)        exp2l(x)
            #define _CCCL_FPMP_EXPM1Q(x)       expm1l(x)
            #define _CCCL_FPMP_LOGQ(x)         logl(x)
            #define _CCCL_FPMP_LOG2Q(x)        log2l(x)
            #define _CCCL_FPMP_LOG10Q(x)       log10l(x)
            #define _CCCL_FPMP_LOG1PQ(x)       log1pl(x)
            #define _CCCL_FPMP_SINQ(x)         sinl(x)
            #define _CCCL_FPMP_COSQ(x)         cosl(x)
            #define _CCCL_FPMP_TANQ(x)         tanl(x)
            #define _CCCL_FPMP_ASINQ(x)        asinl(x)
            #define _CCCL_FPMP_ACOSQ(x)        acosl(x)
            #define _CCCL_FPMP_ATANQ(x)        atanl(x)
            #define _CCCL_FPMP_SINHQ(x)        sinhl(x)
            #define _CCCL_FPMP_COSHQ(x)        coshl(x)
            #define _CCCL_FPMP_TANHQ(x)        tanhl(x)
            #define _CCCL_FPMP_ASINHQ(x)       asinhl(x)
            #define _CCCL_FPMP_ACOSHQ(x)       acoshl(x)
            #define _CCCL_FPMP_ATANHQ(x)       atanhl(x)
            #define _CCCL_FPMP_SQRTQ(x)        sqrtl(x)
            #define _CCCL_FPMP_FABSQ(x)        fabsl(x)
            #define _CCCL_FPMP_POWQ(x,y)       powl((x),(y))
            #define _CCCL_FPMP_CBRTQ(x)        cbrtl(x)
            #define _CCCL_FPMP_ATAN2Q(y,x)     atan2l((y),(x))
            #define _CCCL_FPMP_FMODQ(x,y)      fmodl((x),(y))
            #define _CCCL_FPMP_REMAINDERQ(x,y) remainderl((x),(y))
            #define _CCCL_FPMP_ERFQ(x)         erfl(x)
            #define _CCCL_FPMP_ERFCQ(x)        erfcl(x)
            #define _CCCL_FPMP_FLOORQ(x)       floorl(x)
            #define _CCCL_FPMP_CEILQ(x)        ceill(x)
            #define _CCCL_FPMP_TRUNCQ(x)       truncl(x)
            #define _CCCL_FPMP_ROUNDQ(x)       roundl(x)
            #define _CCCL_FPMP_RINTQ(x)        rintl(x)
            #define _CCCL_FPMP_NEARBYINTQ(x)   nearbyintl(x)
            #define _CCCL_FPMP_EXP10Q(x)       ((long double)powl(10.0L, (x)))
        // ----------------------------------------------------------------------
        // Branch 4 -- CUDA DEVICE WITHOUT a usable native fp128 path
        //             (the fp64 fallback).
        // Target: any device build (x86_64 or AArch64) that did NOT enable the
        // extended NVVM intrinsics (no _CCCL_FPMP_CUDA_FP128_INTRINSICS), or where the
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
            #define _CCCL_FPMP_EXPQ(x)         ((__fpmp_fp128)exp((double)(x)))
            #define _CCCL_FPMP_EXP2Q(x)        ((__fpmp_fp128)exp2((double)(x)))
            #define _CCCL_FPMP_EXP10Q(x)       ((__fpmp_fp128)exp10((double)(x)))
            #define _CCCL_FPMP_EXPM1Q(x)       ((__fpmp_fp128)expm1((double)(x)))
            #define _CCCL_FPMP_LOGQ(x)         ((__fpmp_fp128)log((double)(x)))
            #define _CCCL_FPMP_LOG2Q(x)        ((__fpmp_fp128)log2((double)(x)))
            #define _CCCL_FPMP_LOG10Q(x)       ((__fpmp_fp128)log10((double)(x)))
            #define _CCCL_FPMP_LOG1PQ(x)       ((__fpmp_fp128)log1p((double)(x)))
            #define _CCCL_FPMP_SINQ(x)         ((__fpmp_fp128)sin((double)(x)))
            #define _CCCL_FPMP_COSQ(x)         ((__fpmp_fp128)cos((double)(x)))
            #define _CCCL_FPMP_TANQ(x)         ((__fpmp_fp128)tan((double)(x)))
            #define _CCCL_FPMP_ASINQ(x)        ((__fpmp_fp128)asin((double)(x)))
            #define _CCCL_FPMP_ACOSQ(x)        ((__fpmp_fp128)acos((double)(x)))
            #define _CCCL_FPMP_ATANQ(x)        ((__fpmp_fp128)atan((double)(x)))
            #define _CCCL_FPMP_SINHQ(x)        ((__fpmp_fp128)sinh((double)(x)))
            #define _CCCL_FPMP_COSHQ(x)        ((__fpmp_fp128)cosh((double)(x)))
            #define _CCCL_FPMP_TANHQ(x)        ((__fpmp_fp128)tanh((double)(x)))
            #define _CCCL_FPMP_ASINHQ(x)       ((__fpmp_fp128)asinh((double)(x)))
            #define _CCCL_FPMP_ACOSHQ(x)       ((__fpmp_fp128)acosh((double)(x)))
            #define _CCCL_FPMP_ATANHQ(x)       ((__fpmp_fp128)atanh((double)(x)))
            #define _CCCL_FPMP_SQRTQ(x)        ((__fpmp_fp128)sqrt((double)(x)))
            #define _CCCL_FPMP_FABSQ(x)        ((__fpmp_fp128)fabs((double)(x)))
            #define _CCCL_FPMP_POWQ(x,y)       ((__fpmp_fp128)pow((double)(x), (double)(y)))
            #define _CCCL_FPMP_CBRTQ(x)        ((__fpmp_fp128)cbrt((double)(x)))
            #define _CCCL_FPMP_ATAN2Q(y,x)     ((__fpmp_fp128)atan2((double)(y), (double)(x)))
            #define _CCCL_FPMP_FMODQ(x,y)      ((__fpmp_fp128)fmod((double)(x), (double)(y)))
            #define _CCCL_FPMP_REMAINDERQ(x,y) ((__fpmp_fp128)remainder((double)(x), (double)(y)))
            #define _CCCL_FPMP_ERFQ(x)         ((__fpmp_fp128)erf((double)(x)))
            #define _CCCL_FPMP_ERFCQ(x)        ((__fpmp_fp128)erfc((double)(x)))
            #define _CCCL_FPMP_FLOORQ(x)       ((__fpmp_fp128)floor((double)(x)))
            #define _CCCL_FPMP_CEILQ(x)        ((__fpmp_fp128)ceil((double)(x)))
            #define _CCCL_FPMP_TRUNCQ(x)       ((__fpmp_fp128)trunc((double)(x)))
            #define _CCCL_FPMP_ROUNDQ(x)       ((__fpmp_fp128)round((double)(x)))
            #define _CCCL_FPMP_RINTQ(x)        ((__fpmp_fp128)rint((double)(x)))
            #define _CCCL_FPMP_NEARBYINTQ(x)   ((__fpmp_fp128)nearbyint((double)(x)))
        #endif // _CCCL_FPMP_FP128_MATH_FALLBACK == 1

        /*
         * Simplified dispatch macro: uses __FPMP_*Q wrapper macros which already
         * handle CUDA/libquadmath/long double dispatching internally.
         */
        #define _CCCL_FPMP_CALL_FP64MP2_MATH(dfunc,qfunc,xhi,xlo,reshi,reslo) __fpmp2_from_quad(qfunc(__fpmp2_to_quad(xhi,xlo)),reshi,reslo)
        #define _CCCL_FPMP_CALL_FP64MP2_MATH_2A(dfunc,qfunc,xhi,xlo,yhi,ylo,reshi,reslo) __fpmp2_from_quad(qfunc(__fpmp2_to_quad(xhi,xlo),__fpmp2_to_quad(yhi,ylo)),reshi,reslo)
    #else
        #define _CCCL_FPMP_CALL_FP64MP2_MATH(dfunc,qfunc,xhi,xlo,reshi,reslo) __fpmp2_from_double(::dfunc(__fpmp2_to_double(xhi,xlo)),reshi,reslo)
        #define _CCCL_FPMP_CALL_FP64MP2_MATH_2A(dfunc,qfunc,xhi,xlo,yhi,ylo,reshi,reslo) __fpmp2_from_double(::dfunc(__fpmp2_to_double(xhi,xlo),__fpmp2_to_double(yhi,ylo)),reshi,reslo)
    #endif // _CCCL_FPMP_FP128_MATH_FALLBACK == 1

    template<> _CCCL_API inline void __fpmp2_exp<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(exp, _CCCL_FPMP_EXPQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(log, _CCCL_FPMP_LOGQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log2<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(log2, _CCCL_FPMP_LOG2Q, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log10<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(log10, _CCCL_FPMP_LOG10Q, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log1p<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(log1p, _CCCL_FPMP_LOG1PQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sin<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(sin, _CCCL_FPMP_SINQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cos<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(cos, _CCCL_FPMP_COSQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_asin<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(asin, _CCCL_FPMP_ASINQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_acos<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(acos, _CCCL_FPMP_ACOSQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_atan<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(atan, _CCCL_FPMP_ATANQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sinh<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(sinh, _CCCL_FPMP_SINHQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cosh<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(cosh, _CCCL_FPMP_COSHQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_tanh<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(tanh, _CCCL_FPMP_TANHQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_pow<double>    (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH_2A(pow, _CCCL_FPMP_POWQ, __x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sincos<double> (const double __x_hi, const double __x_lo, double* __sin_hi, double* __sin_lo, double* __cos_hi, double* __cos_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(sin, _CCCL_FPMP_SINQ, __x_hi, __x_lo, __sin_hi, __sin_lo); _CCCL_FPMP_CALL_FP64MP2_MATH(cos, _CCCL_FPMP_COSQ, __x_hi, __x_lo, __cos_hi, __cos_lo); }
    
    // Functions with no 128-bit support in CUDA
    template<> _CCCL_API inline void __fpmp2_erf<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::erf(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);}
    template<> _CCCL_API inline void __fpmp2_erfc<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::erfc(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_boys_f0<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
        double __x = __fpmp2_to_double(__x_hi, __x_lo);
        double __r;
        if (__x < 1e-15) { __r = 1.0; }
        else { __r = 0.5 * ::sqrt(3.14159265358979323846 / __x) * ::erf(::sqrt(__x)); }
        __fpmp2_from_double(__r, __res_hi, __res_lo);
    }
    template<> _CCCL_API inline void __fpmp2_normcdfinv<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
    {
        double __p = __fpmp2_to_double(__x_hi, __x_lo);
    #if defined(__CUDA_ARCH__)
        // Hardcoded value since M_SQRT2 is not guaranteed to be defined on all platforms
        constexpr double __sqrt2_v = 1.41421356237309504880;
        __fpmp2_from_double(-__sqrt2_v * ::erfcinv(2.0 * __p), __res_hi, __res_lo);
    #else
        // Not implemented yet: double precision normcdfinv fallback to float precision
        float __f_hi, __f_lo;
        __fpmp2_normcdfinv(static_cast<float>(__p), 0.0f, &__f_hi, &__f_lo);
        *__res_hi = static_cast<double>(__f_hi) + static_cast<double>(__f_lo);
        *__res_lo = 0.0;
    #endif
    }

    template<> _CCCL_API inline void __fpmp2_cbrt<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept 
    {
        #if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
            __fpmp_fp128 __res = _CCCL_FPMP_CBRTQ(__fpmp2_to_quad(__x_hi, __x_lo));
            __fpmp2_from_quad(__res, __res_hi, __res_lo);
        #else
            double __res = ::cbrt(__fpmp2_to_double(__x_hi, __x_lo));
            __fpmp2_from_double(__res, __res_hi, __res_lo);
        #endif
    }
// Note: On CUDA device, _CCCL_FPMP_ATAN2Q widens through double atan2 (no fp128 intrinsic); _CCCL_FPMP_CBRTQ is reconstructed from __nv_fp128_pow.
    template<> _CCCL_API inline void __fpmp2_atan2<double>  (const double __y_hi, const double __y_lo, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept 
    {
        #if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
            __fpmp_fp128 __res = _CCCL_FPMP_ATAN2Q(__fpmp2_to_quad(__y_hi, __y_lo), __fpmp2_to_quad(__x_hi, __x_lo));
            __fpmp2_from_quad(__res, __res_hi, __res_lo);
        #else
            double __res = ::atan2(__fpmp2_to_double(__y_hi, __y_lo), __fpmp2_to_double(__x_hi, __x_lo));
            __fpmp2_from_double(__res, __res_hi, __res_lo);
        #endif
    } // __fpmp2_atan2<double>

    // Additional fp64mp2 specializations (double-precision fallback for all)
    template<> _CCCL_API inline void __fpmp2_acosh<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(acosh, _CCCL_FPMP_ACOSHQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_asinh<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(asinh, _CCCL_FPMP_ASINHQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_atanh<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(atanh, _CCCL_FPMP_ATANHQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_tan<double>     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(tan, _CCCL_FPMP_TANQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_exp2<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(exp2,  _CCCL_FPMP_EXP2Q,  __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_expm1<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(expm1, _CCCL_FPMP_EXPM1Q, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_logb<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::logb(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo); }
    // Rounding family: fp64mp2 routes through higher precision (fp128) when
    // available; otherwise falls back to fp64 system rounding. This avoids
    // precision loss from collapsing the (hi, lo) pair into a single double.
    template<> _CCCL_API inline void __fpmp2_ceil<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(ceil,      _CCCL_FPMP_CEILQ,      __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_floor<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(floor,     _CCCL_FPMP_FLOORQ,     __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_trunc<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(trunc,     _CCCL_FPMP_TRUNCQ,     __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_round<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(round,     _CCCL_FPMP_ROUNDQ,     __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rint<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(rint,      _CCCL_FPMP_RINTQ,      __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_nearbyint<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH(nearbyint, _CCCL_FPMP_NEARBYINTQ, __x_hi, __x_lo, __res_hi, __res_lo); }
    // __fpmp2_fabs<double>: handled by the type-agnostic primary template above (no double round-trip).
    template<> _CCCL_API inline void __fpmp2_lgamma<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::lgamma(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_tgamma<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::tgamma(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_j0<double>      (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::j0(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__x_hi; (void)__x_lo; assert(0 && "j0: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_j1<double>      (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::j1(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__x_hi; (void)__x_lo; assert(0 && "j1: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_y0<double>      (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::y0(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__x_hi; (void)__x_lo; assert(0 && "y0: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_y1<double>      (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::y1(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__x_hi; (void)__x_lo; assert(0 && "y1: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_cyl_bessel_i0<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::cyl_bessel_i0(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__x_hi; (void)__x_lo; assert(0 && "cyl_bessel_i0: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_cyl_bessel_i1<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::cyl_bessel_i1(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__x_hi; (void)__x_lo; assert(0 && "cyl_bessel_i1: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    // __fpmp2_fmax<double>, __fpmp2_fmin<double>, __fpmp2_max<double>, __fpmp2_min<double>:
    // handled by the type-agnostic primary templates above (no double round-trip).
    template<> _CCCL_API inline void __fpmp2_fmod<double>    (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH_2A(fmod,      _CCCL_FPMP_FMODQ,      __x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_remainder<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { _CCCL_FPMP_CALL_FP64MP2_MATH_2A(remainder, _CCCL_FPMP_REMAINDERQ, __x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_hypot<double>   (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::hypot(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_copysign<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::copysign(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fdim<double>    (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::fdim(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_nextafter<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::nextafter(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rhypot<double>  (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept
    {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::rhypot(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);
    #else
        __fpmp2_from_double(1.0 / ::hypot(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo)), __res_hi, __res_lo);
    #endif
    }
    template<> _CCCL_API inline int __fpmp2_ilogb<double>    (const double __x_hi, const double __x_lo) noexcept { return ::ilogb(__fpmp2_to_double(__x_hi, __x_lo)); }
    template<> _CCCL_API inline long long int __fpmp2_llrint<double> (const double __x_hi, const double __x_lo) noexcept { return ::llrint(__fpmp2_to_double(__x_hi, __x_lo)); }
    template<> _CCCL_API inline long long int __fpmp2_llround<double>(const double __x_hi, const double __x_lo) noexcept { return ::llround(__fpmp2_to_double(__x_hi, __x_lo)); }
    template<> _CCCL_API inline long int __fpmp2_lrint<double>  (const double __x_hi, const double __x_lo) noexcept { return ::lrint(__fpmp2_to_double(__x_hi, __x_lo)); }
    template<> _CCCL_API inline long int __fpmp2_lround<double> (const double __x_hi, const double __x_lo) noexcept { return ::lround(__fpmp2_to_double(__x_hi, __x_lo)); }
    template<> _CCCL_API inline int __fpmp2_isfinite<double> (const double __x_hi, const double __x_lo) noexcept { (void)__x_lo; return (std::isfinite)(__x_hi); }
    template<> _CCCL_API inline int __fpmp2_isinf<double>    (const double __x_hi, const double __x_lo) noexcept { (void)__x_lo; return (std::isinf)(__x_hi); }
    template<> _CCCL_API inline int __fpmp2_isnan<double>    (const double __x_hi, const double __x_lo) noexcept { (void)__x_lo; return (std::isnan)(__x_hi); }
    template<> _CCCL_API inline int __fpmp2_signbit<double>  (const double __x_hi, const double __x_lo) noexcept { (void)__x_lo; return (std::signbit)(__x_hi); }
    template<> _CCCL_API inline void __fpmp2_ldexp<double>   (const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::ldexp(__fpmp2_to_double(__x_hi, __x_lo), __n), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_scalbn<double>  (const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::scalbn(__fpmp2_to_double(__x_hi, __x_lo), __n), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_scalbln<double> (const double __x_hi, const double __x_lo, long int __n, double* __res_hi, double* __res_lo) noexcept { __fpmp2_from_double(::scalbln(__fpmp2_to_double(__x_hi, __x_lo), __n), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_jn<double>      (int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::jn(__n, __fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__n; (void)__x_hi; (void)__x_lo; assert(0 && "jn: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_yn<double>      (int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::yn(__n, __fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__n; (void)__x_hi; (void)__x_lo; assert(0 && "yn: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_frexp<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo, int* __nptr) noexcept { __fpmp2_from_double(::frexp(__fpmp2_to_double(__x_hi, __x_lo), __nptr), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_modf<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo, double* __iptr_hi, double* __iptr_lo) noexcept { double __ip; __fpmp2_from_double(::modf(__fpmp2_to_double(__x_hi, __x_lo), &__ip), __res_hi, __res_lo); __fpmp2_from_double(__ip, __iptr_hi, __iptr_lo); }
    template<> _CCCL_API inline void __fpmp2_remquo<double>  (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo, int* __quo) noexcept { __fpmp2_from_double(::remquo(__fpmp2_to_double(__x_hi, __x_lo), __fpmp2_to_double(__y_hi, __y_lo), __quo), __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_exp10<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
    {
    #if (_CCCL_FPMP_FP128_MATH_FALLBACK == 1)
        /* fp128 path: _CCCL_FPMP_EXP10Q handles every backend (libquadmath
         * powq, CUDA double widen, host long-double powl). */
        _CCCL_FPMP_CALL_FP64MP2_MATH(exp10, _CCCL_FPMP_EXP10Q, __x_hi, __x_lo, __res_hi, __res_lo);
    #else
        /* fp64 fallback: libm has no portable `exp10`; synthesize via
         * pow(10, x).  CUDA device has the intrinsic, prefer it. */
        double __xd = __fpmp2_to_double(__x_hi, __x_lo);
        #if defined(__CUDA_ARCH__)
            __fpmp2_from_double(::exp10(__xd), __res_hi, __res_lo);
        #else
            __fpmp2_from_double(::pow(10.0, __xd), __res_hi, __res_lo);
        #endif
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_sinpi<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
    {
        double __xd = __fpmp2_to_double(__x_hi, __x_lo);
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::sinpi(__xd), __res_hi, __res_lo);
    #else
        __fpmp2_from_double(::sin(__xd * 3.14159265358979323846), __res_hi, __res_lo);
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_cospi<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
    {
        double __xd = __fpmp2_to_double(__x_hi, __x_lo);
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::cospi(__xd), __res_hi, __res_lo);
    #else
        __fpmp2_from_double(::cos(__xd * 3.14159265358979323846), __res_hi, __res_lo);
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_sincospi<double>(const double __x_hi, const double __x_lo, double* __sin_hi, double* __sin_lo, double* __cos_hi, double* __cos_lo) noexcept
    {
        double __xd = __fpmp2_to_double(__x_hi, __x_lo);
    #if defined(__CUDA_ARCH__)
        double __sd, __cd; ::sincospi(__xd, &__sd, &__cd);
        __fpmp2_from_double(__sd, __sin_hi, __sin_lo); __fpmp2_from_double(__cd, __cos_hi, __cos_lo);
    #else
        double __xpi = __xd * 3.14159265358979323846;
        __fpmp2_from_double(::sin(__xpi), __sin_hi, __sin_lo); __fpmp2_from_double(::cos(__xpi), __cos_hi, __cos_lo);
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_normcdf<double> (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
    {
        double __xd = __fpmp2_to_double(__x_hi, __x_lo);
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::normcdf(__xd), __res_hi, __res_lo);
    #else
        __fpmp2_from_double(0.5 * ::erfc(-__xd * 0.70710678118654752440), __res_hi, __res_lo);
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_rcbrt<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept
    {
        double __xd = __fpmp2_to_double(__x_hi, __x_lo);
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::rcbrt(__xd), __res_hi, __res_lo);
    #else
        __fpmp2_from_double(1.0 / ::cbrt(__xd), __res_hi, __res_lo);
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_erfcinv<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::erfcinv(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__x_hi; (void)__x_lo; assert(0 && "erfcinv: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_erfinv<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::erfinv(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__x_hi; (void)__x_lo; assert(0 && "erfinv: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_erfcx<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept {
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::erfcx(__fpmp2_to_double(__x_hi, __x_lo)), __res_hi, __res_lo);
    #else
        (void)__x_hi; (void)__x_lo; assert(0 && "erfcx: no host fallback, returning 0"); *__res_hi = 0.0; *__res_lo = 0.0;
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_norm3d<double>(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, double* __res_hi, double* __res_lo) noexcept {
        double __ad = __fpmp2_to_double(__a_hi, __a_lo), __bd = __fpmp2_to_double(__b_hi, __b_lo), __cd = __fpmp2_to_double(__c_hi, __c_lo);
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::norm3d(__ad, __bd, __cd), __res_hi, __res_lo);
    #else
        __fpmp2_from_double(::sqrt(__ad*__ad + __bd*__bd + __cd*__cd), __res_hi, __res_lo);
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_norm4d<double>(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, const double __d_hi, const double __d_lo, double* __res_hi, double* __res_lo) noexcept {
        double __ad = __fpmp2_to_double(__a_hi, __a_lo), __bd = __fpmp2_to_double(__b_hi, __b_lo), __cd = __fpmp2_to_double(__c_hi, __c_lo), __dd = __fpmp2_to_double(__d_hi, __d_lo);
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::norm4d(__ad, __bd, __cd, __dd), __res_hi, __res_lo);
    #else
        __fpmp2_from_double(::sqrt(__ad*__ad + __bd*__bd + __cd*__cd + __dd*__dd), __res_hi, __res_lo);
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_rnorm3d<double>(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, double* __res_hi, double* __res_lo) noexcept {
        double __ad = __fpmp2_to_double(__a_hi, __a_lo), __bd = __fpmp2_to_double(__b_hi, __b_lo), __cd = __fpmp2_to_double(__c_hi, __c_lo);
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::rnorm3d(__ad, __bd, __cd), __res_hi, __res_lo);
    #else
        __fpmp2_from_double(1.0 / ::sqrt(__ad*__ad + __bd*__bd + __cd*__cd), __res_hi, __res_lo);
    #endif
    }
    template<> _CCCL_API inline void __fpmp2_rnorm4d<double>(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, const double __d_hi, const double __d_lo, double* __res_hi, double* __res_lo) noexcept {
        double __ad = __fpmp2_to_double(__a_hi, __a_lo), __bd = __fpmp2_to_double(__b_hi, __b_lo), __cd = __fpmp2_to_double(__c_hi, __c_lo), __dd = __fpmp2_to_double(__d_hi, __d_lo);
    #if defined(__CUDA_ARCH__)
        __fpmp2_from_double(::rnorm4d(__ad, __bd, __cd, __dd), __res_hi, __res_lo);
    #else
        __fpmp2_from_double(1.0 / ::sqrt(__ad*__ad + __bd*__bd + __cd*__cd + __dd*__dd), __res_hi, __res_lo);
    #endif
    }


#else // _CCCL_FPMP_USE_LIB
    /*
    * ============================================================================
    * Library mode - fp32mp2 declarations
    * ============================================================================
    */
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_exp    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_log    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_log2   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_log10  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_log1p  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_pow    (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_cbrt   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_sin    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_cos    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_sincos (const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_asin   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_acos   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_atan   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_atan2  (const float __y_hi, const float __y_lo, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_sinh   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_cosh   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_tanh   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_erf    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_erfc   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_normcdfinv (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_icdf32     (uint32_t __x, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_icdf64     (uint64_t __x, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_acosh  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_asinh  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_atanh  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_tan    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_exp2   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_exp10  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_expm1  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_logb   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_ceil   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_floor  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_trunc  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_round  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_rint   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_nearbyint(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fabs   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_lgamma (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_tgamma (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_j0     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_j1     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_y0     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_y1     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_cyl_bessel_i0(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_cyl_bessel_i1(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_sinpi  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_cospi  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_normcdf(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_rcbrt  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_erfcinv(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_erfinv (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_erfcx  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_boys_f0(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_norm3d (const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, const float __c_hi, const float __c_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_norm4d (const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, const float __c_hi, const float __c_lo, const float __d_hi, const float __d_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_rnorm3d(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, const float __c_hi, const float __c_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_rnorm4d(const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, const float __c_hi, const float __c_lo, const float __d_hi, const float __d_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fmax   (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fmin   (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_max    (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_min    (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fmod   (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_remainder(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_hypot  (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_copysign(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_fdim   (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_nextafter(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_rhypot (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_remquo (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo, int* __quo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp32mp2_ilogb  (const float __x_hi, const float __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL long long int __fp32mp2_llrint (const float __x_hi, const float __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL long long int __fp32mp2_llround(const float __x_hi, const float __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL long int __fp32mp2_lrint  (const float __x_hi, const float __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL long int __fp32mp2_lround (const float __x_hi, const float __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp32mp2_isfinite(const float __x_hi, const float __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp32mp2_isinf   (const float __x_hi, const float __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp32mp2_isnan   (const float __x_hi, const float __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp32mp2_signbit (const float __x_hi, const float __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_ldexp  (const float __x_hi, const float __x_lo, int __n, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_scalbn (const float __x_hi, const float __x_lo, int __n, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_scalbln(const float __x_hi, const float __x_lo, long int __n, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_jn     (int __n, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_yn     (int __n, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_frexp  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, int* __nptr) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_modf   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, float* __iptr_hi, float* __iptr_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp32mp2_sincospi(const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept;

    /*
    * ============================================================================
    * Library mode - fp64mp2 declarations
    * ============================================================================
    */
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_exp    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_log    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_log2   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_log10  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_log1p  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_pow    (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_cbrt   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_sin    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_cos    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_sincos (const double __x_hi, const double __x_lo, double* __sin_hi, double* __sin_lo, double* __cos_hi, double* __cos_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_asin   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_acos   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_atan   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_atan2  (const double __y_hi, const double __y_lo, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_sinh   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_cosh   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_tanh   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_erf    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_erfc   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_normcdfinv (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_acosh  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_asinh  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_atanh  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_tan    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_exp2   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_exp10  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_expm1  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_logb   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_ceil   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_floor  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_trunc  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_round  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_rint   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_nearbyint(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fabs   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_lgamma (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_tgamma (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_j0     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_j1     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_y0     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_y1     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_cyl_bessel_i0(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_cyl_bessel_i1(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_sinpi  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_cospi  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_normcdf(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_rcbrt  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_erfcinv(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_erfinv (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_erfcx  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_boys_f0(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_norm3d (const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_norm4d (const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, const double __d_hi, const double __d_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_rnorm3d(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_rnorm4d(const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, const double __d_hi, const double __d_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fmax   (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fmin   (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_max    (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_min    (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fmod   (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_remainder(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_hypot  (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_copysign(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_fdim   (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_nextafter(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_rhypot (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_remquo (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo, int* __quo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp64mp2_ilogb  (const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL long long int __fp64mp2_llrint (const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL long long int __fp64mp2_llround(const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL long int __fp64mp2_lrint  (const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL long int __fp64mp2_lround (const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp64mp2_isfinite(const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp64mp2_isinf   (const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp64mp2_isnan   (const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL int  __fp64mp2_signbit (const double __x_hi, const double __x_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_ldexp  (const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_scalbn (const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_scalbln(const double __x_hi, const double __x_lo, long int __n, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_jn     (int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_yn     (int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_frexp  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo, int* __nptr) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_modf   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo, double* __iptr_hi, double* __iptr_lo) noexcept;
    _CCCL_FPMP_BUILTIN_DECL void __fp64mp2_sincospi(const double __x_hi, const double __x_lo, double* __sin_hi, double* __sin_lo, double* __cos_hi, double* __cos_lo) noexcept;

    /*
    * ============================================================================
    * Template declarations and float specializations
    * ============================================================================
    */
    template<typename _Tp> _CCCL_API inline void __fpmp2_exp    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_log    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_log2   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_log10  (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_log1p  (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_pow    (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_cbrt   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_sin    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_cos    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_sincos (const _Tp __x_hi, const _Tp __x_lo, _Tp* __sin_hi, _Tp* __sin_lo, _Tp* __cos_hi, _Tp* __cos_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_asin   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_acos   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_atan   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_atan2  (const _Tp __y_hi, const _Tp __y_lo, const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_sinh   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_cosh   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_tanh   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_erf    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_erfc   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_normcdfinv (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_acosh   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_asinh   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_atanh   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_tan     (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_exp2    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_exp10   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_expm1   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_logb    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_ceil    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_floor   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_trunc   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_round   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_rint    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_nearbyint(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_fabs    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_lgamma  (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_tgamma  (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_j0      (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_j1      (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_y0      (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_y1      (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_cyl_bessel_i0(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_cyl_bessel_i1(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_sinpi   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_cospi   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_normcdf (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_rcbrt   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_erfcinv(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_erfinv (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_erfcx  (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_boys_f0(const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_norm3d (const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, const _Tp __c_hi, const _Tp __c_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_norm4d (const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, const _Tp __c_hi, const _Tp __c_lo, const _Tp __d_hi, const _Tp __d_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_rnorm3d(const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, const _Tp __c_hi, const _Tp __c_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_rnorm4d(const _Tp __a_hi, const _Tp __a_lo, const _Tp __b_hi, const _Tp __b_lo, const _Tp __c_hi, const _Tp __c_lo, const _Tp __d_hi, const _Tp __d_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_fmax    (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_fmin    (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_max     (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_min     (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_fmod    (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_remainder(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_hypot   (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_copysign(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_fdim    (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_nextafter(const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_rhypot  (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_remquo  (const _Tp __x_hi, const _Tp __x_lo, const _Tp __y_hi, const _Tp __y_lo, _Tp* __res_hi, _Tp* __res_lo, int* __quo) noexcept;
    template<typename _Tp> _CCCL_API inline int  __fpmp2_ilogb   (const _Tp __x_hi, const _Tp __x_lo) noexcept;
    template<typename _Tp> _CCCL_API inline long long int __fpmp2_llrint (const _Tp __x_hi, const _Tp __x_lo) noexcept;
    template<typename _Tp> _CCCL_API inline long long int __fpmp2_llround(const _Tp __x_hi, const _Tp __x_lo) noexcept;
    template<typename _Tp> _CCCL_API inline long int __fpmp2_lrint  (const _Tp __x_hi, const _Tp __x_lo) noexcept;
    template<typename _Tp> _CCCL_API inline long int __fpmp2_lround (const _Tp __x_hi, const _Tp __x_lo) noexcept;
    template<typename _Tp> _CCCL_API inline int  __fpmp2_isfinite(const _Tp __x_hi, const _Tp __x_lo) noexcept;
    template<typename _Tp> _CCCL_API inline int  __fpmp2_isinf   (const _Tp __x_hi, const _Tp __x_lo) noexcept;
    template<typename _Tp> _CCCL_API inline int  __fpmp2_isnan   (const _Tp __x_hi, const _Tp __x_lo) noexcept;
    template<typename _Tp> _CCCL_API inline int  __fpmp2_signbit (const _Tp __x_hi, const _Tp __x_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_ldexp   (const _Tp __x_hi, const _Tp __x_lo, int __n, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_scalbn  (const _Tp __x_hi, const _Tp __x_lo, int __n, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_scalbln (const _Tp __x_hi, const _Tp __x_lo, long int __n, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_jn      (int __n, const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_yn      (int __n, const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_frexp   (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo, int* __nptr) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_modf    (const _Tp __x_hi, const _Tp __x_lo, _Tp* __res_hi, _Tp* __res_lo, _Tp* __iptr_hi, _Tp* __iptr_lo) noexcept;
    template<typename _Tp> _CCCL_API inline void __fpmp2_sincospi(const _Tp __x_hi, const _Tp __x_lo, _Tp* __sin_hi, _Tp* __sin_lo, _Tp* __cos_hi, _Tp* __cos_lo) noexcept;

    _CCCL_API inline void __fpmp2_icdf(uint32_t __x, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_icdf32(__x, __res_hi, __res_lo); }
    _CCCL_API inline void __fpmp2_icdf(uint64_t __x, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_icdf64(__x, __res_hi, __res_lo); }

    // Float (fp32) template specializations
    template<> _CCCL_API inline void __fpmp2_exp<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_exp(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_log(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log2<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_log2(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log10<float>  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_log10(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log1p<float>  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_log1p(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_pow<float>    (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_pow(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cbrt<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_cbrt(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sin<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_sin(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cos<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_cos(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sincos<float> (const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept { __fp32mp2_sincos(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo); }
    template<> _CCCL_API inline void __fpmp2_asin<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_asin(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_acos<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_acos(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_atan<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_atan(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_atan2<float>  (const float __y_hi, const float __y_lo, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_atan2(__y_hi, __y_lo, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sinh<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_sinh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cosh<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_cosh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_tanh<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_tanh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erf<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_erf(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erfc<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_erfc(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_normcdfinv<float> (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_normcdfinv(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_acosh<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_acosh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_asinh<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_asinh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_atanh<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_atanh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_tan<float>      (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_tan(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_exp2<float>     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_exp2(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_exp10<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_exp10(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_expm1<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_expm1(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_logb<float>     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_logb(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_ceil<float>     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_ceil(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_floor<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_floor(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_trunc<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_trunc(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_round<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_round(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rint<float>     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_rint(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_nearbyint<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_nearbyint(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fabs<float>     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_fabs(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_lgamma<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_lgamma(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_tgamma<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_tgamma(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_j0<float>       (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_j0(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_j1<float>       (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_j1(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_y0<float>       (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_y0(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_y1<float>       (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_y1(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cyl_bessel_i0<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_cyl_bessel_i0(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cyl_bessel_i1<float>(const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_cyl_bessel_i1(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sinpi<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_sinpi(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cospi<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_cospi(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_normcdf<float>  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_normcdf(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rcbrt<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_rcbrt(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erfcinv<float>  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_erfcinv(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erfinv<float>   (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_erfinv(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erfcx<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_erfcx(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_boys_f0<float>  (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_boys_f0(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_norm3d<float>   (const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, const float __c_hi, const float __c_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_norm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_norm4d<float>   (const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, const float __c_hi, const float __c_lo, const float __d_hi, const float __d_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_norm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rnorm3d<float>  (const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, const float __c_hi, const float __c_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_rnorm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rnorm4d<float>  (const float __a_hi, const float __a_lo, const float __b_hi, const float __b_lo, const float __c_hi, const float __c_lo, const float __d_hi, const float __d_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_rnorm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fmax<float>     (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_fmax(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fmin<float>     (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_fmin(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_max<float>      (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_max(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_min<float>      (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_min(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fmod<float>     (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_fmod(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_remainder<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_remainder(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_hypot<float>    (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_hypot(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_copysign<float> (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_copysign(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fdim<float>     (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_fdim(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_nextafter<float>(const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_nextafter(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rhypot<float>   (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_rhypot(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_remquo<float>   (const float __x_hi, const float __x_lo, const float __y_hi, const float __y_lo, float* __res_hi, float* __res_lo, int* __quo) noexcept { __fp32mp2_remquo(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, __quo); }
    template<> _CCCL_API inline int  __fpmp2_ilogb<float>    (const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_ilogb(__x_hi, __x_lo); }
    template<> _CCCL_API inline long long int __fpmp2_llrint<float> (const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_llrint(__x_hi, __x_lo); }
    template<> _CCCL_API inline long long int __fpmp2_llround<float>(const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_llround(__x_hi, __x_lo); }
    template<> _CCCL_API inline long int __fpmp2_lrint<float>  (const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_lrint(__x_hi, __x_lo); }
    template<> _CCCL_API inline long int __fpmp2_lround<float> (const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_lround(__x_hi, __x_lo); }
    template<> _CCCL_API inline int  __fpmp2_isfinite<float> (const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_isfinite(__x_hi, __x_lo); }
    template<> _CCCL_API inline int  __fpmp2_isinf<float>    (const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_isinf(__x_hi, __x_lo); }
    template<> _CCCL_API inline int  __fpmp2_isnan<float>    (const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_isnan(__x_hi, __x_lo); }
    template<> _CCCL_API inline int  __fpmp2_signbit<float>  (const float __x_hi, const float __x_lo) noexcept { return __fp32mp2_signbit(__x_hi, __x_lo); }
    template<> _CCCL_API inline void __fpmp2_ldexp<float>    (const float __x_hi, const float __x_lo, int __n, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_ldexp(__x_hi, __x_lo, __n, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_scalbn<float>   (const float __x_hi, const float __x_lo, int __n, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_scalbn(__x_hi, __x_lo, __n, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_scalbln<float>  (const float __x_hi, const float __x_lo, long int __n, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_scalbln(__x_hi, __x_lo, __n, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_jn<float>       (int __n, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_jn(__n, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_yn<float>       (int __n, const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo) noexcept { __fp32mp2_yn(__n, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_frexp<float>    (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, int* __nptr) noexcept { __fp32mp2_frexp(__x_hi, __x_lo, __res_hi, __res_lo, __nptr); }
    template<> _CCCL_API inline void __fpmp2_modf<float>     (const float __x_hi, const float __x_lo, float* __res_hi, float* __res_lo, float* __iptr_hi, float* __iptr_lo) noexcept { __fp32mp2_modf(__x_hi, __x_lo, __res_hi, __res_lo, __iptr_hi, __iptr_lo); }
    template<> _CCCL_API inline void __fpmp2_sincospi<float> (const float __x_hi, const float __x_lo, float* __sin_hi, float* __sin_lo, float* __cos_hi, float* __cos_lo) noexcept { __fp32mp2_sincospi(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo); }

    /*
    * ============================================================================
    * Double (fp64) template specializations
    * ============================================================================
    */
    template<> _CCCL_API inline void __fpmp2_exp<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_exp(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_log(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log2<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_log2(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log10<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_log10(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_log1p<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_log1p(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_pow<double>    (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_pow(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cbrt<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_cbrt(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sin<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_sin(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cos<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_cos(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sincos<double> (const double __x_hi, const double __x_lo, double* __sin_hi, double* __sin_lo, double* __cos_hi, double* __cos_lo) noexcept { __fp64mp2_sincos(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo); }
    template<> _CCCL_API inline void __fpmp2_asin<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_asin(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_acos<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_acos(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_atan<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_atan(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_atan2<double>  (const double __y_hi, const double __y_lo, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_atan2(__y_hi, __y_lo, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sinh<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_sinh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cosh<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_cosh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_tanh<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_tanh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erf<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_erf(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erfc<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_erfc(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_normcdfinv<double> (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_normcdfinv(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_acosh<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_acosh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_asinh<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_asinh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_atanh<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_atanh(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_tan<double>      (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_tan(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_exp2<double>     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_exp2(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_exp10<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_exp10(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_expm1<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_expm1(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_logb<double>     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_logb(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_ceil<double>     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_ceil(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_floor<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_floor(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_trunc<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_trunc(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_round<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_round(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rint<double>     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_rint(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_nearbyint<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_nearbyint(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fabs<double>     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_fabs(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_lgamma<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_lgamma(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_tgamma<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_tgamma(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_j0<double>       (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_j0(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_j1<double>       (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_j1(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_y0<double>       (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_y0(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_y1<double>       (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_y1(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cyl_bessel_i0<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_cyl_bessel_i0(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cyl_bessel_i1<double>(const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_cyl_bessel_i1(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_sinpi<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_sinpi(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_cospi<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_cospi(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_normcdf<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_normcdf(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rcbrt<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_rcbrt(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erfcinv<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_erfcinv(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erfinv<double>   (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_erfinv(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_erfcx<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_erfcx(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_boys_f0<double>  (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_boys_f0(__x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_norm3d<double>   (const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_norm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_norm4d<double>   (const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, const double __d_hi, const double __d_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_norm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rnorm3d<double>  (const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_rnorm3d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rnorm4d<double>  (const double __a_hi, const double __a_lo, const double __b_hi, const double __b_lo, const double __c_hi, const double __c_lo, const double __d_hi, const double __d_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_rnorm4d(__a_hi, __a_lo, __b_hi, __b_lo, __c_hi, __c_lo, __d_hi, __d_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fmax<double>     (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_fmax(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fmin<double>     (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_fmin(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_max<double>      (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_max(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_min<double>      (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_min(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fmod<double>     (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_fmod(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_remainder<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_remainder(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_hypot<double>    (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_hypot(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_copysign<double> (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_copysign(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_fdim<double>     (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_fdim(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_nextafter<double>(const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_nextafter(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_rhypot<double>   (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_rhypot(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_remquo<double>   (const double __x_hi, const double __x_lo, const double __y_hi, const double __y_lo, double* __res_hi, double* __res_lo, int* __quo) noexcept { __fp64mp2_remquo(__x_hi, __x_lo, __y_hi, __y_lo, __res_hi, __res_lo, __quo); }
    template<> _CCCL_API inline int  __fpmp2_ilogb<double>    (const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_ilogb(__x_hi, __x_lo); }
    template<> _CCCL_API inline long long int __fpmp2_llrint<double> (const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_llrint(__x_hi, __x_lo); }
    template<> _CCCL_API inline long long int __fpmp2_llround<double>(const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_llround(__x_hi, __x_lo); }
    template<> _CCCL_API inline long int __fpmp2_lrint<double>  (const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_lrint(__x_hi, __x_lo); }
    template<> _CCCL_API inline long int __fpmp2_lround<double> (const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_lround(__x_hi, __x_lo); }
    template<> _CCCL_API inline int  __fpmp2_isfinite<double> (const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_isfinite(__x_hi, __x_lo); }
    template<> _CCCL_API inline int  __fpmp2_isinf<double>    (const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_isinf(__x_hi, __x_lo); }
    template<> _CCCL_API inline int  __fpmp2_isnan<double>    (const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_isnan(__x_hi, __x_lo); }
    template<> _CCCL_API inline int  __fpmp2_signbit<double>  (const double __x_hi, const double __x_lo) noexcept { return __fp64mp2_signbit(__x_hi, __x_lo); }
    template<> _CCCL_API inline void __fpmp2_ldexp<double>    (const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_ldexp(__x_hi, __x_lo, __n, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_scalbn<double>   (const double __x_hi, const double __x_lo, int __n, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_scalbn(__x_hi, __x_lo, __n, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_scalbln<double>  (const double __x_hi, const double __x_lo, long int __n, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_scalbln(__x_hi, __x_lo, __n, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_jn<double>       (int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_jn(__n, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_yn<double>       (int __n, const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo) noexcept { __fp64mp2_yn(__n, __x_hi, __x_lo, __res_hi, __res_lo); }
    template<> _CCCL_API inline void __fpmp2_frexp<double>    (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo, int* __nptr) noexcept { __fp64mp2_frexp(__x_hi, __x_lo, __res_hi, __res_lo, __nptr); }
    template<> _CCCL_API inline void __fpmp2_modf<double>     (const double __x_hi, const double __x_lo, double* __res_hi, double* __res_lo, double* __iptr_hi, double* __iptr_lo) noexcept { __fp64mp2_modf(__x_hi, __x_lo, __res_hi, __res_lo, __iptr_hi, __iptr_lo); }
    template<> _CCCL_API inline void __fpmp2_sincospi<double> (const double __x_hi, const double __x_lo, double* __sin_hi, double* __sin_lo, double* __cos_hi, double* __cos_lo) noexcept { __fp64mp2_sincospi(__x_hi, __x_lo, __sin_hi, __sin_lo, __cos_hi, __cos_lo); }

#endif // ! defined _CCCL_FPMP_USE_LIB

/*
* ============================================================================
* Freestanding API functions for fpmp2 class
* ============================================================================
*/

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> exp (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_exp(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> log (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_log(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> log2 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_log2(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> log10 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_log10(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> log1p (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_log1p(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> pow (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_pow(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> cbrt (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_cbrt(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> sin (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_sin(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> cos (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_cos(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline void sincos (const fpmp2<_FpType, _TypeAcc>& __x, fpmp2<_FpType, _TypeAcc>* __s, fpmp2<_FpType, _TypeAcc>* __c) noexcept 
{ _FpType __sin_hi, __sin_lo, __cos_hi, __cos_lo; __fpmp2_sincos(__x.hi(), __x.lo(), &__sin_hi, &__sin_lo, &__cos_hi, &__cos_lo); *__s = fpmp2<_FpType, _TypeAcc>(__sin_hi, __sin_lo); *__c = fpmp2<_FpType, _TypeAcc>(__cos_hi, __cos_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> asin (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_asin(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> acos (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_acos(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> atan (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_atan(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> atan2 (const fpmp2<_FpType, _TypeAcc>& __y, const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_atan2(__y.hi(), __y.lo(), __x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> sinh (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_sinh(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> cosh (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_cosh(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> tanh (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_tanh(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> erf (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_erf(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> erfc (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_erfc(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> boys_f0 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_boys_f0(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> normcdfinv (const fpmp2<_FpType, _TypeAcc>& __x) noexcept 
{ _FpType __res_hi, __res_lo; __fpmp2_normcdfinv(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<float, _TypeAcc> icdf (uint32_t __x) noexcept 
{ float __res_hi, __res_lo; __fpmp2_icdf(__x, &__res_hi, &__res_lo); return fpmp2<float, _TypeAcc>(__res_hi, __res_lo); }

template <fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<float, _TypeAcc> icdf (uint64_t __x) noexcept 
{ float __res_hi, __res_lo; __fpmp2_icdf(__x, &__res_hi, &__res_lo); return fpmp2<float, _TypeAcc>(__res_hi, __res_lo); }

// Inverse hyperbolic functions
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> acosh (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_acosh(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> asinh (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_asinh(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> atanh (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_atanh(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// Tangent
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> tan (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_tan(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// Additional exponential/logarithmic functions
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> exp2 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_exp2(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> exp10 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_exp10(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> expm1 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_expm1(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> logb (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_logb(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// Rounding functions
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> ceil (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_ceil(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> floor (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_floor(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> trunc (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_trunc(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> round (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_round(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> rint (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_rint(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> nearbyint (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_nearbyint(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// Absolute value
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> fabs (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_fabs(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// Gamma functions
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> lgamma (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_lgamma(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> tgamma (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_tgamma(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// Bessel functions
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> j0 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_j0(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> j1 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_j1(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> y0 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_y0(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> y1 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_y1(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> jn (int __n, const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_jn(__n, __x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> yn (int __n, const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_yn(__n, __x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> cyl_bessel_i0 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_cyl_bessel_i0(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> cyl_bessel_i1 (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_cyl_bessel_i1(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// CUDA-specific trigonometric functions
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> sinpi (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_sinpi(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> cospi (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_cospi(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline void sincospi (const fpmp2<_FpType, _TypeAcc>& __x, fpmp2<_FpType, _TypeAcc>* __s, fpmp2<_FpType, _TypeAcc>* __c) noexcept
{ _FpType __sin_hi, __sin_lo, __cos_hi, __cos_lo; __fpmp2_sincospi(__x.hi(), __x.lo(), &__sin_hi, &__sin_lo, &__cos_hi, &__cos_lo); *__s = fpmp2<_FpType, _TypeAcc>(__sin_hi, __sin_lo); *__c = fpmp2<_FpType, _TypeAcc>(__cos_hi, __cos_lo); }

// Normal distribution CDF and reciprocal functions
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> normcdf (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_normcdf(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> rcbrt (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_rcbrt(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> erfcinv (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_erfcinv(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> erfinv (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_erfinv(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> erfcx (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_erfcx(__x.hi(), __x.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> norm3d (const fpmp2<_FpType, _TypeAcc>& __a, const fpmp2<_FpType, _TypeAcc>& __b, const fpmp2<_FpType, _TypeAcc>& __c) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_norm3d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> norm4d (const fpmp2<_FpType, _TypeAcc>& __a, const fpmp2<_FpType, _TypeAcc>& __b, const fpmp2<_FpType, _TypeAcc>& __c, const fpmp2<_FpType, _TypeAcc>& __d) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_norm4d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), __d.hi(), __d.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> rnorm3d (const fpmp2<_FpType, _TypeAcc>& __a, const fpmp2<_FpType, _TypeAcc>& __b, const fpmp2<_FpType, _TypeAcc>& __c) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_rnorm3d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> rnorm4d (const fpmp2<_FpType, _TypeAcc>& __a, const fpmp2<_FpType, _TypeAcc>& __b, const fpmp2<_FpType, _TypeAcc>& __c, const fpmp2<_FpType, _TypeAcc>& __d) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_rnorm4d(__a.hi(), __a.lo(), __b.hi(), __b.lo(), __c.hi(), __c.lo(), __d.hi(), __d.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// Two-argument functions
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> fmax (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_fmax(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> fmin (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_fmin(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> max (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_max(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> min (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_min(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> fmod (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_fmod(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> remainder (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_remainder(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> hypot (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_hypot(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> copysign (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_copysign(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> fdim (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_fdim(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> nextafter (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_nextafter(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> rhypot (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_rhypot(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// Functions with special signatures
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> remquo (const fpmp2<_FpType, _TypeAcc>& __x, const fpmp2<_FpType, _TypeAcc>& __y, int* __quo) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_remquo(__x.hi(), __x.lo(), __y.hi(), __y.lo(), &__res_hi, &__res_lo, __quo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> ldexp (const fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_ldexp(__x.hi(), __x.lo(), __n, &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> scalbn (const fpmp2<_FpType, _TypeAcc>& __x, int __n) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_scalbn(__x.hi(), __x.lo(), __n, &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> scalbln (const fpmp2<_FpType, _TypeAcc>& __x, long int __n) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_scalbln(__x.hi(), __x.lo(), __n, &__res_hi, &__res_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> frexp (const fpmp2<_FpType, _TypeAcc>& __x, int* __nptr) noexcept
{ _FpType __res_hi, __res_lo; __fpmp2_frexp(__x.hi(), __x.lo(), &__res_hi, &__res_lo, __nptr); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline fpmp2<_FpType, _TypeAcc> modf (const fpmp2<_FpType, _TypeAcc>& __x, fpmp2<_FpType, _TypeAcc>* __iptr) noexcept
{ _FpType __res_hi, __res_lo, __i_hi, __i_lo; __fpmp2_modf(__x.hi(), __x.lo(), &__res_hi, &__res_lo, &__i_hi, &__i_lo); *__iptr = fpmp2<_FpType, _TypeAcc>(__i_hi, __i_lo); return fpmp2<_FpType, _TypeAcc>(__res_hi, __res_lo); }

// Functions returning integer types
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline int ilogb (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ return __fpmp2_ilogb(__x.hi(), __x.lo()); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline long long int llrint (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ return __fpmp2_llrint(__x.hi(), __x.lo()); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline long long int llround (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ return __fpmp2_llround(__x.hi(), __x.lo()); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline long int lrint (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ return __fpmp2_lrint(__x.hi(), __x.lo()); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline long int lround (const fpmp2<_FpType, _TypeAcc>& __x) noexcept
{ return __fpmp2_lround(__x.hi(), __x.lo()); }

// Classification functions
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline int fpmp_isfinite (const fpmp2<_FpType, _TypeAcc>& __x)
_CCCL_FPMP_NOEXCEPT
{ return __fpmp2_isfinite(__x.hi(), __x.lo()); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline int fpmp_isinf (const fpmp2<_FpType, _TypeAcc>& __x)
_CCCL_FPMP_NOEXCEPT
{ return __fpmp2_isinf(__x.hi(), __x.lo()); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline int fpmp_isnan (const fpmp2<_FpType, _TypeAcc>& __x)
_CCCL_FPMP_NOEXCEPT
{ return __fpmp2_isnan(__x.hi(), __x.lo()); }

template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline int fpmp_signbit (const fpmp2<_FpType, _TypeAcc>& __x)
_CCCL_FPMP_NOEXCEPT
{ return __fpmp2_signbit(__x.hi(), __x.lo()); }

// Standard names are provided only when no conflicting macro is active.
#ifndef isfinite
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline int isfinite (const fpmp2<_FpType, _TypeAcc>& __x)
_CCCL_FPMP_NOEXCEPT
{ return fpmp_isfinite(__x); }
#endif

#ifndef isinf
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline int isinf (const fpmp2<_FpType, _TypeAcc>& __x)
_CCCL_FPMP_NOEXCEPT
{ return fpmp_isinf(__x); }
#endif

#ifndef isnan
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline int isnan (const fpmp2<_FpType, _TypeAcc>& __x)
_CCCL_FPMP_NOEXCEPT
{ return fpmp_isnan(__x); }
#endif

#ifndef signbit
template <typename _FpType = float, fpmp2_accuracy _TypeAcc = fpmp2_accuracy::def>
_CCCL_API inline int signbit (const fpmp2<_FpType, _TypeAcc>& __x)
_CCCL_FPMP_NOEXCEPT
{ return fpmp_signbit(__x); }
#endif

/*
* Note: the fpmp2 warp-shuffle overloads (__shfl_sync, __shfl_xor_sync,
* __shfl_down_sync, __shfl_up_sync) are thread-cooperation primitives, not math
* functions, so they live in the core header <cuda/__fp/fpmp.h> (available via
* <cuda/fpmp>) rather than here.
*/

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FP_FPMP_MATH_H
