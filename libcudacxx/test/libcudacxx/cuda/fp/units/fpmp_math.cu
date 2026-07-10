// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp32mp2 / fp64mp2 math functions (device only).
//
//  Sanity test that calls every fpmp2 math function from its own CUDA kernel on
//  a single input and compares the result against the corresponding
//  double-precision reference computed on the device. Kernels are templated on
//  the multi-precision type so the same kernel serves both fp32mp2 and fp64mp2.
//
//  This test relies on CUDA device math intrinsics (rcbrt, normcdf, Bessel, pi-
//  scaled trig, vector norms, ...) that have no host equivalent, so it is
//  device-only: SKIP()ed in a host-only build, run on the device under CUDA.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp_math>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_CUDA_COMPILATION()

#  define CUDA_CHECK(call)                                                                           \
    do                                                                                               \
    {                                                                                                \
      cudaError_t err = (call);                                                                      \
      if (err != cudaSuccess)                                                                        \
      {                                                                                              \
        ::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false;                                                                                \
      }                                                                                              \
    } while (0)

// Result structures stored in managed memory.
struct Result
{
  double fpmp_val;
  double ref_val;
};
struct ResultInt
{
  int fpmp_val;
  int ref_val;
};
struct ResultLL
{
  long long fpmp_val;
  long long ref_val;
};
struct ResultLong
{
  long fpmp_val;
  long ref_val;
};

// One-argument kernels: f(x) -> fpmp2
#  define DEFINE_KERNEL_1A(name)                          \
    template <typename MP2>                               \
    __global__ void kernel_##name(double x_in, Result* r) \
    {                                                     \
      MP2 x       = MP2(x_in);                            \
      MP2 res     = name(x);                              \
      r->fpmp_val = static_cast<double>(res);             \
      r->ref_val  = ::name(x_in);                         \
    }

DEFINE_KERNEL_1A(exp)
DEFINE_KERNEL_1A(log)
DEFINE_KERNEL_1A(log2)
DEFINE_KERNEL_1A(log10)
DEFINE_KERNEL_1A(log1p)
DEFINE_KERNEL_1A(cbrt)
DEFINE_KERNEL_1A(sin)
DEFINE_KERNEL_1A(cos)
DEFINE_KERNEL_1A(asin)
DEFINE_KERNEL_1A(acos)
DEFINE_KERNEL_1A(atan)
DEFINE_KERNEL_1A(sinh)
DEFINE_KERNEL_1A(cosh)
DEFINE_KERNEL_1A(tanh)
DEFINE_KERNEL_1A(erf)
DEFINE_KERNEL_1A(erfc)
DEFINE_KERNEL_1A(acosh)
DEFINE_KERNEL_1A(asinh)
DEFINE_KERNEL_1A(atanh)
DEFINE_KERNEL_1A(tan)
DEFINE_KERNEL_1A(exp2)
DEFINE_KERNEL_1A(exp10)
DEFINE_KERNEL_1A(expm1)
DEFINE_KERNEL_1A(logb)
DEFINE_KERNEL_1A(ceil)
DEFINE_KERNEL_1A(floor)
DEFINE_KERNEL_1A(trunc)
DEFINE_KERNEL_1A(round)
DEFINE_KERNEL_1A(rint)
DEFINE_KERNEL_1A(nearbyint)
DEFINE_KERNEL_1A(fabs)
DEFINE_KERNEL_1A(lgamma)
DEFINE_KERNEL_1A(tgamma)
DEFINE_KERNEL_1A(j0)
DEFINE_KERNEL_1A(j1)
DEFINE_KERNEL_1A(y0)
DEFINE_KERNEL_1A(y1)
DEFINE_KERNEL_1A(cyl_bessel_i0)
DEFINE_KERNEL_1A(cyl_bessel_i1)
DEFINE_KERNEL_1A(sinpi)
DEFINE_KERNEL_1A(cospi)
DEFINE_KERNEL_1A(normcdf)
DEFINE_KERNEL_1A(rcbrt)
DEFINE_KERNEL_1A(erfcinv)
DEFINE_KERNEL_1A(erfinv)
DEFINE_KERNEL_1A(erfcx)

// Three-argument kernels: f(a,b,c) -> fpmp2
#  define DEFINE_KERNEL_3A(name)                                                    \
    template <typename MP2>                                                         \
    __global__ void kernel_##name(double a_in, double b_in, double c_in, Result* r) \
    {                                                                               \
      MP2 a = MP2(a_in), b = MP2(b_in), c = MP2(c_in);                              \
      MP2 res     = name(a, b, c);                                                  \
      r->fpmp_val = static_cast<double>(res);                                       \
      r->ref_val  = ::name(a_in, b_in, c_in);                                       \
    }

DEFINE_KERNEL_3A(norm3d)
DEFINE_KERNEL_3A(rnorm3d)

// Four-argument kernels: f(a,b,c,d) -> fpmp2
#  define DEFINE_KERNEL_4A(name)                                                                 \
    template <typename MP2>                                                                      \
    __global__ void kernel_##name(double a_in, double b_in, double c_in, double d_in, Result* r) \
    {                                                                                            \
      MP2 a = MP2(a_in), b = MP2(b_in), c = MP2(c_in), d = MP2(d_in);                            \
      MP2 res     = name(a, b, c, d);                                                            \
      r->fpmp_val = static_cast<double>(res);                                                    \
      r->ref_val  = ::name(a_in, b_in, c_in, d_in);                                              \
    }

DEFINE_KERNEL_4A(norm4d)
DEFINE_KERNEL_4A(rnorm4d)

// Two-argument kernels: f(x,y) -> fpmp2
#  define DEFINE_KERNEL_2A(name)                                       \
    template <typename MP2>                                            \
    __global__ void kernel_##name(double x_in, double y_in, Result* r) \
    {                                                                  \
      MP2 x       = MP2(x_in);                                         \
      MP2 y       = MP2(y_in);                                         \
      MP2 res     = name(x, y);                                        \
      r->fpmp_val = static_cast<double>(res);                          \
      r->ref_val  = ::name(x_in, y_in);                                \
    }

DEFINE_KERNEL_2A(pow)
DEFINE_KERNEL_2A(atan2)
DEFINE_KERNEL_2A(fmax)
DEFINE_KERNEL_2A(fmin)

template <typename MP2>
__global__ void kernel_max(double x_in, double y_in, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 y       = MP2(y_in);
  MP2 res     = max(x, y);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = (x_in < y_in) ? y_in : x_in;
}

template <typename MP2>
__global__ void kernel_min(double x_in, double y_in, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 y       = MP2(y_in);
  MP2 res     = min(x, y);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = (y_in < x_in) ? y_in : x_in;
}

DEFINE_KERNEL_2A(fmod)
DEFINE_KERNEL_2A(remainder)
DEFINE_KERNEL_2A(hypot)
DEFINE_KERNEL_2A(copysign)
DEFINE_KERNEL_2A(fdim)
DEFINE_KERNEL_2A(nextafter)
DEFINE_KERNEL_2A(rhypot)

// sincos / sincospi (use sin+cos / sinpi+cospi to avoid overload clash).
template <typename MP2>
__global__ void kernel_sincos(double x_in, Result* r_sin, Result* r_cos)
{
  MP2 x           = MP2(x_in);
  r_sin->fpmp_val = static_cast<double>(sin(x));
  r_cos->fpmp_val = static_cast<double>(cos(x));
  double sd, cd;
  ::sincos(x_in, &sd, &cd);
  r_sin->ref_val = sd;
  r_cos->ref_val = cd;
}

template <typename MP2>
__global__ void kernel_sincospi(double x_in, Result* r_sin, Result* r_cos)
{
  MP2 x           = MP2(x_in);
  r_sin->fpmp_val = static_cast<double>(sinpi(x));
  r_cos->fpmp_val = static_cast<double>(cospi(x));
  double sd, cd;
  ::sincospi(x_in, &sd, &cd);
  r_sin->ref_val = sd;
  r_cos->ref_val = cd;
}

// normcdfinv (input must be in (0,1)).
template <typename MP2>
__global__ void kernel_normcdfinv(double x_in, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 res     = normcdfinv(x);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = ::normcdfinv(x_in);
}

// Integer-returning kernels.
template <typename MP2>
__global__ void kernel_ilogb(double x_in, ResultInt* r)
{
  MP2 x       = MP2(x_in);
  r->fpmp_val = ilogb(x);
  r->ref_val  = ::ilogb(x_in);
}

template <typename MP2>
__global__ void kernel_llrint(double x_in, ResultLL* r)
{
  MP2 x       = MP2(x_in);
  r->fpmp_val = llrint(x);
  r->ref_val  = ::llrint(x_in);
}

template <typename MP2>
__global__ void kernel_llround(double x_in, ResultLL* r)
{
  MP2 x       = MP2(x_in);
  r->fpmp_val = llround(x);
  r->ref_val  = ::llround(x_in);
}

template <typename MP2>
__global__ void kernel_lrint(double x_in, ResultLong* r)
{
  MP2 x       = MP2(x_in);
  r->fpmp_val = lrint(x);
  r->ref_val  = ::lrint(x_in);
}

template <typename MP2>
__global__ void kernel_lround(double x_in, ResultLong* r)
{
  MP2 x       = MP2(x_in);
  r->fpmp_val = lround(x);
  r->ref_val  = ::lround(x_in);
}

// Classification kernels.
template <typename MP2>
__global__ void kernel_isfinite(double x_in, ResultInt* r)
{
  MP2 x       = MP2(x_in);
  r->fpmp_val = fpmp_isfinite(x);
  r->ref_val  = isfinite(x_in) ? 1 : 0;
}

template <typename MP2>
__global__ void kernel_isinf(double x_in, ResultInt* r)
{
  MP2 x       = MP2(x_in);
  r->fpmp_val = fpmp_isinf(x);
  r->ref_val  = isinf(x_in) ? 1 : 0;
}

template <typename MP2>
__global__ void kernel_isnan(double x_in, ResultInt* r)
{
  MP2 x       = MP2(x_in);
  r->fpmp_val = fpmp_isnan(x);
  r->ref_val  = isnan(x_in) ? 1 : 0;
}

template <typename MP2>
__global__ void kernel_signbit(double x_in, ResultInt* r)
{
  MP2 x       = MP2(x_in);
  r->fpmp_val = fpmp_signbit(x);
  r->ref_val  = signbit(x_in) ? 1 : 0;
}

// Mixed-signature kernels.
template <typename MP2>
__global__ void kernel_ldexp(double x_in, int n, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 res     = ldexp(x, n);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = ::ldexp(x_in, n);
}

template <typename MP2>
__global__ void kernel_scalbn(double x_in, int n, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 res     = scalbn(x, n);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = ::scalbn(x_in, n);
}

template <typename MP2>
__global__ void kernel_scalbln(double x_in, long int n, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 res     = scalbln(x, n);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = ::scalbln(x_in, n);
}

template <typename MP2>
__global__ void kernel_jn(int n, double x_in, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 res     = jn(n, x);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = ::jn(n, x_in);
}

template <typename MP2>
__global__ void kernel_yn(int n, double x_in, Result* r)
{
  MP2 x       = MP2(x_in);
  MP2 res     = yn(n, x);
  r->fpmp_val = static_cast<double>(res);
  r->ref_val  = ::yn(n, x_in);
}

template <typename MP2>
__global__ void kernel_frexp(double x_in, Result* r, ResultInt* r_exp)
{
  using FpType = decltype(MP2().hi());
  MP2 x        = MP2(x_in);
  int nptr;
  FpType frac_hi, frac_lo;
  __fpmp2_frexp(x.hi(), x.lo(), &frac_hi, &frac_lo, &nptr);
  r->fpmp_val     = static_cast<double>(MP2(frac_hi, frac_lo));
  r_exp->fpmp_val = nptr;
  int ref_exp;
  r->ref_val     = ::frexp(x_in, &ref_exp);
  r_exp->ref_val = ref_exp;
}

template <typename MP2>
__global__ void kernel_modf(double x_in, Result* r_frac, Result* r_int)
{
  using FpType = decltype(MP2().hi());
  MP2 x        = MP2(x_in);
  FpType frac_hi, frac_lo, ipart_hi, ipart_lo;
  __fpmp2_modf(x.hi(), x.lo(), &frac_hi, &frac_lo, &ipart_hi, &ipart_lo);
  r_frac->fpmp_val = static_cast<double>(MP2(frac_hi, frac_lo));
  r_int->fpmp_val  = static_cast<double>(MP2(ipart_hi, ipart_lo));
  double iref;
  r_frac->ref_val = ::modf(x_in, &iref);
  r_int->ref_val  = iref;
}

template <typename MP2>
__global__ void kernel_remquo(double x_in, double y_in, Result* r, ResultInt* r_quo)
{
  using FpType = decltype(MP2().hi());
  MP2 x = MP2(x_in), y = MP2(y_in);
  int quo;
  FpType res_hi, res_lo;
  __fpmp2_remquo(x.hi(), x.lo(), y.hi(), y.lo(), &res_hi, &res_lo, &quo);
  r->fpmp_val     = static_cast<double>(MP2(res_hi, res_lo));
  r_quo->fpmp_val = quo;
  int ref_quo;
  r->ref_val     = ::remquo(x_in, y_in, &ref_quo);
  r_quo->ref_val = ref_quo;
}

// Test runner helpers.
static bool approx_eq(double a, double b, double tol)
{
  if (a == b)
  {
    return true;
  }
  double diff = ::fabs(a - b);
  double mag  = ::fmax(::fabs(a), ::fabs(b));
  if (mag == 0.0)
  {
    return diff < tol;
  }
  return (diff / mag) < tol;
}

static bool check(const char* label, double fpmp_val, double ref_val, double tol)
{
  bool ok = approx_eq(fpmp_val, ref_val, tol);
  if (ok)
  {
    ::printf("  PASS  %-38s  fpmp=%.12e  ref=%.12e\n", label, fpmp_val, ref_val);
  }
  else
  {
    ::printf(
      "  FAIL  %-38s  fpmp=%.12e  ref=%.12e  (diff=%.3e)\n", label, fpmp_val, ref_val, ::fabs(fpmp_val - ref_val));
  }
  return ok;
}

static bool check_int(const char* label, long long fpmp_val, long long ref_val)
{
  bool ok = (fpmp_val == ref_val);
  if (ok)
  {
    ::printf("  PASS  %-38s  fpmp=%lld  ref=%lld\n", label, fpmp_val, ref_val);
  }
  else
  {
    ::printf("  FAIL  %-38s  fpmp=%lld  ref=%lld\n", label, fpmp_val, ref_val);
  }
  return ok;
}

// Templated test runner — works for both fp32mp2 and fp64mp2.
template <typename MP2>
static bool run_tests(const char* type_name, double tol)
{
  const double x_val = 1.234567890123;
  const double y_val = 2.345678901234;
  const double p_val = 0.3;
  const int n_val    = 3;

  Result *r1, *r2;
  ResultInt* ri;
  ResultLL* rll;
  ResultLong* rl;

  CUDA_CHECK(cudaMallocManaged(&r1, sizeof(Result)));
  CUDA_CHECK(cudaMallocManaged(&r2, sizeof(Result)));
  CUDA_CHECK(cudaMallocManaged(&ri, sizeof(ResultInt)));
  CUDA_CHECK(cudaMallocManaged(&rll, sizeof(ResultLL)));
  CUDA_CHECK(cudaMallocManaged(&rl, sizeof(ResultLong)));

  bool ok = true;
  ::printf("\n  ====== %s sanity test ======\n", type_name);

#  define RUN_1A(name, xv)                                   \
    kernel_##name<MP2><<<1, 1>>>(xv, r1);                    \
    CUDA_CHECK(cudaDeviceSynchronize());                     \
    {                                                        \
      char lbl[80];                                          \
      ::snprintf(lbl, sizeof(lbl), #name "(%.6g)", xv);      \
      ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok; \
    }

#  define RUN_2A(name, xv, yv)                                    \
    kernel_##name<MP2><<<1, 1>>>(xv, yv, r1);                     \
    CUDA_CHECK(cudaDeviceSynchronize());                          \
    {                                                             \
      char lbl[80];                                               \
      ::snprintf(lbl, sizeof(lbl), #name "(%.6g, %.6g)", xv, yv); \
      ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;      \
    }

#  define RUN_3A(name, av, bv, cv)                                          \
    kernel_##name<MP2><<<1, 1>>>(av, bv, cv, r1);                           \
    CUDA_CHECK(cudaDeviceSynchronize());                                    \
    {                                                                       \
      char lbl[80];                                                         \
      ::snprintf(lbl, sizeof(lbl), #name "(%.6g, %.6g, %.6g)", av, bv, cv); \
      ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;                \
    }

#  define RUN_4A(name, av, bv, cv, dv)                                                \
    kernel_##name<MP2><<<1, 1>>>(av, bv, cv, dv, r1);                                 \
    CUDA_CHECK(cudaDeviceSynchronize());                                              \
    {                                                                                 \
      char lbl[80];                                                                   \
      ::snprintf(lbl, sizeof(lbl), #name "(%.6g, %.6g, %.6g, %.6g)", av, bv, cv, dv); \
      ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;                          \
    }

  // Exponential / Logarithmic
  RUN_1A(exp, x_val)
  RUN_1A(log, x_val)
  RUN_1A(log2, x_val)
  RUN_1A(log10, x_val)
  RUN_1A(log1p, x_val)
  RUN_1A(exp2, x_val)
  RUN_1A(exp10, x_val)
  RUN_1A(expm1, x_val)
  RUN_1A(logb, x_val)

  // Power / Root
  RUN_1A(cbrt, x_val)
  RUN_1A(rcbrt, x_val)

  // Trigonometric
  RUN_1A(sin, x_val)
  RUN_1A(cos, x_val)
  RUN_1A(tan, x_val)
  RUN_1A(asin, 0.5)
  RUN_1A(acos, 0.5)
  RUN_1A(atan, x_val)

  // Hyperbolic
  RUN_1A(sinh, x_val)
  RUN_1A(cosh, x_val)
  RUN_1A(tanh, x_val)
  RUN_1A(acosh, x_val)
  RUN_1A(asinh, x_val)
  RUN_1A(atanh, 0.5)

  // Error / Probability
  RUN_1A(erf, x_val)
  RUN_1A(erfc, x_val)
  RUN_1A(erfcinv, p_val)
  RUN_1A(erfinv, p_val)
  RUN_1A(erfcx, x_val)
  RUN_1A(normcdf, x_val)

  // Gamma
  RUN_1A(lgamma, x_val)
  RUN_1A(tgamma, x_val)

  // Rounding
  RUN_1A(ceil, x_val)
  RUN_1A(floor, x_val)
  RUN_1A(trunc, x_val)
  RUN_1A(round, x_val)
  RUN_1A(rint, x_val)
  RUN_1A(nearbyint, x_val)

  // Absolute value
  RUN_1A(fabs, -x_val)

  // Bessel
  RUN_1A(j0, x_val)
  RUN_1A(j1, x_val)
  RUN_1A(y0, x_val)
  RUN_1A(y1, x_val)
  RUN_1A(cyl_bessel_i0, x_val)
  RUN_1A(cyl_bessel_i1, x_val)

  // CUDA trigonometric (pi-scaled)
  RUN_1A(sinpi, x_val)
  RUN_1A(cospi, x_val)

  // Inverse CDF
  kernel_normcdfinv<MP2><<<1, 1>>>(p_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "normcdfinv(%.6g)", p_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
  }

  // Two-argument
  RUN_2A(pow, x_val, y_val)
  RUN_2A(atan2, x_val, y_val)
  RUN_2A(fmax, x_val, y_val)
  RUN_2A(fmin, x_val, y_val)
  kernel_max<MP2><<<1, 1>>>(x_val, y_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "max(%.6g, %.6g)", x_val, y_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
  }
  kernel_min<MP2><<<1, 1>>>(x_val, y_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "min(%.6g, %.6g)", x_val, y_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
  }
  RUN_2A(fmod, x_val, y_val)
  RUN_2A(remainder, x_val, y_val)
  RUN_2A(hypot, x_val, y_val)
  RUN_2A(copysign, x_val, y_val)
  RUN_2A(fdim, x_val, y_val)
  RUN_2A(nextafter, x_val, y_val)
  RUN_2A(rhypot, x_val, y_val)

  // Vector norm (3/4 args)
  RUN_3A(norm3d, x_val, y_val, p_val)
  RUN_3A(rnorm3d, x_val, y_val, p_val)
  RUN_4A(norm4d, x_val, y_val, p_val, 0.7)
  RUN_4A(rnorm4d, x_val, y_val, p_val, 0.7)

  // sincos / sincospi
  kernel_sincos<MP2><<<1, 1>>>(x_val, r1, r2);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "sincos_sin(%.6g)", x_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
    ::snprintf(lbl, sizeof(lbl), "sincos_cos(%.6g)", x_val);
    ok = check(lbl, r2->fpmp_val, r2->ref_val, tol) && ok;
  }

  kernel_sincospi<MP2><<<1, 1>>>(x_val, r1, r2);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "sincospi_sin(%.6g)", x_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
    ::snprintf(lbl, sizeof(lbl), "sincospi_cos(%.6g)", x_val);
    ok = check(lbl, r2->fpmp_val, r2->ref_val, tol) && ok;
  }

  // Integer-returning
  kernel_ilogb<MP2><<<1, 1>>>(x_val, ri);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "ilogb(%.6g)", x_val);
    ok = check_int(lbl, ri->fpmp_val, ri->ref_val) && ok;
  }
  kernel_llrint<MP2><<<1, 1>>>(x_val, rll);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "llrint(%.6g)", x_val);
    ok = check_int(lbl, rll->fpmp_val, rll->ref_val) && ok;
  }
  kernel_llround<MP2><<<1, 1>>>(x_val, rll);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "llround(%.6g)", x_val);
    ok = check_int(lbl, rll->fpmp_val, rll->ref_val) && ok;
  }
  kernel_lrint<MP2><<<1, 1>>>(x_val, rl);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "lrint(%.6g)", x_val);
    ok = check_int(lbl, (long long) rl->fpmp_val, (long long) rl->ref_val) && ok;
  }
  kernel_lround<MP2><<<1, 1>>>(x_val, rl);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "lround(%.6g)", x_val);
    ok = check_int(lbl, (long long) rl->fpmp_val, (long long) rl->ref_val) && ok;
  }

  // Classification
  kernel_isfinite<MP2><<<1, 1>>>(x_val, ri);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "isfinite(%.6g)", x_val);
    ok = check_int(lbl, ri->fpmp_val, ri->ref_val) && ok;
  }
  kernel_isinf<MP2><<<1, 1>>>(x_val, ri);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "isinf(%.6g)", x_val);
    ok = check_int(lbl, ri->fpmp_val, ri->ref_val) && ok;
  }
  kernel_isnan<MP2><<<1, 1>>>(x_val, ri);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "isnan(%.6g)", x_val);
    ok = check_int(lbl, ri->fpmp_val, ri->ref_val) && ok;
  }
  kernel_signbit<MP2><<<1, 1>>>(x_val, ri);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "signbit(%.6g)", x_val);
    ok = check_int(lbl, ri->fpmp_val, ri->ref_val) && ok;
  }

  // Mixed signature (fp, int)
  kernel_ldexp<MP2><<<1, 1>>>(x_val, n_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "ldexp(%.6g, %d)", x_val, n_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
  }
  kernel_scalbn<MP2><<<1, 1>>>(x_val, n_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "scalbn(%.6g, %d)", x_val, n_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
  }
  kernel_scalbln<MP2><<<1, 1>>>(x_val, (long) n_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "scalbln(%.6g, %d)", x_val, n_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
  }
  kernel_jn<MP2><<<1, 1>>>(n_val, x_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "jn(%d, %.6g)", n_val, x_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
  }
  kernel_yn<MP2><<<1, 1>>>(n_val, x_val, r1);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "yn(%d, %.6g)", n_val, x_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
  }

  // frexp / modf / remquo
  kernel_frexp<MP2><<<1, 1>>>(x_val, r1, ri);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "frexp_frac(%.6g)", x_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
    ::snprintf(lbl, sizeof(lbl), "frexp_exp(%.6g)", x_val);
    ok = check_int(lbl, ri->fpmp_val, ri->ref_val) && ok;
  }
  kernel_modf<MP2><<<1, 1>>>(x_val, r1, r2);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "modf_frac(%.6g)", x_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
    ::snprintf(lbl, sizeof(lbl), "modf_int(%.6g)", x_val);
    ok = check(lbl, r2->fpmp_val, r2->ref_val, tol) && ok;
  }
  kernel_remquo<MP2><<<1, 1>>>(x_val, y_val, r1, ri);
  CUDA_CHECK(cudaDeviceSynchronize());
  {
    char lbl[80];
    ::snprintf(lbl, sizeof(lbl), "remquo_rem(%.6g, %.6g)", x_val, y_val);
    ok = check(lbl, r1->fpmp_val, r1->ref_val, tol) && ok;
    ::snprintf(lbl, sizeof(lbl), "remquo_quo(%.6g, %.6g)", x_val, y_val);
    ok = check_int(lbl, ri->fpmp_val, ri->ref_val) && ok;
  }

#  undef RUN_1A
#  undef RUN_2A
#  undef RUN_3A
#  undef RUN_4A

  cudaFree(r1);
  cudaFree(r2);
  cudaFree(ri);
  cudaFree(rll);
  cudaFree(rl);
  return ok;
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp math functions", "[fpmp][math]")
{
#if !_CCCL_CUDA_COMPILATION()
  SKIP("fpmp2 math functions rely on CUDA device math intrinsics (device-only)");
#else
  fp_ran_on_device();
  REQUIRE(run_tests<fp32mp2>("fp32mp2", 1e-5));
  REQUIRE(run_tests<fp64mp2>("fp64mp2", 1e-12));
#endif // _CCCL_CUDA_COMPILATION()
}
