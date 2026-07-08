//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/*
    fp32mp2_math.cpp - Dedicated fp32mp2 Math Functions Demo
    ======================================================================================================

    Minimal example exercising every dedicated fp32mp2 math function provided by fpmp_math.h.
    No accuracy checks, no reference comparisons - just call each function and print the result.

    Dedicated fp32mp2 implementations (all pure float-float, no fp64 ops) - exercised below:
    -------------------------------------------------------------------------
      Exponential / Logarithmic : exp, log, pow
      Power                     : cbrt, rcbrt
      Trigonometric             : sin, cos, sincos
      Hyperbolic                : tanh
      Error functions           : erf, erfc
      Special                   : boys_f0
      Probability / Statistics  : normcdfinv
      Rounding                  : floor, ceil, round, trunc
      Absolute / Min / Max      : fabs, fmin, fmax, min, max

    Also dedicated but not exercised here (specialized Gaussian-sampling helpers):
      icdf(uint32), icdf(uint64)  - integer uniform -> Gaussian via normcdfinv

    Fallback math functions (placeholders delegating to higher-precision fp64 / system math) -
    available through the same API as the dedicated ones, but NOT exercised here:
    -------------------------------------------------------------------------
      Exponential / Logarithmic : log2, log10, log1p, exp2, exp10, expm1, logb, ilogb
      Trigonometric             : tan, sinpi, cospi, sincospi, asin, acos, atan, atan2
      Hyperbolic                : sinh, cosh, acosh, asinh, atanh
      Error functions           : erfcinv, erfinv, erfcx                (device only)
      Probability / Statistics  : normcdf
      Gamma                     : lgamma, tgamma
      Bessel                    : j0, j1, jn, y0, y1, yn,
                                  cyl_bessel_i0, cyl_bessel_i1
      Rounding                  : rint, nearbyint, lrint, lround,
                                  llrint, llround
      Floating-point manipulation: copysign, ldexp, scalbn, scalbln, frexp,
                                   modf, nextafter
      Min/Max/Difference        : fdim
      Remainder                 : fmod, remainder, remquo
      Distance                  : hypot, rhypot
      Vector norms              : norm3d, norm4d, rnorm3d, rnorm4d
      Classification            : fpmp_isfinite, fpmp_isinf, fpmp_isnan, fpmp_signbit
                                  (+ standard `isinf`/`isnan`/... overloads where the
                                   system <math.h> macros do not conflict)

    Build (using the provided Makefile):
        make EXAMPLE=fp32mp2_math               # GPU
        make EXAMPLE=fp32mp2_math TARGET=host   # CPU
        make run EXAMPLE=fp32mp2_math
*/
#include <cstdio>

#include <cuda/std/cstdint>
#include <cuda/std/cstdlib>

// FPMP library headers
#include <cuda/fpmp_math>  // Multi-precision type, operations, and math functions

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

using fptype_t = fp32mp2;

// Number of (hi, lo) result pairs we will store back to the host.
// Each math result is recorded as a pair of doubles (component-precision).
#define NUM_RESULTS 32

_CCCL_HOST_DEVICE void store_pair(double* out, int idx, const fptype_t& v)
{
    out[2 * idx + 0] = static_cast<double>(v.hi());
    out[2 * idx + 1] = static_cast<double>(v.lo());
}

_CCCL_HOST_DEVICE void fp32mp2_math_demo(double* out)
{
    // Inputs chosen to keep every function inside its meaningful domain
    fptype_t a (0.5);            // small magnitude, good for exp / sin / cos / tanh / erf
    fptype_t b (2.0);            // base for log / pow
    fptype_t c (3.5);            // exponent for pow
    fptype_t d (8.0);            // perfect cube for cbrt / rcbrt
    fptype_t p (0.975);          // probability for normcdfinv
    fptype_t z (1.5);            // input for boys_f0
    fptype_t r1(2.7);            // rounding samples
    fptype_t r2(-3.5);

    // Exponential / Logarithmic
    store_pair(out,  0, exp(a));                       // exp(0.5)
    store_pair(out,  1, log(b));                       // log(2)
    store_pair(out,  2, pow(b, c));                    // pow(2, 3.5)

    // Power
    store_pair(out,  3, cbrt(d));                      // cbrt(8) = 2
    store_pair(out,  4, rcbrt(fptype_t(27.0)));        // rcbrt(27) = 1/3

    // Trigonometric
    store_pair(out,  5, sin(a));                       // sin(0.5)
    store_pair(out,  6, cos(a));                       // cos(0.5)
    fptype_t s, k;
    sincos(a, &s, &k);                                 // sincos(0.5)
    store_pair(out,  7, s);
    store_pair(out,  8, k);

    // Hyperbolic
    store_pair(out,  9, tanh(a));                      // tanh(0.5)

    // Error functions
    store_pair(out, 10, erf(a));                       // erf(0.5)
    store_pair(out, 11, erfc(a));                      // erfc(0.5)

    // Special
    store_pair(out, 12, boys_f0(z));                   // F_0(1.5)

    // Probability / Statistics
    store_pair(out, 13, normcdfinv(p));                // normcdfinv(0.975) ~ 1.96

    // Rounding
    store_pair(out, 14, floor(r1));                    // floor( 2.7) = 2
    store_pair(out, 15, ceil (r1));                    // ceil ( 2.7) = 3
    store_pair(out, 16, round(r2));                    // round(-3.5) = -4 (ties-away)
    store_pair(out, 17, trunc(r2));                    // trunc(-3.5) = -3

    // Absolute / Min / Max
    store_pair(out, 18, fabs(r2));                     // fabs(-3.5) = 3.5
    store_pair(out, 19, fmin(b, c));                   // fmin(2, 3.5) = 2
    store_pair(out, 20, fmax(b, c));                   // fmax(2, 3.5) = 3.5
    store_pair(out, 21, min (b, c));                   // min (2, 3.5) = 2
    store_pair(out, 22, max (b, c));                   // max (2, 3.5) = 3.5
} // fp32mp2_math_demo

#if _CCCL_CUDA_COMPILATION()
__global__ void fp32mp2_math_kernel(double* out)
{
    if ((blockIdx.x * blockDim.x + threadIdx.x) == 0)
        fp32mp2_math_demo(out);
}
#endif

static void print_row(const char* label, const double* out, int idx)
{
    printf("  %-22s = %22.17g  (lo = %12.4e)\n",
           label, out[2 * idx + 0], out[2 * idx + 1]);
}

int main()
{
    printf("\n");
    printf("================================================================================\n");
    printf("  FP32MP2 DEDICATED MATH FUNCTIONS DEMO\n");
    printf("================================================================================\n\n");

    double* out = nullptr;
    const size_t n_doubles = 2 * NUM_RESULTS;

#if _CCCL_CUDA_COMPILATION()
    cudaMallocManaged(&out, n_doubles * sizeof(double));
    printf("Running on GPU...\n");
    fp32mp2_math_kernel<<<1, 1>>>(out);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(out);
        return 1;
    }
#else
    out = (double*)malloc(n_doubles * sizeof(double));
    printf("Running on CPU...\n");
    fp32mp2_math_demo(out);
#endif

    printf("\nResults (each line shows hi component and lo magnitude):\n");
    printf("--------------------------------------------------------------------------------\n");

    int i = 0;
    print_row("exp(0.5)",            out, i++);
    print_row("log(2.0)",            out, i++);
    print_row("pow(2.0, 3.5)",       out, i++);
    print_row("cbrt(8.0)",           out, i++);
    print_row("rcbrt(27.0)",         out, i++);
    print_row("sin(0.5)",            out, i++);
    print_row("cos(0.5)",            out, i++);
    print_row("sincos(0.5).sin",     out, i++);
    print_row("sincos(0.5).cos",     out, i++);
    print_row("tanh(0.5)",           out, i++);
    print_row("erf(0.5)",            out, i++);
    print_row("erfc(0.5)",           out, i++);
    print_row("boys_f0(1.5)",        out, i++);
    print_row("normcdfinv(0.975)",   out, i++);
    print_row("floor(2.7)",          out, i++);
    print_row("ceil(2.7)",           out, i++);
    print_row("round(-3.5)",         out, i++);
    print_row("trunc(-3.5)",         out, i++);
    print_row("fabs(-3.5)",          out, i++);
    print_row("fmin(2.0, 3.5)",      out, i++);
    print_row("fmax(2.0, 3.5)",      out, i++);
    print_row("min(2.0, 3.5)",       out, i++);
    print_row("max(2.0, 3.5)",       out, i++);

    printf("\n");
    printf("================================================================================\n");
    printf("  DEMO COMPLETED\n");
    printf("================================================================================\n\n");

#if _CCCL_CUDA_COMPILATION()
    cudaFree(out);
#else
    free(out);
#endif
    return 0;
} // main
