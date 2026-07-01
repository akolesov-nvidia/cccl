/*
    fp64mp2_math.cpp - fp64mp2 Math Functions Demo
    ======================================================================================================

    Minimal example exercising every math function in the fpmp_math.h API for the
    fp64mp2 (double-double) type.  No accuracy checks, no reference comparisons -
    just call each function and print the result.

    Note on the fp64mp2 implementation backend:
    -------------------------------------------------------------------------
    Unlike fp32mp2 where most math functions are dedicated double-float implementations,
    fp64mp2 math functions delegate to higher-precision fallbacks:
      - When FPMP_FP128_MATH_FALLBACK = 1 (host x86 with libquadmath, or CUDA arch >= 1000):
        most calls go through __float128 (~113-bit) for accurate reference results.
      - Otherwise: fall back to system fp64 math (limited to ~52-bit accuracy).

    Functions exercised here (same API surface as fp32mp2_math):
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

    Note: icdf (integer uniform -> Gaussian, dedicated on fp32mp2 only) is a
    specialized Gaussian-sampling helper and is not exercised here.

    Additional fallback math functions exposed by the API (same on fp64mp2 as on fp32mp2),
    NOT exercised here:
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
        make EXAMPLE=fp64mp2_math               # GPU
        make EXAMPLE=fp64mp2_math TARGET=host   # CPU (needs libquadmath for FP128 fallback)
        make run EXAMPLE=fp64mp2_math
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// FPMP library headers
#include <cuda/fpmp_math>  // Multi-precision type, operations, and math functions

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if FPMP_FP64MP2_ENABLE != 1

int main()
{
    fprintf(stderr,
            "Example skipped: fp64mp2_math requires FPMP_FP64MP2_ENABLE=1\n"
            "Rebuild with -DFPMP_FP64MP2_ENABLE=1 to enable fp64mp2 support.\n");
    return 0;
}

#else // FPMP_FP64MP2_ENABLE == 1

using fptype_t = fp64mp2;

#if defined(__CUDACC__)
    #define HOST_DEVICE __host__ __device__
    #define KERNEL      __global__
#else
    #define HOST_DEVICE
    #define KERNEL
#endif

// Number of result slots
#define NUM_RESULTS 32

HOST_DEVICE void store_pair(double* out, int idx, const fptype_t& v)
{
    out[2 * idx + 0] = v.hi();
    out[2 * idx + 1] = v.lo();
}

HOST_DEVICE void fp64mp2_math_demo(double* out)
{
    // Inputs chosen to keep every function inside its meaningful domain
    fptype_t a (0.5);
    fptype_t b (2.0);
    fptype_t c (3.5);
    fptype_t d (8.0);
    fptype_t p (0.975);
    fptype_t z (1.5);
    fptype_t r1(2.7);
    fptype_t r2(-3.5);

    // Exponential / Logarithmic
    store_pair(out,  0, exp(a));
    store_pair(out,  1, log(b));
    store_pair(out,  2, pow(b, c));

    // Power
    store_pair(out,  3, cbrt(d));
    store_pair(out,  4, rcbrt(fptype_t(27.0)));

    // Trigonometric
    store_pair(out,  5, sin(a));
    store_pair(out,  6, cos(a));
    fptype_t s, k;
    sincos(a, &s, &k);
    store_pair(out,  7, s);
    store_pair(out,  8, k);

    // Hyperbolic
    store_pair(out,  9, tanh(a));

    // Error functions
    store_pair(out, 10, erf(a));
    store_pair(out, 11, erfc(a));

    // Special
    store_pair(out, 12, boys_f0(z));

    // Probability / Statistics
    store_pair(out, 13, normcdfinv(p));

    // Rounding
    store_pair(out, 14, floor(r1));
    store_pair(out, 15, ceil (r1));
    store_pair(out, 16, round(r2));
    store_pair(out, 17, trunc(r2));

    // Absolute / Min / Max
    store_pair(out, 18, fabs(r2));
    store_pair(out, 19, fmin(b, c));
    store_pair(out, 20, fmax(b, c));
    store_pair(out, 21, min (b, c));
    store_pair(out, 22, max (b, c));
} // fp64mp2_math_demo

#if defined(__CUDACC__)
KERNEL void fp64mp2_math_kernel(double* out)
{
    if ((blockIdx.x * blockDim.x + threadIdx.x) == 0)
        fp64mp2_math_demo(out);
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
    printf("  FP64MP2 MATH FUNCTIONS DEMO\n");
    printf("================================================================================\n\n");

    double* out = nullptr;
    const size_t n_doubles = 2 * NUM_RESULTS;

#if defined(__CUDACC__)
    cudaMallocManaged(&out, n_doubles * sizeof(double));
    printf("Running on GPU...\n");
    fp64mp2_math_kernel<<<1, 1>>>(out);
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
    fp64mp2_math_demo(out);
#endif

    printf("\nResults (each line shows hi component and lo magnitude):\n");
    printf("--------------------------------------------------------------------------------\n");

    int i = 0;
    print_row("exp(0.5)",          out, i++);
    print_row("log(2.0)",          out, i++);
    print_row("pow(2.0, 3.5)",     out, i++);
    print_row("cbrt(8.0)",         out, i++);
    print_row("rcbrt(27.0)",       out, i++);
    print_row("sin(0.5)",          out, i++);
    print_row("cos(0.5)",          out, i++);
    print_row("sincos(0.5).sin",   out, i++);
    print_row("sincos(0.5).cos",   out, i++);
    print_row("tanh(0.5)",         out, i++);
    print_row("erf(0.5)",          out, i++);
    print_row("erfc(0.5)",         out, i++);
    print_row("boys_f0(1.5)",      out, i++);
    print_row("normcdfinv(0.975)", out, i++);
    print_row("floor(2.7)",        out, i++);
    print_row("ceil(2.7)",         out, i++);
    print_row("round(-3.5)",       out, i++);
    print_row("trunc(-3.5)",       out, i++);
    print_row("fabs(-3.5)",        out, i++);
    print_row("fmin(2.0, 3.5)",    out, i++);
    print_row("fmax(2.0, 3.5)",    out, i++);
    print_row("min(2.0, 3.5)",     out, i++);
    print_row("max(2.0, 3.5)",     out, i++);

    printf("\n");
    printf("================================================================================\n");
    printf("  DEMO COMPLETED\n");
    printf("================================================================================\n\n");

#if defined(__CUDACC__)
    cudaFree(out);
#else
    free(out);
#endif
    return 0;
} // main

#endif // FPMP_FP64MP2_ENABLE == 1
