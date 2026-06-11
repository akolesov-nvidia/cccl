/*
    limits.cpp - Unit Test for cuda::std::numeric_limits<fpmp2_t> specialization
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2026

    This test validates the cuda::std::numeric_limits<> specialization for the multi-precision
    double-word types fp32mp2 (double-float) and fp64mp2 (double-double).

    Test Approach:
    -------------------------------------------------------------------------
    1. Compile-time (static_assert) checks of every reported characteristic:
       - the integer traits (digits, digits10, max_digits10, exponents, radix, flags),
       - the exact (hi, lo) components of min()/max()/lowest()/epsilon(), which are all powers of two,
       - cv-qualified forwarding and the low/high accuracy variants.
       These run for both host (g++) and device (nvcc) builds.
    2. A small host/device kernel that exercises the value members at runtime:
       (1 + epsilon) != 1, infinity() > max(), quiet_NaN() != quiet_NaN(), lowest() < 0.

    Conventions follow docs/libcudacxx/fp/fpmp_spec.rst: a normalized non-overlapping double-word
    carries 2*p - 2 contiguous mantissa bits (fp32mp2 -> 46, fp64mp2 -> 104).
*/

#include <stdio.h>
#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

namespace cs = cuda::std;

template <class T>
using nl = cs::numeric_limits<T>;

//==========================================================================================
// Compile-time checks (evaluated for both host and device builds)
//==========================================================================================

// ----- fp32mp2 (double-float) -----
static_assert(nl<fp32mp2>::is_specialized, "fp32mp2 must be specialized");
static_assert(nl<fp32mp2>::is_signed, "fp32mp2 is signed");
static_assert(!nl<fp32mp2>::is_integer, "fp32mp2 is not integer");
static_assert(!nl<fp32mp2>::is_exact, "fp32mp2 is not exact");
static_assert(nl<fp32mp2>::radix == 2, "fp32mp2 radix is 2");
static_assert(nl<fp32mp2>::digits == 46, "fp32mp2 has 2*24-2 = 46 mantissa bits");
static_assert(nl<fp32mp2>::digits10 == 13, "fp32mp2 digits10");
static_assert(nl<fp32mp2>::max_digits10 == 15, "fp32mp2 max_digits10");
static_assert(nl<fp32mp2>::max_exponent == nl<float>::max_exponent, "fp32mp2 shares float's max exponent");
static_assert(nl<fp32mp2>::min_exponent == nl<float>::min_exponent + nl<float>::digits, "fp32mp2 min exponent");
static_assert(!nl<fp32mp2>::is_iec559, "double-word is not an IEEE-754 format");
static_assert(nl<fp32mp2>::is_bounded, "fp32mp2 is bounded");
static_assert(nl<fp32mp2>::has_infinity, "fp32mp2 has infinity");
static_assert(nl<fp32mp2>::has_quiet_NaN, "fp32mp2 has quiet NaN");
static_assert(nl<fp32mp2>::round_style == cs::round_to_nearest, "fp32mp2 rounds to nearest");

// exact (hi, lo) constants -- all powers of two, so equality is exact.
static_assert(nl<fp32mp2>::epsilon().hi() == 0x1p-45f, "fp32mp2 epsilon = 2^(1-46) = 2^-45");
static_assert(nl<fp32mp2>::epsilon().lo() == 0.0f, "fp32mp2 epsilon lo is zero");
static_assert(nl<fp32mp2>::max().hi() == nl<float>::max(), "fp32mp2 max hi = FLT_MAX");
static_assert(nl<fp32mp2>::max().lo() == nl<float>::max() * 0x1p-25f, "fp32mp2 max lo");
static_assert(nl<fp32mp2>::min().hi() == 0x1p-102f, "fp32mp2 min hi = 2^-102 (smallest all-normal)");
static_assert(nl<fp32mp2>::lowest().hi() == -nl<float>::max(), "fp32mp2 lowest = -max");
static_assert(nl<fp32mp2>::round_error().hi() == 0.5f, "fp32mp2 round_error = 0.5");

// ----- fp64mp2 (double-double) -----
static_assert(nl<fp64mp2>::is_specialized, "fp64mp2 must be specialized");
static_assert(nl<fp64mp2>::is_signed, "fp64mp2 is signed");
static_assert(nl<fp64mp2>::radix == 2, "fp64mp2 radix is 2");
static_assert(nl<fp64mp2>::digits == 104, "fp64mp2 has 2*53-2 = 104 mantissa bits");
static_assert(nl<fp64mp2>::digits10 == 31, "fp64mp2 digits10");
static_assert(nl<fp64mp2>::max_digits10 == 33, "fp64mp2 max_digits10");
static_assert(nl<fp64mp2>::max_exponent == nl<double>::max_exponent, "fp64mp2 shares double's max exponent");
static_assert(nl<fp64mp2>::min_exponent == nl<double>::min_exponent + nl<double>::digits, "fp64mp2 min exponent");
static_assert(nl<fp64mp2>::min_exponent == -968, "fp64mp2 min exponent matches __ibm128 (-968)");
static_assert(!nl<fp64mp2>::is_iec559, "double-word is not an IEEE-754 format");
static_assert(nl<fp64mp2>::has_infinity, "fp64mp2 has infinity");
static_assert(nl<fp64mp2>::has_quiet_NaN, "fp64mp2 has quiet NaN");

// exact (hi, lo) constants.
static_assert(nl<fp64mp2>::epsilon().hi() == 0x1p-103, "fp64mp2 epsilon = 2^(1-104) = 2^-103");
static_assert(nl<fp64mp2>::max().hi() == nl<double>::max(), "fp64mp2 max hi = DBL_MAX");
static_assert(nl<fp64mp2>::max().lo() == nl<double>::max() * 0x1p-54, "fp64mp2 max lo");
static_assert(nl<fp64mp2>::min().hi() == 0x1p-969, "fp64mp2 min hi = 2^-969 (matches __ibm128)");
static_assert(nl<fp64mp2>::lowest().hi() == -nl<double>::max(), "fp64mp2 lowest = -max");

// ----- accuracy variants and cv-qualified forwarding -----
static_assert(nl<fp32mp2_low>::digits == 46, "low variant is specialized");
static_assert(nl<fp32mp2_high>::is_specialized, "high variant is specialized");
static_assert(nl<fp64mp2_low>::digits == 104, "low variant is specialized");
static_assert(nl<fp64mp2_high>::max_exponent == nl<double>::max_exponent, "high variant is specialized");
static_assert(nl<const fp32mp2>::digits == 46, "const-qualified forwards to the specialization");
static_assert(nl<volatile fp64mp2>::digits == 104, "volatile-qualified forwards to the specialization");

// the value members are usable in a constexpr context (they use the constexpr (hi, lo) ctor).
static constexpr fp32mp2 kEps32 = nl<fp32mp2>::epsilon();
static constexpr fp64mp2 kMax64 = nl<fp64mp2>::max();
static_assert(kEps32.hi() > 0.0f && kMax64.hi() > 0.0, "constexpr value members");

//==========================================================================================
// Runtime checks (host call or device kernel)
//==========================================================================================

#if __CUDACC__
    #define MALLOC(x, s) cudaMallocManaged(&x, s)
    #define FREE(x) cudaFree(x)
    #define RUN_CHECKS(r) run_checks<<<1, 1>>>(r)
    #define DEVICE_SYNCHRONIZE() cudaDeviceSynchronize()
    #define TARGET_DEVICE __global__
#else
    #define MALLOC(x, s) x = (int*) malloc(s)
    #define FREE(x) free(x)
    #define RUN_CHECKS(r) run_checks(r)
    #define DEVICE_SYNCHRONIZE()
    #define TARGET_DEVICE
#endif

enum
{
    CHK_EPS32 = 0, // (1 + epsilon) != 1 for fp32mp2
    CHK_EPS64,     // (1 + epsilon) != 1 for fp64mp2
    CHK_INF32,     // infinity() > max() for fp32mp2
    CHK_NAN64,     // quiet_NaN() != quiet_NaN() for fp64mp2
    CHK_LOWEST,    // lowest() < 0 for fp64mp2
    NCHK
};

TARGET_DEVICE void run_checks(int* r)
{
    const fp32mp2 one32 = fp32mp2(1.0f);
    r[CHK_EPS32] = ((one32 + nl<fp32mp2>::epsilon()) != one32) ? 1 : 0;

    const fp64mp2 one64 = fp64mp2(1.0);
    r[CHK_EPS64] = ((one64 + nl<fp64mp2>::epsilon()) != one64) ? 1 : 0;

    const double inf32 = (double) nl<fp32mp2>::infinity();
    r[CHK_INF32] = (inf32 > (double) nl<fp32mp2>::max()) ? 1 : 0;

    const double nan64 = (double) nl<fp64mp2>::quiet_NaN();
    r[CHK_NAN64] = (nan64 != nan64) ? 1 : 0; // NaN compares unequal to itself

    r[CHK_LOWEST] = ((double) nl<fp64mp2>::lowest() < 0.0) ? 1 : 0;
}

int main()
{
    printf("  ** numeric_limits<> specialization for fp32mp2 / fp64mp2\n\n");

    printf("  fp32mp2: digits=%d digits10=%d max_digits10=%d min_exp=%d max_exp=%d\n",
           nl<fp32mp2>::digits, nl<fp32mp2>::digits10, nl<fp32mp2>::max_digits10,
           nl<fp32mp2>::min_exponent, nl<fp32mp2>::max_exponent);
    printf("           min=%.6e max=%.6e epsilon=%.6e\n",
           (double) nl<fp32mp2>::min(), (double) nl<fp32mp2>::max(), (double) nl<fp32mp2>::epsilon());
    printf("  fp64mp2: digits=%d digits10=%d max_digits10=%d min_exp=%d max_exp=%d\n",
           nl<fp64mp2>::digits, nl<fp64mp2>::digits10, nl<fp64mp2>::max_digits10,
           nl<fp64mp2>::min_exponent, nl<fp64mp2>::max_exponent);
    printf("           min=%.6e max=%.6e epsilon=%.6e\n\n",
           (double) nl<fp64mp2>::min(), (double) nl<fp64mp2>::max(), (double) nl<fp64mp2>::epsilon());

    int* r;
    MALLOC(r, NCHK * sizeof(int));
    for (int i = 0; i < NCHK; ++i)
    {
        r[i] = 0;
    }

    RUN_CHECKS(r);
    DEVICE_SYNCHRONIZE();

    static const char* names[NCHK] = {
        "(1 + eps) != 1  [fp32mp2]",
        "(1 + eps) != 1  [fp64mp2]",
        "infinity() > max() [fp32mp2]",
        "quiet_NaN() != quiet_NaN() [fp64mp2]",
        "lowest() < 0    [fp64mp2]",
    };

    int fails = 0;
    for (int i = 0; i < NCHK; ++i)
    {
        if (r[i])
        {
            printf("  PASS: %s\n", names[i]);
        }
        else
        {
            printf("  ERROR: %s\n", names[i]);
            ++fails;
        }
    }

    FREE(r);

    printf("\n  %s (compile-time checks passed; %d/%d runtime checks passed)\n",
           fails ? "FAILED" : "ALL PASS", NCHK - fails, NCHK);
    return fails ? 1 : 0;
}
