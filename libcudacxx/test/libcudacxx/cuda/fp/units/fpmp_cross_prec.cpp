/*
    fpmp_cross_prec.cpp - Unit test for cross-precision fp32mp2 <-> fp64mp2 conversion
    =================================================================================
    Author:  generated
    Date:    2026

    fpmp.h provides cross-precision converting constructors and assignment
    operators between fp32mp2 (double-float) and fp64mp2 (double-double):

      * Upconvert  fp32mp2 -> fp64mp2 : implicit, lossless. Uses fast_two_sum
        on the float -> double casts so the residual is preserved even when
        the input's (hi, lo) exponent gap exceeds 53 bits — i.e. when the
        value is not representable as a single double, like (1.0f, 2^-100f).

      * Downconvert fp64mp2 -> fp32mp2 : _CCCL_FPMP_EXPLICIT (matches the
        existing double -> fp32mp2 narrowing). Splits each double component
        into a (float, float) pair via __nv_fpmp2_from_double<float>, then
        renormalizes with __nv_fpmp2_add<float>.

    This unit test verifies the contract:

      1. The convertibility / assignability matrix (compile-time):
           - Upconvert is implicit (lossless widening, like float -> double).
           - Downconvert honors _CCCL_FPMP_EXPLICIT: implicit under
             CCCL_FPMP_EXPLICIT_CASTS=0 (default), explicit under =1.
           - All source accuracy tags (def / low / high) work on both sides
             and the destination accuracy tag is preserved.

      2. Upconvert mathematical correctness (runtime):
           - Bit-exact: long-double reference of (hi + lo) matches the
             fp64mp2 result's (hi + lo) for every input, including the
             pathological wide-exponent-gap case (1.0f, 2^-100f) which a
             naive (double)hi + (double)lo collapse would silently round to
             1.0 in double precision.
           - Renormalized output: |dst.lo()| <= ulp_double(dst.hi()) / 2.

      3. Downconvert mathematical correctness (runtime):
           - Relative error <= ~2 ulp at fp32mp2 precision (~48 bits) for
             values that fit, exact for values originally born in fp32mp2.
           - All three explicit-conversion forms (direct-init, functional
             cast, static_cast) and the assignment operator produce bit-
             identical fp32mp2 results.

      4. Lossless round trip fp32mp2 -> fp64mp2 -> fp32mp2 (bit-exact),
         even for the pathological case.

    All runtime checks execute on both host and device (when compiled with
    nvcc) via the same TARGET_DEVICE-decorated worker, matching the pattern
    already used by units/fpmp_accuracy_conv.cpp.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <type_traits>
#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if __CUDACC__
    #define TARGET_DEVICE __global__
    #define LAUNCH(fn, ...) fn<<<1,1>>>(__VA_ARGS__); cudaDeviceSynchronize()
#else
    #define TARGET_DEVICE
    #define LAUNCH(fn, ...) fn(__VA_ARGS__)
#endif

using namespace fpmp;

/* =====================================================================
 * Compile-time contract
 * ===================================================================== */

/* --- Upconvert (fp32mp2 -> fp64mp2): implicit, lossless ----------------- */
static_assert(std::is_constructible<fp64mp2,          fp32mp2>         ::value, "");
static_assert(std::is_constructible<fp64mp2,          fp32mp2_low>    ::value, "");
static_assert(std::is_constructible<fp64mp2,          fp32mp2_high>::value, "");
static_assert(std::is_constructible<fp64mp2_low,     fp32mp2>         ::value, "");
static_assert(std::is_constructible<fp64mp2_high, fp32mp2>         ::value, "");

static_assert(std::is_convertible<fp32mp2,          fp64mp2>         ::value,
              "fp32mp2 -> fp64mp2 must be implicit (lossless widening)");
static_assert(std::is_convertible<fp32mp2_low,     fp64mp2_high>::value,
              "cross-accuracy upconvert must be implicit");
static_assert(std::is_convertible<fp32mp2_high, fp64mp2_low>    ::value,
              "cross-accuracy upconvert must be implicit");

static_assert(std::is_assignable<fp64mp2&,          fp32mp2>         ::value, "");
static_assert(std::is_assignable<fp64mp2_low&,     fp32mp2_high>::value, "");
static_assert(std::is_assignable<fp64mp2_high&, fp32mp2_low>    ::value, "");

/* The destination accuracy tag is preserved across precision conversion. */
static_assert(std::is_same<decltype(fp64mp2_low(std::declval<fp32mp2_high>())),
                           fp64mp2_low>::value, "");

/* --- Downconvert (fp64mp2 -> fp32mp2): explicit-macro-driven ------------ */
static_assert(std::is_constructible<fp32mp2,          fp64mp2>         ::value, "");
static_assert(std::is_constructible<fp32mp2,          fp64mp2_low>    ::value, "");
static_assert(std::is_constructible<fp32mp2,          fp64mp2_high>::value, "");
static_assert(std::is_constructible<fp32mp2_low,     fp64mp2>         ::value, "");
static_assert(std::is_constructible<fp32mp2_high, fp64mp2>         ::value, "");

/* Convertibility (implicit conversion) depends on CCCL_FPMP_EXPLICIT_CASTS.
 * The new converting constructor uses _CCCL_FPMP_EXPLICIT which mirrors the
 * existing (double -> fp32mp2) narrowing-constructor convention. */
#if CCCL_FPMP_EXPLICIT_CASTS == 1
    static_assert(!std::is_convertible<fp64mp2, fp32mp2>::value,
                  "downconvert must NOT be implicit under EXPLICIT_CASTS=1");
#else
    static_assert(std::is_convertible<fp64mp2, fp32mp2>::value,
                  "downconvert is implicit under EXPLICIT_CASTS=0 (default)");
#endif

/* Assignment from fp64mp2 to fp32mp2 is always available (dedicated
 * operator= overload), independent of EXPLICIT_CASTS. */
static_assert(std::is_assignable<fp32mp2&,          fp64mp2>         ::value, "");
static_assert(std::is_assignable<fp32mp2_low&,     fp64mp2_high>::value, "");
static_assert(std::is_assignable<fp32mp2_high&, fp64mp2_low>    ::value, "");

static_assert(std::is_same<decltype(fp32mp2_high(std::declval<fp64mp2_low>())),
                           fp32mp2_high>::value, "");


/* =====================================================================
 * Runtime: shared counters + helpers
 * ===================================================================== */
static int g_pass = 0;
static int g_fail = 0;

/* Plain (hi, lo) input/output triples shared between host and device. */
struct F32 { float  hi, lo; };
struct F64 { double hi, lo; };

/* =====================================================================
 * Runtime: data construction in the same target as the conversions.
 *
 * Computing the input/expected values inside a TARGET_DEVICE kernel keeps
 * host and device runs on identical bits (any difference between IEEE-
 * compliant host casts and CUDA device casts would surface here).
 * ===================================================================== */
TARGET_DEVICE void compute_test_data(F32* in32, F64* up_out, F64* in64, F32* down_out)
{
    /* === Upconvert inputs: span the renormalization spectrum =========== */
    /* 0. ordinary value with non-trivial lo (typical case). */
    in32[0] = { 1.2345678f, 1.0e-9f };

    /* 1. PATHOLOGICAL CASE: lo is many bits below ulp(hi)/2. A renormalized
     *    fp32mp2 may legally hold lo at any exponent, including far below
     *    the float ulp of hi. (hi=1.0, lo=2^-100) represents 1 + 2^-100,
     *    which CANNOT be expressed as a single double — verifies that the
     *    upconvert preserves the residual via fast_two_sum. */
    in32[1] = { 1.0f, 0x1.0p-100f };

    /* 2. negative hi, opposite-sign tiny lo. */
    in32[2] = { -3.1415927f, 1.5e-8f };

    /* 3. wide exponent gap, also pathological. */
    in32[3] = { 1.0e+30f, 0x1.0p-110f };

    /* 4. boundary: hi near float max, lo at near-min. */
    in32[4] = { 0x1.0p+120f, 0x1.0p-149f };

    /* 5. zero pair. */
    in32[5] = { 0.0f, 0.0f };

    /* 6. hi only, lo == 0. */
    in32[6] = { 3.14f, 0.0f };

    /* 7. typical |lo| ~ ulp(hi)/2 (fits in single double exactly). */
    in32[7] = { 1.0f, 0x1.0p-25f };

    /* Compute each upconvert via the new implicit converting ctor and
     * stash the resulting fp64mp2 pair. */
    for (int i = 0; i < 8; ++i) {
        fp32mp2 src(in32[i].hi, in32[i].lo);
        fp64mp2 dst = src;                /* implicit upconvert */
        up_out[i] = { dst.hi(), dst.lo() };
    }

    /* === Downconvert inputs: values needing > 24 bits ================== */
    /* 0. canonical double value: needs both fp32mp2 components. */
    in64[0] = { 1.234567890123456, 1.0e-18 };

    /* 1. large in-float-range magnitude with a non-trivial lo. */
    in64[1] = { 1.0e+30, 1.5e+13 };

    /* 2. negative. */
    in64[2] = { -2.7182818284590452, 1.5e-17 };

    /* 3. value originally born in fp32mp2 (should round-trip bit-exact
     *    through fp64mp2 then back). */
    in64[3] = { (double)1.2345678f, (double)1.0e-9f };

    for (int i = 0; i < 4; ++i) {
        fp64mp2 src(in64[i].hi, in64[i].lo);
        fp32mp2 dst(src);                 /* explicit downconvert (direct-init) */
        down_out[i] = { dst.hi(), dst.lo() };
    }
}

/* =====================================================================
 * Runtime: host-side validators
 * ===================================================================== */

/* Verify the fp32mp2 -> fp64mp2 upconvert with three independent checks:
 *
 *  1. BIT-EXACT vs. independently derived fast_two_sum reference.
 *     A wider-than-double "reference sum" (e.g. long double) is NOT
 *     sufficient for the pathological (hi=1, lo=2^-100) case because
 *     even long double's 64-bit mantissa cannot capture a residual that
 *     many bits below the unit; a buggy collapse to (double)hi+(double)lo
 *     would round to 1.0 and the long-double reference would round to the
 *     same 1.0, hiding the regression.
 *
 *     Instead we recompute the expected (ref_hi, ref_lo) pair by inlining
 *     the error-free fast_two_sum transformation on the exact float -> double
 *     promotions (both casts are bit-exact). This is well-defined, has a
 *     unique correct result, and catches ANY loss of the low residual.
 *
 *  2. Mathematical-value equality at component granularity:
 *     dst.lo must be exactly the rounding error of (double)src.hi + (double)src.lo,
 *     so (dst.hi - (double)src.hi) + dst.lo == (double)src.lo in exact math.
 *     Since fp32mp2 components widen losslessly to double, the equation
 *     reduces to bit-exact double arithmetic which we can verify directly.
 *
 *  3. Renormalization invariant: |dst.lo| <= ulp_double(dst.hi) / 2.
 *     Required so the result is a well-formed fp64mp2.
 */
static void check_upconvert(const char* label, const F32& src, const F64& dst)
{
    /* float -> double is exact (subset of representable values). */
    const double a = (double)src.hi;
    const double b = (double)src.lo;

    /* Inline reference: fast_two_sum(a, b) -- unique correct output. */
    const double ref_hi = a + b;
    const double z      = ref_hi - a;   /* portion of b that fit in ref_hi */
    const double ref_lo = b - z;        /* exact residual */

    const bool bit_exact = (dst.hi == ref_hi) && (dst.lo == ref_lo);

    /* Renormalization invariant on the produced fp64mp2 pair. */
    bool renorm;
    if (dst.hi == 0.0) {
        renorm = (dst.lo == 0.0);
    } else {
        const double ulp_hi = std::ldexp(1.0, std::ilogb(dst.hi) - 52);
        renorm = std::fabs(dst.lo) <= 0.5 * ulp_hi;
    }

    if (bit_exact && renorm) {
        printf("  PASS  %-44s (hi=%a,lo=%a) -> (hi=%a,lo=%a)\n",
               label, src.hi, src.lo, dst.hi, dst.lo);
        g_pass++;
    } else {
        printf("  FAIL  %-44s (hi=%a,lo=%a) -> (hi=%a,lo=%a)  bit_exact=%d renorm=%d\n",
               label, src.hi, src.lo, dst.hi, dst.lo, (int)bit_exact, (int)renorm);
        printf("        expected (hi=%a, lo=%a)\n", ref_hi, ref_lo);
        g_fail++;
    }
}

/* Explicit "residual must be non-zero" assertion for the inputs whose
 * naive (double)hi+(double)lo collapse would silently round the residual
 * to zero -- i.e. fp32mp2 values that CANNOT be represented as a single
 * double. A regression to such a collapse would set dst.lo == 0 here. */
static void check_residual_preserved(const char* label, const F64& dst)
{
    if (dst.lo != 0.0) {
        printf("  PASS  %-44s residual preserved (dst.lo=%a)\n", label, dst.lo);
        g_pass++;
    } else {
        printf("  FAIL  %-44s residual lost (dst.lo == 0; the naive sum collapsed)\n",
               label);
        g_fail++;
    }
}

/* Downconvert is lossy in the general case but bit-exact for values that
 * are originally fp32mp2-representable. We measure relative error against
 * the long-double sum of the fp64mp2 input; the fp32mp2 tolerance is one
 * ulp at 48 bits (~2^-47 ~ 7.1e-15), with a small safety factor. */
static void check_downconvert(const char* label, const F64& src, const F32& dst,
                              long double tol)
{
    const long double ref = (long double)src.hi + (long double)src.lo;
    const long double got = (long double)dst.hi + (long double)dst.lo;
    const long double abs_ref = std::fabs(ref);
    const long double rel_err = (abs_ref > 0.0L)
                                ? std::fabs((got - ref) / ref)
                                : std::fabs(got - ref);

    if (rel_err <= tol) {
        printf("  PASS  %-44s rel_err=%.3Le (tol=%.1Le)\n",
               label, rel_err, tol);
        g_pass++;
    } else {
        printf("  FAIL  %-44s rel_err=%.3Le > tol=%.1Le  (got=(hi=%a,lo=%a))\n",
               label, rel_err, tol, dst.hi, dst.lo);
        g_fail++;
    }
}

/* =====================================================================
 * Runtime: round-trip + explicit-form coverage worker (runs target-side
 * so device behavior is exercised identically).
 * ===================================================================== */
TARGET_DEVICE void compute_extras(F32* round_trip_in, F32* round_trip_out,
                                  F32* explicit_forms_out)
{
    /* Round trip: fp32mp2 -> fp64mp2 -> fp32mp2 must be lossless even for
     * the pathological wide-exponent-gap case. */
    round_trip_in[0] = { 1.0f, 0x1.0p-100f };  /* pathological */
    round_trip_in[1] = { 1.2345678f, 1.0e-9f };
    round_trip_in[2] = { 0x1.0p+120f, 0x1.0p-149f };
    for (int i = 0; i < 3; ++i) {
        fp32mp2 a(round_trip_in[i].hi, round_trip_in[i].lo);
        fp64mp2 b = a;                          /* implicit upconvert */
        fp32mp2 c = static_cast<fp32mp2>(b);  /* explicit downconvert */
        round_trip_out[i] = { c.hi(), c.lo() };
    }

    /* Five explicit conversion forms must all yield the same fp32mp2 pair. */
    fp64mp2 src(1.234567890123456, 1.0e-18);
    fp32mp2 a(src);                             /* 0. direct-init */
    fp32mp2 b = fp32mp2(src);                 /* 1. functional cast */
    fp32mp2 c = static_cast<fp32mp2>(src);    /* 2. static_cast */
    fp32mp2 d;
    d = static_cast<fp32mp2>(src);              /* 3. assign via cast */
    fp32mp2 e;
    e = src;                                      /* 4. operator= overload */
    explicit_forms_out[0] = { a.hi(), a.lo() };
    explicit_forms_out[1] = { b.hi(), b.lo() };
    explicit_forms_out[2] = { c.hi(), c.lo() };
    explicit_forms_out[3] = { d.hi(), d.lo() };
    explicit_forms_out[4] = { e.hi(), e.lo() };
}

/* =====================================================================
 * main: orchestrate kernel launch + host-side validation
 * ===================================================================== */
int main()
{
    printf("\n  fpmp_cross_prec: fp32mp2 <-> fp64mp2 cross-precision conversion\n");
    printf("  ==================================================================\n");

    constexpr int N_UP   = 8;
    constexpr int N_DOWN = 4;
    constexpr int N_RT   = 3;
    constexpr int N_EXP  = 5;

    F32 in32[N_UP],   round_trip_in[N_RT], round_trip_out[N_RT], explicit_forms[N_EXP];
    F64 up_out[N_UP], in64[N_DOWN];
    F32 down_out[N_DOWN];

#if __CUDACC__
    F32 *d_in32, *d_rt_in, *d_rt_out, *d_explicit, *d_down_out;
    F64 *d_up_out, *d_in64;
    cudaMallocManaged(&d_in32,      sizeof(in32));
    cudaMallocManaged(&d_up_out,    sizeof(up_out));
    cudaMallocManaged(&d_in64,      sizeof(in64));
    cudaMallocManaged(&d_down_out,  sizeof(down_out));
    cudaMallocManaged(&d_rt_in,     sizeof(round_trip_in));
    cudaMallocManaged(&d_rt_out,    sizeof(round_trip_out));
    cudaMallocManaged(&d_explicit,  sizeof(explicit_forms));

    LAUNCH(compute_test_data, d_in32, d_up_out, d_in64, d_down_out);
    LAUNCH(compute_extras,    d_rt_in, d_rt_out, d_explicit);

    for (int i = 0; i < N_UP;   ++i) { in32[i]           = d_in32[i];     up_out[i]   = d_up_out[i]; }
    for (int i = 0; i < N_DOWN; ++i) { in64[i]           = d_in64[i];     down_out[i] = d_down_out[i]; }
    for (int i = 0; i < N_RT;   ++i) { round_trip_in[i]  = d_rt_in[i];    round_trip_out[i] = d_rt_out[i]; }
    for (int i = 0; i < N_EXP;  ++i) { explicit_forms[i] = d_explicit[i]; }

    cudaFree(d_in32); cudaFree(d_up_out); cudaFree(d_in64); cudaFree(d_down_out);
    cudaFree(d_rt_in); cudaFree(d_rt_out); cudaFree(d_explicit);
#else
    compute_test_data(in32, up_out, in64, down_out);
    compute_extras(round_trip_in, round_trip_out, explicit_forms);
#endif

    /* ---- Upconvert: bit-exact value + renormalized output ---- */
    printf("\n  --- Upconvert fp32mp2 -> fp64mp2 (implicit, lossless) ---\n");
    static const char* const up_labels[] = {
        "[0] ordinary",
        "[1] pathological (hi=1, lo=2^-100)",
        "[2] negative hi",
        "[3] wide exponent gap",
        "[4] near float max",
        "[5] zero",
        "[6] hi only",
        "[7] |lo| ~ ulp(hi)/2",
    };
    for (int i = 0; i < N_UP; ++i) {
        check_upconvert(up_labels[i], in32[i], up_out[i]);
    }

    /* For the wide-exponent-gap cases the fp32mp2 value is NOT representable
     * as a single double, so a buggy upconvert that collapsed via
     * (double)hi + (double)lo would have set dst.lo == 0 and silently lost
     * the residual. These checks fail loudly in that scenario. */
    printf("\n  --- Residual must survive in dst.lo for inputs > fp64 single-double range ---\n");
    check_residual_preserved(up_labels[1], up_out[1]);
    check_residual_preserved(up_labels[3], up_out[3]);
    check_residual_preserved(up_labels[4], up_out[4]);

    /* ---- Downconvert: bounded relative error ---- */
    printf("\n  --- Downconvert fp64mp2 -> fp32mp2 (explicit, lossy) ---\n");
    /* fp32mp2 holds ~48 bits; 2 ulp at that precision is ~7e-15. */
    const long double tol_general = 1.5e-14L;
    /* The fp32mp2-originated value must round-trip bit-exact. */
    const long double tol_exact   = 0.0L;
    static const char* const down_labels[] = {
        "[0] canonical double",
        "[1] large magnitude",
        "[2] negative",
        "[3] fp32mp2-born (must be exact)",
    };
    const long double tols[] = { tol_general, tol_general, tol_general, tol_exact };
    for (int i = 0; i < N_DOWN; ++i) {
        check_downconvert(down_labels[i], in64[i], down_out[i], tols[i]);
    }

    /* ---- Round trip: fp32mp2 -> fp64mp2 -> fp32mp2 is bit-exact ---- */
    printf("\n  --- Round trip fp32mp2 -> fp64mp2 -> fp32mp2 (lossless) ---\n");
    static const char* const rt_labels[] = {
        "[0] pathological (1.0, 2^-100)",
        "[1] ordinary",
        "[2] near float max",
    };
    for (int i = 0; i < N_RT; ++i) {
        const bool ok = (round_trip_in[i].hi == round_trip_out[i].hi)
                     && (round_trip_in[i].lo == round_trip_out[i].lo);
        if (ok) {
            printf("  PASS  %-44s (hi=%a,lo=%a)\n",
                   rt_labels[i], round_trip_in[i].hi, round_trip_in[i].lo);
            g_pass++;
        } else {
            printf("  FAIL  %-44s in=(hi=%a,lo=%a) out=(hi=%a,lo=%a)\n",
                   rt_labels[i],
                   round_trip_in[i].hi, round_trip_in[i].lo,
                   round_trip_out[i].hi, round_trip_out[i].lo);
            g_fail++;
        }
    }

    /* ---- Explicit-form consistency: direct-init / functional / static_cast /
     *      assign-via-cast / operator= all produce the same fp32mp2. ---- */
    printf("\n  --- All explicit conversion forms produce bit-identical fp32mp2 ---\n");
    {
        const F32& ref = explicit_forms[0];
        bool all_same = true;
        for (int i = 1; i < N_EXP; ++i) {
            if (explicit_forms[i].hi != ref.hi || explicit_forms[i].lo != ref.lo) {
                all_same = false;
                printf("    diverged at form[%d]: (hi=%a,lo=%a)\n",
                       i, explicit_forms[i].hi, explicit_forms[i].lo);
            }
        }
        if (all_same) {
            printf("  PASS  direct-init / functional / static_cast / assign-cast / operator= all match\n");
            g_pass++;
        } else {
            printf("  FAIL  one of the five explicit conversion forms diverged\n");
            g_fail++;
        }
    }


    printf("\n  ==================================================================\n");
    printf("  Total: %d passed, %d failed\n\n", g_pass, g_fail);
    return g_fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
