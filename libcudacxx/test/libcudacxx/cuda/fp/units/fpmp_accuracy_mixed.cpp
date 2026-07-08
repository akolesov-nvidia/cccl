/*
    fpmp_accuracy_mixed.cpp - Unit test for mixed-type accuracy-explicit overloads
    =========================================================================
    Author:  generated
    Date:    2026

    Companion to fpmp_accuracy.cpp.  Tests the mixed-type overloads of the
    accuracy-explicit free functions add<m>, sub<m>, mul<m>, div<m>, fma<m>,
    mad<m>: a single symmetric template per operation that accepts any
    combination of fpmp2 and built-in arithmetic operands as long
    as at least one operand is fpmp2.  Example call sites:

        ffloat r1 = sub<fpmp2_accuracy::high>(a, 1.0f);   // ffloat - float
        ffloat r2 = sub<fpmp2_accuracy::high>(1.0f, a);   // float  - ffloat
        ffloat r3 = mad<fpmp2_accuracy::low>(2.0f, a, b);    // float * ffloat + ffloat

    Verifies:
      1. Each mixed-type call is bit-identical to the equivalent strict
         (all-mp2) call where the scalar(s) are explicitly wrapped in the
         participating fpmp2 type.  Wrapper logic is accuracy-independent,
         so def / low / high are spot-checked across operations.
      2. Result type preserves the participating fpmp2 specialization
         (compile-time `static_assert` over fp32mp2, fp32mp2_low and
         fp32mp2_high).
      3. Both argument orders for binary ops, and every scalar position
         for ternary ops, route correctly through the symmetric overload.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <utility>
#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if __CUDACC__
    #define TARGET_DEVICE __global__
    #define LAUNCH(fn, ...) fn<<<1,1>>>(__VA_ARGS__); cudaDeviceSynchronize()
#else
    #define TARGET_DEVICE
    #define LAUNCH(fn, ...) fn(__VA_ARGS__)
#endif


/* ------------------------------------------------------------------
 * Compile-time checks: result type must match the participating
 * fpmp2 type, regardless of operand order or which arg is the scalar.
 * ------------------------------------------------------------------*/
static_assert(std::is_same<
    decltype(add<fpmp2_accuracy::high>(std::declval<fp32mp2>(), 1.0f)),
    fp32mp2>::value, "add(mp2_def, float) must return mp2_def");

static_assert(std::is_same<
    decltype(sub<fpmp2_accuracy::low>(1.0f, std::declval<fp32mp2_low>())),
    fp32mp2_low>::value, "sub(float, mp2_low) must return mp2_low");

static_assert(std::is_same<
    decltype(mul<fpmp2_accuracy::high>(std::declval<fp32mp2_high>(), 2)),
    fp32mp2_high>::value, "mul(mp2_high, int) must return mp2_high");

static_assert(std::is_same<
    decltype(fma<fpmp2_accuracy::low>(1.0f, std::declval<fp32mp2_low>(), 2.0f)),
    fp32mp2_low>::value, "fma(float, mp2_low, float) must return mp2_low");

static_assert(std::is_same<
    decltype(mad<fpmp2_accuracy::def>(1.0f, 2.0f, std::declval<fp32mp2>())),
    fp32mp2>::value, "mad(float, float, mp2_def) must return mp2_def");

/* ------------------------------------------------------------------
 * Runtime test infrastructure.
 * ------------------------------------------------------------------*/
static int g_pass = 0;
static int g_fail = 0;

static void check(const char* label, double got, double expected) {
    bool ok = (got == expected);
    if (ok) {
        printf("  PASS  %-62s  got=%.15e\n", label, got);
        g_pass++;
    } else {
        printf("  FAIL  %-62s  got=%.15e  exp=%.15e  diff=%.3e\n",
               label, got, expected, std::fabs(got - expected));
        g_fail++;
    }
}

struct Results {
    /* binary, both argument orders */
    double add_lr, add_rl;
    double sub_lr, sub_rl;
    double mul_lr, mul_rl;
    double div_lr, div_rl;
    /* ternary, every scalar position
     *   _mss = (mp2, scalar, scalar), _sms = (scalar, mp2, scalar), etc. */
    double fma_mss, fma_sms, fma_ssm;
    double mad_mms, mad_msm, mad_smm;
};

/* The mixed-type wrappers should produce bit-identical results to the
 * explicit strict-form calls.  Accuracy levels chosen to cover def / low /
 * high across the six operations. */
TARGET_DEVICE void compute_mixed(Results* r) {
    fp32mp2_low a(1.234567890f), b(2.345678901f);
    const float    s = 0.5f;
    const float    t = 3.0f;

    /* binary */
    r->add_lr = static_cast<double>(add<fpmp2_accuracy::high>(a, s));
    r->add_rl = static_cast<double>(add<fpmp2_accuracy::high>(s, a));
    r->sub_lr = static_cast<double>(sub<fpmp2_accuracy::high>(a, s));
    r->sub_rl = static_cast<double>(sub<fpmp2_accuracy::high>(s, a));
    r->mul_lr = static_cast<double>(mul<fpmp2_accuracy::low>    (a, s));
    r->mul_rl = static_cast<double>(mul<fpmp2_accuracy::low>    (s, a));
    r->div_lr = static_cast<double>(div<fpmp2_accuracy::def>     (a, s));
    r->div_rl = static_cast<double>(div<fpmp2_accuracy::def>     (s, a));

    /* ternary: one mp2 operand, scalar in each position */
    r->fma_mss = static_cast<double>(fma<fpmp2_accuracy::high>(a, s, t));
    r->fma_sms = static_cast<double>(fma<fpmp2_accuracy::high>(s, a, t));
    r->fma_ssm = static_cast<double>(fma<fpmp2_accuracy::high>(s, t, a));

    /* ternary: two mp2 operands, one scalar position */
    r->mad_mms = static_cast<double>(mad<fpmp2_accuracy::low>(a, b, s));
    r->mad_msm = static_cast<double>(mad<fpmp2_accuracy::low>(a, s, b));
    r->mad_smm = static_cast<double>(mad<fpmp2_accuracy::low>(s, a, b));
}

TARGET_DEVICE void compute_reference(Results* r) {
    using ff = fp32mp2_low;
    ff a(1.234567890f), b(2.345678901f);
    const float s = 0.5f;
    const float t = 3.0f;

    /* Same calls, scalars manually wrapped in the fpmp2 type — this is
     * literally what the mixed-type wrapper expands to. */
    r->add_lr = static_cast<double>(add<fpmp2_accuracy::high>(a,     ff(s)));
    r->add_rl = static_cast<double>(add<fpmp2_accuracy::high>(ff(s), a));
    r->sub_lr = static_cast<double>(sub<fpmp2_accuracy::high>(a,     ff(s)));
    r->sub_rl = static_cast<double>(sub<fpmp2_accuracy::high>(ff(s), a));
    r->mul_lr = static_cast<double>(mul<fpmp2_accuracy::low>    (a,     ff(s)));
    r->mul_rl = static_cast<double>(mul<fpmp2_accuracy::low>    (ff(s), a));
    r->div_lr = static_cast<double>(div<fpmp2_accuracy::def>     (a,     ff(s)));
    r->div_rl = static_cast<double>(div<fpmp2_accuracy::def>     (ff(s), a));

    r->fma_mss = static_cast<double>(fma<fpmp2_accuracy::high>(a,     ff(s), ff(t)));
    r->fma_sms = static_cast<double>(fma<fpmp2_accuracy::high>(ff(s), a,     ff(t)));
    r->fma_ssm = static_cast<double>(fma<fpmp2_accuracy::high>(ff(s), ff(t), a));

    r->mad_mms = static_cast<double>(mad<fpmp2_accuracy::low>(a,     b,     ff(s)));
    r->mad_msm = static_cast<double>(mad<fpmp2_accuracy::low>(a,     ff(s), b));
    r->mad_smm = static_cast<double>(mad<fpmp2_accuracy::low>(ff(s), a,     b));
}

int main() {
    printf("\n  fpmp_accuracy_mixed: unit test for mixed-type accuracy-explicit overloads\n");
    printf("  =====================================================================\n");

    Results res;
    Results ref;

#if __CUDACC__
    Results *d_res;
    Results *d_ref;
    cudaMallocManaged(&d_res, sizeof(Results));
    cudaMallocManaged(&d_ref, sizeof(Results));
    LAUNCH(compute_mixed,     d_res);
    LAUNCH(compute_reference, d_ref);
    res = *d_res;
    ref = *d_ref;
    cudaFree(d_res);
    cudaFree(d_ref);
#else
    compute_mixed(&res);
    compute_reference(&ref);
#endif

    printf("\n  --- Binary mixed (mp2, scalar) and (scalar, mp2) ---\n");
    check("add<high>(mp2, scalar) == add(mp2, ff(scalar))", res.add_lr, ref.add_lr);
    check("add<high>(scalar, mp2) == add(ff(scalar), mp2)", res.add_rl, ref.add_rl);
    check("sub<high>(mp2, scalar) == sub(mp2, ff(scalar))", res.sub_lr, ref.sub_lr);
    check("sub<high>(scalar, mp2) == sub(ff(scalar), mp2)", res.sub_rl, ref.sub_rl);
    check("mul<low> (mp2, scalar) == mul(mp2, ff(scalar))", res.mul_lr, ref.mul_lr);
    check("mul<low> (scalar, mp2) == mul(ff(scalar), mp2)", res.mul_rl, ref.mul_rl);
    check("div<def>     (mp2, scalar) == div(mp2, ff(scalar))", res.div_lr, ref.div_lr);
    check("div<def>     (scalar, mp2) == div(ff(scalar), mp2)", res.div_rl, ref.div_rl);

    printf("\n  --- Ternary fma<m>: scalar in every position ---\n");
    check("fma<high>(mp2, scalar, scalar) == strict",   res.fma_mss, ref.fma_mss);
    check("fma<high>(scalar, mp2, scalar) == strict",   res.fma_sms, ref.fma_sms);
    check("fma<high>(scalar, scalar, mp2) == strict",   res.fma_ssm, ref.fma_ssm);

    printf("\n  --- Ternary mad<m>: one scalar, two mp2 operands ---\n");
    check("mad<low>(mp2, mp2, scalar)        == strict",   res.mad_mms, ref.mad_mms);
    check("mad<low>(mp2, scalar, mp2)        == strict",   res.mad_msm, ref.mad_msm);
    check("mad<low>(scalar, mp2, mp2)        == strict",   res.mad_smm, ref.mad_smm);

    printf("\n  ===========================================================\n");
    printf("  Total: %d passed, %d failed\n\n", g_pass, g_fail);
    return g_fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
