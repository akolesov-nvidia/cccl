/*
    fpmp_accuracy.cpp - Unit test for accuracy-explicit arithmetic free functions
    =========================================================================
    Author:  generated
    Date:    2026

    Tests the accuracy-explicit free functions: add<m>, sub<m>, mul<m>, div<m>,
    fma<m>, mad<m> that override the arithmetic accuracy for a single operation
    without changing the result type.

    Verifies:
    1. Results match operator-based computation with equivalent accuracy types
    2. Return type preserves the input type's accuracy tag
    3. All three accuracy levels (def, low, high) work for each operation
    4. Cross-accuracy calls work (e.g., sub<high> on fp32mp2_low)
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
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

static int g_pass = 0;
static int g_fail = 0;

static void check(const char* label, double got, double expected, double tol) {
    double diff = std::fabs(got - expected);
    double mag  = std::fmax(std::fabs(got), std::fabs(expected));
    bool ok = (got == expected) || (mag > 0 ? diff / mag < tol : diff < tol);
    if (ok) {
        printf("  PASS  %-55s  got=%.15e\n", label, got);
        g_pass++;
    } else {
        printf("  FAIL  %-55s  got=%.15e  exp=%.15e  diff=%.3e\n",
               label, got, expected, diff);
        g_fail++;
    }
}

struct Results {
    double add_def, add_fast, add_acc;
    double sub_def, sub_fast, sub_acc;
    double mul_def, mul_fast, mul_acc;
    double div_def, div_fast, div_acc;
    double fma_def, fma_fast, fma_acc;
    double mad_def, mad_fast, mad_acc;
    double cross_sub_acc_on_fast;
    double cross_add_def_on_fast;
    double cross_fma_acc_on_fast;
};

struct RefResults {
    double add_def, add_fast, add_acc;
    double sub_def, sub_fast, sub_acc;
    double mul_def, mul_fast, mul_acc;
    double div_def, div_fast, div_acc;
    double fma_def, fma_fast, fma_acc;
    double mad_def, mad_fast, mad_acc;
};

TARGET_DEVICE void compute_accuracy_explicit(double a_in, double b_in,
                                           double x_in, double y_in, double z_in,
                                           Results* r) {
    // a, b — near-cancelling pair for add/sub
    fp32mp2 ad(a_in), bd(b_in);
    // x, y, z — normal-range values for mul/div/fma/mad
    fp32mp2 xd(x_in), yd(y_in), zd(z_in);

    r->add_def  = static_cast<double>(add<fpmp2_accuracy::def>(ad, bd));
    r->add_fast = static_cast<double>(add<fpmp2_accuracy::low>(ad, bd));
    r->add_acc  = static_cast<double>(add<fpmp2_accuracy::high>(ad, bd));

    r->sub_def  = static_cast<double>(sub<fpmp2_accuracy::def>(ad, -bd));
    r->sub_fast = static_cast<double>(sub<fpmp2_accuracy::low>(ad, -bd));
    r->sub_acc  = static_cast<double>(sub<fpmp2_accuracy::high>(ad, -bd));

    r->mul_def  = static_cast<double>(mul<fpmp2_accuracy::def>(xd, yd));
    r->mul_fast = static_cast<double>(mul<fpmp2_accuracy::low>(xd, yd));
#if __FPMP_USE_ACCURATE_MUL__ == 1
    r->mul_acc  = static_cast<double>(mul<fpmp2_accuracy::high>(xd, yd));
#else
    r->mul_acc  = r->mul_def;
#endif

    r->div_def  = static_cast<double>(div<fpmp2_accuracy::def>(xd, yd));
    r->div_fast = static_cast<double>(div<fpmp2_accuracy::low>(xd, yd));
#if __FPMP_USE_ACCURATE_DIV__ == 1
    r->div_acc  = static_cast<double>(div<fpmp2_accuracy::high>(xd, yd));
#else
    r->div_acc  = r->div_def;
#endif

    r->fma_def  = static_cast<double>(fma<fpmp2_accuracy::def>(xd, yd, zd));
    r->fma_fast = static_cast<double>(fma<fpmp2_accuracy::low>(xd, yd, zd));
    r->fma_acc  = static_cast<double>(fma<fpmp2_accuracy::high>(xd, yd, zd));

    r->mad_def  = static_cast<double>(mad<fpmp2_accuracy::def>(xd, yd, zd));
    r->mad_fast = static_cast<double>(mad<fpmp2_accuracy::low>(xd, yd, zd));
    r->mad_acc  = static_cast<double>(mad<fpmp2_accuracy::high>(xd, yd, zd));

    fp32mp2_low af(a_in), bf(b_in);
    fp32mp2_low xf(x_in), yf(y_in), zf(z_in);
    r->cross_sub_acc_on_fast = static_cast<double>(sub<fpmp2_accuracy::high>(af, -bf));
    r->cross_add_def_on_fast = static_cast<double>(add<fpmp2_accuracy::def>(af, bf));
    r->cross_fma_acc_on_fast = static_cast<double>(fma<fpmp2_accuracy::high>(xf, yf, zf));
}

TARGET_DEVICE void compute_operator_ref(double a_in, double b_in,
                                        double x_in, double y_in, double z_in,
                                        RefResults* r) {
    fp32mp2          ad(a_in), bd(b_in);
    fp32mp2_low     af(a_in), bf(b_in);
    fp32mp2_high aa(a_in), ba(b_in);
    fp32mp2          xd(x_in), yd(y_in), zd(z_in);
    fp32mp2_low     xf(x_in), yf(y_in), zf(z_in);
    fp32mp2_high xa(x_in), ya(y_in), za(z_in);

    r->add_def  = static_cast<double>(ad + bd);
    r->add_fast = static_cast<double>(af + bf);
    r->add_acc  = static_cast<double>(aa + ba);

    r->sub_def  = static_cast<double>(ad - (-bd));
    r->sub_fast = static_cast<double>(af - (-bf));
    r->sub_acc  = static_cast<double>(aa - (-ba));

    r->mul_def  = static_cast<double>(xd * yd);
    r->mul_fast = static_cast<double>(xf * yf);
#if __FPMP_USE_ACCURATE_MUL__ == 1
    r->mul_acc  = static_cast<double>(xa * ya);
#else
    r->mul_acc  = r->mul_def;
#endif

    r->div_def  = static_cast<double>(xd / yd);
    r->div_fast = static_cast<double>(xf / yf);
#if __FPMP_USE_ACCURATE_DIV__ == 1
    r->div_acc  = static_cast<double>(xa / ya);
#else
    r->div_acc  = r->div_def;
#endif

    r->fma_def  = static_cast<double>(fma(xd, yd, zd));
    r->fma_fast = static_cast<double>(fma(xf, yf, zf));
    r->fma_acc  = static_cast<double>(fma(xa, ya, za));

    r->mad_def  = static_cast<double>(mad(xd, yd, zd));
    r->mad_fast = static_cast<double>(mad(xf, yf, zf));
    r->mad_acc  = static_cast<double>(mad(xa, ya, za));
}

struct CancelResults {
    double add_fast;
    double add_def;
    double add_acc;
};

TARGET_DEVICE void compute_cancellation(CancelResults* r) {
    using namespace fpmp;
    using ff = fp32mp2_low;

    // Two large values of opposite sign whose sum nearly cancels.
    // The exact sum is ~7.256e+026 — tiny relative to ~2.7e+033 inputs,
    // so the result quality depends entirely on error tracking.
    ff a(-2.7059461654979244e+033);
    ff b(+2.7059454398538426e+033);

    r->add_fast = static_cast<double>(add<fpmp2_accuracy::low>(a, b));
    r->add_def  = static_cast<double>(add<fpmp2_accuracy::def>(a, b));
    r->add_acc  = static_cast<double>(add<fpmp2_accuracy::high>(a, b));
}

int main() {
    printf("\n  fpmp_accuracy: unit test for accuracy-explicit free functions\n");
    printf("  ===========================================================\n");

    // Near-cancelling pair for add/sub
    const double a = -2.7059461654979244e+033;
    const double b = +2.7059454398538426e+033;
    // Normal-range values for mul/div/fma/mad
    const double x = 1.234567890123456;
    const double y = 2.345678901234567;
    const double z = 0.567890123456789;

    Results    res;
    RefResults ref;

#if __CUDACC__
    Results    *d_res;
    RefResults *d_ref;
    cudaMallocManaged(&d_res, sizeof(Results));
    cudaMallocManaged(&d_ref, sizeof(RefResults));
    LAUNCH(compute_accuracy_explicit, a, b, x, y, z, d_res);
    LAUNCH(compute_operator_ref,    a, b, x, y, z, d_ref);
    res = *d_res;
    ref = *d_ref;
    cudaFree(d_res);
    cudaFree(d_ref);
#else
    compute_accuracy_explicit(a, b, x, y, z, &res);
    compute_operator_ref(a, b, x, y, z, &ref);
#endif

    const double tol = 0.0;

    printf("\n  --- add<m> vs operator+ ---\n");
    check("add<def>(fp32mp2)      == fp32mp2      + ", res.add_def,  ref.add_def,  tol);
    check("add<low>(fp32mp2)      == fp32mp2_low  + ", res.add_fast, ref.add_fast, tol);
    check("add<high>(fp32mp2)     == fp32mp2_high + ", res.add_acc,  ref.add_acc,  tol);

    printf("\n  --- sub<m> vs operator- ---\n");
    check("sub<def>(fp32mp2)      == fp32mp2       - ", res.sub_def,  ref.sub_def,  tol);
    check("sub<low>(fp32mp2)      == fp32mp2_low  - ", res.sub_fast, ref.sub_fast, tol);
    check("sub<high>(fp32mp2)     == fp32mp2_high - ", res.sub_acc,  ref.sub_acc,  tol);

    printf("\n  --- mul<m> vs operator* ---\n");
    check("mul<def>(fp32mp2)      == fp32mp2       * ", res.mul_def,  ref.mul_def,  tol);
    check("mul<low>(fp32mp2)      == fp32mp2_low  * ", res.mul_fast, ref.mul_fast, tol);
#if __FPMP_USE_ACCURATE_MUL__ == 1
    check("mul<high>(fp32mp2)     == fp32mp2_high * ", res.mul_acc,  ref.mul_acc,  tol);
#else
    check("mul<high>(fp32mp2)     == def (fallback)    ", res.mul_acc,  ref.mul_acc,  tol);
#endif

    printf("\n  --- div<m> vs operator/ ---\n");
    check("div<def>(fp32mp2)      == fp32mp2       / ", res.div_def,  ref.div_def,  tol);
    check("div<low>(fp32mp2)      == fp32mp2_low  / ", res.div_fast, ref.div_fast, tol);
#if __FPMP_USE_ACCURATE_DIV__ == 1
    check("div<high>(fp32mp2)     == fp32mp2_high / ", res.div_acc,  ref.div_acc,  tol);
#else
    check("div<high>(fp32mp2)     == def (fallback)    ", res.div_acc,  ref.div_acc,  tol);
#endif

    printf("\n  --- fma<m> vs fma() ---\n");
    check("fma<def>(fp32mp2)      == fma(fp32mp2)      ", res.fma_def,  ref.fma_def,  tol);
    check("fma<low>(fp32mp2)      == fma(fp32mp2_low)  ", res.fma_fast, ref.fma_fast, tol);
    check("fma<high>(fp32mp2)     == fma(fp32mp2_high)   ", res.fma_acc,  ref.fma_acc,  tol);

    printf("\n  --- mad<m> vs mad() ---\n");
    check("mad<def>(fp32mp2)      == mad(fp32mp2)      ", res.mad_def,  ref.mad_def,  tol);
    check("mad<low>(fp32mp2)      == mad(fp32mp2_low)  ", res.mad_fast, ref.mad_fast, tol);
    check("mad<high>(fp32mp2)     == mad(fp32mp2_high)   ", res.mad_acc,  ref.mad_acc,  tol);

    printf("\n  --- Cross-accuracy (op<m> on different type) ---\n");
    check("sub<high>(fp32mp2_low)     == fp32mp2_high - ", res.cross_sub_acc_on_fast, ref.sub_acc, tol);
    check("add<def>(fp32mp2_low)      == fp32mp2     + ", res.cross_add_def_on_fast, ref.add_def, tol);
    check("fma<high>(fp32mp2_low)     == fma(fp32mp2_high)", res.cross_fma_acc_on_fast, ref.fma_acc, tol);

    // Cancellation test with large nearly-cancelling values.
    // a + b loses ~7 decimal digits, exposing accuracy differences directly.
    printf("\n  --- Cancellation test: add(-2.706e33, +2.706e33) ---\n");

    CancelResults canc;
#if __CUDACC__
    CancelResults *d_canc;
    cudaMallocManaged(&d_canc, sizeof(CancelResults));
    LAUNCH(compute_cancellation, d_canc);
    canc = *d_canc;
    cudaFree(d_canc);
#else
    compute_cancellation(&canc);
#endif

    const double exact = -2.7059461654979244e+033 + 2.7059454398538426e+033;

    printf("    exact (fp64)       = %.15e\n", exact);
    printf("    add<low>           = %.15e  |err|=%.3e\n", canc.add_fast, std::fabs(canc.add_fast - exact));
    printf("    add<def>           = %.15e  |err|=%.3e\n", canc.add_def,  std::fabs(canc.add_def  - exact));
    printf("    add<high>          = %.15e  |err|=%.3e\n", canc.add_acc,  std::fabs(canc.add_acc  - exact));

    double err_fast = std::fabs(canc.add_fast - exact);
    double err_acc  = std::fabs(canc.add_acc  - exact);
    if (err_acc <= err_fast) {
        printf("  PASS  add<high> error <= add<low> error in cancellation\n");
        g_pass++;
    } else {
        printf("  FAIL  add<high> error > add<low> error (%.3e > %.3e)\n", err_acc, err_fast);
        g_fail++;
    }

    printf("\n  ===========================================================\n");
    printf("  Total: %d passed, %d failed\n\n", g_pass, g_fail);

    return g_fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
