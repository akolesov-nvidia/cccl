/*
    fpmp_accuracy_conv.cpp - Unit test for cross-accuracy conversion semantics
    =====================================================================
    Author:  generated
    Date:    2026

    Different specializations of fpmp2_t<FpType, met> share the same
    (hi, lo) representation; only the accuracy tag (which selects the
    arithmetic algorithm used downstream) differs. Without a dedicated
    converting constructor, the compiler would fall back on
    operator double() + fpmp2_t(double) — an expensive round trip
    that is especially painful on GPUs with limited FP64 throughput.

    fpmp.h provides an `explicit` cross-accuracy converting constructor
    that simply bit-copies (hi, lo).  This unit verifies the contract:

      1. Cross-accuracy conversion is `explicit`:
           - copy-init  (`fp32mp2_low d = b;`)  is ill-formed,
           - copy-assign (`a = b;`)                 is ill-formed.
         Both checked at compile time via `std::is_convertible`.

      2. Cross-accuracy conversion via direct-init / functional cast /
         static_cast is allowed and produces a bit-exact (hi, lo) copy
         (no rounding through double / FpType).

      3. Same-type implicit conversion (copy) is unaffected.

      4. Cross-FpType behavior is unchanged: implicit conversion is still
         forbidden (the only existing path was the two-UDC double chain,
         which is already ill-formed in C++ overload resolution).

    All checks are compile-time (`static_assert`) plus a runtime bit-equality
    check across a few values, all three accuracy levels, and both FpType variants.
*/

#include <cstdio>
#include <cstdlib>
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
 * Compile-time contract for fp32mp2 (FpType == float)
 * ===================================================================== */
/* explicit construction across accuracy levels is allowed... */
static_assert(std::is_constructible<fp32mp2_low,     fp32mp2_high>::value, "");
static_assert(std::is_constructible<fp32mp2_low,     fp32mp2>         ::value, "");
static_assert(std::is_constructible<fp32mp2,          fp32mp2_low>    ::value, "");
static_assert(std::is_constructible<fp32mp2,          fp32mp2_high>::value, "");
static_assert(std::is_constructible<fp32mp2_high, fp32mp2>         ::value, "");
static_assert(std::is_constructible<fp32mp2_high, fp32mp2_low>    ::value, "");

/* ...but implicit conversion across accuracy levels is NOT. */
static_assert(!std::is_convertible<fp32mp2_high, fp32mp2_low>    ::value, "");
static_assert(!std::is_convertible<fp32mp2,          fp32mp2_low>    ::value, "");
static_assert(!std::is_convertible<fp32mp2_low,     fp32mp2>         ::value, "");
static_assert(!std::is_convertible<fp32mp2_high, fp32mp2>         ::value, "");
static_assert(!std::is_convertible<fp32mp2,          fp32mp2_high>::value, "");
static_assert(!std::is_convertible<fp32mp2_low,     fp32mp2_high>::value, "");

/* Same-type implicit conversion (copy) is unaffected. */
static_assert(std::is_convertible<fp32mp2_low,     fp32mp2_low>    ::value, "");
static_assert(std::is_convertible<fp32mp2,          fp32mp2>         ::value, "");
static_assert(std::is_convertible<fp32mp2_high, fp32mp2_high>::value, "");

/* Cross-FpType conversion contract:
 *   - Upconvert (fp32mp2 -> fp64mp2) is implicit (lossless widening, like
 *     IEEE float -> double).
 *   - Downconvert (fp64mp2 -> fp32mp2) honors _CCCL_FPMP_EXPLICIT: implicit
 *     under CCCL_FPMP_EXPLICIT_CASTS=0 (default), explicit under =1.
 *   The dedicated cross-precision invariants are exercised in detail by
 *   units/fpmp_cross_prec.cpp. */
static_assert(std::is_convertible<fp32mp2, fp64mp2>::value,
              "fp32mp2 -> fp64mp2 must be implicit (lossless upconvert)");
  #if CCCL_FPMP_EXPLICIT_CASTS == 1
    static_assert(!std::is_convertible<fp64mp2, fp32mp2>::value,
                  "fp64mp2 -> fp32mp2 must be explicit under EXPLICIT_CASTS=1");
  #else
    static_assert(std::is_convertible<fp64mp2, fp32mp2>::value,
                  "fp64mp2 -> fp32mp2 implicit by default (matches double -> fp32mp2)");
  #endif

/* Assignment side of the contract.
 *
 * `a = b` resolves to `a.operator=(b)` and must bind `b` to the parameter of
 * the defaulted copy-assignment via an implicit conversion sequence — which
 * skips `explicit` constructors. So the new explicit cross-accuracy ctor is
 * deliberately not visible here, and cross-accuracy assignment must fail.
 * Same-type assignment, of course, still works. */
static_assert(!std::is_assignable<fp32mp2_low&,     fp32mp2_high>::value, "");
static_assert(!std::is_assignable<fp32mp2_low&,     fp32mp2>         ::value, "");
static_assert(!std::is_assignable<fp32mp2&,          fp32mp2_low>    ::value, "");
static_assert(!std::is_assignable<fp32mp2&,          fp32mp2_high>::value, "");
static_assert(!std::is_assignable<fp32mp2_high&, fp32mp2>         ::value, "");
static_assert(!std::is_assignable<fp32mp2_high&, fp32mp2_low>    ::value, "");
static_assert( std::is_assignable<fp32mp2_low&,     fp32mp2_low>    ::value, "");
static_assert( std::is_assignable<fp32mp2&,          fp32mp2>         ::value, "");
static_assert( std::is_assignable<fp32mp2_high&, fp32mp2_high>::value, "");

/* The new ctor preserves result type (no type inference surprises) */
static_assert(std::is_same<decltype(fp32mp2_low(std::declval<fp32mp2_high>())),
                           fp32mp2_low>::value, "");

/* =====================================================================
 * Compile-time contract for fp64mp2 (FpType == double)
 * ===================================================================== */
static_assert( std::is_constructible<fp64mp2_low,     fp64mp2_high>::value, "");
static_assert( std::is_constructible<fp64mp2_high, fp64mp2>         ::value, "");
static_assert(!std::is_convertible <fp64mp2_high, fp64mp2_low>::value, "");
static_assert(!std::is_convertible <fp64mp2,          fp64mp2_low>::value, "");
static_assert(!std::is_assignable  <fp64mp2_low&,    fp64mp2_high>::value, "");
static_assert(!std::is_assignable  <fp64mp2_low&,    fp64mp2>         ::value, "");

/* =====================================================================
 * Runtime bit-equality check
 * ===================================================================== */
static int g_pass = 0;
static int g_fail = 0;

template<typename Src, typename Dst>
static void check_bit_exact(const char* label, Src src) {
    Dst dst(src);  /* explicit cross-accuracy ctor */
    bool ok = (dst.hi() == src.hi()) && (dst.lo() == src.lo());
    if (ok) {
        printf("  PASS  %-55s  hi=%a lo=%a\n", label, (double)dst.hi(), (double)dst.lo());
        g_pass++;
    } else {
        printf("  FAIL  %-55s  src hi=%a lo=%a / dst hi=%a lo=%a\n",
               label,
               (double)src.hi(), (double)src.lo(),
               (double)dst.hi(), (double)dst.lo());
        g_fail++;
    }
}

struct F32 { float hi, lo; };
struct F64 { double hi, lo; };

TARGET_DEVICE void compute_pairs(F32* p32, F64* p64) {
    /* Pick representative (hi, lo) pairs: regular, denormal-tiny lo,
     * large magnitude, tiny magnitude. */
    p32[0] = { 1.2345678f,            1.0e-9f };
    p32[1] = { 0x1.fffffep+126f,     -0x1.0p+102f };  /* near float max */
    p32[2] = { 1.0e-30f,              1.0e-38f };     /* tiny */
    p32[3] = { -3.1415927f,           1.5e-8f };      /* negative */
    p64[0] = { 1.234567890123456,     1.0e-18 };
    p64[1] = { 1.0e+300,              1.0e+283 };
    p64[2] = { -2.7182818284590452,   1.5e-17 };
    p64[3] = { 1.0e-200,              1.0e-217 };
}

int main() {
    printf("\n  fpmp_accuracy_conv: cross-accuracy explicit converting constructor\n");
    printf("  ==================================================================\n");

    F32 p32[4]; F64 p64[4];
#if __CUDACC__
    F32 *d_p32; F64 *d_p64;
    cudaMallocManaged(&d_p32, sizeof(p32));
    cudaMallocManaged(&d_p64, sizeof(p64));
    LAUNCH(compute_pairs, d_p32, d_p64);
    for (int i = 0; i < 4; ++i) { p32[i] = d_p32[i]; p64[i] = d_p64[i]; }
    cudaFree(d_p32); cudaFree(d_p64);
#else
    compute_pairs(p32, p64);
#endif

    printf("\n  --- fp32mp2: every cross-accuracy pair, every test value ---\n");
    for (int i = 0; i < 4; ++i) {
        char lbl[80];
        snprintf(lbl, sizeof(lbl), "[%d] def  -> low",  i);
        check_bit_exact<fp32mp2,          fp32mp2_low>    (lbl, fp32mp2         (p32[i].hi, p32[i].lo));
        snprintf(lbl, sizeof(lbl), "[%d] def  -> high", i);
        check_bit_exact<fp32mp2,          fp32mp2_high>(lbl, fp32mp2         (p32[i].hi, p32[i].lo));
        snprintf(lbl, sizeof(lbl), "[%d] low  -> def",  i);
        check_bit_exact<fp32mp2_low,     fp32mp2>         (lbl, fp32mp2_low    (p32[i].hi, p32[i].lo));
        snprintf(lbl, sizeof(lbl), "[%d] low  -> high", i);
        check_bit_exact<fp32mp2_low,     fp32mp2_high>(lbl, fp32mp2_low    (p32[i].hi, p32[i].lo));
        snprintf(lbl, sizeof(lbl), "[%d] high -> def",  i);
        check_bit_exact<fp32mp2_high, fp32mp2>         (lbl, fp32mp2_high(p32[i].hi, p32[i].lo));
        snprintf(lbl, sizeof(lbl), "[%d] high -> low",  i);
        check_bit_exact<fp32mp2_high, fp32mp2_low>    (lbl, fp32mp2_high(p32[i].hi, p32[i].lo));
    }

    printf("\n  --- fp64mp2: spot-check cross-accuracy pairs ---\n");
    for (int i = 0; i < 4; ++i) {
        char lbl[80];
        snprintf(lbl, sizeof(lbl), "[%d] fp64 high -> low", i);
        check_bit_exact<fp64mp2_high, fp64mp2_low>(lbl,
            fp64mp2_high(p64[i].hi, p64[i].lo));
        snprintf(lbl, sizeof(lbl), "[%d] fp64 def  -> low", i);
        check_bit_exact<fp64mp2,          fp64mp2_low>(lbl,
            fp64mp2         (p64[i].hi, p64[i].lo));
    }

    /* Sanity: exercise the four explicit-conversion shapes a user may write
     * and confirm each routes through the bit-exact cross-accuracy ctor.
     *
     * Implicit forms (`d = src;`, `a_fast = src_acc;`) are not exercised at
     * runtime — they are ill-formed by design and are pinned down by the
     * static_assert(!is_convertible<...>) / static_assert(!is_assignable<...>)
     * checks at the top of this file. */
    {
        fp32mp2_high src(0x1.23p+4f, 0x1.0p-20f);
        fp32mp2_low a(src);                              // direct-init
        fp32mp2_low b = fp32mp2_low(src);             // functional cast in copy-init
        fp32mp2_low c = static_cast<fp32mp2_low>(src);// static_cast in copy-init
        fp32mp2_low d;
        d = fp32mp2_low(src);                            // explicit assign (functional)
        fp32mp2_low e;
        e = static_cast<fp32mp2_low>(src);               // explicit assign (static_cast)
        bool ok = (a.hi() == src.hi() && a.lo() == src.lo() &&
                   b.hi() == src.hi() && b.lo() == src.lo() &&
                   c.hi() == src.hi() && c.lo() == src.lo() &&
                   d.hi() == src.hi() && d.lo() == src.lo() &&
                   e.hi() == src.hi() && e.lo() == src.lo());
        if (ok) {
            printf("  PASS  direct-init / functional / static_cast / explicit-assign all bit-exact\n");
            g_pass++;
        } else {
            printf("  FAIL  one of the explicit conversion forms diverged from src\n");
            g_fail++;
        }
    }

    printf("\n  ==================================================================\n");
    printf("  Total: %d passed, %d failed\n\n", g_pass, g_fail);
    return g_fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
