/*
    gauss.cpp - Double-Precision Gaussian RNG Performance Benchmark
    ================================================================
    Author:  Andrei Kolesov
    Date:    2025

    Measures throughput and statistical correctness of double-precision
    Gaussian random number generation on GPU using three methods:

      curand_normal_double   - cuRAND built-in Box-Muller transform
      uniform + normcdfinv   - cuRAND uniform double + CUDA normcdfinv intrinsic
      fpmp icdf(uint64)      - two curand() calls combined into uint64 + fpmp ICDF
                               (fp32mp2 precision, cast to double)

    Each kernel generates NUM_SAMPLES * REPS Gaussian variates per launch,
    accumulating four moments (sum of z, z^2, z^3, z^4) per thread to compute
    mean, stddev, skewness, and excess kurtosis without storing individual
    samples.

    Compile-time configuration (via Makefile or -D flags):
      NUM_SAMPLES        - samples per launch (default 16M)
      REPS               - inner repetitions (default 128)
      THREADS_PER_BLOCK  - CUDA block size (default 256)
      NUM_BLOCKS         - CUDA grid size  (default 2048)
      NUM_ITERATIONS     - timing iterations for averaging (default 10)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if !defined(__CUDACC__)
int main() {
    printf("This benchmark requires CUDA. Build with nvcc.\n");
    return 0;
}
#else

#include <cuda/fpmp>
#include <cuda/fpmp_math>

using namespace cuda::experimental;
using namespace cuda::experimental::fpmp;

#include <curand_kernel.h>

/* ------------------------------------------------------------------ */
/* Compile-time defaults                                               */
/* ------------------------------------------------------------------ */
#ifndef NUM_SAMPLES
  #if defined(__CUDACC__)
    #define NUM_SAMPLES (1024*1024*16)
  #else
    #define NUM_SAMPLES (1024*16)
  #endif
#endif
#ifndef REPS
  #define REPS 128
#endif
#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK 256
#endif
#ifndef NUM_BLOCKS
  #define NUM_BLOCKS 2048
#endif
#ifndef NUM_ITERATIONS
  #define NUM_ITERATIONS 10
#endif

static constexpr int TOTAL_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(1);                                                        \
    }                                                                   \
} while (0)

/* ------------------------------------------------------------------ */
/* Method tags                                                         */
/* ------------------------------------------------------------------ */
enum class GaussMethod { BoxMuller, Normcdfinv, FpmpIcdf };

/* ------------------------------------------------------------------ */
/* CUDA kernels                                                        */
/*                                                                     */
/* Each thread generates num_items * REPS Gaussian doubles and         */
/* accumulates moments 1-4 for statistical validation:                 */
/*   sum(z), sum(z^2), sum(z^3), sum(z^4)                             */
/* ------------------------------------------------------------------ */
template <GaussMethod method>
__global__ void gauss_kernel(double* __restrict__ p_m1,
                             double* __restrict__ p_m2,
                             double* __restrict__ p_m3,
                             double* __restrict__ p_m4,
                             const int num_items,
                             const unsigned long long seed)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    double m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0;

    for (int rep = 0; rep < REPS; ++rep) {
        for (int i = tid; i < num_items; i += stride) {
            double z;

            if constexpr (method == GaussMethod::BoxMuller) {
                z = curand_normal_double(&state);
            }
            else if constexpr (method == GaussMethod::Normcdfinv) {
                double u = curand_uniform_double(&state);
                z = normcdfinv(fmin(u, 0x1.fffffffffffffp-1));
            }
            else {
                uint64_t u = ((uint64_t)curand(&state) << 32) | curand(&state);
                z = (double)icdf(u);
            }

            double z2 = z * z;
            m1 += z;
            m2 += z2;
            m3 += z2 * z;
            m4 += z2 * z2;
        }
    }

    p_m1[tid] = m1;
    p_m2[tid] = m2;
    p_m3[tid] = m3;
    p_m4[tid] = m4;
}

/* ------------------------------------------------------------------ */
/* Benchmark result                                                    */
/* ------------------------------------------------------------------ */
struct BenchResult {
    double time_ms;
    double samples_per_sec;
    double mean;
    double stddev;
    double skewness;
    double ex_kurtosis;
};

/* ------------------------------------------------------------------ */
/* Run benchmark for a single method                                   */
/*                                                                     */
/* Allocates partial-moment buffers, warms up, runs NUM_ITERATIONS     */
/* timed launches, reduces moments on host, computes statistics.       */
/* ------------------------------------------------------------------ */
template <GaussMethod method>
static BenchResult run_bench(int num_samples)
{
    const size_t buf_sz = TOTAL_THREADS * sizeof(double);
    double *d_m1 = nullptr, *d_m2 = nullptr, *d_m3 = nullptr, *d_m4 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_m1, buf_sz));
    CUDA_CHECK(cudaMalloc(&d_m2, buf_sz));
    CUDA_CHECK(cudaMalloc(&d_m3, buf_sz));
    CUDA_CHECK(cudaMalloc(&d_m4, buf_sz));

    const unsigned long long seed = 42ULL;

    auto launch = [&]() {
        gauss_kernel<method><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_m1, d_m2, d_m3, d_m4, num_samples, seed);
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0));
        launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_ms += ms;
        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
    }

    double *h_m1 = new double[TOTAL_THREADS], *h_m2 = new double[TOTAL_THREADS];
    double *h_m3 = new double[TOTAL_THREADS], *h_m4 = new double[TOTAL_THREADS];
    CUDA_CHECK(cudaMemcpy(h_m1, d_m1, buf_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m2, d_m2, buf_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m3, d_m3, buf_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m4, d_m4, buf_sz, cudaMemcpyDeviceToHost));

    long double s1 = 0.0L, s2 = 0.0L, s3 = 0.0L, s4 = 0.0L;
    for (int i = 0; i < TOTAL_THREADS; ++i) {
        s1 += (long double)h_m1[i];
        s2 += (long double)h_m2[i];
        s3 += (long double)h_m3[i];
        s4 += (long double)h_m4[i];
    }

    const long double N = (long double)num_samples * REPS;
    long double mu  = s1 / N;
    long double var = s2 / N - mu * mu;
    long double sd  = sqrtl(var);
    long double sd3 = sd * sd * sd;

    BenchResult res;
    res.time_ms        = total_ms / NUM_ITERATIONS;
    res.samples_per_sec = (double)(N / (res.time_ms * 1e-3));
    res.mean           = (double)mu;
    res.stddev         = (double)sd;
    res.skewness       = (double)((s3 / N - 3.0L * mu * var - mu * mu * mu) / sd3);
    res.ex_kurtosis    = (double)((s4 / N - 4.0L * mu * s3 / N
                                   + 6.0L * mu * mu * s2 / N
                                   - 3.0L * mu * mu * mu * mu)
                                  / (var * var) - 3.0L);

    delete[] h_m1; delete[] h_m2; delete[] h_m3; delete[] h_m4;
    CUDA_CHECK(cudaFree(d_m1)); CUDA_CHECK(cudaFree(d_m2));
    CUDA_CHECK(cudaFree(d_m3)); CUDA_CHECK(cudaFree(d_m4));
    return res;
}

/* ------------------------------------------------------------------ */
/* Deterministic ICDF tail-edge test                                   */
/*                                                                     */
/* Directly probes known boundary inputs that random sampling cannot   */
/* reach.  Verifies that extreme discretization points (x=0, x=1,     */
/* midpoint, mirrored extremes) all produce finite results within a    */
/* reasonable range (|z| < 10, i.e. no infinities or NaNs).           */
/*                                                                     */
/* Also checks that deep-tail points (beyond 5.5 sigma) produce       */
/* non-zero res_lo, confirming full float-float precision from the     */
/* tail 2 polynomial (not single-float fallback).                      */
/* ------------------------------------------------------------------ */
static int test_icdf_tails()
{
    struct TestCase {
        uint64_t input;
        const char* label;
        bool expect_lo;  /* true if res_lo should be non-zero */
    };

    /*
     * After x >>= 16, only the top 48 bits matter.  The low 16 bits are
     * shifted out, so we place test values in the top 48 bits.
     *
     *   x48 = 0                 → deepest negative tail  (p ≈ 0.5/2^48)
     *   x48 = 2^47              → midpoint               (p ≈ 0.5)
     *   x48 = 2^48 - 1          → mirrors to 0, positive (p ≈ 0.5/2^48)
     */
    const TestCase cases[] = {
        { 0x0000000000000000ULL, "x48=0          neg deep tail", true  },
        { 0x0000000000010000ULL, "x48=1          neg near-deep", true  },
        { 0x0000000000020000ULL, "x48=2          neg tail",      true  },
        { 0x00000000FFFF0000ULL, "x48=0xFFFF     neg moderate",  true  },
        { 0x0000FFFFFFFF0000ULL, "x48=2^32-1     neg bulk",      true  },
        { 0x7FFF000000000000ULL, "x48=2^47-1<<24 neg near-zero", false },
        { 0x7FFFFFFFFFFF0000ULL, "x48=2^47-1     neg ~0",        false },
        { 0x8000000000000000ULL, "x48=2^47       midpoint (~0)", false },
        { 0x8000000000010000ULL, "x48=2^47+1     pos ~0",        false },
        { 0xFFFFFFFFFFFE0000ULL, "x48 mirrors→1  pos near-deep", true  },
        { 0xFFFFFFFFFFFF0000ULL, "x48 mirrors→0  pos deep tail", true  },
    };
    const int N = sizeof(cases) / sizeof(cases[0]);

    printf("ICDF tail-edge test (deterministic, %d cases)\n", N);
    printf("%-22s  %-28s  %14s  %14s  %16s  %s\n",
           "Input (hex)", "Description", "res_hi", "res_lo",
           "as double", "Check");
    printf("%-22s  %-28s  %14s  %14s  %16s  %s\n",
           "----------------------", "----------------------------",
           "--------------", "--------------",
           "----------------", "-----");

    int failures = 0;
    int lo_checks = 0;
    for (int i = 0; i < N; ++i) {
        float hi, lo;
        __nv_fpmp2_icdf(cases[i].input, &hi, &lo);
        double z = (double)hi + (double)lo;

        bool finite_ok = isfinite(hi) && isfinite(lo);
        bool range_ok  = fabs(z) < 10.0;
        bool ok = finite_ok && range_ok;
        if (!ok) ++failures;

        if (cases[i].expect_lo && lo == 0.0f && finite_ok && range_ok) {
            ++lo_checks;
        }

        printf("0x%016llx  %-28s  % 14.7e  % 14.7e  % 16.10g  %s\n",
               (unsigned long long)cases[i].input, cases[i].label,
               hi, lo, z,
               ok ? "OK" : "FAIL");
    }

    if (lo_checks > 0) {
        printf("\n  WARNING: %d deep-tail case(s) produced res_lo=0 (single-float precision)\n", lo_checks);
    }

    printf("\nTail-edge result: %s\n\n", failures ? "FAILURES DETECTED" : "ALL PASSED");
    return failures;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main()
{
    constexpr int num_samples = NUM_SAMPLES;

    printf("Double-Precision Gaussian RNG Performance Benchmark\n");
    printf("===================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Target: GPU (%s)\n\n", prop.name);

    printf("Samples: %d  REPS: %d\n", num_samples, REPS);
    printf("Threads: %d  Timing iterations: %d\n\n",
           TOTAL_THREADS, NUM_ITERATIONS);

    /* -- Deterministic tail-edge test (host-side, no random sampling) -- */
    int tail_failures = test_icdf_tails();

    const long long effective_n = (long long)num_samples * REPS;
    printf("Effective samples per launch: %lld\n\n", effective_n);

    printf("Methods:\n");
    printf("  curand_normal_double  - cuRAND built-in Box-Muller\n");
    printf("  uniform + normcdfinv  - cuRAND uniform + CUDA normcdfinv\n");
    printf("  fpmp icdf(uint64)     - two curand() -> uint64 -> fpmp ICDF -> double\n");
    printf("                          (fp32mp2 precision, ~46-bit mantissa)\n\n");

    printf("Running benchmarks...\n\n");

    auto res_bm   = run_bench<GaussMethod::BoxMuller>(num_samples);
    auto res_ncdf = run_bench<GaussMethod::Normcdfinv>(num_samples);
    auto res_icdf = run_bench<GaussMethod::FpmpIcdf>(num_samples);

    auto fmt_thousands = [](char* buf, size_t sz, double val) {
        long long v = (long long)val;
        if (v < 1000) { snprintf(buf, sz, "%lld", v); return; }
        char tmp[32]; int n = 0;
        bool neg = v < 0; if (neg) v = -v;
        while (v > 0) { tmp[n++] = '0' + (int)(v % 10); v /= 10; }
        int pos = 0;
        if (neg) buf[pos++] = '-';
        for (int i = n - 1; i >= 0; --i) {
            buf[pos++] = tmp[i];
            if (i > 0 && i % 3 == 0) buf[pos++] = '\'';
        }
        buf[pos] = '\0';
    };

    const double tol_mean   = 5.0 / sqrt((double)effective_n);
    const double tol_stddev = 5.0 / sqrt(2.0 * (double)effective_n);
    const double tol_skew   = 5.0 * sqrt(6.0 / (double)effective_n);
    const double tol_kurt   = 5.0 * sqrt(24.0 / (double)effective_n);

    int failures = 0;

    /* -- Accuracy table ------------------------------------------------ */
    printf("Statistical accuracy (expected: mean=0, stddev=1, skewness=0, excess kurtosis=0)\n\n");

    printf("%-24s  %13s  %12s  %12s  %12s  %s\n",
           "Method", "Mean", "StdDev", "Skewness", "Ex.Kurt", "Check");
    printf("%-24s  %13s  %12s  %12s  %12s  %s\n",
           "------------------------", "-------------", "------------",
           "------------", "------------", "-----");

    auto acc_row = [&](const char* name, const BenchResult& r) {
        bool mean_ok   = fabs(r.mean) < tol_mean;
        bool stddev_ok = fabs(r.stddev - 1.0) < tol_stddev;
        bool skew_ok   = fabs(r.skewness) < tol_skew;
        bool kurt_ok   = fabs(r.ex_kurtosis) < tol_kurt;
        bool ok = mean_ok && stddev_ok && skew_ok && kurt_ok;
        if (!ok) ++failures;
        printf("%-24s  % 13.6e  %12.10f  % 12.6e  % 12.6e  %s\n",
               name, r.mean, r.stddev, r.skewness, r.ex_kurtosis,
               ok ? "OK" : "FAIL");
    };

    acc_row("curand_normal_double",  res_bm);
    acc_row("uniform + normcdfinv",  res_ncdf);
    acc_row("fpmp icdf(uint64)",     res_icdf);

    constexpr double jb_crit = 5.991;

    printf("\nJarque-Bera normality test (chi-sq(2), critical=%.3f at p=0.05):\n", jb_crit);
    printf("  %-24s  %12s  %s\n", "------------------------", "--------", "-----");

    auto jb_row = [&](const char* name, const BenchResult& r) {
        double jb = ((double)effective_n / 6.0)
                     * (r.skewness * r.skewness
                        + r.ex_kurtosis * r.ex_kurtosis / 4.0);
        bool ok = jb < jb_crit;
        if (!ok) ++failures;
        printf("  %-24s  %8.4f  %s\n", name, jb, ok ? "OK" : "FAIL");
    };

    jb_row("curand_normal_double",  res_bm);
    jb_row("uniform + normcdfinv",  res_ncdf);
    jb_row("fpmp icdf(uint64)",     res_icdf);

    /* -- Performance table --------------------------------------------- */
    printf("\nPerformance:\n\n");

    printf("%-24s  %10s  %18s\n", "Method", "Time(ms)", "Samples/sec");
    printf("%-24s  %10s  %18s\n",
           "------------------------", "----------", "------------------");

    auto perf_row = [&](const char* name, const BenchResult& r) {
        char sps[32];
        fmt_thousands(sps, sizeof(sps), r.samples_per_sec);
        printf("%-24s  %10.3f  %18s\n", name, r.time_ms, sps);
    };

    perf_row("curand_normal_double",  res_bm);
    perf_row("uniform + normcdfinv",  res_ncdf);
    perf_row("fpmp icdf(uint64)",     res_icdf);

    printf("\nSpeedups vs curand_normal_double:\n");
    printf("  %-24s  %6s\n", "------------------------", "------");
    printf("  %-24s  %5.2fx\n",
           "uniform + normcdfinv", res_bm.time_ms / res_ncdf.time_ms);
    printf("  %-24s  %5.2fx\n",
           "fpmp icdf(uint64)",    res_bm.time_ms / res_icdf.time_ms);

    failures += tail_failures;

    printf("\nResult: %s (5-sigma tolerance: mean<%.2e, |stddev-1|<%.2e, |skew|<%.2e, |exkurt|<%.2e)\n",
           failures == 0 ? "ALL PASSED" : "FAILURES DETECTED",
           tol_mean, tol_stddev, tol_skew, tol_kurt);

    return 0;
}

#endif /* __CUDACC__ */
