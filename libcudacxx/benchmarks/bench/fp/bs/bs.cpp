/*
    bs.cpp - Black-Scholes Option Pricing Benchmark
    ================================================
    Author:  Andrei Kolesov
    Date:    2025

    Prices N random European options via the analytical Black-Scholes
    formula and compares throughput and accuracy across eight floating-point
    data types:

      float                - IEEE-754 binary32 (single precision, 23-bit mantissa)
      fp32mp2_low          - Double-float, low accuracy (~46-bit mantissa)
      fp32mp2              - Double-float, default (mid) accuracy (~46-bit mantissa)
      double               - IEEE-754 binary64 (double precision, 53-bit mantissa)
      fp64emu_mid          - Emulated fp64 via fp32 arithmetic (mid accuracy)
      fp64emu              - Emulated fp64 via fp32 arithmetic (high accuracy, library default)
      fp64mp2              - Double-double (pair of doubles, ~104-bit mantissa)
      fp128_t              - IEEE-754 binary128 quad precision (~113-bit mantissa)
                             (optional, requires FP128=1 and ARCH=100+ for GPU)

    The benchmark is fully templated: every type goes through the same
    code path (norm_cdf, d1/d2, discounting) so that timing differences
    reflect arithmetic cost alone.  Results are validated against a
    quad-precision reference using libquadmath's erfcq on x86
    (default, FP128_REF=1), or a double-precision reference (FP128_REF=0).

    When the float128 reference is active, accuracy comparisons are
    accumulated in quad precision.  If FP128=1 is also enabled, partial-sum
    reduction uses the same quad type via inline helpers.

    Supports both CUDA (device) and CPU (host) targets.

    Compile-time configuration (via Makefile or -D flags):
      NUM_OPTIONS        - options per launch (default 1M GPU / 32K CPU)
      REPS               - inner repetitions per launch (default 128 GPU / 2 CPU)
      THREADS_PER_BLOCK  - CUDA block size (default 256)
      NUM_BLOCKS         - CUDA grid size  (default 2048)
      NUM_ITERATIONS     - timing iterations for averaging (default 10)
      FP128              - include fp128_t as benchmarked type (0 or 1)
      FP128_REF          - use quad-precision reference (default 1)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <unistd.h>

#include <cuda/fpmp>
#include <cuda/fpmp_math>
#include <cuda/fpemu>

using namespace cuda::experimental;

#include "bs_fp128_math.hpp"

#if __FPMP_FP128_ENABLED__ == 1
using fp128_t = __bs_fp128;
#endif

/* ------------------------------------------------------------------ */
/* Host/device compatibility                                           */
/* ------------------------------------------------------------------ */
#if defined(__CUDACC__)
  #define HOST_DEVICE __host__ __device__
  #define KERNEL      __global__
  #define USE_CUDA    1
#else
  #define HOST_DEVICE
  #define KERNEL
  #define USE_CUDA    0
#endif

#ifndef NUM_OPTIONS
  #if defined(__CUDACC__)
      #define NUM_OPTIONS (1024*1024)
  #else
      #define NUM_OPTIONS (1024*32)
  #endif
#endif
#ifndef REPS
  #if defined(__CUDACC__)
      #define REPS 128
  #else
      #define REPS 2
  #endif
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

#if USE_CUDA
#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        char _msg[256];                                    \
        int _n = snprintf(_msg, sizeof(_msg),              \
            "CUDA error at %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(err));  \
        (void)!write(STDOUT_FILENO, _msg, _n);             \
        exit(1);                                           \
    }                                                      \
} while (0)
#endif


/* Float128 reference (bs_ref.cpp, compiled by g++ with libquadmath). */
#if defined(__USE_FP128_REFERENCE__)
extern double bs_ref_price_f128(double S, double K, double r,
                                double q, double sigma, double Tmat,
                                bool is_call);
extern void bs_accuracy_f128(const double* prices, const double* ref_prices,
                             int N,
                             double* out_max_re, double* out_avg_re);
#endif

/* Convert any T to __bs_fp128 with maximum precision.                */
/* Default path goes through double; fpmp2 overload uses hi()+lo().   */
#if defined(__USE_FP128_REFERENCE__) && (__FPMP_FP128_ENABLED__ == 1)
template<typename T>
static __bs_fp128 to_f128_val(const T& x) {
    return (__bs_fp128)(double)x;
}

template<typename FpType, fpmp2_accuracy met>
static __bs_fp128 to_f128_val(const fpmp2_t<FpType, met>& x) {
    return (__bs_fp128)x.hi() + (__bs_fp128)x.lo();
}

static __bs_fp128 to_f128_val(const __bs_fp128& x) { return x; }
#endif /* __USE_FP128_REFERENCE__ && __FPMP_FP128_ENABLED__ */

/* ------------------------------------------------------------------ */
/* Black-Scholes analytical formula (templated)                       */
/*                                                                    */
/* N(x) = 0.5 * erfc(-x / sqrt(2))                                     */
/* d1   = (ln(S/K) + (r-q+0.5*sigma^2)*T) / (sigma*sqrt(T))           */
/* d2   = d1 - sigma*sqrt(T)                                          */
/* Call  = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)                          */
/* Put   = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)                        */
/* ------------------------------------------------------------------ */
template <typename T>
HOST_DEVICE T norm_cdf(T x)
{
    return T(0.5) * erfc(-x * T(M_SQRT1_2));
}

template <typename T, bool IsCall>
HOST_DEVICE T bs_price(T S, T K, T r, T q, T sigma, T Tmat)
{
    T sqrtT   = sqrt(Tmat);
    T vsqrtT  = sigma * sqrtT;
    T d1      = (log(S / K) + (r - q + T(0.5) * sigma * sigma) * Tmat) / vsqrtT;
    T d2      = d1 - vsqrtT;
    T disc_r  = exp(-r * Tmat);
    T disc_q  = exp(-q * Tmat);
    T S_disc  = S * disc_q;
    T K_disc  = K * disc_r;

    if (IsCall)
        return S_disc * norm_cdf(d1) - K_disc * norm_cdf(d2);
    else
        return K_disc * norm_cdf(-d2) - S_disc * norm_cdf(-d1);
} // bs_price

/* Reference price — float128 (with custom erfcq) or double fallback */
static double bs_ref_price(double S, double K, double r,
                           double q, double sigma, double Tmat,
                           bool is_call)
{
#if defined(__USE_FP128_REFERENCE__)
    return bs_ref_price_f128(S, K, r, q, sigma, Tmat, is_call);
#else
    double sqrtT  = sqrt(Tmat);
    double vsqrtT = sigma * sqrtT;
    double d1     = (log(S / K) + (r - q + 0.5 * sigma * sigma) * Tmat) / vsqrtT;
    double d2     = d1 - vsqrtT;
    double Nd1    = 0.5 * erfc(-d1 * M_SQRT1_2);
    double Nd2    = 0.5 * erfc(-d2 * M_SQRT1_2);
    double disc_r = exp(-r * Tmat);
    double disc_q = exp(-q * Tmat);

    if (is_call)
        return S * disc_q * Nd1 - K * disc_r * Nd2;
    else
        return K * disc_r * (0.5 * erfc(d2 * M_SQRT1_2)) -
               S * disc_q * (0.5 * erfc(d1 * M_SQRT1_2));
#endif
} // bs_ref_price

/* ------------------------------------------------------------------ */
/* CUDA kernel                                                         */
/* ------------------------------------------------------------------ */
#if USE_CUDA

template <typename T, bool IsCall>
__global__ void bs_kernel(const double* __restrict__ d_S,
                          const double* __restrict__ d_K,
                          const double* __restrict__ d_sigma,
                          const double* __restrict__ d_Tmat,
                          T* __restrict__ partial_sums,
                          double* __restrict__ prices,
                          const double r, const double q,
                          const int N, const int reps)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const T   r_t    = T(r);
    const T   q_t    = T(q);
    T sum(0.0);

    for (int rep = 0; rep < reps; ++rep)
    {
        for (int i = tid; i < N; i += stride)
        {
            T px = bs_price<T, IsCall>(T(__ldg(&d_S[i])),
                                       T(__ldg(&d_K[i])),
                                       r_t, q_t,
                                       T(__ldg(&d_sigma[i])),
                                       T(__ldg(&d_Tmat[i])));
            sum = sum + px;
            if (prices && rep == 0)
                prices[i] = (double)px;
        } // for i
    } // for rep
    partial_sums[tid] = sum;
} // bs_kernel

#endif /* USE_CUDA */

/* ------------------------------------------------------------------ */
/* Host simulation                                                     */
/* ------------------------------------------------------------------ */
#if !USE_CUDA

template <typename T, bool IsCall>
static void bs_host_run(const double* h_S, const double* h_K,
                        const double* h_sigma, const double* h_Tmat,
                        T* partial_sums, double* prices,
                        double r, double q, int N, int reps)
{
    const T r_t = T(r);
    const T q_t = T(q);

    /* Threads with tid >= N do no work; only iterate the active set    */
    /* and zero the tail of partial_sums so the host reduction is       */
    /* unaffected.                                                       */
    const int active = (TOTAL_THREADS < N) ? TOTAL_THREADS : N;

    for (int tid = 0; tid < active; ++tid)
    {
        T sum(0.0);
        for (int rep = 0; rep < reps; ++rep)
        {
            for (int i = tid; i < N; i += TOTAL_THREADS)
            {
                T px = bs_price<T, IsCall>(T(h_S[i]), T(h_K[i]),
                                           r_t, q_t,
                                           T(h_sigma[i]), T(h_Tmat[i]));
                sum = sum + px;
                if (prices && rep == 0)
                    prices[i] = (double)px;
            } // for i
        } // for rep
        partial_sums[tid] = sum;
    } // for tid

    for (int tid = active; tid < TOTAL_THREADS; ++tid)
        partial_sums[tid] = T(0.0);
} // bs_host_run

#endif /* !USE_CUDA */

/* ------------------------------------------------------------------ */
/* Benchmark result                                                    */
/* ------------------------------------------------------------------ */
struct BSResult 
{
    double mean_price;
    double time_ms;
    double opts_per_sec;
    double max_rel_err;
    double avg_rel_err;
    double mape;
};

/* ------------------------------------------------------------------ */
/* Run benchmark for a single type                                     */
/* ------------------------------------------------------------------ */
template <typename T>
static BSResult run_bs(const double* params_S, const double* params_K,
                       const double* params_sigma, const double* params_Tmat,
                       const double* ref_prices,
                       int N, double r, double q, bool is_call)
{
    const int reps = REPS;
    T* partial = nullptr;

#if USE_CUDA
    double *d_S, *d_K, *d_sigma, *d_Tmat;
    const size_t param_sz = N * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_S,     param_sz));
    CUDA_CHECK(cudaMalloc(&d_K,     param_sz));
    CUDA_CHECK(cudaMalloc(&d_sigma, param_sz));
    CUDA_CHECK(cudaMalloc(&d_Tmat,  param_sz));
    CUDA_CHECK(cudaMemcpy(d_S,     params_S,     param_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K,     params_K,     param_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma, params_sigma, param_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Tmat,  params_Tmat,  param_sz, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&partial, TOTAL_THREADS * sizeof(T)));

    double* d_prices;
    CUDA_CHECK(cudaMalloc(&d_prices, N * sizeof(double)));

    /* Warmup — also captures per-option prices for accuracy */
    if (is_call)
        bs_kernel<T, true><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_S, d_K, d_sigma, d_Tmat, partial, d_prices, r, q, N, reps);
    else
        bs_kernel<T, false><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            d_S, d_K, d_sigma, d_Tmat, partial, d_prices, r, q, N, reps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed iterations (no accuracy output) */
    double total_ms = 0;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (is_call)
            bs_kernel<T, true><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
                d_S, d_K, d_sigma, d_Tmat, partial, nullptr, r, q, N, reps);
        else
            bs_kernel<T, false><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
                d_S, d_K, d_sigma, d_Tmat, partial, nullptr, r, q, N, reps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    } // for iter

    T* h_partial = new T[TOTAL_THREADS];
    CUDA_CHECK(cudaMemcpy(h_partial, partial,
                          TOTAL_THREADS * sizeof(T), cudaMemcpyDeviceToHost));

    double* h_prices = new double[N];
    CUDA_CHECK(cudaMemcpy(h_prices, d_prices,
                          N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_prices));

#else // host

    partial = new T[TOTAL_THREADS];
    double* h_prices = new double[N];

    /* First run — also captures per-option prices for accuracy */
    if (is_call)
        bs_host_run<T, true>(params_S, params_K, params_sigma, params_Tmat,
                             partial, h_prices, r, q, N, reps);
    else
        bs_host_run<T, false>(params_S, params_K, params_sigma, params_Tmat,
                              partial, h_prices, r, q, N, reps);

    /* Timed iterations (no accuracy output) */
    double total_ms = 0;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (is_call)
            bs_host_run<T, true>(params_S, params_K, params_sigma, params_Tmat,
                                 partial, nullptr, r, q, N, reps);
        else
            bs_host_run<T, false>(params_S, params_K, params_sigma, params_Tmat,
                                  partial, nullptr, r, q, N, reps);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    T* h_partial = partial;

#endif // USE_CUDA

    double total_sum;
#if defined(__USE_FP128_REFERENCE__) && (__FPMP_FP128_ENABLED__ == 1)
    {
        __bs_fp128 sum_q = (__bs_fp128)0.0;
        for (int i = 0; i < TOTAL_THREADS; ++i)
            sum_q += to_f128_val(h_partial[i]);
        total_sum = (double)sum_q;
    }
#else
    total_sum = 0.0;
    for (int i = 0; i < TOTAL_THREADS; ++i)
        total_sum += (double)h_partial[i];
#endif

    const double effective_n = (double)N * reps;

    /* Compare per-option prices against reference; skip near-zero       */
    /* reference prices to avoid division-by-zero noise.                  */
    double max_re = 0.0, avg_re = 0.0;

#if defined(__USE_FP128_REFERENCE__)
    bs_accuracy_f128(h_prices, ref_prices, N, &max_re, &avg_re);
#else
    double sum_re = 0.0;
    int n_acc = 0;
    for (int i = 0; i < N; ++i) {
        double ref = ref_prices[i];
        if (fabs(ref) < 1e-12) continue;
        double re = fabs((h_prices[i] - ref) / ref);
        if (re > max_re) max_re = re;
        sum_re += re;
        ++n_acc;
    }
    avg_re = n_acc > 0 ? sum_re / n_acc : 0.0;
#endif

    BSResult res;
    res.mean_price   = total_sum / effective_n;
    res.time_ms      = total_ms / NUM_ITERATIONS;
    res.opts_per_sec = effective_n / (res.time_ms * 1e-3);
    res.max_rel_err  = max_re;
    res.avg_rel_err  = avg_re;
    res.mape         = avg_re * 100.0;

    delete[] h_prices;
#if USE_CUDA
    delete[] h_partial;
    CUDA_CHECK(cudaFree(partial));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_sigma));
    CUDA_CHECK(cudaFree(d_Tmat));
#else
    delete[] partial;
#endif
    return res;
} // run_bs

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);

    constexpr int N = NUM_OPTIONS;
    constexpr double r = 0.02, q = 0.01;
    constexpr bool is_call = true;

    printf("Black-Scholes Option Pricing Benchmark\n");
    printf("======================================\n");

#if USE_CUDA
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Target: GPU (%s)\n", prop.name);
#if __FPMP_FP128_ENABLED__ == 1
    printf("Types:  float(%zu) fp32mp2_low(%zu) fp32mp2(%zu) double(%zu) fp64emu_mid(%zu) fp64emu(%zu) fp64mp2(%zu) fp128(%zu)\n\n",
           sizeof(float), sizeof(fp32mp2_low), sizeof(fp32mp2),
           sizeof(double), sizeof(fp64emu_mid), sizeof(fp64emu), sizeof(fp64mp2),
           sizeof(fp128_t));
#else
    printf("Types:  float(%zu) fp32mp2_low(%zu) fp32mp2(%zu) double(%zu) fp64emu_mid(%zu) fp64emu(%zu) fp64mp2(%zu)\n\n",
           sizeof(float), sizeof(fp32mp2_low), sizeof(fp32mp2),
           sizeof(double), sizeof(fp64emu_mid), sizeof(fp64emu), sizeof(fp64mp2));
#endif
#else
    printf("Target: CPU (host)\n");
#if __FPMP_FP128_ENABLED__ == 1
    printf("Types:  float(%zu) fp32mp2_low(%zu) fp32mp2(%zu) double(%zu) fp64emu_mid(%zu) fp64emu(%zu) fp64mp2(%zu) fp128(%zu)\n\n",
           sizeof(float), sizeof(fp32mp2_low), sizeof(fp32mp2),
           sizeof(double), sizeof(fp64emu_mid), sizeof(fp64emu), sizeof(fp64mp2),
           sizeof(fp128_t));
#else
    printf("Types:  float(%zu) fp32mp2_low(%zu) fp32mp2(%zu) double(%zu) fp64emu_mid(%zu) fp64emu(%zu) fp64mp2(%zu)\n\n",
           sizeof(float), sizeof(fp32mp2_low), sizeof(fp32mp2),
           sizeof(double), sizeof(fp64emu_mid), sizeof(fp64emu), sizeof(fp64mp2));
#endif
#endif

    printf("Options: %d  REPS: %d  r=%.4f  q=%.4f  %s\n",
           N, REPS, r, q, is_call ? "Call" : "Put");
    printf("Threads: %d  Timing iterations: %d\n\n", TOTAL_THREADS, NUM_ITERATIONS);

    /* Generate random option parameters (seed can be overridden for testing) */
#ifndef RNG_SEED
    #define RNG_SEED 123456789ULL
#endif
    const uint64_t seed = RNG_SEED;
    printf("RNG seed: %llu\n\n", (unsigned long long)seed);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uS(10.0, 200.0);
    std::uniform_real_distribution<double> uK(10.0, 200.0);
    std::uniform_real_distribution<double> uT(0.05, 5.0);
    std::uniform_real_distribution<double> usig(0.05, 1.0);

    double* h_S     = new double[N];
    double* h_K     = new double[N];
    double* h_sigma = new double[N];
    double* h_Tmat  = new double[N];
    for (int i = 0; i < N; ++i) 
    {
        h_S[i]     = uS(rng);
        h_K[i]     = uK(rng);
        h_Tmat[i]  = uT(rng);
        h_sigma[i] = usig(rng);
    }

    /* Compute reference prices */
    double* ref = new double[N];
    {
        double ref_sum = 0.0;
        for (int i = 0; i < N; ++i) 
        {
            ref[i] = bs_ref_price(h_S[i], h_K[i], r, q, h_sigma[i], h_Tmat[i], is_call);
            ref_sum += ref[i];
        }
#if defined(__USE_FP128_REFERENCE__)
        printf("Reference mean price (float128): %.10f\n\n", ref_sum / N);
#else
        printf("Reference mean price (double): %.10f\n\n", ref_sum / N);
#endif
    }

    /* Run benchmark for all types */
    auto res_f    = run_bs<float>             (h_S, h_K, h_sigma, h_Tmat, ref, N, r, q, is_call);
    auto res_ff_f = run_bs<fp32mp2_low>    (h_S, h_K, h_sigma, h_Tmat, ref, N, r, q, is_call);
    auto res_ff   = run_bs<fp32mp2>         (h_S, h_K, h_sigma, h_Tmat, ref, N, r, q, is_call);
    auto res_d    = run_bs<double>            (h_S, h_K, h_sigma, h_Tmat, ref, N, r, q, is_call);
    auto res_de   = run_bs<fp64emu_mid>       (h_S, h_K, h_sigma, h_Tmat, ref, N, r, q, is_call);
    auto res_dea  = run_bs<fp64emu>(h_S, h_K, h_sigma, h_Tmat, ref, N, r, q, is_call);
    auto res_dd   = run_bs<fp64mp2>         (h_S, h_K, h_sigma, h_Tmat, ref, N, r, q, is_call);
#if __FPMP_FP128_ENABLED__ == 1
    auto res_q    = run_bs<fp128_t>           (h_S, h_K, h_sigma, h_Tmat, ref, N, r, q, is_call);
#endif

    /* -- Accuracy table ------------------------------------------------ */
#if defined(__USE_FP128_REFERENCE__)
    printf("Accuracy vs float128 reference:\n\n");
#else
    printf("Accuracy vs double reference:\n\n");
#endif

    printf("%-24s  %12s  %14s  %14s  %12s  %-22s  %s\n",
           "Type", "Mean Price", "Max|RelErr|", "Avg|RelErr|",
           "MAPE(%)", "Quality", "Status");
    printf("%-24s  %12s  %14s  %14s  %12s  %-22s  %s\n",
           "------------------------", "------------", "--------------",
           "--------------", "------------", "----------------------", "------");

#if defined(__USE_FP128_REFERENCE__)
    const double tol_float   = 5e-3;
    const double tol_fp32mp2 = 1e-6;
    const double tol_fp64    = 1e-8;
    const double tol_fp64mp2 = 1e-8;
#else
    const double tol_float   = 5e-3;
    const double tol_fp32mp2 = 1e-6;
    const double tol_fp64    = 1e-9;
    const double tol_fp64mp2 = 1e-9;
#endif

    /* MAPE (Mean Absolute Percentage Error) quality scale */
    auto mape_quality = [](double mape) -> const char* 
    {
        if (mape < 1e-11) return "near machine eps";
        if (mape < 1e-7)  return "excellent";
        if (mape < 1e-3)  return "good";
        if (mape < 1e-1)  return "rough approximation";
        return                   "very inaccurate";
    };

    auto acc_row = [&](const char* name, const BSResult& r, double tol,
                       const char* status_override = nullptr) 
    {
        const char* status = status_override ? status_override
                           : (r.max_rel_err <= tol) ? "PASS" : "FAIL";
        printf("%-24s  %12.6f  %14.8e  %14.8e  %12.2e  %-22s  %s\n",
               name, r.mean_price, r.max_rel_err, r.avg_rel_err,
               r.mape, mape_quality(r.mape), status);
    };

    acc_row("float",              res_f,    tol_float);
    acc_row("fp32mp2_low",     res_ff_f, tol_fp32mp2);
    acc_row("fp32mp2",          res_ff,   tol_fp32mp2);
    acc_row("double",             res_d,    tol_fp64);
    acc_row("fp64emu_mid",          res_de,   tol_fp64);
    acc_row("fp64emu", res_dea,  tol_fp64);
    acc_row("fp64mp2",          res_dd,   tol_fp64mp2);
#if __FPMP_FP128_ENABLED__ == 1
    acc_row("fp128_t",            res_q,    1e-13);
#endif

    printf("\nMAPE (Mean Absolute Percentage Error) quality scale:\n");
    printf("  < 1e-11  near machine eps      at the limit of floating-point representation\n");
    printf("  < 1e-7   excellent             well-optimized double-precision algorithm\n");
    printf("  < 1e-3   good                  typical double-precision math library\n");
    printf("  < 1e-1   rough approximation   reduced precision or single-precision path\n");
    printf("  >= 1e-1  very inaccurate       significant precision loss\n");

    /* -- Performance table --------------------------------------------- */
    auto fmt_thousands = [](char* buf, size_t sz, double val) 
    {
        long long v = (long long)val;
        if (v < 1000) { snprintf(buf, sz, "%lld", v); return; }
        char tmp[32]; int n = 0;
        bool neg = v < 0; if (neg) v = -v;
        while (v > 0) { tmp[n++] = '0' + (int)(v % 10); v /= 10; }
        int pos = 0;
        if (neg) buf[pos++] = '-';
        for (int i = n - 1; i >= 0; --i) 
        {
            buf[pos++] = tmp[i];
            if (i > 0 && i % 3 == 0) buf[pos++] = '\'';
        }
        buf[pos] = '\0';
    };

    printf("\nPerformance:\n\n");

    printf("%-24s  %10s  %18s\n", "Type", "Time(ms)", "Options/sec");
    printf("%-24s  %10s  %18s\n",
           "------------------------", "----------", "------------------");

    auto perf_row = [&](const char* name, const BSResult& r) 
    {
        char ops[32];
        fmt_thousands(ops, sizeof(ops), r.opts_per_sec);
        printf("%-24s  %10.3f  %18s\n", name, r.time_ms, ops);
    };

    perf_row("float",              res_f);
    perf_row("fp32mp2_low",     res_ff_f);
    perf_row("fp32mp2",          res_ff);
    perf_row("double",             res_d);
    perf_row("fp64emu_mid",          res_de);
    perf_row("fp64emu", res_dea);
    perf_row("fp64mp2",          res_dd);
#if __FPMP_FP128_ENABLED__ == 1
    perf_row("fp128_t",            res_q);
#endif

    printf("\nSpeedups vs double:\n");
    printf("  %-24s  %6s\n", "------------------------", "------");
    printf("  %-24s  %5.2fx\n", "float",              res_d.time_ms / res_f.time_ms);
    printf("  %-24s  %5.2fx\n", "fp32mp2_low",     res_d.time_ms / res_ff_f.time_ms);
    printf("  %-24s  %5.2fx\n", "fp32mp2",          res_d.time_ms / res_ff.time_ms);
    printf("  %-24s  %5.2fx\n", "fp64emu_mid",          res_d.time_ms / res_de.time_ms);
    printf("  %-24s  %5.2fx\n", "fp64emu", res_d.time_ms / res_dea.time_ms);
    printf("  %-24s  %5.2fx\n", "fp64mp2",          res_d.time_ms / res_dd.time_ms);
#if __FPMP_FP128_ENABLED__ == 1
    printf("  %-24s  %5.2fx\n", "fp128_t",            res_d.time_ms / res_q.time_ms);
#endif

    printf("\nResult: COMPLETED\n");

    delete[] h_S;
    delete[] h_K;
    delete[] h_sigma;
    delete[] h_Tmat;
    delete[] ref;
    return 0;
} // main
