/*
    mc.cpp - Monte Carlo European Option Pricing Benchmark
    ======================================================
    Author:  Andrei Kolesov
    Date:    2025

    Prices a European option via Monte Carlo simulation and compares
    accuracy and throughput across multiple floating-point data types:

      float                - IEEE-754 binary32 (single precision, 24-bit mantissa)
      fp32mp2              - Double-float (pair of floats, ~46-bit mantissa)
      double               - IEEE-754 binary64 (double precision, 53-bit mantissa)
      fp64emu_mid          - Emulated fp64 via fp32 arithmetic (mid accuracy)
      fp64emu              - Emulated fp64 via fp32 arithmetic (high accuracy, library default)
      fp64emu_unpacked_mid - Emulated fp64 via fp32 arithmetic (unpacked: sign/exp/mantissa)
      fp64mp2              - Double-double (pair of doubles, ~104-bit mantissa)

    The benchmark is fully templated: every type goes through the same
    code path (GBM asset evolution, payoff evaluation, reduction) so that
    timing differences reflect arithmetic cost alone.

    Asset evolution (per path):
      S_T = S0 * exp(drift + vol_sqrt_t * Z)
    where Z is a standard normal variate.  Payoff is max(S_T - K, 0) for
    calls or max(K - S_T, 0) for puts, discounted by exp(-r*T).

    Gaussian variate generation (MC_RNG_ICDF):
      0 - Box-Muller (curand_normal / std::normal_distribution)
      1 - CUDA normcdfinv applied to a uniform variate
      2 - FPMP icdf: integer curand → fpmp normcdfinv (fp32mp2 path only)

    Variance reduction:
      Antithetic variates: each draw Z produces two paths (Z and -Z),
      halving the number of RNG calls while reducing variance.

    Supports both CUDA (device) and CPU (host) targets.

    Output tables:
      1. Performance: pregen  — timing/throughput with pre-generated normals
      2. Performance: inline  — timing/throughput with inline RNG
         Both show MC price, time, paths/sec, speedup vs double.
      3. Accuracy: inline RNG — |error vs BS analytical| across four
         parameter sets (ATM, OTM, deep OTM, extreme) to show how MC
         accuracy varies with option moneyness and volatility.
      4. Arithmetic precision — pregen prices compared against double,
         isolating pure floating-point arithmetic error from MC noise.

    Reference price:
      By default (__USE_FP128_REFERENCE__), the Black-Scholes analytical
      price is computed in __float128 via libquadmath (see mc_host.cpp).
      This provides ~34-digit accuracy for the BS reference.

    Compile-time configuration (via Makefile or -D flags):
      NUM_PATHS          - total MC paths per launch (default 16M GPU / 32K CPU)
      REPS               - inner repetitions per launch (default 128 GPU / 16 CPU)
      THREADS_PER_BLOCK  - CUDA block size (default 256)
      NUM_BLOCKS         - CUDA grid size  (default 2048)
      NUM_ITERATIONS     - timing iterations for averaging (default 10)
      MC_RNG_ICDF        - Gaussian generation method (0/1/2, see above)
      MC_FPEMU_UNPACKED  - 1: include fp64emu_unpacked_mid in the benchmark (default 0)
      MC_STRIKE          - strike price for performance tables (default 100.0)
      MC_SIGMA           - volatility for performance tables (default 0.2)
      __USE_FP128_REFERENCE__ - use quad-precision BS reference (default, via FP128_REF=1)

    REPS multiplies GPU work without extra memory.  The MC price is
    divided by the effective path count (paths * REPS) so it is unchanged.
    Higher REPS = longer kernels = more stable timing measurements.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <chrono>
#include <type_traits>

#include <cuda/fpmp>
#include <cuda/fpmp_math>
#include <cuda/fpemu>

using namespace cuda::experimental;
using namespace cuda::experimental::fpmp;

#ifndef MC_RNG_ICDF
  #define MC_RNG_ICDF 0
#endif
#ifndef MC_FPEMU_UNPACKED
  #define MC_FPEMU_UNPACKED 0
#endif

#if defined(__CUDACC__)
  #include <curand_kernel.h>
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

#ifndef NUM_PATHS
  #if defined(__CUDACC__)
      #define NUM_PATHS (1024*1024*16)
  #else
      #define NUM_PATHS (1024*16)
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
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(1);                                                        \
    }                                                                   \
} while (0)
#endif

/* ------------------------------------------------------------------ */
/* Black-Scholes closed-form price (host)                              */
/*                                                                     */
/* Computes the analytical European option price using the standard    */
/* Black-Scholes formula.  Used as the reference value to validate     */
/* Monte Carlo results across all data types.                          */
/*                                                                     */
/* When __USE_FP128_REFERENCE__ is defined, delegates to mc_host.cpp   */
/* (mc_ref_price_f128) which computes entirely in __float128 via       */
/* libquadmath for maximum reference accuracy (~34 decimal digits).    */
/* ------------------------------------------------------------------ */
#if defined(__USE_FP128_REFERENCE__)
extern double mc_ref_price_f128(double S, double K, double r,
                                double q, double sigma, double Tmat,
                                bool is_call);
#endif

static double bs_price(double S, double K, double r, double q,
                       double v, double T, bool is_call)
{
#if defined(__USE_FP128_REFERENCE__)
    return mc_ref_price_f128(S, K, r, q, v, T, is_call);
#else
    double d1 = (log(S / K) + (r - q + 0.5 * v * v) * T) / (v * sqrt(T));
    double d2 = d1 - v * sqrt(T);
    double Nd1 = 0.5 * erfc(-d1 * M_SQRT1_2);
    double Nd2 = 0.5 * erfc(-d2 * M_SQRT1_2);
    double call_px = S * exp(-q * T) * Nd1 - K * exp(-r * T) * Nd2;
    return is_call ? call_px : call_px - S * exp(-q * T) + K * exp(-r * T);
#endif
} // bs_price

/* ------------------------------------------------------------------ */
/* MC payoff core (shared between host and device)                     */
/*                                                                     */
/* Computes the option payoff for a single Gaussian draw z using       */
/* antithetic variates: two paths (z and -z) are evaluated per draw,   */
/* halving the RNG cost while reducing variance.                       */
/* ------------------------------------------------------------------ */
template <typename T>
HOST_DEVICE void mc_payoff(const T z, const T drift, const T vol_sqrt_t,
                           const T S0, const T K, const bool is_call, T& sum)
{
    T vz  = vol_sqrt_t * z;
    T st1 = S0 * exp(drift + vz);
    T st2 = S0 * exp(drift - vz);
    T p1 = is_call ? (st1 > K ? st1 - K : T(0.0))
                   : (K > st1 ? K - st1 : T(0.0));
    T p2 = is_call ? (st2 > K ? st2 - K : T(0.0))
                   : (K > st2 ? K - st2 : T(0.0));
    sum = sum + p1 + p2;
} // mc_payoff

/* ------------------------------------------------------------------ */
/* CUDA kernel (device only)                                           */
/*                                                                     */
/* MC kernel for pre-generated or inline RNG with antithetic variates. */
/* The use_pregen template parameter selects the RNG path at compile   */
/* time via if constexpr.                                              */
/* ------------------------------------------------------------------ */
#if USE_CUDA

/* Generates a Gaussian variate from cuRAND state.                     */
/* When use_icdf=true, uses fpmp inverse CDF (icdf) instead of cuRAND  */
/* Box-Muller.  Two curand() calls are combined into a uint64_t to     */
/* match the 64-bit entropy that curand_normal_double uses internally.  */
/* The icdf path produces fp32mp2 precision (~46-bit mantissa);        */
/* for double the result is cast from fp32mp2.                         */
/*   float      : curand_normal (Box-Muller, always)                   */
/*   fp32mp2  : curand_normal_double or icdf(uint64)                 */
/*   double/etc : curand_normal_double or icdf(uint64)                 */
template <typename T, bool use_icdf = false>
__device__ __forceinline__
T mc_randn(curandStatePhilox4_32_10_t* state)
{
    if constexpr (std::is_same<T, fp32mp2>::value)
    {
        if constexpr (use_icdf) 
        {
            uint64_t u = ((uint64_t)curand(state) << 32) | curand(state);
            return T(icdf(u));
        } 
        else 
        {
            return T(curand_normal_double(state));
        }
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        return curand_normal(state);
    }
    else // double & fp64mp2
    {
        if constexpr (use_icdf) 
        {
            uint64_t u = ((uint64_t)curand(state) << 32) | curand(state);
            return T(icdf(u));
        } 
        else 
        {
            return T(curand_normal_double(state));
        }
    }
} // mc_randn

/*
Kernel for MC simulation
- normals: pre-generated normals (if use_pregen is true)
- partial_sums: partial sums of the payoff
- drift: drift term
- vol_sqrt_t: volatility term
- disc: discount factor
- S0: initial stock price
- K: strike price
*/
template <typename T, bool use_pregen, bool use_icdf = false>
__global__ void mc_kernel(const double* __restrict__ normals,
                          T* __restrict__ partial_sums,
                          const T drift, const T vol_sqrt_t,
                          const T disc, const T S0, const T K,
                          const bool is_call, const int num_items,
                          const unsigned long long seed)
{
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    T sum(0.0);
    if constexpr (!use_pregen) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, tid, 0, &state);
        for (int rep = 0; rep < REPS; ++rep) {
            for (int i = tid; i < num_items; i += stride) {
                mc_payoff<T>(mc_randn<T, use_icdf>(&state),
                    drift, vol_sqrt_t, S0, K, is_call, sum);
            }
        }
    } else {
        for (int rep = 0; rep < REPS; ++rep) {
            for (int i = tid; i < num_items; i += stride) {
                mc_payoff<T>(T(normals[i]),
                    drift, vol_sqrt_t, S0, K, is_call, sum);
            }
        }
    }
    partial_sums[tid] = disc * sum;
} // mc_kernel

#endif /* USE_CUDA */

/* ------------------------------------------------------------------ */
/* Host simulation (CPU path)                                          */
/*                                                                     */
/* Host MC loop with antithetic variates for pre-generated or inline   */
/* RNG.  Each "thread" is emulated by a loop iteration over            */
/* TOTAL_THREADS.                                                      */
/* ------------------------------------------------------------------ */
#if !USE_CUDA

#if MC_RNG_ICDF
extern double mc_host_normcdfinv(double p);
#endif

/* Host Gaussian variate generator.  Wraps std::mt19937_64 with either */
/* a normal distribution (Box-Muller) or a uniform distribution +      */
/* mc_host_normcdfinv (from mc_host.cpp), depending on MC_RNG_ICDF.    */
struct mc_host_randn {
    std::mt19937_64 rng;
#if MC_RNG_ICDF
    std::uniform_real_distribution<double> dist{1e-15, 1.0 - 1e-15};
#else
    std::normal_distribution<double> dist{0.0, 1.0};
#endif
    mc_host_randn(unsigned long long seed) : rng(seed) {}
    double operator()() {
#if MC_RNG_ICDF
        return mc_host_normcdfinv(dist(rng));
#else
        return dist(rng);
#endif
    }
};

template <typename T, bool use_pregen>
static void mc_host_run(const double* normals, T* partial_sums,
                        const T drift, const T vol_sqrt_t,
                        const T disc, const T S0, const T K,
                        const bool is_call, const int num_items,
                        unsigned long long seed)
{
    for (int tid = 0; tid < TOTAL_THREADS; ++tid) {
        T sum(0.0);
        if constexpr (!use_pregen) {
            mc_host_randn randn(seed + tid);
            for (int rep = 0; rep < REPS; ++rep) {
                for (int i = tid; i < num_items; i += TOTAL_THREADS) {
                    mc_payoff<T>(T(randn()),
                        drift, vol_sqrt_t, S0, K, is_call, sum);
                }
            }
        } else {
            for (int rep = 0; rep < REPS; ++rep) {
                for (int i = tid; i < num_items; i += TOTAL_THREADS) {
                    mc_payoff<T>(T(normals[i]),
                        drift, vol_sqrt_t, S0, K, is_call, sum);
                }
            }
        }
        partial_sums[tid] = disc * sum;
    }
} // mc_host_run

#endif /* !USE_CUDA */

/* ------------------------------------------------------------------ */
/* Run benchmark for a single type                                     */
/*                                                                     */
/* Orchestrates the full MC pipeline for data type T:                  */
/*   1. Converts market parameters to type T                           */
/*   2. Allocates partial-sum buffer (device or host)                  */
/*   3. Warms up with one un-timed launch                              */
/*   4. Runs NUM_ITERATIONS timed launches and averages elapsed time   */
/*   5. Reduces partial sums to compute the MC option price            */
/* Returns MCResult with the price, average time, and throughput.      */
/* ------------------------------------------------------------------ */
struct MCResult {
    double price;
    double time_ms;
    double paths_per_sec;
};

template <typename T, bool use_pregen, bool use_icdf = false>
static MCResult run_mc(const double* normals [[maybe_unused]], int num_paths,
                       double S0_d, double K_d, double r, double q,
                       double sigma, double Tmat, bool is_call,
                       int num_iter = NUM_ITERATIONS)
{
    T S0(S0_d), K_val(K_d);
    T drift((r - q - 0.5 * sigma * sigma) * Tmat);
    T vol_sqrt_t(sigma * sqrt(Tmat));
    T disc(exp(-r * Tmat));

    const int num_draws    = num_paths / 2;
    const int actual_paths = num_draws * 2;

    T* partial = nullptr;

#if USE_CUDA
    CUDA_CHECK(cudaMalloc(&partial, TOTAL_THREADS * sizeof(T)));

    const unsigned long long seed = 42ULL;

    auto launch = [&]() {
        mc_kernel<T, use_pregen, use_icdf><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
            normals, partial, drift, vol_sqrt_t, disc,
            S0, K_val, is_call, num_draws, seed);
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    for (int iter = 0; iter < num_iter; ++iter) {
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

    T* h_partial = new T[TOTAL_THREADS];
    CUDA_CHECK(cudaMemcpy(h_partial, partial,
                          TOTAL_THREADS * sizeof(T), cudaMemcpyDeviceToHost));
#else
    partial = new T[TOTAL_THREADS];

    const unsigned long long seed = 42ULL;

    auto launch = [&]() {
        mc_host_run<T, use_pregen>(
            normals, partial, drift, vol_sqrt_t, disc,
            S0, K_val, is_call, num_draws, seed);
    };

    launch();

    double total_ms = 0;
    for (int iter = 0; iter < num_iter; ++iter) {
        auto t0 = std::chrono::high_resolution_clock::now();
        launch();
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    T* h_partial = partial;
#endif

    long double sum = 0.0L;
    for (int i = 0; i < TOTAL_THREADS; ++i)
        sum += (long double)(double)h_partial[i];

    const long double effective_paths = (long double)actual_paths * REPS;

    MCResult res;
    res.price         = (double)(sum / effective_paths);
    res.time_ms       = num_iter > 0 ? (double)total_ms / num_iter : 0.0;
    res.paths_per_sec = res.time_ms > 0.0 ? (double)effective_paths / (res.time_ms * 1e-3) : 0.0;

#if USE_CUDA
    delete[] h_partial;
    CUDA_CHECK(cudaFree(partial));
#else
    delete[] partial;
#endif
    return res;
} // run_mc

/* ------------------------------------------------------------------ */
/* Main                                                                */
/*                                                                     */
/* Prints configuration, runs run_mc<T> for every enabled data type,   */
/* and prints a summary table with prices, timings, throughput, and    */
/* accuracy vs Black-Scholes.                                          */
/* ------------------------------------------------------------------ */
int main()
{
#ifndef MC_STRIKE
    #define MC_STRIKE 100.0
#endif
#ifndef MC_SIGMA
    #define MC_SIGMA 0.2
#endif
    constexpr double S0 = 100.0, K = MC_STRIKE, r = 0.05, q = 0.0, sigma = MC_SIGMA, T = 1.0;
    constexpr bool is_call = true;
    constexpr int num_paths = NUM_PATHS;

    printf("Monte Carlo European Option Pricing Benchmark\n");
    printf("==============================================\n");

#if USE_CUDA
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Target: GPU (%s)\n\n", prop.name);
#else
    printf("Target: CPU (host)\n\n");
#endif

    printf("Option: S0=%.2f K=%.2f r=%.4f q=%.4f sigma=%.4f T=%.4f %s\n",
           S0, K, r, q, sigma, T, is_call ? "Call" : "Put");
    printf("Paths: %d  Antithetic: yes  REPS: %d\n", num_paths, REPS);
    printf("RNG (pregen): std::mt19937_64 normal distribution\n");
#if USE_CUDA
    #if MC_RNG_ICDF == 1
    printf("RNG (inline): cuRAND Philox4x32 + normcdfinv\n");
    #elif MC_RNG_ICDF == 2
    printf("RNG (inline): cuRAND Philox4x32 + fpmp normcdfinv\n");
    #else
    printf("RNG (inline): cuRAND Philox4x32 Box-Muller\n");
    #endif
#else
    #if MC_RNG_ICDF
    printf("RNG (inline): std::mt19937_64 + mc_host_normcdfinv\n");
    #else
    printf("RNG (inline): std::mt19937_64 Box-Muller\n");
    #endif
#endif
    printf("Threads: %d  Timing iterations: %d\n",
           TOTAL_THREADS, NUM_ITERATIONS);

    struct ParamSet {
        const char* label;
        double K;
        double sigma;
    };
    static const ParamSet acc_sets[] = {
        {"ATM",       100.0, 0.2},
        {"OTM",       130.0, 0.2},
        {"deep OTM",  200.0, 0.4},
        {"extreme",   500.0, 1.0},
    };
    static constexpr int N_ACC = (int)(sizeof(acc_sets) / sizeof(acc_sets[0]));

    double bs_acc[N_ACC];
    for (int p = 0; p < N_ACC; ++p)
        bs_acc[p] = bs_price(S0, acc_sets[p].K, r, q, acc_sets[p].sigma, T, is_call);

    printf("\nParam sets (S0=%.0f, r=%.4f, q=%.4f, T=%.1f, %s):\n",
           S0, r, q, T, is_call ? "Call" : "Put");
    printf("----------------------------------------------\n");
    for (int p = 0; p < N_ACC; ++p)
        printf("  %-12s K=%-6.0f sigma=%-4.1f  BS=%.10f\n",
               acc_sets[p].label, acc_sets[p].K, acc_sets[p].sigma, bs_acc[p]);

    double bs_ref = bs_price(S0, K, r, q, sigma, T, is_call);
#if defined(__USE_FP128_REFERENCE__)
    printf("\nBlack-Scholes analytical: %.10f (float128 reference)\n", bs_ref);
#else
    printf("\nBlack-Scholes analytical: %.10f (double reference)\n", bs_ref);
#endif
    printf("==============================================\n");

    const int num_draws = num_paths / 2;

    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    double* h_normals = new double[num_draws];
    for (int i = 0; i < num_draws; ++i)
        h_normals[i] = dist(rng);

    const double* normals_ptr = h_normals;

#if USE_CUDA
    double* d_normals;
    CUDA_CHECK(cudaMalloc(&d_normals, num_draws * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_normals, h_normals,
                          num_draws * sizeof(double), cudaMemcpyHostToDevice));
    normals_ptr = d_normals;
#endif

    const int actual_paths = num_draws * 2;
    printf("Running %d paths (%d draws) x %d reps per type...\n\n",
           actual_paths, num_draws, REPS);

    auto run_all = [&](auto tag) {
        constexpr bool pg = decltype(tag)::value;
        struct Results {
            MCResult f, ff, d, de, dea, dd;
#if MC_FPEMU_UNPACKED
            MCResult du;
#endif
        } res;
        res.f   = run_mc<float,              pg>(normals_ptr, num_paths, S0, K, r, q, sigma, T, is_call);
        res.ff  = run_mc<fp32mp2,          pg>(normals_ptr, num_paths, S0, K, r, q, sigma, T, is_call);
        res.d   = run_mc<double,             pg>(normals_ptr, num_paths, S0, K, r, q, sigma, T, is_call);
        res.de  = run_mc<fp64emu_mid,        pg>(normals_ptr, num_paths, S0, K, r, q, sigma, T, is_call);
        res.dea = run_mc<fp64emu, pg>(normals_ptr, num_paths, S0, K, r, q, sigma, T, is_call);
#if MC_FPEMU_UNPACKED
        res.du  = run_mc<fp64emu_unpacked_mid, pg>(normals_ptr, num_paths, S0, K, r, q, sigma, T, is_call);
#endif
        res.dd  = run_mc<fp64mp2,          pg>(normals_ptr, num_paths, S0, K, r, q, sigma, T, is_call);
        return res;
    };

    auto res_pg  = run_all(std::true_type{});
    auto res_rng = run_all(std::false_type{});

#if USE_CUDA
    MCResult res_icdf_ff = run_mc<fp32mp2, false, true>(normals_ptr, num_paths, S0, K, r, q, sigma, T, is_call);
    MCResult res_icdf_d  = run_mc<double,    false, true>(normals_ptr, num_paths, S0, K, r, q, sigma, T, is_call);
#endif

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

    auto print_perf_header = [&]() {
        printf("%-24s  %14s  %10s  %18s  %8s\n",
               "Type", "MC Price", "Time(ms)", "Paths/sec", "Speedup vs double");
        printf("%-24s  %14s  %10s  %18s  %8s\n",
               "------------------------", "--------------", "----------",
               "------------------", "--------");
    };

    auto row = [&](const char* name, const MCResult& res, double ref_time) {
        char pps[32];
        fmt_thousands(pps, sizeof(pps), res.paths_per_sec);
        printf("%-24s  %14.10f  %10.3f  %18s  %6.2fx\n",
               name, res.price, res.time_ms, pps,
               ref_time / res.time_ms);
    };

    const double t_pg = res_pg.d.time_ms;
    const double t_rng = res_rng.d.time_ms;

    printf("=== Performance: pregen (K=%.0f, sigma=%.1f) ===\n", K, sigma);
    printf("Pre-generated normals, measures arithmetic cost only.\n\n");
    print_perf_header();
    row("float",              res_pg.f,   t_pg);
    row("fp32mp2",          res_pg.ff,  t_pg);
    row("double",             res_pg.d,   t_pg);
    row("fp64emu_mid",          res_pg.de,  t_pg);
    row("fp64emu", res_pg.dea, t_pg);
#if MC_FPEMU_UNPACKED
    row("fp64emu_unpacked_mid", res_pg.du,  t_pg);
#endif
    row("fp64mp2",          res_pg.dd,  t_pg);

    printf("\n=== Performance: inline RNG (K=%.0f, sigma=%.1f) ===\n", K, sigma);
    printf("RNG inside kernel, measures arithmetic + RNG cost.\n");
#if USE_CUDA
    printf("(icdf) = fpmp icdf(uint64) instead of Box-Muller.\n");
#endif
    printf("\n");
    print_perf_header();
    row("float",              res_rng.f,  t_rng);
    row("fp32mp2",          res_rng.ff, t_rng);
#if USE_CUDA
    row("fp32mp2 (icdf)",   res_icdf_ff, t_rng);
#endif
    row("double",             res_rng.d,  t_rng);
#if USE_CUDA
    row("double (icdf)",      res_icdf_d,  t_rng);
#endif
    row("fp64emu_mid",          res_rng.de, t_rng);
    row("fp64emu", res_rng.dea,t_rng);
#if MC_FPEMU_UNPACKED
    row("fp64emu_unpacked_mid", res_rng.du, t_rng);
#endif
    row("fp64mp2",          res_rng.dd, t_rng);

    auto print_acc_header = [&]() {
        printf("  %-24s", "Type");
        for (int p = 0; p < N_ACC; ++p)
            printf("  %12s", acc_sets[p].label);
        printf("\n");
        printf("  %-24s", "------------------------");
        for (int p = 0; p < N_ACC; ++p)
            printf("  %12s", "------------");
        printf("\n");
    };

    auto acc_row = [&](const char* name, const double* errs) {
        printf("  %-24s", name);
        for (int p = 0; p < N_ACC; ++p)
            printf("  %12.2e", errs[p]);
        printf("\n");
    };

    // --- Inline accuracy ---
    double acc_f[N_ACC], acc_ff[N_ACC], acc_d[N_ACC];
    double acc_de[N_ACC], acc_dea[N_ACC], acc_dd[N_ACC];
#if MC_FPEMU_UNPACKED
    double acc_du[N_ACC];
#endif
#if USE_CUDA
    double acc_icdf_ff[N_ACC], acc_icdf_d[N_ACC];
#endif

    for (int p = 0; p < N_ACC; ++p) {
        double Kp = acc_sets[p].K, sp = acc_sets[p].sigma;
        double bs = bs_acc[p];
        acc_f[p]   = fabs(run_mc<float,              false>(normals_ptr, num_paths, S0, Kp, r, q, sp, T, is_call, 0).price - bs);
        acc_ff[p]  = fabs(run_mc<fp32mp2,          false>(normals_ptr, num_paths, S0, Kp, r, q, sp, T, is_call, 0).price - bs);
        acc_d[p]   = fabs(run_mc<double,             false>(normals_ptr, num_paths, S0, Kp, r, q, sp, T, is_call, 0).price - bs);
        acc_de[p]  = fabs(run_mc<fp64emu_mid,        false>(normals_ptr, num_paths, S0, Kp, r, q, sp, T, is_call, 0).price - bs);
        acc_dea[p] = fabs(run_mc<fp64emu, false>(normals_ptr, num_paths, S0, Kp, r, q, sp, T, is_call, 0).price - bs);
        acc_dd[p]  = fabs(run_mc<fp64mp2,          false>(normals_ptr, num_paths, S0, Kp, r, q, sp, T, is_call, 0).price - bs);
#if MC_FPEMU_UNPACKED
        acc_du[p]  = fabs(run_mc<fp64emu_unpacked_mid, false>(normals_ptr, num_paths, S0, Kp, r, q, sp, T, is_call, 0).price - bs);
#endif
#if USE_CUDA
        acc_icdf_ff[p] = fabs(run_mc<fp32mp2, false, true>(normals_ptr, num_paths, S0, Kp, r, q, sp, T, is_call, 0).price - bs);
        acc_icdf_d[p]  = fabs(run_mc<double,    false, true>(normals_ptr, num_paths, S0, Kp, r, q, sp, T, is_call, 0).price - bs);
#endif
    }

#if defined(__USE_FP128_REFERENCE__)
    printf("\n=== Accuracy: inline RNG (|error vs float128 BS|) ===\n");
#else
    printf("\n=== Accuracy: inline RNG (|error vs double BS|) ===\n");
#endif
    printf("Each type uses its own RNG sequence.\n\n");
    print_acc_header();
    acc_row("float",              acc_f);
    acc_row("fp32mp2",          acc_ff);
#if USE_CUDA
    acc_row("fp32mp2 (icdf)",   acc_icdf_ff);
#endif
    acc_row("double",             acc_d);
#if USE_CUDA
    acc_row("double (icdf)",      acc_icdf_d);
#endif
    acc_row("fp64emu_mid",          acc_de);
    acc_row("fp64emu", acc_dea);
#if MC_FPEMU_UNPACKED
    acc_row("fp64emu_unpacked_mid", acc_du);
#endif
    acc_row("fp64mp2",          acc_dd);

    // --- Arithmetic precision (pregen, vs double) ---
    printf("\n=== Arithmetic precision (pregen, vs double) ===\n");
    printf("Same normals, same payoff: isolates pure arithmetic error.\n\n");
    const double ref_price = res_pg.d.price;
    int failures = 0;
    auto arith_row = [&](const char* name, double price, double tol, const char* status_override = nullptr) {
        double rel = ref_price != 0.0 ? fabs(price - ref_price) / fabs(ref_price) : 0.0;
        const char* status = status_override ? status_override
                           : (rel <= tol) ? "OK" : "FAIL";
        if (!status_override && rel > tol) failures++;
        printf("  %-24s  MC Price %16.10f   |rel err vs double| = %.2e  %s\n", name, price, rel, status);
    };
    arith_row("float",              res_pg.f.price,   0,     "- (lower precision)");
    arith_row("fp32mp2",          res_pg.ff.price,  1e-14);
    arith_row("fp64emu_mid",          res_pg.de.price,  1e-12);
    arith_row("fp64emu", res_pg.dea.price, 1e-15);
#if MC_FPEMU_UNPACKED
    arith_row("fp64emu_unpacked_mid", res_pg.du.price,  1e-12);
#endif

    printf("\nResult: %s\n", failures == 0 ? "PASS" : "FAIL");

    delete[] h_normals;
#if USE_CUDA
    CUDA_CHECK(cudaFree(d_normals));
#endif
    return 0;
} // main
