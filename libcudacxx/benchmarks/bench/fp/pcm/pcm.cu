/*
 * pcm.cu - PCM Off-Diagonal Matrix-Vector Product Benchmark (CUDA)
 *
 * Benchmarks the accumulate_A_offdiagonal kernel from Polarizable Continuum
 * Model (PCM) quantum chemistry code.  The kernel computes the off-diagonal
 * contribution of the PCM cavity matrix A applied to a vector:
 *
 *   lhs[i] += sum_{j!=i} A(i,j) * rhs[j]
 *
 * where A(i,j) = erf(zeta_ij * r_ij) / r_ij, with zeta_ij derived from
 * per-point Gaussian exponents.  Complexity is O(npoint^2).
 *
 * Compares performance across two data types:
 *   double      - IEEE-754 binary64 (native hardware)
 *   fp32mp2     - Double-float (pair of floats, ~46-bit mantissa)
 *
 * Compile-time configuration (via Makefile or -D flags):
 *   NPOINT              - number of surface points (default: 10000)
 *   THREADS_PER_BLOCK   - CUDA block size (default: 256)
 *   NUM_BLOCKS          - CUDA grid size  (default: 512)
 *   NUM_ITERATIONS      - timing iterations for averaging (default: 10)
 *   MIN_BLOCKS_PER_SM   - min blocks/SM for boys kernel (default: 1)
 *
 * Author:  Melisa Alcan, Andrei Kolesov
 * Date:    2026
 */

#include <cstdio>

#if !defined(__CUDACC__)
int main() {
    printf("PCM benchmark requires CUDA. Build with nvcc.\n");
    return 0;
}
#else

#include <cstdlib>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

#include <cuda/fpmp>
#include <cuda/fpmp_math>

using namespace cuda::experimental;

/* ------------------------------------------------------------------ */
/* ffloat: accuracy-switchable fp32mp2 alias                           */
/*   __METHOD__ = def (default), low, high                             */
/* ------------------------------------------------------------------ */
#ifndef __METHOD__
  #define __METHOD__ def
#endif

using ffloat = fpmp2<float, fpmp2_accuracy::__METHOD__>;

typedef struct ffloat4_32a {
    ffloat x, y, z, w;
} __attribute__((aligned(32))) ffloat4_32a;

#define STRINGIFY(x) #x
#define METHOD_STR(x) STRINGIFY(x)

/* ------------------------------------------------------------------ */
/* Compile-time defaults                                               */
/* ------------------------------------------------------------------ */
#ifndef NPOINT
  #define NPOINT 10000
#endif
#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK 256
#endif
#ifndef NUM_BLOCKS
  #define NUM_BLOCKS 512
#endif
#ifndef NUM_ITERATIONS
  #define NUM_ITERATIONS 10
#endif
#ifndef MIN_BLOCKS_PER_SM
  #define MIN_BLOCKS_PER_SM 1
#endif

/* ------------------------------------------------------------------ */
/* CUDA error checking                                                 */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(1);                                                        \
    }                                                                   \
} while (0)

/* ------------------------------------------------------------------ */
/* Vec4 types                                                          */
/* ------------------------------------------------------------------ */
/* double4_32a is provided by CUDA vector_types.h (CUDA >= 13.0)       */
/* ffloat4_32a is defined above (method-switchable)                     */

/* ------------------------------------------------------------------ */
/* fast_rsqrt: type-dispatched reciprocal square root                  */
/* ------------------------------------------------------------------ */
namespace detail {

template<typename T>
__host__ __device__ __forceinline__ T fast_rsqrt(T const& x) { return rsqrt(x); }

template<>
__host__ __device__ __forceinline__ double fast_rsqrt<double>(double const& x) {
#if 1 & __CUDA_ARCH__ != 0
    double r{};
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(x));
    double t = __dmul_rn(r, r);
    double e = __fma_rn(x, -t, 1.0);
    t = __fma_rn(0.375, e, 0.5);
    e = __dmul_rn(e, r);
    return __fma_rn(t, e, r);
#else
    return 1.0 / sqrt(x);
#endif
}

template<>
__host__ __device__ __forceinline__ ffloat fast_rsqrt<ffloat>(ffloat const& x) {
    return rsqrt(x);
}

} // namespace detail



/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */
constexpr double INV_SQRT_PI = 0.5641895835477563;

/* ------------------------------------------------------------------ */
/* Kernel: accumulate_A_offdiagonal (original, erf-based)              */
/*                                                                     */
/* Used for double precision.  Each thread processes one or more "row"  */
/* indices via grid-stride loop, iterating over all npoint columns.     */
/* ------------------------------------------------------------------ */
template <typename T, typename T4>
__global__
void accumulate_A_offdiagonal_kernel(
    size_t npoint,
    const T4* __restrict__ xyzz,
    T* __restrict__ lhs_vec,
    const T* __restrict__ rhs_vec)
{
    const T inv_sqrt_pi = static_cast<T>(INV_SQRT_PI);

    for (size_t indexA = threadIdx.x + blockDim.x * blockIdx.x;
         indexA < npoint;
         indexA += blockDim.x * gridDim.x)
    {
        T4 xyzzA = xyzz[indexA];
        const T xA = xyzzA.x;
        const T yA = xyzzA.y;
        const T zA = xyzzA.z;
        const T zetaA = xyzzA.w;
        const T zetaA_sq = zetaA * zetaA;

        T local_sum = static_cast<T>(0.0);

        for (size_t indexB = 0; indexB < npoint; indexB++) {
            if (indexA == indexB) continue;
            T4 xyzzB = xyzz[indexB];
            const T xB = xyzzB.x;
            const T yB = xyzzB.y;
            const T zB = xyzzB.z;
            const T zetaB = xyzzB.w;

            T zeta_denom = zetaA_sq;
            zeta_denom = zeta_denom + zetaB * zetaB;
            const T zeta = zetaA * zetaB * detail::fast_rsqrt<T>(zeta_denom);

            const T dx = xA - xB;
            const T dy = yA - yB;
            const T dz = zA - zB;
            T r2 = dx * dx;
            r2 = r2 + dy * dy;
            r2 = r2 + dz * dz;

            T A_AB;
            if (r2 == static_cast<T>(0.0)) {
                A_AB = static_cast<T>(2.0) * zeta * inv_sqrt_pi;
            } else {
                const T inv_rAB = detail::fast_rsqrt<T>(r2);
                A_AB = erf(zeta * r2 * inv_rAB) * inv_rAB;
            }

            local_sum = local_sum + A_AB * rhs_vec[indexB];
        }

        lhs_vec[indexA] = lhs_vec[indexA] + local_sum;
    }
}

/* ------------------------------------------------------------------ */
/* Kernel: accumulate_A_offdiagonal_boys (Boys F_0 based)              */
/*                                                                     */
/* ffloat-only variant.  Uses boys_f0(zeta²·r²) instead of            */
/* erf(zeta·r)/r, eliminating one rsqrt and the r²==0 branch.         */
/*                                                                     */
/*   A(i,j) = erf(ζ·r)/r = (2ζ/√π) · F_0(ζ²·r²)                     */
/* ------------------------------------------------------------------ */
__global__ __launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
void accumulate_A_offdiagonal_boys_kernel(
    size_t npoint,
    const ffloat4_32a* __restrict__ xyzz,
    ffloat* __restrict__ lhs_vec,
    const ffloat* __restrict__ rhs_vec)
{
    const ffloat two_inv_sqrt_pi = static_cast<ffloat>(2.0 * INV_SQRT_PI);

    for (size_t indexA = threadIdx.x + blockDim.x * blockIdx.x;
         indexA < npoint;
         indexA += blockDim.x * gridDim.x)
    {
        ffloat4_32a xyzzA = xyzz[indexA];
        const ffloat xA = xyzzA.x;
        const ffloat yA = xyzzA.y;
        const ffloat zA = xyzzA.z;
        const ffloat zetaA = xyzzA.w;
        const ffloat zetaA_sq = zetaA * zetaA;

        ffloat local_sum = static_cast<ffloat>(0.0);

        for (size_t indexB = 0; indexB < npoint; indexB++) {
            if (indexA == indexB) continue;
            ffloat4_32a xyzzB = xyzz[indexB];
            const ffloat xB = xyzzB.x;
            const ffloat yB = xyzzB.y;
            const ffloat zB = xyzzB.z;
            const ffloat zetaB = xyzzB.w;

            ffloat zeta_denom = zetaA_sq;
            zeta_denom = zeta_denom + zetaB * zetaB;
            const ffloat zeta = zetaA * zetaB * detail::fast_rsqrt<ffloat>(zeta_denom);

            const ffloat dx = xA - xB;
            const ffloat dy = yA - yB;
            const ffloat dz = zA - zB;
            ffloat r2 = dx * dx;
            r2 = r2 + dy * dy;
            r2 = r2 + dz * dz;

            const ffloat t = zeta * zeta * r2;
            const ffloat A_AB = two_inv_sqrt_pi * zeta * boys_f0(t);

            local_sum = local_sum + A_AB * rhs_vec[indexB];
        }

        lhs_vec[indexA] = lhs_vec[indexA] + local_sum;
    }
}

/* ------------------------------------------------------------------ */
/* Benchmark result                                                    */
/* ------------------------------------------------------------------ */
struct BenchResult {
    double time_ms;
    double gflops;
    double checksum;
};

/* ------------------------------------------------------------------ */
/* run_bench: run the kernel for a given type                          */
/*                                                                     */
/* 1. Allocate & fill device arrays with data converted from double    */
/* 2. Warm up with one un-timed launch                                 */
/* 3. Time NUM_ITERATIONS launches (cudaEvent)                         */
/* 4. Copy back lhs_vec, compute host-side checksum (sum of doubles)   */
/* ------------------------------------------------------------------ */
template <typename T, typename T4>
static BenchResult run_bench(size_t npoint,
                             const double* h_xyzz_flat,
                             const double* h_rhs,
                             int threadsPerBlock,
                             int numBlocks)
{
    const size_t xyzz_bytes = npoint * sizeof(T4);
    const size_t vec_bytes  = npoint * sizeof(T);

    T4* d_xyzz  = nullptr;
    T*  d_lhs   = nullptr;
    T*  d_rhs   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyzz, xyzz_bytes));
    CUDA_CHECK(cudaMalloc(&d_lhs,  vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs,  vec_bytes));

    T4* h_xyzz = new T4[npoint];
    T*  h_rhs_T = new T[npoint];
    T*  h_lhs_T = new T[npoint];

    for (size_t i = 0; i < npoint; ++i) {
        h_xyzz[i].x = static_cast<T>(h_xyzz_flat[4*i + 0]);
        h_xyzz[i].y = static_cast<T>(h_xyzz_flat[4*i + 1]);
        h_xyzz[i].z = static_cast<T>(h_xyzz_flat[4*i + 2]);
        h_xyzz[i].w = static_cast<T>(h_xyzz_flat[4*i + 3]);
        h_rhs_T[i]  = static_cast<T>(h_rhs[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_xyzz, h_xyzz, xyzz_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs,  h_rhs_T, vec_bytes,  cudaMemcpyHostToDevice));

    auto reset_lhs = [&]() {
        CUDA_CHECK(cudaMemset(d_lhs, 0, vec_bytes));
    };

    auto launch = [&]() {
        accumulate_A_offdiagonal_kernel<T, T4>
            <<<numBlocks, threadsPerBlock>>>(npoint, d_xyzz, d_lhs, d_rhs);
    };

    reset_lhs();
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        reset_lhs();
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

    CUDA_CHECK(cudaMemcpy(h_lhs_T, d_lhs, vec_bytes, cudaMemcpyDeviceToHost));

    long double checksum = 0.0L;
    for (size_t i = 0; i < npoint; ++i)
        checksum += (long double)(double)h_lhs_T[i];

    /*
     * Per (indexA, indexB) pair: ~20 FLOPs (distances, zeta, rsqrt, erf, fma).
     * Total pairs: npoint * (npoint - 1).
     */
    const double flops_per_pair = 20.0;
    const double total_flops = flops_per_pair * (double)npoint * ((double)npoint - 1.0);

    BenchResult res;
    res.time_ms  = (double)total_ms / NUM_ITERATIONS;
    res.gflops   = total_flops / (res.time_ms * 1e-3) / 1e9;
    res.checksum = (double)checksum;

    delete[] h_xyzz;
    delete[] h_rhs_T;
    delete[] h_lhs_T;
    CUDA_CHECK(cudaFree(d_xyzz));
    CUDA_CHECK(cudaFree(d_lhs));
    CUDA_CHECK(cudaFree(d_rhs));
    return res;
}

/* ------------------------------------------------------------------ */
/* run_bench_boys: ffloat-only, uses the Boys-F0-based kernel          */
/* ------------------------------------------------------------------ */
static BenchResult run_bench_boys(size_t npoint,
                                  const double* h_xyzz_flat,
                                  const double* h_rhs,
                                  int threadsPerBlock,
                                  int numBlocks)
{
    const size_t xyzz_bytes = npoint * sizeof(ffloat4_32a);
    const size_t vec_bytes  = npoint * sizeof(ffloat);

    ffloat4_32a* d_xyzz = nullptr;
    ffloat*      d_lhs  = nullptr;
    ffloat*      d_rhs  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xyzz, xyzz_bytes));
    CUDA_CHECK(cudaMalloc(&d_lhs,  vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs,  vec_bytes));

    ffloat4_32a* h_xyzz  = new ffloat4_32a[npoint];
    ffloat*      h_rhs_T = new ffloat[npoint];
    ffloat*      h_lhs_T = new ffloat[npoint];

    for (size_t i = 0; i < npoint; ++i) {
        h_xyzz[i].x = static_cast<ffloat>(h_xyzz_flat[4*i + 0]);
        h_xyzz[i].y = static_cast<ffloat>(h_xyzz_flat[4*i + 1]);
        h_xyzz[i].z = static_cast<ffloat>(h_xyzz_flat[4*i + 2]);
        h_xyzz[i].w = static_cast<ffloat>(h_xyzz_flat[4*i + 3]);
        h_rhs_T[i]  = static_cast<ffloat>(h_rhs[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_xyzz, h_xyzz, xyzz_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs,  h_rhs_T, vec_bytes,  cudaMemcpyHostToDevice));

    auto reset_lhs = [&]() {
        CUDA_CHECK(cudaMemset(d_lhs, 0, vec_bytes));
    };

    auto launch = [&]() {
        accumulate_A_offdiagonal_boys_kernel
            <<<numBlocks, threadsPerBlock>>>(npoint, d_xyzz, d_lhs, d_rhs);
    };

    reset_lhs();
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    float total_ms = 0;
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        reset_lhs();
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

    CUDA_CHECK(cudaMemcpy(h_lhs_T, d_lhs, vec_bytes, cudaMemcpyDeviceToHost));

    long double checksum = 0.0L;
    for (size_t i = 0; i < npoint; ++i)
        checksum += (long double)(double)h_lhs_T[i];

    const double flops_per_pair = 20.0;
    const double total_flops = flops_per_pair * (double)npoint * ((double)npoint - 1.0);

    BenchResult res;
    res.time_ms  = (double)total_ms / NUM_ITERATIONS;
    res.gflops   = total_flops / (res.time_ms * 1e-3) / 1e9;
    res.checksum = (double)checksum;

    delete[] h_xyzz;
    delete[] h_rhs_T;
    delete[] h_lhs_T;
    CUDA_CHECK(cudaFree(d_xyzz));
    CUDA_CHECK(cudaFree(d_lhs));
    CUDA_CHECK(cudaFree(d_rhs));
    return res;
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    int npoint = (argc > 1) ? atoi(argv[1]) : NPOINT;
    int threadsPerBlock = (argc > 2) ? atoi(argv[2]) : THREADS_PER_BLOCK;
    int numBlocks       = (argc > 3) ? atoi(argv[3]) : NUM_BLOCKS;

    if (npoint < 1 || npoint > 100000) {
        fprintf(stderr, "Error: npoint must be between 1 and 100000, got %d\n", npoint);
        return 1;
    }
    if (threadsPerBlock <= 0 || threadsPerBlock > 1024) {
        fprintf(stderr, "Error: threads per block must be between 1 and 1024, got %d\n",
                threadsPerBlock);
        return 1;
    }

    printf("PCM Off-Diagonal Matrix-Vector Product Benchmark\n");
    printf("=================================================\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n\n", prop.name);

    printf("npoint         = %d\n", npoint);
    printf("threadsPerBlock= %d\n", threadsPerBlock);
    printf("numBlocks      = %d\n", numBlocks);
    printf("timing iters   = %d\n", NUM_ITERATIONS);
    printf("pairs          = %lld\n\n",
           (long long)npoint * ((long long)npoint - 1));

    /*
     * Generate random input data:
     *   xyz  in [-10, 10]   (Bohr-scale molecular coordinates)
     *   zeta in [ 0.5, 5.0] (Gaussian exponents)
     *   rhs  in [-1, 1]     (charge-like vector)
     */
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist_xyz(-10.0, 10.0);
    std::uniform_real_distribution<double> dist_zeta(0.5, 5.0);
    std::uniform_real_distribution<double> dist_rhs(-1.0, 1.0);

    double* h_xyzz_flat = new double[4 * npoint];
    double* h_rhs       = new double[npoint];

    for (int i = 0; i < npoint; ++i) {
        h_xyzz_flat[4*i + 0] = dist_xyz(rng);
        h_xyzz_flat[4*i + 1] = dist_xyz(rng);
        h_xyzz_flat[4*i + 2] = dist_xyz(rng);
        h_xyzz_flat[4*i + 3] = dist_zeta(rng);
        h_rhs[i] = dist_rhs(rng);
    }

    printf("Running double precision (erf)...\n");
    BenchResult res_d = run_bench<double, double4_32a>(
        npoint, h_xyzz_flat, h_rhs, threadsPerBlock, numBlocks);

    printf("Running ffloat (erf, method=%s)...\n", METHOD_STR(__METHOD__));
    BenchResult res_ff = run_bench<ffloat, ffloat4_32a>(
        npoint, h_xyzz_flat, h_rhs, threadsPerBlock, numBlocks);

    printf("Running ffloat (boys_f0, method=%s)...\n\n", METHOD_STR(__METHOD__));
    BenchResult res_boys = run_bench_boys(
        npoint, h_xyzz_flat, h_rhs, threadsPerBlock, numBlocks);

    printf("=== Performance ===\n\n");
    printf("%-20s  %10s  %12s  %8s  %22s\n",
           "Type", "Time(ms)", "GFLOP/s", "Speedup", "Checksum");
    printf("%-20s  %10s  %12s  %8s  %22s\n",
           "--------------------", "----------", "------------",
           "--------", "----------------------");
    printf("%-20s  %10.3f  %12.2f  %8s  %22.15e\n",
           "double(erf)", res_d.time_ms, res_d.gflops, "1.00x", res_d.checksum);

    char ff_erf_label[32];
    snprintf(ff_erf_label, sizeof(ff_erf_label), "ffloat(%s,erf)", METHOD_STR(__METHOD__));
    printf("%-20s  %10.3f  %12.2f  %7.2fx  %22.15e\n",
           ff_erf_label, res_ff.time_ms, res_ff.gflops,
           res_d.time_ms / res_ff.time_ms, res_ff.checksum);

    char ff_boys_label[32];
    snprintf(ff_boys_label, sizeof(ff_boys_label), "ffloat(%s,boys)", METHOD_STR(__METHOD__));
    printf("%-20s  %10.3f  %12.2f  %7.2fx  %22.15e\n",
           ff_boys_label, res_boys.time_ms, res_boys.gflops,
           res_d.time_ms / res_boys.time_ms, res_boys.checksum);

    printf("\n=== Accuracy (vs double) ===\n\n");

    double rel_err_erf = (res_d.checksum != 0.0)
        ? fabs(res_ff.checksum - res_d.checksum) / fabs(res_d.checksum)
        : 0.0;
    double rel_err_boys = (res_d.checksum != 0.0)
        ? fabs(res_boys.checksum - res_d.checksum) / fabs(res_d.checksum)
        : 0.0;

    printf("  Checksum (double,erf)  = %.15e\n", res_d.checksum);
    printf("  Checksum (ffloat,erf)  = %.15e   |rel err| = %.2e  %s\n",
           res_ff.checksum, rel_err_erf, rel_err_erf < 1e-6 ? "OK" : "FAIL");
    printf("  Checksum (ffloat,boys) = %.15e   |rel err| = %.2e  %s\n",
           res_boys.checksum, rel_err_boys, rel_err_boys < 1e-6 ? "OK" : "FAIL");

    delete[] h_xyzz_flat;
    delete[] h_rhs;
    return 0;
}

#endif /* __CUDACC__ */
