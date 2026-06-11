/*!
	\file   hpc_exp_error_gpu.cu
	\brief  CUDA implementation of the composite-exp benchmark.

	Performance-loop methodology:
	  * Fixed grid: numBlocks * threadsPerBlock CUDA threads, regardless of
	    numSamples. Each thread walks chains via a grid-stride loop.
	  * Inner `reps` loop INSIDE the kernel: one launch performs `reps`
	    passes over the entire numSamples set, amortising launch overhead
	    and producing a long-running kernel suitable for stable timing.
	  * One warm-up launch (also captures per-sample outputs for accuracy
	    on rep == 0).
	  * `numIterations` timed launches; chrono around cudaDeviceSynchronize;
	    final time = average per launch.

	Throughput = (numSamples * chainLen * reps) / avgKernelSeconds.

	The same kernel is reused at variable depth for the alpha sweep with
	reps == 1 (no timing).

	Defines HPC_EXP_ERROR_HAVE_CUDA so the host TU's stubs are excluded.
*/

#define HPC_EXP_ERROR_HAVE_CUDA 1

#include "hpc_exp_error.h"
#include "hpc_exp_error_kernels.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                          \
	do {                                                                          \
		cudaError_t _err = (call);                                                \
		if (_err != cudaSuccess) {                                                \
			std::fprintf(stderr,                                                  \
				"[hpc_exp_error_gpu] CUDA error at %s:%d : %s\n",                 \
				__FILE__, __LINE__, cudaGetErrorString(_err));                    \
			std::abort();                                                         \
		}                                                                         \
	} while (0)

namespace fix::hpc {

bool gpuSupportCompiledIn() { return true; }

bool gpuRuntimeAvailable()
{
	int devCount = 0;
	const cudaError_t e = cudaGetDeviceCount(&devCount);
	if (e != cudaSuccess)
	{
		// Reset the sticky error so subsequent CUDA calls (in case the caller
		// proceeds anyway) start from a clean slate.
		(void)cudaGetLastError();
		return false;
	}
	return devCount > 0;
}

namespace {

// ---------------------------------------------------------------------------
// Conversion kernel: double[] -> T[]. Run ONCE per variant, outside the
// timed loop, so that double<->T conversions (which are non-trivial for
// fp32mp2 and even costly for `float` on FP64-throttled GPUs) do not
// pollute the perf measurement.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void convertToTKernel(const double* __restrict__ src,
                                 T*            __restrict__ dst,
                                 long long                  n)
{
	const long long tid    = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
	for (long long i = tid; i < n; i += stride)
		dst[i] = T(src[i]);
}

// ---------------------------------------------------------------------------
// Kernel: fixed grid + grid-stride loop over chains, with an inner `reps`
// loop. Each thread processes (numSamples / TOTAL_THREADS) chains per rep
// (rounded up), and `reps` passes are performed within a single launch.
//
// xs is laid out COLUMN-MAJOR / SoA on device, ALREADY in working type T:
// xs[i * numSamples + s] is the i-th exponent of the s-th sample, stored
// natively as T. Pre-conversion (done outside the timed region) means this
// kernel measures pure compute (exp + mul/add in T) plus a coalesced T-load,
// with no implicit double->T conversion on the hot path.
// ---------------------------------------------------------------------------
template <typename T, KernelType K>
__global__ void runChainsKernel(const T*    __restrict__ xs,
                                int                      depth,
                                int                      numSamples,
                                int                      reps,
                                double*     __restrict__ out)
{
	const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = gridDim.x * blockDim.x;

	for (int rep = 0; rep < reps; ++rep)
	{
		for (int s = tid; s < numSamples; s += stride)
		{
			const T* base = xs + s;               // SoA column 0 of sample s
			T r;
			if constexpr (K == KernelType::Compose)
				r = kernel_compose<T>(base, depth, numSamples);
			else
				r = kernel_expsum<T>(base, depth, numSamples);
			if (rep == 0 && out)
				out[s] = static_cast<double>(r);  // single per-chain cast, ignorable
		}
	}
}

// Launch the kernel once with the given depth/reps. Used both for the perf
// loop (reps from cfg) and for the alpha sweep (reps == 1).
template <typename T, KernelType K>
void launchOnce(const ExpErrorConfig& cfg,
                const T*              d_xs_T,
                int                   depth,
                int                   reps,
                double*               d_out)
{
	runChainsKernel<T, K><<<cfg.numBlocks, cfg.threadsPerBlock>>>(
		d_xs_T, depth, cfg.numSamples, reps, d_out);
	CUDA_CHECK(cudaGetLastError());
}

// Run one (T, K) perf variant: warm-up + numIterations chrono-timed launches,
// each launch internally repeating `cfg.reps` passes over the data.
// Returns the average per-launch wall-clock seconds. Copies the warm-up's
// per-sample outputs (= a deterministic "rep == 0" pass) to h_out.
template <typename T, KernelType K>
double runOneVariantTimed(const ExpErrorConfig& cfg,
                          const T*              d_xs_T,
                          double*               d_out,
                          double*               h_out)
{
	// Warm-up. This call also populates d_out (rep == 0) so accuracy can
	// be evaluated against the same kernel's output. Subsequent timed
	// launches pass d_out=nullptr to skip the per-sample store.
	launchOnce<T, K>(cfg, d_xs_T, cfg.chainLen, cfg.reps, d_out);
	CUDA_CHECK(cudaDeviceSynchronize());

	double totalMs = 0.0;
	for (int it = 0; it < cfg.numIterations; ++it)
	{
		const auto t0 = std::chrono::high_resolution_clock::now();
		launchOnce<T, K>(cfg, d_xs_T, cfg.chainLen, cfg.reps, /*d_out=*/nullptr);
		CUDA_CHECK(cudaDeviceSynchronize());
		const auto t1 = std::chrono::high_resolution_clock::now();
		totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
	}

	CUDA_CHECK(cudaMemcpy(h_out, d_out,
	                      sizeof(double) * cfg.numSamples,
	                      cudaMemcpyDeviceToHost));

	const double avgMs = totalMs / static_cast<double>(cfg.numIterations);
	return avgMs * 1e-3;
}

// Allocate d_xs_T, run the conversion kernel, and synchronise. Caller frees.
template <typename T>
T* allocAndConvert(const double* d_xs_double, long long total)
{
	T* d_xs_T = nullptr;
	CUDA_CHECK(cudaMalloc(&d_xs_T, sizeof(T) * static_cast<size_t>(total)));

	// Big linear conversion - 256 threads/block, enough blocks to fill the
	// SM count generously.
	constexpr int TPB = 256;
	const long long blocks = (total + TPB - 1) / TPB;
	const int gridX = static_cast<int>(blocks > 65535 ? 65535 : blocks);
	convertToTKernel<T><<<gridX, TPB>>>(d_xs_double, d_xs_T, total);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	return d_xs_T;
}

// Aggregate per-sample double approximations against a contiguous reference.
void aggregateBlock(const double*  approx,
                    const double*  ref,
                    int            n,
                    double&        maxAbs,
                    double&        meanAbs,
                    double&        rmsAbs,
                    double&        maxRel,
                    double&        meanRel)
{
	double sumAbs = 0.0, sumRel = 0.0, sumSq = 0.0;
	maxAbs = 0.0; maxRel = 0.0;
	for (int s = 0; s < n; ++s)
	{
		const double absErr = std::abs(approx[s] - ref[s]);
		const double relErr = relErrFp128(approx[s], ref[s]);
		sumAbs += absErr;
		sumRel += relErr;
		sumSq  += absErr * absErr;
		if (absErr > maxAbs) maxAbs = absErr;
		if (relErr > maxRel) maxRel = relErr;
	}
	const double inv = 1.0 / static_cast<double>(n);
	meanAbs = sumAbs * inv;
	rmsAbs  = std::sqrt(sumSq * inv);
	meanRel = sumRel * inv;
}

double fitLogLogSlopeImpl(const std::vector<DepthErrorRow>& rows, int& nOut)
{
	double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
	int    n = 0;
	for (const auto& r : rows)
	{
		if (r.meanRelError > 0.0 && r.depth > 0)
		{
			const double x = std::log(static_cast<double>(r.depth));
			const double y = std::log(r.meanRelError);
			sumX  += x; sumY += y; sumXX += x*x; sumXY += x*y;
			++n;
		}
	}
	nOut = n;
	if (n < 2) return 0.0;
	const double denom = static_cast<double>(n) * sumXX - sumX * sumX;
	if (denom == 0.0) return 0.0;
	return (static_cast<double>(n) * sumXY - sumX * sumY) / denom;
}

} // anonymous

// ---------------------------------------------------------------------------
// Fixed-chainLen GPU benchmark.
// ---------------------------------------------------------------------------
std::vector<VariantResult> runGpuBenchmark(const ExpErrorConfig& cfg,
                                           const double*         xs,
                                           const double*         refExpSum)
{
	std::vector<VariantResult> out;
	out.reserve(6);

	int devCount = 0;
	if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0)
	{
		std::fprintf(stderr, "[hpc_exp_error_gpu] no CUDA device available\n");
		return out;
	}

	const long long totalExps = static_cast<long long>(cfg.numSamples) *
	                            static_cast<long long>(cfg.chainLen);

	double *d_xs = nullptr, *d_out = nullptr;
	CUDA_CHECK(cudaMalloc(&d_xs,  sizeof(double) * totalExps));
	CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * cfg.numSamples));
	CUDA_CHECK(cudaMemcpy(d_xs, xs, sizeof(double) * totalExps,
	                      cudaMemcpyHostToDevice));

	std::vector<double> h_out(cfg.numSamples);

	// Pre-converted T-typed input buffers. Conversion done OUTSIDE the
	// per-variant timed loop so that double <-> T conversion cost (which
	// is non-trivial for fp32mp2 and even for `float` on FP64-throttled
	// hardware) is invariant and not part of the perf measurement.
	float*     d_xs_f  = allocAndConvert<float    >(d_xs, totalExps);
	double*    d_xs_d  = allocAndConvert<double   >(d_xs, totalExps); // copy is cheap; keeps the code uniform
	ffloat*    d_xs_ff = allocAndConvert<ffloat   >(d_xs, totalExps);

	auto runOne = [&](WorkingType wt, KernelType kt, auto runner, auto* d_xs_T) {
		const double secs = runner(cfg, d_xs_T, d_out, h_out.data());
		VariantResult r{};
		r.wtype     = wt;
		r.ktype     = kt;
		r.ranOnGpu  = true;
		r.timeSec   = secs;
		// Work per launch = numSamples * reps chains * chainLen ops/chain.
		// We use `numChains` as the chain-count denominator;
		// `numChains * chainLen / timeSec` then yields exps/s; the host
		// reporter scales by 1e-9 to print Gexps/s.
		r.numChains = static_cast<long long>(cfg.numSamples) *
		              static_cast<long long>(cfg.reps);
		r.chainLen  = cfg.chainLen;
		aggregateBlock(h_out.data(), refExpSum, cfg.numSamples,
		               r.maxAbsError, r.meanAbsError, r.rmsAbsError,
		               r.maxRelError, r.meanRelError);
		out.push_back(r);
	};

	runOne(WorkingType::Float,  KernelType::Compose, runOneVariantTimed<float,     KernelType::Compose>, d_xs_f );
	runOne(WorkingType::Float,  KernelType::ExpSum,  runOneVariantTimed<float,     KernelType::ExpSum >, d_xs_f );
	runOne(WorkingType::Double, KernelType::Compose, runOneVariantTimed<double,    KernelType::Compose>, d_xs_d );
	runOne(WorkingType::Double, KernelType::ExpSum,  runOneVariantTimed<double,    KernelType::ExpSum >, d_xs_d );
	runOne(WorkingType::Fpmp2,  KernelType::Compose, runOneVariantTimed<ffloat,    KernelType::Compose>, d_xs_ff);
	runOne(WorkingType::Fpmp2,  KernelType::ExpSum,  runOneVariantTimed<ffloat,    KernelType::ExpSum >, d_xs_ff);

	CUDA_CHECK(cudaFree(d_xs_f));
	CUDA_CHECK(cudaFree(d_xs_d));
	CUDA_CHECK(cudaFree(d_xs_ff));
	CUDA_CHECK(cudaFree(d_xs));
	CUDA_CHECK(cudaFree(d_out));
	return out;
}

// ---------------------------------------------------------------------------
// GPU alpha sweep: for each (T, K), launch the kernel at every requested
// depth (no timing), aggregate per-sample errors against the matching
// fp128 reference column, and fit a log-log slope at the end.
// ---------------------------------------------------------------------------
std::vector<VariantSweep> runGpuAlphaSweep(const ExpErrorConfig&   cfg,
                                           const std::vector<int>& depths,
                                           const double*           xs,
                                           const double*           refExpSum2D)
{
	std::vector<VariantSweep> out;
	out.reserve(6);

	if (depths.empty()) return out;

	int devCount = 0;
	if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0)
	{
		std::fprintf(stderr, "[hpc_exp_error_gpu] no CUDA device available\n");
		return out;
	}

	const long long totalExps = static_cast<long long>(cfg.numSamples) *
	                            static_cast<long long>(cfg.chainLen);
	const int nDepth = static_cast<int>(depths.size());

	double *d_xs = nullptr, *d_out = nullptr;
	CUDA_CHECK(cudaMalloc(&d_xs,  sizeof(double) * totalExps));
	CUDA_CHECK(cudaMalloc(&d_out, sizeof(double) * cfg.numSamples));
	CUDA_CHECK(cudaMemcpy(d_xs, xs, sizeof(double) * totalExps,
	                      cudaMemcpyHostToDevice));

	std::vector<double> h_out(cfg.numSamples);
	std::vector<double> refCol(cfg.numSamples);

	// Pre-converted T-typed inputs (alpha sweep is an accuracy probe; the
	// conversion would not affect timing, but using the same path keeps
	// behaviour identical to the perf benchmark).
	float*     d_xs_f  = allocAndConvert<float    >(d_xs, totalExps);
	double*    d_xs_d  = allocAndConvert<double   >(d_xs, totalExps);
	ffloat*    d_xs_ff = allocAndConvert<ffloat   >(d_xs, totalExps);

	auto runVariant = [&](WorkingType wt, KernelType kt, auto launcher, auto* d_xs_T) {
		VariantSweep sw{};
		sw.wtype = wt; sw.ktype = kt; sw.ranOnGpu = true;
		sw.rows.reserve(nDepth);

		for (int di = 0; di < nDepth; ++di)
		{
			const int d = depths[di];

			// Alpha sweep is purely an accuracy probe -> reps == 1.
			launcher(cfg, d_xs_T, d, /*reps=*/1, d_out);
			CUDA_CHECK(cudaDeviceSynchronize());
			CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
			                      sizeof(double) * cfg.numSamples,
			                      cudaMemcpyDeviceToHost));

			for (int s = 0; s < cfg.numSamples; ++s)
				refCol[s] = refExpSum2D[static_cast<size_t>(s) * nDepth + di];

			DepthErrorRow row{};
			row.depth = d;
			aggregateBlock(h_out.data(), refCol.data(), cfg.numSamples,
			               row.maxAbsError, row.meanAbsError, row.rmsAbsError,
			               row.maxRelError, row.meanRelError);
			sw.rows.push_back(row);
		}

		sw.alpha = fitLogLogSlopeImpl(sw.rows, sw.nFitPoints);
		out.push_back(std::move(sw));
	};

	runVariant(WorkingType::Float,  KernelType::Compose, launchOnce<float,     KernelType::Compose>, d_xs_f );
	runVariant(WorkingType::Float,  KernelType::ExpSum,  launchOnce<float,     KernelType::ExpSum >, d_xs_f );
	runVariant(WorkingType::Double, KernelType::Compose, launchOnce<double,    KernelType::Compose>, d_xs_d );
	runVariant(WorkingType::Double, KernelType::ExpSum,  launchOnce<double,    KernelType::ExpSum >, d_xs_d );
	runVariant(WorkingType::Fpmp2,  KernelType::Compose, launchOnce<ffloat,    KernelType::Compose>, d_xs_ff);
	runVariant(WorkingType::Fpmp2,  KernelType::ExpSum,  launchOnce<ffloat,    KernelType::ExpSum >, d_xs_ff);

	CUDA_CHECK(cudaFree(d_xs_f));
	CUDA_CHECK(cudaFree(d_xs_d));
	CUDA_CHECK(cudaFree(d_xs_ff));
	CUDA_CHECK(cudaFree(d_xs));
	CUDA_CHECK(cudaFree(d_out));
	return out;
}

} // namespace fix::hpc
