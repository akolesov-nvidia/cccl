/*!
	\file   hpc_exp_error_host.cpp
	\brief  Host-side driver for the composite-exp benchmark.

	Responsibilities:
	  * RNG (kept entirely outside the kernels).
	  * CPU implementation of the same eight (T x kernel) variants. The CPU
	    benchmark is no longer wired into the default flow (the project is
	    GPU-only by directive) but the entry point is kept available for
	    offline accuracy checks and as a fallback when no GPU is present.
	  * CPU implementation of the alpha sweep (used as a fallback).
	  * All shared reporting routines:
	      - reportVariants               (per-variant accuracy stats)
	      - reportSpeedupsVsBaseline     (relative to (Double, ExpSum))
	      - reportAlphaSweep             (per-variant depth table + alpha fit)
	  * Host-only stubs of runGpuBenchmark / runGpuAlphaSweep / gpuSupportCompiledIn
	    that get replaced by the .cu TU when TARGET=device.
*/

#include "hpc_exp_error.h"
#include "hpc_exp_error_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fix::hpc {

const char* toString(WorkingType t)
{
	switch (t) {
		case WorkingType::Float:  return "float";
		case WorkingType::Double: return "double";
		case WorkingType::Fpmp2:  return "fp32mp2";
	}
	return "?";
}

const char* toString(KernelType k)
{
	switch (k) {
		case KernelType::Compose: return "compose";
		case KernelType::ExpSum:  return "expsum";
	}
	return "?";
}

// ---------------------------------------------------------------------------
// Random exponent generation (RNG is here, not inside any kernel).
//
// Layout: COLUMN-MAJOR / SoA. The element at "exponent index i, sample s" is
// stored at out[(long long)i * numSamples + s]. With this layout, on the GPU
// the 32 threads of a warp at a fixed iteration `i` access 32 consecutive
// doubles -> a single coalesced 128 / 256-byte transaction services the whole
// warp, instead of 32 separate cache lines as in a row-major (AoS) layout.
//
// On Ada-class GPUs (FP64:FP32 = 1:64) the row-major layout is so badly
// uncoalesced that all working types end up bandwidth-limited and look the
// same speed; with this SoA layout the kernels become compute-bound and
// fp32mp2 starts beating double as expected.
//
// To keep the per-(sample, index) random sequence identical to a row-major
// generator (same seed -> same numerical experiment), we still walk samples
// in the outer loop but place each draw at the SoA address.
// ---------------------------------------------------------------------------
void generateExponents(const ExpErrorConfig& cfg, double* out)
{
	std::mt19937_64 rng(cfg.seed);
	const double lo = cfg.negativeOnly ? -cfg.dtMax :  cfg.dtMin;
	const double hi = cfg.negativeOnly ? -cfg.dtMin :  cfg.dtMax;
	std::uniform_real_distribution<double> dist(lo, hi);

	const int N  = cfg.chainLen;
	const int NS = cfg.numSamples;
	for (int s = 0; s < NS; ++s)
	{
		for (int i = 0; i < N; ++i)
		{
			out[static_cast<long long>(i) * NS + s] = dist(rng);
		}
	}
}

// ---------------------------------------------------------------------------
// CPU helpers (templated kernel runners and error aggregation).
// ---------------------------------------------------------------------------
namespace {

template <typename T, KernelType K>
inline T runOneChain(const T* base, int n, int stride)
{
	if constexpr (K == KernelType::Compose)
		return kernel_compose<T>(base, n, stride);
	else
		return kernel_expsum<T>(base, n, stride);
}

// Pre-convert SoA double[] -> T[]. Run ONCE per variant, OUTSIDE the timed
// region, so that double <-> T conversion cost is invariant and not part of
// the perf measurement.
template <typename T>
std::vector<T> convertExponents(const double* xs, long long total)
{
	std::vector<T> out(static_cast<size_t>(total));
	for (long long k = 0; k < total; ++k)
		out[static_cast<size_t>(k)] = T(xs[k]);
	return out;
}

// Single-pass CPU runner over numSamples chains. `writeOut` controls whether
// per-sample results are stored (used to mirror the GPU's "rep == 0 stores,
// other reps don't" semantics for fair timing).
//
// OpenMP-parallel across the per-sample loop: each chain is independent
// (no cross-chain reduction). On TARGET=host the host wall-clock numbers
// are not reported in the summary table anyway -> the OMP parallelisation
// is purely a "make the warmup + the alpha-sweep finish in a reasonable
// time at GPU-aligned (numSamples, chainLen)" optimisation. When the TU
// is built without -fopenmp the pragma degrades silently and the loop is
// sequential, which is fine for small (numSamples * chainLen).
template <typename T, KernelType K>
void runCpuOnePass(const ExpErrorConfig& cfg,
                   const T*              xs_T,
                   double*               out,
                   bool                  writeOut)
{
	const int N  = cfg.chainLen;
	const int NS = cfg.numSamples;
#if defined(_OPENMP)
	#pragma omp parallel for schedule(static)
#endif
	for (int s = 0; s < NS; ++s)
	{
		const T* base = xs_T + s; // column 0 of sample s in SoA
		const T r = runOneChain<T, K>(base, N, NS);
		if (writeOut) out[s] = static_cast<double>(r);
	}
}

// CPU performance loop mirroring the GPU pattern (warm-up then numIterations
// timed runs, each timed run executing `reps` passes over the data). Returns
// the average per-launch wall-clock seconds. The warm-up populates `out` so
// accuracy stats can be evaluated against the same kernel's output.
//
// xs is passed as const double*; the typed input buffer is built ONCE here,
// before the timed region, so timing measures only the working-type kernel
// arithmetic.
template <typename T, KernelType K>
double runCpuVariant(const ExpErrorConfig& cfg,
                     const double*         xs,
                     double*               out)
{
	const long long total = static_cast<long long>(cfg.numSamples) *
	                        static_cast<long long>(cfg.chainLen);
	std::vector<T> xs_T = convertExponents<T>(xs, total);

	// Warm-up: 1 pass, writes `out`.
	runCpuOnePass<T, K>(cfg, xs_T.data(), out, /*writeOut=*/true);

	double totalSec = 0.0;
	for (int it = 0; it < cfg.numIterations; ++it)
	{
		const auto t0 = std::chrono::steady_clock::now();
		for (int rep = 0; rep < cfg.reps; ++rep)
			runCpuOnePass<T, K>(cfg, xs_T.data(), out, /*writeOut=*/false);
		const auto t1 = std::chrono::steady_clock::now();
		totalSec += std::chrono::duration<double>(t1 - t0).count();
	}
	return totalSec / static_cast<double>(cfg.numIterations);
}

// Single-pass CPU runner used by the alpha sweep at a custom depth (no
// timing). The caller pre-converts xs to T to keep this consistent with the
// perf path. OMP-parallel across the per-sample loop for the same reasons
// as runCpuOnePass.
template <typename T, KernelType K>
void runCpuVariantAtDepth(const ExpErrorConfig& cfg,
                          const T*              xs_T,
                          int                   depth,
                          double*               out)
{
	const int NS = cfg.numSamples;
#if defined(_OPENMP)
	#pragma omp parallel for schedule(static)
#endif
	for (int s = 0; s < NS; ++s)
	{
		const T* base = xs_T + s;
		out[s] = static_cast<double>(runOneChain<T, K>(base, depth, NS));
	}
}

void aggregate(const double*  approx,
               const double*  ref,
               int            numSamples,
               double&        maxAbs,
               double&        meanAbs,
               double&        rmsAbs,
               double&        maxRel,
               double&        meanRel)
{
	double sumAbs = 0.0, sumRel = 0.0, sumSq = 0.0;
	maxAbs = 0.0; maxRel = 0.0;
	for (int s = 0; s < numSamples; ++s)
	{
		const double absErr = std::abs(approx[s] - ref[s]);
		const double relErr = relErrFp128(approx[s], ref[s]);
		sumAbs += absErr;
		sumRel += relErr;
		sumSq  += absErr * absErr;
		if (absErr > maxAbs) maxAbs = absErr;
		if (relErr > maxRel) maxRel = relErr;
	}
	const double inv = 1.0 / static_cast<double>(numSamples);
	meanAbs = sumAbs * inv;
	rmsAbs  = std::sqrt(sumSq * inv);
	meanRel = sumRel * inv;
}

void aggregateInto(const double* approx, const double* ref, int n, VariantResult& r)
{
	aggregate(approx, ref, n,
	          r.maxAbsError, r.meanAbsError, r.rmsAbsError,
	          r.maxRelError, r.meanRelError);
}

void aggregateInto(const double* approx, const double* ref, int n, DepthErrorRow& r)
{
	aggregate(approx, ref, n,
	          r.maxAbsError, r.meanAbsError, r.rmsAbsError,
	          r.maxRelError, r.meanRelError);
}

double thrPerExp(const VariantResult& r)
{
	if (r.timeSec <= 0.0) return 0.0;
	return static_cast<double>(r.numChains) *
	       static_cast<double>(r.chainLen) / r.timeSec;
}

// Log-log linear regression. Skips depths where meanRelErr <= 0.
double fitLogLogSlope(const std::vector<DepthErrorRow>& rows, int& nFitOut)
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
	nFitOut = n;
	if (n < 2) return 0.0;
	const double denom = static_cast<double>(n) * sumXX - sumX * sumX;
	if (denom == 0.0) return 0.0;
	return (static_cast<double>(n) * sumXY - sumX * sumY) / denom;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// CPU benchmark (fixed chainLen).
// ---------------------------------------------------------------------------
std::vector<VariantResult> runCpuBenchmark(const ExpErrorConfig& cfg,
                                           const double*         xs,
                                           const double*         refExpSum)
{
	std::vector<VariantResult> out;
	out.reserve(6);
	std::vector<double> approx(cfg.numSamples);

	auto runOne = [&](WorkingType wt, KernelType kt, auto runner)
	{
		const double secs = runner(cfg, xs, approx.data());
		VariantResult r{};
		r.wtype = wt; r.ktype = kt; r.ranOnGpu = false;
		r.timeSec   = secs;
		r.numChains = static_cast<long long>(cfg.numSamples) *
		              static_cast<long long>(cfg.reps);
		r.chainLen  = cfg.chainLen;
		aggregateInto(approx.data(), refExpSum, cfg.numSamples, r);
		out.push_back(r);
	};

	runOne(WorkingType::Float,  KernelType::Compose, runCpuVariant<float,     KernelType::Compose>);
	runOne(WorkingType::Float,  KernelType::ExpSum,  runCpuVariant<float,     KernelType::ExpSum >);
	runOne(WorkingType::Double, KernelType::Compose, runCpuVariant<double,    KernelType::Compose>);
	runOne(WorkingType::Double, KernelType::ExpSum,  runCpuVariant<double,    KernelType::ExpSum >);
	runOne(WorkingType::Fpmp2,  KernelType::Compose, runCpuVariant<ffloat,    KernelType::Compose>);
	runOne(WorkingType::Fpmp2,  KernelType::ExpSum,  runCpuVariant<ffloat,    KernelType::ExpSum >);
	return out;
}

// ---------------------------------------------------------------------------
// CPU alpha sweep.
// ---------------------------------------------------------------------------
std::vector<VariantSweep> runCpuAlphaSweep(const ExpErrorConfig&   cfg,
                                           const std::vector<int>& depths,
                                           const double*           xs,
                                           const double*           refExpSum2D)
{
	std::vector<VariantSweep> out;
	out.reserve(6);
	std::vector<double> approx(cfg.numSamples);

	const int       nDepth = static_cast<int>(depths.size());
	const long long total  = static_cast<long long>(cfg.numSamples) *
	                         static_cast<long long>(cfg.chainLen);

	// Pre-converted T-typed inputs (alpha sweep is an accuracy probe; the
	// conversion would not affect timing, but using the same path keeps
	// behaviour identical to the perf benchmark).
	std::vector<float    > xs_f  = convertExponents<float    >(xs, total);
	std::vector<double   > xs_d  = convertExponents<double   >(xs, total);
	std::vector<ffloat   > xs_ff = convertExponents<ffloat   >(xs, total);

	auto runVariant = [&](WorkingType wt, KernelType kt, auto runner, const auto* xs_T)
	{
		VariantSweep sw{};
		sw.wtype = wt; sw.ktype = kt; sw.ranOnGpu = false;
		sw.rows.reserve(nDepth);

		std::vector<double> refCol(cfg.numSamples);
		for (int di = 0; di < nDepth; ++di)
		{
			runner(cfg, xs_T, depths[di], approx.data());
			DepthErrorRow row{};
			row.depth = depths[di];
			for (int s = 0; s < cfg.numSamples; ++s)
				refCol[s] = refExpSum2D[static_cast<size_t>(s) * nDepth + di];
			aggregateInto(approx.data(), refCol.data(), cfg.numSamples, row);
			sw.rows.push_back(row);
		}

		sw.alpha = fitLogLogSlope(sw.rows, sw.nFitPoints);
		out.push_back(std::move(sw));
	};

	runVariant(WorkingType::Float,  KernelType::Compose, runCpuVariantAtDepth<float,     KernelType::Compose>, xs_f .data());
	runVariant(WorkingType::Float,  KernelType::ExpSum,  runCpuVariantAtDepth<float,     KernelType::ExpSum >, xs_f .data());
	runVariant(WorkingType::Double, KernelType::Compose, runCpuVariantAtDepth<double,    KernelType::Compose>, xs_d .data());
	runVariant(WorkingType::Double, KernelType::ExpSum,  runCpuVariantAtDepth<double,    KernelType::ExpSum >, xs_d .data());
	runVariant(WorkingType::Fpmp2,  KernelType::Compose, runCpuVariantAtDepth<ffloat,    KernelType::Compose>, xs_ff.data());
	runVariant(WorkingType::Fpmp2,  KernelType::ExpSum,  runCpuVariantAtDepth<ffloat,    KernelType::ExpSum >, xs_ff.data());
	return out;
}

// ---------------------------------------------------------------------------
// GPU stubs (overridden by hpc_exp_error_gpu.cu when TARGET=device).
// ---------------------------------------------------------------------------
#ifndef HPC_EXP_ERROR_HAVE_CUDA
bool gpuSupportCompiledIn() { return false; }
bool gpuRuntimeAvailable() { return false; }

std::vector<VariantResult> runGpuBenchmark(const ExpErrorConfig&,
                                           const double*,
                                           const double*)
{
	return {};
}

std::vector<VariantSweep> runGpuAlphaSweep(const ExpErrorConfig&,
                                           const std::vector<int>&,
                                           const double*,
                                           const double*)
{
	return {};
}
#endif

// ---------------------------------------------------------------------------
// Reporting.
// ---------------------------------------------------------------------------
void reportVariants(std::ostream&                     os,
                    const ExpErrorConfig&             cfg,
                    const std::vector<VariantResult>& results,
                    const char*                       heading)
{
	os << "\n=== " << heading << " ===\n";
	os << "config: numSamples=" << cfg.numSamples
	   << " chainLen="          << cfg.chainLen
	   << " dt in ["            << cfg.dtMin << "," << cfg.dtMax << "]"
	   << " negOnly="           << (cfg.negativeOnly ? 1 : 0)
	   << " reps="              << cfg.reps
	   << " numIterations="     << cfg.numIterations
	   << " grid=("             << cfg.numBlocks
	   << "x"                   << cfg.threadsPerBlock << ")"
	   << "\n";

	std::ios_base::fmtflags f = os.flags();
	std::streamsize p = os.precision();

	os << std::left;
	os.width(14); os << "dtype";
	os.width(10); os << "kernel";
	os << std::right;
	os.width(14); os << "maxAbsErr";
	os.width(14); os << "meanAbsErr";
	os.width(14); os << "rmsAbsErr";
	os.width(14); os << "maxRelErr";
	os.width(14); os << "meanRelErr";
	os << "\n";
	os << std::string(14 + 10 + 14 * 5, '-') << "\n";

	os << std::scientific << std::setprecision(3);
	for (const auto& r : results)
	{
		os << std::left;
		os.width(14); os << toString(r.wtype);
		os.width(10); os << toString(r.ktype);
		os << std::right;
		os.width(14); os << r.maxAbsError;
		os.width(14); os << r.meanAbsError;
		os.width(14); os << r.rmsAbsError;
		os.width(14); os << r.maxRelError;
		os.width(14); os << r.meanRelError;
		os << "\n";
	}

	os.flags(f);
	os.precision(p);
}

namespace {

const char* shortAlphaConclusion(double a)
{
	if      (a < 0.3) return "sublinear";
	else if (a < 0.7) return "sqrt(N)";
	else if (a < 1.3) return "linear";
	else              return "super-linear";
}

const VariantSweep* findSweep(const std::vector<VariantSweep>& sweeps,
                              WorkingType wt, KernelType kt)
{
	for (const auto& s : sweeps)
		if (s.wtype == wt && s.ktype == kt) return &s;
	return nullptr;
}

} // anonymous

void reportSpeedupsVsBaseline(std::ostream&                     os,
                              const std::vector<VariantResult>& results,
                              const std::vector<VariantSweep>&  sweeps,
                              const char*                       heading)
{
	// Per-row timing is meaningful only on GPU; on host the wall-clock
	// numbers are dominated by noise (sub-ms regimes, dtype-specific
	// vectorization quirks) and would mislead the reader. We strip the
	// time/throughput/vs_fp64 columns whenever no row ran on GPU.
	bool hasGpu = false;
	for (const auto& r : results) if (r.ranOnGpu) { hasGpu = true; break; }

	os << "\n=== " << heading << " ===\n";
	if (results.empty())
	{
		os << "(no results)\n";
		return;
	}

	const bool haveSweeps = !sweeps.empty();

	// Per-kernel native FP64 baselines: (Double, Compose) and (Double,
	// ExpSum). Apples-to-apples ("how much faster/slower than native FP64
	// on the SAME kernel?"), unaffected by the compose-vs-expsum gap.
	double doubleComposeThr = 0.0;
	double doubleExpsumThr  = 0.0;
	if (hasGpu)
	{
		for (const auto& r : results)
		{
			if (r.wtype != WorkingType::Double) continue;
			const double thr = thrPerExp(r);
			if (r.ktype == KernelType::Compose) doubleComposeThr = thr;
			else                                doubleExpsumThr  = thr;
		}
		if (doubleComposeThr <= 0.0 && doubleExpsumThr <= 0.0)
		{
			os << "(no double rows present, cannot compute vs_fp64)\n";
			return;
		}
	}

	std::ios_base::fmtflags f = os.flags();
	std::streamsize p = os.precision();

	os << std::left;
	os.width(14); os << "dtype";
	os.width(10); os << "kernel";
	os << std::right;
	int dividerWidth = 14 + 10;
	if (hasGpu)
	{
		os.width(13); os << "time[ms]";
		os.width(15); os << "Gexps/s";
		os.width(13); os << "vs_fp64";
		dividerWidth += 13 + 15 + 13;
	}
	os.width(15); os << "maxRelErr";
	dividerWidth += 15;
	if (haveSweeps)
	{
		os.width(10); os << "alpha";
		os << "  ";
		os << std::left;
		os.width(14); os << "conclusion";
		os << std::right;
		dividerWidth += 10 + 2 + 14;
	}
	os << "\n";
	os << std::string(static_cast<size_t>(dividerWidth), '-') << "\n";

	for (const auto& r : results)
	{
		os << std::left;
		os.width(14); os << toString(r.wtype);
		os.width(10); os << toString(r.ktype);
		os << std::right;

		if (hasGpu)
		{
			const double thr = thrPerExp(r);
			const double sameKernelDoubleThr =
				(r.ktype == KernelType::Compose) ? doubleComposeThr
				                                 : doubleExpsumThr;
			const bool sameKernelBaseline = (r.wtype == WorkingType::Double);
			const double vsFp64 = (sameKernelDoubleThr > 0.0)
				? thr / sameKernelDoubleThr
				: 0.0;

			os << std::fixed << std::setprecision(3);
			os.width(13); os << r.timeSec * 1.0e3;
			os.width(15); os << thr * 1e-9;
			os.width(11); os << vsFp64;
			os.width(2);  os << (sameKernelBaseline ? "x*" : "x ");
		}

		os << std::scientific << std::setprecision(3);
		os.width(15); os << r.maxRelError;

		if (haveSweeps)
		{
			const VariantSweep* sw = findSweep(sweeps, r.wtype, r.ktype);
			if (sw && sw->nFitPoints >= 2)
			{
				os << std::fixed << std::setprecision(3);
				os.width(10); os << sw->alpha;
				os << "  ";
				os << std::left;
				os.width(14); os << shortAlphaConclusion(sw->alpha);
				os << std::right;
			}
			else
			{
				os.width(10); os << "n/a";
				os << "  ";
				os << std::left;
				os.width(14); os << "-";
				os << std::right;
			}
		}
		os << "\n";
	}
	const bool footerNeeded = hasGpu || haveSweeps;
	if (footerNeeded)
	{
		os << "(";
		const char* sep = "";
		if (hasGpu)
		{
			os << "vs_fp64* = vs (double, <same kernel>) per-kernel baseline";
			sep = "; ";
		}
		if (haveSweeps)
		{
			os << sep
			   << "alpha fitted from meanRelErr ~ N^alpha across the alpha-sweep depths";
		}
		os << ")\n";
	}
	os.flags(f);
	os.precision(p);
}

void reportAlphaSweep(std::ostream&                    os,
                      const std::vector<VariantSweep>& sweeps)
{
	os << "\n=== Error-propagation alpha sweep ===\n";
	if (sweeps.empty())
	{
		os << "(no sweeps)\n";
		return;
	}

	std::ios_base::fmtflags f = os.flags();
	std::streamsize p = os.precision();

	for (const auto& sw : sweeps)
	{
		os << "\n[" << toString(sw.wtype) << " / " << toString(sw.ktype)
		   << " on " << (sw.ranOnGpu ? "GPU" : "CPU") << "]\n";

		os << std::right;
		os.width(10); os << "N";
		os.width(16); os << "maxAbsErr";
		os.width(16); os << "meanAbsErr";
		os.width(16); os << "rmsAbsErr";
		os.width(16); os << "maxRelErr";
		os.width(16); os << "meanRelErr";
		os << "\n";
		os << std::string(10 + 16 * 5, '-') << "\n";

		os << std::scientific << std::setprecision(3);
		for (const auto& r : sw.rows)
		{
			os.width(10); os << r.depth;
			os.width(16); os << r.maxAbsError;
			os.width(16); os << r.meanAbsError;
			os.width(16); os << r.rmsAbsError;
			os.width(16); os << r.maxRelError;
			os.width(16); os << r.meanRelError;
			os << "\n";
		}

		os << std::fixed << std::setprecision(4);
		if (sw.nFitPoints >= 2)
		{
			os << "  fitted exponent  alpha (meanRelErr ~ N^alpha) = "
			   << sw.alpha
			   << "  [" << sw.nFitPoints << " points]\n";

			const double a = sw.alpha;
			os << "  conclusion       : ";
			if      (a < 0.3)  os << "SUBLINEAR (benign, ~O(N^" << a << "))\n";
			else if (a < 0.7)  os << "~O(sqrt(N)) - random-walk-like accumulation\n";
			else if (a < 1.3)  os << "approximately LINEAR in N\n";
			else               os << "SUPER-LINEAR (alpha=" << a << ") - potential concern\n";
		}
		else
		{
			os << "  (insufficient positive-meanRelErr points for an alpha fit)\n";
		}
	}

	os.flags(f);
	os.precision(p);
}

} // namespace fix::hpc
