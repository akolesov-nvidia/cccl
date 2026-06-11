#pragma once

/*!
	\file   hpc_exp_error.h
	\brief  Public API for the composite-exp() GPU benchmark + error sweep.

	The benchmark studies how the FP64 exp() composition scenario behaves
	when the working type is changed to {float, double, fp32mp2_*} on a
	GPU, with two semantically equivalent kernels:
	  * compose<T>  ->  prod_i exp(x_i)        (n exps, n-1 muls)
	  * expsum<T>   ->  exp(sum_i x_i)         (1 exp,  n-1 adds)

	Reference oracle: __float128 expsum (host, libquadmath).

	The driver prints three sections:
	  1. Per-variant accuracy + GPU throughput (6 rows)
	  2. GPU speedup table (all 6 variants vs the SAME-kernel `double` row)
	  3. Error-propagation alpha sweep (per variant: meanRelErr at multiple
	     chain depths, plus a fitted log-log slope alpha).

	The double-float type used here is the default-accuracy variant
	  fp32mp2 = fpmp2_t<float, fpmp2_accuracy::def>
	(2-Sum-style add with low-part renormalisation).
*/

#include <cstdint>
#include <ostream>
#include <vector>

namespace fix::hpc {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the fixed-chain GPU benchmark.
///
/// Performance-loop methodology:
///   warm-up launch -> (numIterations x [chrono-start ; one kernel launch
///   that internally repeats `reps` passes over the data ; sync ; chrono-end]).
/// Throughput = (numSamples * chainLen * reps) / (avgKernelTime).
///
/// Defaults match the original problem statement this benchmark was derived
/// from: numSamples = 10000, maxDepth = 10000, depthStep = 500 -> 20
/// checkpoints. At 10000 x 10000 the per-variant device buffers are
/// ~0.4 - 1.6 GB each (3.6 GB total across float / double / fp32mp2 /
/// fp128-ref), which fits comfortably on any modern data-center or
/// consumer GPU.
struct ExpErrorConfig
{
	int           numSamples      = 10000;       //!< independent sample chains per launch
	int           chainLen        = 10000;       //!< exponents per chain (also = max alpha-sweep depth)
	double        dtMin           = 1.0e-4;      //!< min |x_i|
	double        dtMax           = 1.0e-2;      //!< max |x_i|
	bool          negativeOnly    = true;        //!< sample x_i in [-dtMax, -dtMin]
	unsigned long seed            = 42;
	int           reps            = 8;           //!< inner repetitions inside ONE kernel launch
	int           numIterations   = 10;          //!< timed launches averaged for the perf number
	int           threadsPerBlock = 256;         //!< CUDA block size
	int           numBlocks       = 2048;        //!< CUDA grid size (fixed, grid-stride loop)
	int           sweepNumDepths  = 20;          //!< number of depth checkpoints in the alpha sweep
};

enum class WorkingType : int {
	Float  = 0,
	Double = 1,
	Fpmp2  = 2,
};
enum class KernelType  : int { Compose = 0, ExpSum = 1 };

// ---------------------------------------------------------------------------
// Single-config benchmark result types
// ---------------------------------------------------------------------------

/// Single (working type, kernel) variant at fixed chainLen.
struct VariantResult
{
	WorkingType wtype;
	KernelType  ktype;
	bool        ranOnGpu;
	double      maxAbsError;
	double      meanAbsError;
	double      rmsAbsError;
	double      maxRelError;
	double      meanRelError;
	double      timeSec;        //!< average wall-clock per launch (one launch = `reps` passes)
	long long   numChains;      //!< numSamples * reps  (work per launch, used for throughput)
	long long   chainLen;
};

/// One row in the per-variant alpha-sweep table.
struct DepthErrorRow
{
	int    depth;
	double maxAbsError;
	double meanAbsError;
	double rmsAbsError;
	double maxRelError;
	double meanRelError;
};

/// Alpha-sweep results for one (working type, kernel) variant.
struct VariantSweep
{
	WorkingType                wtype;
	KernelType                 ktype;
	bool                       ranOnGpu;
	std::vector<DepthErrorRow> rows;
	double                     alpha;       //!< log-log slope of meanRelErr vs depth
	int                        nFitPoints;  //!< number of points used in the fit
};

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// Pre-generate a numSamples x chainLen FP64 exponent matrix.
void generateExponents(const ExpErrorConfig& cfg, double* out);

/// Compute float128 reference values for the full chainLen.
///   refExpSum[s]  = exp(sum_{i=0..chainLen-1} xs[s*chainLen+i])
///   refCompose[s] = prod_{i=0..chainLen-1} exp(xs[s*chainLen+i])
/// Both are evaluated in __float128 and cast back to double.
void computeReference(const ExpErrorConfig& cfg,
                      const double*         xs,
                      double*               refExpSum,
                      double*               refCompose);

/// Multi-depth float128 reference for the alpha sweep.
/// `depths` must be ascending and all <= cfg.chainLen.
/// `refExpSum2D` is laid out as numSamples x depths.size(), row-major:
///   refExpSum2D[s * depths.size() + k] = exp(sum_{i=0..depths[k]-1} xs[s*chainLen+i])
void computeReferenceForDepths(const ExpErrorConfig&   cfg,
                               const double*           xs,
                               const std::vector<int>& depths,
                               double*                 refExpSum2D);

/// Float128-based relative error helper, exposed for downstream stats.
double relErrFp128(double approx, double ref);

/// Run the 8 (working type x kernel) variants on the CPU at fixed chainLen.
/// Kept available for offline accuracy sanity-checks; no longer called from
/// the default driver flow (per the GPU-only mandate).
std::vector<VariantResult> runCpuBenchmark(const ExpErrorConfig& cfg,
                                           const double*         xs,
                                           const double*         refExpSum);

/// Run the 6 variants on CUDA. Empty result on a non-device build or when
/// no GPU is available at runtime.
std::vector<VariantResult> runGpuBenchmark(const ExpErrorConfig& cfg,
                                           const double*         xs,
                                           const double*         refExpSum);

/// Multi-depth alpha-sweep on CUDA. Same layout assumption for refExpSum2D
/// as in computeReferenceForDepths.
std::vector<VariantSweep> runGpuAlphaSweep(const ExpErrorConfig&   cfg,
                                           const std::vector<int>& depths,
                                           const double*           xs,
                                           const double*           refExpSum2D);

/// CPU implementation of the alpha sweep (fallback when no GPU is present).
std::vector<VariantSweep> runCpuAlphaSweep(const ExpErrorConfig&   cfg,
                                           const std::vector<int>& depths,
                                           const double*           xs,
                                           const double*           refExpSum2D);

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Per-variant accuracy + throughput table.
void reportVariants(std::ostream&                     os,
                    const ExpErrorConfig&             cfg,
                    const std::vector<VariantResult>& results,
                    const char*                       heading);

/// Speedup table: every variant's throughput compared to native FP64
/// on the SAME kernel ("vs_fp64" = thr / thr(Double, <same kernel>)).
/// Heading describes the backend ("GPU"/"CPU"). When `sweeps` is non-empty,
/// two extra columns are emitted per row: the fitted alpha (meanRelErr ~
/// N^alpha) and a short propagation conclusion taken from the matching
/// variant in `sweeps` (matched on (wtype, ktype)).
void reportSpeedupsVsBaseline(std::ostream&                     os,
                              const std::vector<VariantResult>& results,
                              const std::vector<VariantSweep>&  sweeps,
                              const char*                       heading);

/// Alpha-sweep report: one block per variant (depth | meanRelErr | maxRelErr ...)
/// followed by the fitted alpha.
void reportAlphaSweep(std::ostream&                    os,
                      const std::vector<VariantSweep>& sweeps);

// ---------------------------------------------------------------------------
// Build-time / run-time capabilities
// ---------------------------------------------------------------------------

/// True iff this binary was built with TARGET=device (CUDA TU linked in).
bool gpuSupportCompiledIn();

/// True iff a usable CUDA device is visible to the runtime right now.
/// Returns false unconditionally when gpuSupportCompiledIn() is false.
/// Cheap probe: wraps cudaGetDeviceCount; safe to call before any allocations.
bool gpuRuntimeAvailable();

const char* toString(WorkingType t);
const char* toString(KernelType  k);

} // namespace fix::hpc
