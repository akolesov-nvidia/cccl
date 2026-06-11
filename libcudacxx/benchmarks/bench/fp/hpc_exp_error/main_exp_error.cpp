/*!
	\file   main_exp_error.cpp
	\brief  Composite-exp() error & GPU performance benchmark - CLI driver.

	usage: ./hpc_exp_error [numSamples] [chainLen] [reps] [numIter] [sweepNumDepths]

	Build-time defaults (overridable via -D from the Makefile):
	  NUM_OPTIONS, CHAIN_LEN, REPS, NUM_ITERATIONS, THREADS_PER_BLOCK, NUM_BLOCKS

	Output order (per user request):
	  1. Alpha sweep (multi-depth error propagation tables + fitted alphas).
	  2. Accuracy table at fixed chainLen (per-variant errors vs fp128 ref).
	  3. Performance + speedups table (each variant's vs_fp64 ratio against
	     the SAME-kernel `double` row, with alpha and conclusion columns
	     from the alpha sweep).
*/

#include "hpc_exp_error.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// ---- Build-time defaults ---------------------------------------------------
// Override at compile time via -DNUM_OPTIONS=..., -DREPS=..., etc.
//
// Defaults match the original problem statement this benchmark was derived
// from:
//     numSamples = 10000, maxDepth = 10000, depthStep = 500 (-> 20 checkpoints)
//
// Host and GPU runs share the same problem size so the host's report
// (`make TARGET=host`) lists maxRelErr / alpha at exactly the same
// (numSamples, chainLen) as the GPU run, making the two reports directly
// comparable. The host pass is OMP-parallel across independent chains and
// finishes the accuracy snapshot in a few seconds at this size.
#ifndef NUM_OPTIONS
  #define NUM_OPTIONS 10000
#endif
#ifndef CHAIN_LEN
  #define CHAIN_LEN 10000
#endif
#ifndef REPS
  #ifdef HPC_EXP_ERROR_HAVE_CUDA
    #define REPS 8
  #else
    #define REPS 2
  #endif
#endif
#ifndef NUM_ITERATIONS
  #define NUM_ITERATIONS 10
#endif
#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK 256
#endif
#ifndef NUM_BLOCKS
  #define NUM_BLOCKS 2048
#endif
#ifndef SWEEP_NUM_DEPTHS
  #define SWEEP_NUM_DEPTHS 20
#endif

namespace {

std::vector<int> makeSweepDepths(int chainLen, int numDepths)
{
	std::vector<int> out;
	if (numDepths <= 0 || chainLen <= 0) return out;
	out.reserve(numDepths);
	for (int k = 1; k <= numDepths; ++k)
	{
		long long d = static_cast<long long>(chainLen) * k / numDepths;
		if (d < 1)         d = 1;
		if (d > chainLen)  d = chainLen;
		// Avoid duplicate depths from integer rounding when numDepths > chainLen.
		if (!out.empty() && out.back() == static_cast<int>(d)) continue;
		out.push_back(static_cast<int>(d));
	}
	return out;
}

} // anonymous

int main(int argc, char** argv)
{
	using namespace fix::hpc;

	ExpErrorConfig cfg;
	cfg.numSamples      = (NUM_OPTIONS);
	cfg.chainLen        = (CHAIN_LEN);
	cfg.reps            = (REPS);
	cfg.numIterations   = (NUM_ITERATIONS);
	cfg.threadsPerBlock = (THREADS_PER_BLOCK);
	cfg.numBlocks       = (NUM_BLOCKS);
	cfg.sweepNumDepths  = (SWEEP_NUM_DEPTHS);

	if (argc > 1) cfg.numSamples     = std::atoi(argv[1]);
	if (argc > 2) cfg.chainLen       = std::atoi(argv[2]);
	if (argc > 3) cfg.reps           = std::atoi(argv[3]);
	if (argc > 4) cfg.numIterations  = std::atoi(argv[4]);
	if (argc > 5) cfg.sweepNumDepths = std::atoi(argv[5]);

	if (cfg.numSamples <= 0 || cfg.chainLen <= 0 ||
	    cfg.reps <= 0 || cfg.numIterations <= 0 ||
	    cfg.sweepNumDepths <= 0 || cfg.numBlocks <= 0 ||
	    cfg.threadsPerBlock <= 0)
	{
		std::fprintf(stderr,
			"usage: %s [numSamples>0] [chainLen>0] [reps>0] [numIter>0] [sweepNumDepths>0]\n",
			argv[0]);
		return 1;
	}

	std::cout << "[hpc_exp_error] numSamples="    << cfg.numSamples
	          << " chainLen="                     << cfg.chainLen
	          << " reps="                         << cfg.reps
	          << " numIterations="                << cfg.numIterations
	          << " grid=("                        << cfg.numBlocks
	          << "x"                              << cfg.threadsPerBlock << ")"
	          << " sweepNumDepths="               << cfg.sweepNumDepths
	          << " dt in ["                       << cfg.dtMin
	          << "," << cfg.dtMax << "]"
	          << " (negativeOnly="                << (cfg.negativeOnly ? "yes" : "no")
	          << ")\n";
	std::cout << "[hpc_exp_error] GPU compiled in: "
	          << (gpuSupportCompiledIn() ? "YES" : "NO") << "\n";

	// ---------------------------------------------------------------------
	// Decide effective execution mode early. Three cases:
	//   * GPU build, GPU runtime present -> full GPU run.
	//   * GPU build, no GPU runtime      -> exit cleanly. The CPU fallback
	//                                       at GPU-sized defaults is multi-
	//                                       minutes per variant; not useful.
	//                                       Re-run with TARGET=host for a
	//                                       CPU-sized benchmark.
	//   * Host build (no CUDA compiled)  -> CPU run with host-sized defaults.
	// ---------------------------------------------------------------------
	const bool useGpu = gpuSupportCompiledIn() && gpuRuntimeAvailable();
	if (gpuSupportCompiledIn() && !useGpu)
	{
		std::cout
			<< "[hpc_exp_error] no usable CUDA device visible to the runtime\n"
			<< "[hpc_exp_error] skipping benchmark: GPU-sized defaults are too\n"
			<< "[hpc_exp_error]   slow for a CPU fallback (multi-minutes per\n"
			<< "[hpc_exp_error]   variant). Re-run with TARGET=host to run a\n"
			<< "[hpc_exp_error]   CPU-sized benchmark instead.\n";
		return 0;
	}

	// ---------------------------------------------------------------------
	// Random exponent matrix (RNG outside any kernel).
	// ---------------------------------------------------------------------
	const long long total =
		static_cast<long long>(cfg.numSamples) *
		static_cast<long long>(cfg.chainLen);
	std::cout << "[hpc_exp_error] generating " << total << " random exponents...\n";

	std::vector<double> xs(static_cast<size_t>(total));
	{
		const auto t0 = std::chrono::steady_clock::now();
		generateExponents(cfg, xs.data());
		const auto t1 = std::chrono::steady_clock::now();
		std::cout << "[hpc_exp_error] RNG took "
		          << std::chrono::duration<double>(t1 - t0).count() << " s\n";
	}

	// ---------------------------------------------------------------------
	// fp128 reference at full chainLen + multi-depth fp128 reference.
	// We compute both up front so all benchmarks below need only float128
	// arrays and never re-touch libquadmath.
	// ---------------------------------------------------------------------
	std::vector<double> refExpSum (cfg.numSamples);
	std::vector<double> refCompose(cfg.numSamples);
	{
		std::cout << "[hpc_exp_error] computing fp128 reference at N="
		          << cfg.chainLen << "...\n";
		const auto t0 = std::chrono::steady_clock::now();
		computeReference(cfg, xs.data(), refExpSum.data(), refCompose.data());
		const auto t1 = std::chrono::steady_clock::now();
		std::cout << "[hpc_exp_error] fp128 reference took "
		          << std::chrono::duration<double>(t1 - t0).count() << " s\n";

		double maxRel = 0.0;
		for (int s = 0; s < cfg.numSamples; ++s)
		{
			const double rel = relErrFp128(refCompose[s], refExpSum[s]);
			if (rel > maxRel) maxRel = rel;
		}
		std::cout << "[hpc_exp_error] fp128 self-check |compose-expsum|/|expsum| max = "
		          << std::scientific << std::setprecision(3) << maxRel << "\n";
	}

	const std::vector<int> depths = makeSweepDepths(cfg.chainLen, cfg.sweepNumDepths);
	std::cout << "[hpc_exp_error] alpha-sweep depths (count=" << depths.size() << "):";
	for (int d : depths) std::cout << " " << d;
	std::cout << "\n";

	std::vector<double> refExpSum2D(
		static_cast<size_t>(cfg.numSamples) * depths.size());
	{
		std::cout << "[hpc_exp_error] computing fp128 multi-depth reference ("
		          << depths.size() << " depths)...\n";
		const auto t0 = std::chrono::steady_clock::now();
		computeReferenceForDepths(cfg, xs.data(), depths, refExpSum2D.data());
		const auto t1 = std::chrono::steady_clock::now();
		std::cout << "[hpc_exp_error] multi-depth reference took "
		          << std::chrono::duration<double>(t1 - t0).count() << " s\n";
	}

	// At this point execution mode is fixed: useGpu == true means a CUDA
	// build with a usable device; useGpu == false means a TARGET=host build
	// running CPU-sized work. The GPU-built-but-no-device case has already
	// returned above.

	// ---------------------------------------------------------------------
	// 1) ALPHA SWEEP (printed first, per directive).
	// ---------------------------------------------------------------------
	std::vector<VariantSweep> sweeps;
	if (useGpu)
	{
		std::cout << "[hpc_exp_error] running GPU alpha sweep...\n";
		sweeps = runGpuAlphaSweep(cfg, depths, xs.data(), refExpSum2D.data());
	}
	else
	{
		std::cout << "[hpc_exp_error] running host alpha sweep "
		             "(TARGET=host build)...\n";
		sweeps = runCpuAlphaSweep(cfg, depths, xs.data(), refExpSum2D.data());
	}
	reportAlphaSweep(std::cout, sweeps);

	// ---------------------------------------------------------------------
	// 2) ACCURACY (per-variant errors at fixed chainLen).
	// 3) PERFORMANCE + per-kernel speedups vs same-kernel `double` row,
	//    with alpha + conclusion columns from the alpha sweep.
	// Both come from the same benchmark run.
	// ---------------------------------------------------------------------
	std::vector<VariantResult> bench;
	if (useGpu)
	{
		std::cout << "\n[hpc_exp_error] running GPU benchmark...\n";
		bench = runGpuBenchmark(cfg, xs.data(), refExpSum.data());
	}
	else
	{
		std::cout << "\n[hpc_exp_error] running host benchmark "
		             "(TARGET=host build)...\n";
		bench = runCpuBenchmark(cfg, xs.data(), refExpSum.data());
	}

	const char* benchHeading = useGpu
		? "GPU accuracy : 6 variants vs float128 reference"
		: "Host accuracy : 6 variants vs float128 reference";
	reportVariants(std::cout, cfg, bench, benchHeading);

	reportSpeedupsVsBaseline(std::cout, bench, sweeps,
	                         useGpu ? "GPU summary + performance"
	                                : "Host summary");

	std::cout << std::flush;
	return 0;
}
