/*!
	\file   hpc_exp_error_ref.cpp
	\brief  Float128 reference oracle for the composite-exp benchmark.

	Compiled by g++ with `-fext-numeric-literals -lquadmath`. Provides the
	float128 (binary128) values used as the ground truth against which all
	float / double / fp32mp2 variants are compared.

	The reference is `exp(sum_i x_i)` evaluated in __float128. We also
	compute `prod_i exp(x_i)` in __float128 (to confirm that, at quad
	precision, the two are numerically equal to within ~1e-30 — i.e. the
	choice of kernel is irrelevant once the working precision is high
	enough).
*/

#include "hpc_exp_error.h"
#include "hpc_exp_error_kernels.hpp"

#include <quadmath.h>
#include <cmath>

// __float128 specialisations of the kernel-level helpers. They sit in
// the global namespace because the templated kernel calls them via ADL
// + `using std::exp`.

namespace {

inline __float128 abs_q(__float128 x) { return x < 0 ? -x : x; }

} // anonymous

namespace fix::hpc {

void computeReference(const ExpErrorConfig& cfg,
                      const double*         xs,
                      double*               refExpSum,
                      double*               refCompose)
{
	const int N  = cfg.chainLen;
	const int NS = cfg.numSamples;

	// One-shot host computation: not on a hot path, so we inline a loop
	// that promotes each double to __float128 just-in-time. (The kernel
	// templates now take a `const T*` and we don't want to materialise a
	// numSamples * chainLen __float128 buffer just for the reference.)
	for (int s = 0; s < NS; ++s)
	{
		const double* base = xs + s;

		__float128 sum  = 0;
		__float128 prod = 1;
		for (int i = 0; i < N; ++i)
		{
			const __float128 xi =
				static_cast<__float128>(base[static_cast<long long>(i) * NS]);
			sum  += xi;
			prod *= expq(xi);
		}
		refExpSum [s] = static_cast<double>(expq(sum));
		refCompose[s] = static_cast<double>(prod);
	}
}

/*!
	\brief Multi-depth fp128 reference for the alpha sweep.

	For each sample row we walk forward once accumulating the sum in fp128
	and snapshot expq(sum) at every requested depth checkpoint, so the cost
	is O(numSamples * maxDepth) fp128 adds + O(numSamples * numDepths)
	expq() calls regardless of how many checkpoints are requested.
*/
void computeReferenceForDepths(const ExpErrorConfig&   cfg,
                               const double*           xs,
                               const std::vector<int>& depths,
                               double*                 refExpSum2D)
{
	if (depths.empty()) return;

	const int       nDepth = static_cast<int>(depths.size());
	const int       maxD   = depths.back();
	const long long NS     = cfg.numSamples;

	for (long long s = 0; s < NS; ++s)
	{
		// SoA layout: sample s lives at xs + s, stride = NS.
		const double* base = xs + s;

		__float128 sum = 0;
		int        di  = 0;

		for (int i = 0; i < maxD; ++i)
		{
			sum += static_cast<__float128>(base[static_cast<long long>(i) * NS]);
			while (di < nDepth && depths[di] == i + 1)
			{
				refExpSum2D[static_cast<size_t>(s) * nDepth + di] =
					static_cast<double>(expq(sum));
				++di;
			}
			if (di >= nDepth) break;
		}
	}
}

// Helper used by the CPU driver below: relative error in float128, returned
// as double for downstream stat aggregation. Done in fp128 so that the
// reported relative error itself is accurate to ~30 digits even when the
// approximate value is FP64-noise-floor close to the reference.
double relErrFp128(double approx, double ref)
{
	if (ref == 0.0)
		return std::abs(approx);
	const __float128 a = static_cast<__float128>(approx);
	const __float128 r = static_cast<__float128>(ref);
	const __float128 e = abs_q((a - r) / r);
	return static_cast<double>(e);
}

} // namespace fix::hpc
