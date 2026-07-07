/*!
	\file   hpc_exp_error_kernels.hpp
	\brief  Templated, host+device kernels for the composite-exp() error study.

	Two single-chain kernels, each parametrised on the working type T:
	  * compose<T>(xs, n)  =  prod_{i=0..n-1} exp(xs[i])
	  * expsum<T>(xs, n)   =  exp(sum_{i=0..n-1} xs[i])

	The two are mathematically identical for x_i in real numbers, but their
	numerical behaviour under finite precision is very different — the
	'compose' variant has n-1 multiplications and n exp() calls, while the
	'expsum' variant has one exp() call and n-1 additions. T can be:
	  - float
	  - double
	  - fp32mp2  (double-float, ~46-bit mantissa)
	  - __float128 (host only, used as the reference oracle)

	IMPORTANT: inputs are taken as `const T*`, i.e. ALREADY in the working
	type. Per-element `double -> T` conversions (which are non-trivial for
	fp32mp2 and even costly for `float` on FP64-throttled GPUs) are NOT
	part of these kernels. Callers must convert their double exponent
	stream to T outside any timed region.
*/

#ifndef HPC_EXP_ERROR_KERNELS_HPP
#define HPC_EXP_ERROR_KERNELS_HPP

#if defined(__CUDACC__)
	#define HPC_EXP_HD __host__ __device__
#else
	#define HPC_EXP_HD
#endif

// Per-thread unroll factor / number of independent partial accumulators
// in compose / expsum. The Makefile passes -DHPC_UNROLL_FACTOR=<UNROLL>;
// 4 is the default. Setting to 1 disables ILP and falls back to a naive
// single-accumulator loop.
#ifndef HPC_UNROLL_FACTOR
	#define HPC_UNROLL_FACTOR 4
#endif

// Compiler-specific full-unroll pragma:
//   * nvcc / nvc++ / PGI: `#pragma unroll N`
//   * gcc / g++ (real)  : `#pragma GCC unroll N`
//   * everything else   : no-op (silent)
// NVHPC (`nvc++`) and PGI define `__GNUC__` for gcc compatibility but only
// recognise the `unroll N` form, so the NVIDIA-compiler check must come
// before the generic GCC fallback.
// Two-level stringification is required so that HPC_UNROLL_FACTOR (a
// preprocessor macro) is expanded before being turned into a string.
#define HPC_EXP_STRINGIFY_(x) #x
#define HPC_EXP_STRINGIFY(x)  HPC_EXP_STRINGIFY_(x)
#if defined(__CUDACC__) || defined(__NVCOMPILER) || defined(__PGI)
	#define HPC_EXP_UNROLL(N) _Pragma(HPC_EXP_STRINGIFY(unroll N))
#elif defined(__GNUC__) || defined(__GNUG__)
	#define HPC_EXP_UNROLL(N) _Pragma(HPC_EXP_STRINGIFY(GCC unroll N))
#else
	#define HPC_EXP_UNROLL(N)
#endif

#include <cmath>

#include <cuda/fpmp>
#include <cuda/fpmp_math>

using namespace cuda::experimental;

using ffloat = fp32mp2;

namespace fix::hpc {

// Bring std::exp into the namespace so unqualified `exp(x)` calls inside
// the kernels below resolve correctly for every working type:
//   * float / double : finds std::exp via this using-declaration.
//   * __fpmp2_t<> : finds the freestanding exp() declared at global
//                      scope in fpmp_math.h via argument-dependent
//                      lookup (the type's associated namespace).
// __float128 is intentionally NOT used through these kernels — the
// reference oracle in hpc_exp_error_ref.cpp inlines its own expq() loop.
using std::exp;

/*!
	\brief Composite-exp kernel: y = prod_i exp(xs[i*stride]).
	\param xs       starting address of this chain's exponent stream, in T
	\param n        chain length (number of compositions)
	\param stride   distance (in elements) between consecutive exponents.
	                For SoA / column-major layouts on GPU pass `numSamples`
	                so that warp-adjacent threads access warp-adjacent
	                elements and a single transaction services the warp.
	                For tightly-packed AoS / row-major data pass 1.
	\returns        composed product, in working type T

	The `#pragma unroll` directive lets the compiler unroll the loop by
	HPC_UNROLL_FACTOR; tune via `make UNROLL=N`. Per-thread ILP is
	limited by the serial dependency chain through `r`; on GPU this is
	hidden by parallelism across threads.
*/
template <typename T>
HPC_EXP_HD inline T kernel_compose(const T* xs, int n, int stride)
{
	T r = T(1.0);
	HPC_EXP_UNROLL(HPC_UNROLL_FACTOR)
	for (long long i = 0; i < n; ++i)
		r = r * exp(xs[i * stride]);
	return r;
}

/*!
	\brief Exp-of-sum kernel: y = exp(sum_i xs[i*stride]).
	\param xs       starting address of this chain's exponent stream, in T
	\param n        chain length (number of additions inside exp)
	\param stride   stride between consecutive exponents (see kernel_compose)
	\returns        exp of summed exponent, in working type T
*/
template <typename T>
HPC_EXP_HD inline T kernel_expsum(const T* xs, int n, int stride)
{
	T s = T(0.0);
	HPC_EXP_UNROLL(HPC_UNROLL_FACTOR)
	for (long long i = 0; i < n; ++i)
		s = s + xs[i * stride];
	return exp(s);
}

} // namespace fix::hpc

#endif // HPC_EXP_ERROR_KERNELS_HPP
