//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/*
    fp64mp2.cpp - Double-Double (Quad-Precision-Like) Extended Precision Arithmetic Demo
    ======================================================================================================

    This example demonstrates the usage and basic operations of double-double extended precision 
    arithmetic on both CPU and GPU (CUDA).

    What is Double-Double Arithmetic?
    -------------------------------------------------------------------------
    Double-double arithmetic represents a number as the unevaluated sum of two 64-bit doubles,
    providing up to ~104 effective mantissa bits (compared to 53 bits for standard double). This 
    enables quad-precision-like accuracy for many use cases without requiring full IEEE-754 binary128 
    hardware or emulation.

    Features Demonstrated:
    -------------------------------------------------------------------------
    - Construction and assignment of double-double values
    - Basic arithmetic: addition, subtraction, multiplication, division
    - Compound operations: sqrt, rsqrt, FMA (fused multiply-add)
    - High-precision computation demonstration (capturing precision lost by double)
    - Construction from __float128 on CUDA arch >= 1000 (when enabled/available)
    - Comparison with double precision reference
    - CPU/GPU host/device compatibility

    For math functions on fp64mp2 (exp, log, pow, sin, cos, sincos, tanh, erf, erfc,
    boys_f0, normcdfinv, cbrt, rcbrt, floor, ceil, round, trunc, fabs, fmin, fmax,
    min, max), see the companion fp64mp2_math.cpp example.  Note: most fp64mp2 math
    functions delegate to higher-precision fallbacks (libquadmath or fp64), unlike
    fp32mp2 where most are dedicated double-float implementations.

    Build Instructions:
    -------------------------------------------------------------------------
    Using the provided Makefile (recommended):
        make                    # Build for GPU (CUDA)
        make TARGET=host        # Build for CPU only
        make run                # Build and run

    Manual compilation on CPU:
        g++ -std=c++17 -O2 -I../include fp64mp2.cpp -o fp64mp2.exe

    Manual compilation with CUDA:
        nvcc -std=c++17 -O2 -I../include fp64mp2.cpp -o fp64mp2.exe

    Add -quadmath to the linker flags to enable quad-precision math fallbacks
    when using fpmp_math.h functions.

    Example:
    nvcc -std=c++17 -O2 -I../include fp64mp2.cpp -o fp64mp2.exe -lquadmath

    Output:
    -------------------------------------------------------------------------
    The program performs extended-precision calculations and compares results with IEEE-754 double 
    precision, demonstrating how double-double captures precision that standard double loses in 
    cancellation scenarios.

    Types Used:
    -------------------------------------------------------------------------
    - fp64mp2      : Default double-double type with Dekker normalization
    - fp64mp2_low : Fast mode without renormalization

    Configuration Macros:
    -------------------------------------------------------------------------
    - CCCL_FPMP_EXPLICIT_CASTS: When 1 (default), conversions like quad to fp64mp2 require
      explicit casts for type safety. Set to 0 for easier migration from standard types.
    - _CCCL_FPMP_FP128_ENABLE: Auto-detected from compiler/CUDA. Enables __float128 construction
      and conversion for fp64mp2. Set to 0 to disable for older compilers.
    - _CCCL_FPMP_FP128_MATH_FALLBACK: When 1, transcendental functions (exp, sin, log, etc.)
      use quad-precision (__float128) for ~113-bit accuracy. Requires libquadmath linkage,
      slower compilation, larger code. When 0 (default), falls back to double precision—
      faster builds, smaller code, but limited accuracy for math functions.
*/
#include <cstdio>

#include <cuda/std/cmath>
#include <cuda/std/cstdlib>

// FPMP library headers
#include <cuda/fpmp>  // Core multi-precision type and operations

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the multi-precision floating-point types (double-double)
using fptype_t      = fp64mp2;
using fptype_fast_t = fp64mp2_low;

// Test values
#define VALUE1 1.234567890123456789
#define VALUE2 9.876543210987654321
#define VALUE3 2.71828182f
#define VALUE4 5u

// Simple kernel/function to test double-double operations
_CCCL_HOST_DEVICE void double_double_operations(double* results) 
{
    // Construction and basic arithmetic
    fptype_t a = VALUE1;
    fptype_t b = VALUE2;
    fptype_t c = VALUE3;
    fptype_t d = VALUE4;
    
    fptype_t sum  = a + b;
    fptype_t diff = a - b;
    fptype_t prod = a * b;
    fptype_t quot = a / b;
    
    results[0] = (double)sum;
    results[1] = (double)diff;
    results[2] = (double)prod;
    results[3] = (double)quot;
    
    // Advanced operations
    fptype_t sqrt_x  = sqrt(d);
    fptype_t rsqrt_x = rsqrt(d);
    results[4]       = (double)sqrt_x;
    results[5]       = (double)rsqrt_x;
    
    // FMA
    fptype_t fma_result = fma(a, b, c);
    results[6]          = (double)fma_result;
    
    // Comparisons (store as 0.0 or 1.0)
    results[7] = (a > b)  ? 1.0 : 0.0;
    results[8] = (a < b)  ? 1.0 : 0.0;
    results[9] = (a == a) ? 1.0 : 0.0;
    
    // Compound operations
    fptype_t temp = a;
    temp += b;
    results[10] = (double)temp;
    
    temp = a;

    temp *= b;
    results[11] = (double)temp;

    // High precision demonstration - compute (1 + 1e-14)^2 - 1 - 2e-14
    // This should be exactly 1e-28, but double precision loses it
    fptype_t one = 1.0;
    fptype_t epsilon = 1e-14;
    fptype_t result = (one + epsilon) * (one + epsilon) - one - 2.0 * epsilon;
    results[12] = (double)result;

    // Construction from __float128 (only available on CUDA architectures >= 1000)
    double d_pi = 3.141592653589793238462;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Demonstrate construction from __float128 higher precision type
    // Using a value that benefits from __float128's ~113 bits of mantissa
    __float128 q_pi2 = (__float128)(d_pi) * (__float128)(d_pi);
    fptype_t dd_pi2  = fptype_t(q_pi2);
#else
    // Fallback for CPU or older CUDA architectures
    double d_pi2     = d_pi * d_pi;
    fptype_t dd_pi2  = fptype_t(d_pi2);
#endif
    results[13]      = dd_pi2.hi();
    results[14]      = dd_pi2.lo();

    // Accuracy-explicit add: high-accuracy addition for near-cancellation
    fptype_t p = fptype_t(-2.7059461654979244e+033);
    fptype_t q = fptype_t(+2.7059454398538426e+033);
    results[15] = static_cast<double>(add<fpmp2_accuracy::high>(p, q));
} // double_double_operations

#if _CCCL_CUDA_COMPILATION()
// CUDA kernel wrapper
__global__ void double_double_kernel(double* results) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        double_double_operations(results);
    }
}
#endif

void run_comparison_test() 
{
    printf("\n");
    printf("================================================================================\n");
    printf("  DOUBLE-DOUBLE (FP64MP2) STANDALONE DEMO\n");
    printf("  Comparing Double-Double vs Double Precision\n");
    printf("================================================================================\n\n");
    
    // Input values
    double a_d = VALUE1;
    double b_d = VALUE2;
    double c_d = VALUE3;
    double d_d = VALUE4;
    
    // Double precision reference
    double ref_sum   = a_d + b_d;
    double ref_diff  = a_d - b_d;
    double ref_prod  = a_d * b_d;
    double ref_quot  = a_d / b_d;
    double ref_sqrt  = ::cuda::std::sqrt(d_d);
    double ref_rsqrt = 1.0 / ::cuda::std::sqrt(d_d);
    double ref_fma   = ::cuda::std::fma(a_d, b_d, c_d);
    double ref_add_acc = -2.7059461654979244e+033 + 2.7059454398538426e+033;

    // High precision test reference (double loses this precision)
    double one = 1.0;
    double epsilon = 1e-14;
    double ref_precision_test = (one + epsilon) * (one + epsilon) - one - 2.0 * epsilon;
    
    printf("Input values:\n");
    printf("  a = %.17f\n", a_d);
    printf("  b = %.17f\n", b_d);
    printf("  c = %.17f\n", c_d);
    printf("  d = %.17f\n\n", d_d);
    
    printf("Precision comparison:\n");
    printf("  Double:        ~15-17 significant decimal digits (53-bit mantissa)\n");
    printf("  Double-Double: ~31 significant decimal digits (up to ~104 effective mantissa bits)\n\n");
    
    // Allocate memory for results
    double* results;
    
#if _CCCL_CUDA_COMPILATION()
    // CUDA path
    cudaMallocManaged(&results, 16 * sizeof(double));
    
    printf("Running on GPU...\n");
    double_double_kernel<<<1, 1>>>(results);
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(results);
        return;
    }
    printf("GPU execution completed.\n\n");
#else
    // CPU path
    results = (double*)malloc(16 * sizeof(double));
    printf("Running on CPU...\n");
    double_double_operations(results);
    printf("CPU execution completed.\n\n");
#endif
    
    // Display results
    printf("Results:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Operation          | Double Reference       | Double-Double Result   | Abs Error\n");
    printf("--------------------------------------------------------------------------------\n");
    
    printf("a + b              | %22.15f | %22.15f | %.2e\n", ref_sum,   results[0], ::cuda::std::fabs(results[0] - ref_sum));
    printf("a - b              | %22.15f | %22.15f | %.2e\n", ref_diff,  results[1], ::cuda::std::fabs(results[1] - ref_diff));
    printf("a * b              | %22.15f | %22.15f | %.2e\n", ref_prod,  results[2], ::cuda::std::fabs(results[2] - ref_prod));
    printf("a / b              | %22.15f | %22.15f | %.2e\n", ref_quot,  results[3], ::cuda::std::fabs(results[3] - ref_quot));
    printf("sqrt(d)            | %22.15f | %22.15f | %.2e\n", ref_sqrt,  results[4], ::cuda::std::fabs(results[4] - ref_sqrt));
    printf("rsqrt(d)           | %22.15f | %22.15f | %.2e\n", ref_rsqrt, results[5], ::cuda::std::fabs(results[5] - ref_rsqrt));
    printf("fma(a, b, c)       | %22.15f | %22.15f | %.2e\n", ref_fma,   results[6], ::cuda::std::fabs(results[6] - ref_fma));
    printf("add<high>(p,q)     | %22.6e | %22.6e | %.2e\n", ref_add_acc, results[15], ::cuda::std::fabs(results[15] - ref_add_acc));

    printf("\nComparisons (1.0 = true, 0.0 = false):\n");
    printf("  a > b:  %.1f\n", results[7]);
    printf("  a < b:  %.1f\n", results[8]);
    printf("  a == a: %.1f\n", results[9]);
    
    printf("\nCompound operations:\n");
    printf("  a += b: %.15f\n", results[10]);
    printf("  a *= b: %.15f\n", results[11]);

    printf("\n--------------------------------------------------------------------------------\n");
    printf("High Precision Demonstration:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Computing: (1 + 1e-14)^2 - 1 - 2e-14\n");
    printf("Exact result: 1e-28\n");
    printf("  Double result:        %.6e (loses precision)\n", ref_precision_test);
    printf("  Double-Double result: %.6e\n", results[12]);
    printf("  (Double-double can capture precision lost by standard double)\n");

    printf("\n--------------------------------------------------------------------------------\n");
    printf("Construction from __float128 (CUDA arch >= 1000 only):\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  pi^2 __float128 value: \n");
    printf("  fp64mp2 result:            %.17le + %.17le\n", results[13], results[14]);
    printf("  (On CUDA arch >= 1000: constructed from __float128; otherwise from double)\n");
    
    printf("\n");
    printf("================================================================================\n");
    printf("  DEMO COMPLETED SUCCESSFULLY\n");
    printf("================================================================================\n");
    
    // Cleanup
#if _CCCL_CUDA_COMPILATION()
    cudaFree(results);
#else
    free(results);
#endif
} // run_comparison_test

int main() 
{
    run_comparison_test();
    
    printf("\nThis standalone demo demonstrated:\n");
    printf("  Double-double (fp64mp2) construction from double values\n");
    printf("  Construction from __float128 (CUDA arch >= 1000)\n");
    printf("  Basic arithmetic operations (+, -, *, /)\n");
    printf("  Advanced operations (sqrt, rsqrt, fma)\n");
    printf("  Accuracy-explicit add<high> for near-cancellation precision\n");
    printf("  Comparison operators\n");
    printf("  Compound assignments\n");
    printf("  Accuracy comparison with double precision\n");
    printf("  High-precision computation demonstration\n");
#if _CCCL_CUDA_COMPILATION()
    printf("  GPU/CUDA execution\n");
#else
    printf("  CPU execution\n");
#endif
    printf("\n");
    printf("Key advantages of double-double (fp64mp2):\n");
    printf("  - Up to ~104 effective mantissa bits (vs 53 for double)\n");
    printf("  - Quad-like accuracy without requiring full IEEE-754 binary128 hardware\n");
    printf("  - Software emulation using error-free transformations\n");
    printf("  - Cross-platform reproducibility\n");
    printf("\n");
    
    return 0;
} // main

