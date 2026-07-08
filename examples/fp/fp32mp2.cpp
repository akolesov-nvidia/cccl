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
    fp32mp2.cpp - Float-Float (Double-Float) Extended Precision Arithmetic Demo
    ======================================================================================================

    This example demonstrates the usage and basic operations of float-float extended precision 
    arithmetic on both CPU and GPU (CUDA).

    What is Float-Float Arithmetic?
    -------------------------------------------------------------------------
    Float-float arithmetic, also known as double-float, represents a number as the unevaluated sum
    of two 32-bit floats, allowing for higher precision than a single float (up to ~46 effective 
    mantissa bits vs 24 bits; accuracy/algorithm dependent). This is particularly useful when native 
    double-precision is unavailable/slow or for applications requiring extended intermediate precision.

    Features Demonstrated:
    -------------------------------------------------------------------------
    - Construction and assignment of float-float values
    - Basic arithmetic: addition, subtraction, multiplication, division
    - Compound operations: sqrt, rsqrt, FMA (fused multiply-add)
    - Fast arithmetic with renormalization for dot products
    - Comparison operators
    - CPU/GPU host/device compatibility

    For dedicated math functions (exp, log, pow, sin, cos, sincos, tanh, erf, erfc,
    boys_f0, normcdfinv, icdf, cbrt, rcbrt, floor, ceil, round, trunc, fabs, fmin,
    fmax, min, max), see the companion fp32mp2_math.cpp example.

    Build Instructions:
    -------------------------------------------------------------------------
    Using the provided Makefile (recommended):
        make                    # Build for GPU (CUDA)
        make TARGET=host        # Build for CPU only
        make run                # Build and run

    Manual compilation on CPU:
        g++ -std=c++17 -O2 -I../include fp32mp2.cpp -o fp32mp2.exe

    Manual compilation with CUDA:
        nvcc -std=c++17 -O2 -I../include fp32mp2.cpp -o fp32mp2.exe

    Output:
    -------------------------------------------------------------------------
    The program performs extended-precision calculations and compares results with IEEE-754 double 
    precision, displaying absolute errors to demonstrate the behavior of the float-float implementation.

    Types Used:
    -------------------------------------------------------------------------
    - fp32mp2      : Default float-float type with Dekker normalization
    - fp32mp2_low : Fast mode (typically used with explicit renormalize() in accumulation patterns)

    Configuration Macros:
    -------------------------------------------------------------------------
    - CCCL_FPMP_EXPLICIT_CASTS: When 1 (default), conversions like double to fp32mp2 require
      explicit casts for type safety. Set to 0 for easier migration from standard types.
    - _CCCL_FPMP_FP128_ENABLE: Auto-detected; set to 0 to disable __float128 support.
    - _CCCL_FPMP_FP128_MATH_FALLBACK: When 1, fp64mp2 math uses quad-precision (requires
      libquadmath). When 0, uses double fallback (faster builds, smaller code).
*/
#include <cstdio>

#include <cuda/std/cmath>
#include <cuda/std/cstdlib>

// FPMP library headers
#include <cuda/fpmp>       // Core multi-precision type and operations (sqrt, rsqrt, fma)

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the multi-precision floating-point types
using fptype_t      = fp32mp2;
using fptype_fast_t = fp32mp2_low;

#define VALUE1 1.234567890123456789
#define VALUE2 9.876543210987654321
#define VALUE3 2.3243546f
#define VALUE4 5u

// Simple kernel/function to test float-float operations
_CCCL_HOST_DEVICE void float_float_operations(double* results) 
{
    // Construction and basic arithmetic
    fptype_t a = fptype_t(VALUE1);  // explicit: double literal -> fp32mp2 (constexpr)
    fptype_t b = fptype_t(VALUE2);
    fptype_t c = VALUE3;
    fptype_t d = VALUE4;
    
    fptype_t sum  = a + b;
    fptype_t diff = a - b;
    fptype_t prod = a * b;
    fptype_t quot = a / b;
    
    results[0] = sum;
    results[1] = diff;
    results[2] = prod;
    results[3] = quot;
    
    // Advanced operations
    fptype_t sqrt_x  = sqrt(d);
    fptype_t rsqrt_x = rsqrt(d);
    results[4]       = sqrt_x;
    results[5]       = rsqrt_x;
    
    // FMA
    fptype_t fma_result = fma(a, b, c);
    results[6]          = fma_result;
    
    // Comparisons (store as 0.0 or 1.0)
    results[7] = (a > b)  ? 1.0 : 0.0;
    results[8] = (a < b)  ? 1.0 : 0.0;
    results[9] = (a == a) ? 1.0 : 0.0;
    
    // Compound operations
    fptype_t temp = a;
    temp += b;
    results[10] = temp;
    
    temp = a;
    temp *= b;
    results[11] = temp;

    // Fast dot product with renormalization
    fptype_fast_t fast_a = fptype_fast_t(VALUE1);  // explicit: double literal -> fp32mp2 (constexpr)
    fptype_fast_t fast_b = fptype_fast_t(VALUE2);
    fptype_fast_t fast_c = VALUE3;
    fptype_fast_t fast_d = VALUE4;
    fptype_fast_t fast_prod = renormalize(fast_a * fast_c + fast_b * fast_d);
    results[12] = fast_prod;

    // Accuracy-explicit add: high-accuracy addition for near-cancellation
    fptype_t p = fptype_t(-2.7059461654979244e+033);
    fptype_t q = fptype_t(+2.7059454398538426e+033);
    results[13] = static_cast<double>(add<fpmp2_accuracy::high>(p, q));
} // float_float_operations

#if _CCCL_CUDA_COMPILATION()
// CUDA kernel wrapper
__global__ void float_float_kernel(double* results) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        float_float_operations(results);
    }
}
#endif

void run_comparison_test() 
{
    printf("\n");
    printf("================================================================================\n");
    printf("  FLOAT-FLOAT STANDALONE DEMO\n");
    printf("  Comparing Float-Float vs Double Precision\n");
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
    double ref_prod_fast = a_d * c_d + b_d * d_d;
    double ref_add_acc = -2.7059461654979244e+033 + 2.7059454398538426e+033;
    
    printf("Input values:\n");
    printf("  a = %.17f\n", a_d);
    printf("  b = %.17f\n", b_d);
    printf("  c = %.17f\n", c_d);
    printf("  d = %.17f\n\n", d_d);
    
    // Allocate memory for results
    double* results;
    
#if _CCCL_CUDA_COMPILATION()
    // CUDA path
    cudaMallocManaged(&results, 14 * sizeof(double));
    
    printf("Running on GPU...\n");
    float_float_kernel<<<1, 1>>>(results);
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
    results = (double*)malloc(14 * sizeof(double));
    printf("Running on CPU...\n");
    float_float_operations(results);
    printf("CPU execution completed.\n\n");
#endif
    
    // Display results
    printf("Results:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Operation          | Double Reference   | Float-Float Result | Abs Error\n");
    printf("--------------------------------------------------------------------------------\n");
    
    printf("a + b              | %18.15f | %18.15f | %.2e\n", ref_sum,   results[0], ::cuda::std::fabs(results[0] - ref_sum));
    printf("a - b              | %18.15f | %18.15f | %.2e\n", ref_diff,  results[1], ::cuda::std::fabs(results[1] - ref_diff));
    printf("a * b              | %18.15f | %18.15f | %.2e\n", ref_prod,  results[2], ::cuda::std::fabs(results[2] - ref_prod));
    printf("a / b              | %18.15f | %18.15f | %.2e\n", ref_quot,  results[3], ::cuda::std::fabs(results[3] - ref_quot));
    printf("sqrt(x)            | %18.15f | %18.15f | %.2e\n", ref_sqrt,  results[4], ::cuda::std::fabs(results[4] - ref_sqrt));
    printf("rsqrt(x)           | %18.15f | %18.15f | %.2e\n", ref_rsqrt, results[5], ::cuda::std::fabs(results[5] - ref_rsqrt));
    printf("fma(a, b, c)       | %18.15f | %18.15f | %.2e\n", ref_fma,   results[6], ::cuda::std::fabs(results[6] - ref_fma));
    printf("dot(a, b, c, d)    | %18.15f | %18.15f | %.2e\n", ref_prod_fast,  results[12], ::cuda::std::fabs(results[12] - ref_prod_fast));
    printf("add<high>(p,q)     | %18.6e | %18.6e | %.2e\n", ref_add_acc, results[13], ::cuda::std::fabs(results[13] - ref_add_acc));
    
    printf("\nComparisons (1.0 = true, 0.0 = false):\n");
    printf("  a > b:  %.1f\n", results[7]);
    printf("  a < b:  %.1f\n", results[8]);
    printf("  a == a: %.1f\n", results[9]);
    
    printf("\nCompound operations:\n");
    printf("  a += b: %.15f\n", results[10]);
    printf("  a *= b: %.15f\n", results[11]);


    
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
    printf("  Float-float construction from double values\n");
    printf("  Basic arithmetic operations (+, -, *, /)\n");
    printf("  Advanced operations (sqrt, rsqrt, fma, dot product)\n");
    printf("  Accuracy-explicit add<high> for near-cancellation precision\n");
    printf("  Comparison operators\n");
    printf("  Compound assignments\n");
    printf("  Accuracy comparison with double precision\n");
#if _CCCL_CUDA_COMPILATION()
    printf("  GPU/CUDA execution\n");
#else
    printf("  CPU execution\n");
#endif
    printf("\n");
    
    return 0;
} // main

