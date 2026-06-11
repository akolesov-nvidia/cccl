/*
    fp64_tool.cpp - FP64 Precision Emulation Tool - Minimal Example
    ======================================================================================================

    This example shows how to use fp64_tool.hpp to emulate reduced floating-point
    precision using native FP64 hardware.

    Steps demonstrated:
    -------------------------------------------------------------------------
    1. Set mantissa and exponent bits via #define macros
    2. Create fp64_tool variables (drop-in replacement for double)
    3. Perform arithmetic — precision callbacks are applied automatically
    4. Print results side-by-side with native double

    Build Instructions:
    -------------------------------------------------------------------------
    Using the provided Makefile (recommended):
        make EXAMPLE=fp64_tool              # Build for GPU (CUDA)
        make TARGET=host EXAMPLE=fp64_tool  # Build for CPU only
        make run EXAMPLE=fp64_tool          # Build and run

    Manual compilation on CPU:
        g++ -std=c++17 -O2 -I../include fp64_tool.cpp -o fp64_tool.exe -lm

    Manual compilation with CUDA:
        nvcc -std=c++17 -O2 -I../include -x cu fp64_tool.cpp -o fp64_tool.exe
*/

#include <cstdio>
#include <cmath>

//=============================================================================
// Step 1: Configure precision BEFORE including the header
//
//   FP64_TOOL_MANTISSA_BITS  – number of mantissa bits to keep  (1-52, default 52)
//   FP64_TOOL_EXPONENT_BITS  – number of exponent bits to keep  (1-11, default 11)
//
// Common configurations:
//   FP32  : mantissa=23, exponent=8
//   BF16  : mantissa=7,  exponent=8
//   FP16  : mantissa=10, exponent=5
//   TF32  : mantissa=10, exponent=8
//=============================================================================
#define FP64_TOOL_MANTISSA_BITS 23   // Float-like mantissa (23 out of 52)
#define FP64_TOOL_EXPONENT_BITS  8   // Float-like exponent  (8 out of 11)
#include <cuda/fptool>

// The FP SDK lives in the cuda::experimental namespace (will be reduced to cuda:: later).
using namespace cuda::experimental;

//=============================================================================
// Host/Device compatibility
//=============================================================================
#if defined(__CUDACC__)
    #define HOST_DEVICE __host__ __device__
    #define KERNEL      __global__
#else
    #define HOST_DEVICE
    #define KERNEL
#endif

//=============================================================================
// A simple computation performed in both native double and fp64_tool
//=============================================================================
struct Result { double native; double reduced; };

HOST_DEVICE void compute(Result* r)
{
    // Step 2: Create variables — fp64_tool is a drop-in replacement for double
    double      a_native = 1.123456789012345;
    double      b_native = 2.987654321098765;

    fp64_tool a = 1.123456789012345;
    fp64_tool b = 2.987654321098765;

    // Step 3: Perform arithmetic — precision callbacks are applied automatically
    double      sum_native = a_native + b_native;
    fp64_tool sum        = a + b;

    // Store results (cast fp64_tool back to double for printing)
    r->native  = sum_native;
    r->reduced = (double)sum;
}

//=============================================================================
// CUDA kernel wrapper
//=============================================================================
#if defined(__CUDACC__)
KERNEL void compute_kernel(Result* r)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        compute(r);
}
#endif

//=============================================================================
// Main
//=============================================================================
int main()
{
    Result* r;

#if defined(__CUDACC__)
    cudaMallocManaged(&r, sizeof(Result));
    compute_kernel<<<1, 1>>>(r);
    cudaDeviceSynchronize();
#else
    r = new Result;
    compute(r);
#endif

    // Step 4: Print results side-by-side
    printf("FP64 Precision Tool example\n");
    printf("======================================\n");
    printf("Configuration: mantissa = %d bits, exponent = %d bits\n\n",
           FP64_TOOL_MANTISSA_BITS, FP64_TOOL_EXPONENT_BITS);

    printf("  a = 1.123456789012345\n");
    printf("  b = 2.987654321098765\n\n");

    printf("  a + b (native double): %.17f (%a)\n", r->native, r->native);
    printf("  a + b (reduced prec.): %.17f (%a)\n", r->reduced, r->reduced);

#if defined(__CUDACC__)
    cudaFree(r);
#else
    delete r;
#endif

    return 0;
}
