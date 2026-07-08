/*
    api.cpp - Unit Test for Float-Float Arithmetic API
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This unit test compares native double-precision arithmetic operations with float-float (ffloat) type:
    - Basic arithmetic: multiplication, addition, division, and subtraction
    - Fused multiply-add (FMA) operations
    - Uses fp32mp2 (i.e. fpmp2<float, fpmp2_accuracy::def>) for extended precision via (hi, lo) components

    Test Approach:
    -------------------------------------------------------------------------
    The test performs computations using:
    1. Native double-precision arithmetic (fp64)
    2. Float-float extended precision type (ffloat)

    Results are compared to illustrate differences between double and float-float arithmetic.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the multi-precision floating-point type
using ffloat = fp32mp2;


// Conditional compilation macros for CUDA vs. CPU execution
// When compiled with nvcc (__CUDACC__ defined), use CUDA functions
// When compiled with standard C++ compiler, use host functions
#if __CUDACC__
    #define MALLOC(x,s) cudaMallocManaged(&x,s)          // CUDA Unified Memory allocation
    #define FREE(x) cudaFree(x)                          // CUDA memory deallocation
    #define RUN_EXAMPLE(x,y,z) run_example<<<1, 1>>>(x,y,z)  // Launch as CUDA kernel
    #define DEVICE_SYNCHRONIZE() cudaDeviceSynchronize() // Wait for GPU completion
    #define TARGET_DEVICE __global__                     // Mark as device kernel
#else
    #define MALLOC(x,s) x = (double*)malloc(s)           // Standard host memory allocation
    #define FREE(x) free(x)                              // Standard host memory deallocation
    #define RUN_EXAMPLE(x,y,z) run_example(x,y,z)        // Direct function call
    #define DEVICE_SYNCHRONIZE()                         // No-op for host execution
    #define TARGET_DEVICE                                // No qualifier for host function
#endif


/**
 * @brief Comparison function that performs arithmetic operations using both fp64 and ffloat types
 * 
 * @param inp   Input array with 3 double values
 * @param out   Output array for native fp64 results (5 values)
 * @param ffout Output array for float-float results (5 values)
 */
TARGET_DEVICE void run_example(double *inp, double *out, double* ffout) 
{
    // Load input values from memory into double-precision variables
    double dx = inp[0];
    double dy = inp[1];
    double dz = inp[2];

    // Convert double values to float-float (ffloat) type with extended precision.
    // Explicit construction (double -> fp32mp2 is a narrowing conversion).
    ffloat ex = ffloat(dx);
    ffloat ey = ffloat(dy);
    ffloat ez = ffloat(dz);

    // Perform native double-precision (fp64) arithmetic operations
    // These use standard IEEE 754 double-precision (53-bit mantissa)
    out[0] = dx * dy;           // Multiplication
    out[1] = dx + dy;           // Addition
    out[2] = dx / dy;           // Division
    out[3] = dx - dy;           // Subtraction
    out[4] = dx * dy + dz;      // Multiply-add (may not be fused)

    // Perform float-float (ffloat) arithmetic with extended precision
    // These use multi-precision representation with 2 float32 accumulators
    // providing higher accuracy than standard fp64 for many operations
    ffout[0] = ex * ey;         // Extended precision multiplication
    ffout[1] = ex + ey;         // Extended precision addition
    ffout[2] = ex / ey;         // Extended precision division
    ffout[3] = ex - ey;         // Extended precision subtraction
    ffout[4] = fma(ex, ey, ez); // Fused multiply-add with extended precision

    return;
}

int main(int argc, char** argv) 
{
    (void)argv; // Suppress unused parameter warning
    // Declare pointers for input, output and ffloat results
    double* inp;
    double* out;
    double* ffout;

    // Allocate Unified Memory – accessible from both CPU and GPU
    // inp:   3 input values for the computation
    // out:   5 results from native fp64 arithmetic
    // ffout: 5 results from float-float extended precision arithmetic
    MALLOC(inp,          3 * sizeof(double));
    MALLOC(out,          5 * sizeof(double));
    MALLOC(ffout,        5 * sizeof(double));

    // Initialize input values with high-precision constants
    // Multiply by argc to prevent compiler from optimizing away computations
    inp[0] = 1.123456782345678936 * argc;
    inp[1] = 2.234567891234567856 * argc;
    inp[2] = 3.345678901234567892 * argc;

    // Launch computation (CUDA kernel on GPU or function call on CPU)
    RUN_EXAMPLE(inp, out, ffout);

    // Wait for GPU to finish before accessing results on CPU
    DEVICE_SYNCHRONIZE();

    // Print results comparing three accuracy levels:
    // 1. Host: CPU computation using native double precision
    // 2. Device (fp64): GPU/device computation using native double precision
    // 3. Device (ffloat): GPU/device computation using float-float extended precision
    printf("  ** This example demonstrates the usage of float-float (fp32mp2) C++ API\n     with multi-precision arithmetic using fpmp2<float, fpmp2_accuracy::def>\n\n");
    
    // Multiplication comparison
    printf("                    Host:         %.4f * %.4f  = %.18f\n",   inp[0], inp[1], inp[0] * inp[1]);
    printf("           Device (fp64):         %.4f * %.4f  = %.18f\n",   inp[0], inp[1], out[0]);
    printf("         Device (ffloat):         %.4f * %.4f  = %.18f\n",   inp[0], inp[1], ffout[0]);

    // Addition comparison
    printf("                    Host:         %.4f + %.4f  = %.18f\n",   inp[0], inp[1], inp[0] + inp[1]);
    printf("           Device (fp64):         %.4f + %.4f  = %.18f\n",   inp[0], inp[1], out[1]);
    printf("         Device (ffloat):         %.4f + %.4f  = %.18f\n",   inp[0], inp[1], ffout[1]);

    // Division comparison
    printf("                    Host:          %.4f / %.4f = %.18f\n",   inp[0], inp[1], inp[0] / inp[1]);
    printf("           Device (fp64):          %.4f / %.4f = %.18f\n",   inp[0], inp[1], out[2]);
    printf("         Device (ffloat):          %.4f / %.4f = %.18f\n",   inp[0], inp[1], ffout[2]);

    // Subtraction comparison
    printf("                    Host:          %.4f - %.4f = %.18f\n",   inp[0], inp[1], inp[0] - inp[1]);
    printf("           Device (fp64):          %.4f - %.4f = %.18f\n",   inp[0], inp[1], out[3]);
    printf("         Device (ffloat):          %.4f - %.4f = %.18f\n",   inp[0], inp[1], ffout[3]);

    // Fused multiply-add comparison
    printf("                    Host:          fma(%.4f, %.4f, %.4f) = %.18f\n",   inp[0], inp[1], inp[2], std::fma(inp[0],inp[1],inp[2]));
    printf("           Device (fp64):          fma(%.4f, %.4f, %.4f) = %.18f\n",   inp[0], inp[1], inp[2], out[4]);
    printf("         Device (ffloat):          fma(%.4f, %.4f, %.4f) = %.18f\n",   inp[0], inp[1], inp[2], ffout[4]);

    // Free allocated memory (Unified Memory or host memory depending on compilation)
    FREE(inp);
    FREE(out);
    FREE(ffout);

    return 0;
}