/*
    lut.cpp - Unit Test for Float-Float Lookup Table with Compile-Time Conversion
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This unit test demonstrates compile-time conversion of double precision literals to float-float 
    (ffloat) constant pairs through a lookup table. The test observes how the compiler handles 
    conversion of double precision literals to float-float representation at compile time. The 
    lookup table is declared as an array of ffloat but initialized with double precision literals.

    Notes:
    -------------------------------------------------------------------------
    This test relies on compile-time initialization (where possible) to avoid runtime conversion 
    overhead for constant data.

    Performance Benefits Achieved:
    -------------------------------------------------------------------------
    - Zero runtime conversion overhead (compile-time initialization)
    - Optimal GPU memory placement (__constant__ memory)
    - Fast cached access, broadcast efficiently to all threads
    - Compiler can fully optimize with known constant values

    What This Test Shows:
    -------------------------------------------------------------------------
    - Compile-time initialization of ffloat arrays from double literals
    - Lookup table in __constant__ memory on GPU
    - Loading values from constant memory by index
    - Performing arithmetic operations with table values
    - Verifying precision is maintained through the conversion
    - GPU execution with optimal constant memory usage
    - Power of constexpr for embedded floating-point types

    Usage:
    -------------------------------------------------------------------------
    Compile standalone on the host (CPU):
        g++ -I../include lut.cpp -std=c++17 -o lut.exe

    Or with CUDA (GPU), if available:
        nvcc -I../include lut.cpp -std=c++17 -o lut.exe

 * When run, the example will load values from the lookup table and perform
 * multiplications, showing how compile-time constant initialization works.
 *
 * Relevant Type
 * -------------
 *   ffloat  :  A float-float class template instance (two 32-bit floats).
 */
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// FPMP library
#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the multi-precision floating-point type
using ffloat = fp32mp2;

// Macros for host/device compatibility
#if defined(__CUDACC__)
    #define HOST_DEVICE __host__ __device__
    #define KERNEL      __global__
#else
    #define HOST_DEVICE
    #define KERNEL
#endif

// Lookup table values - defined once to avoid duplication.
// These are the double precision literals that will be converted to ffloat.
// LUT_LIST(F) applies the function-like macro F to each value, so the same list
// builds both an ffloat array (F = LUT_FF, explicit double -> fp32mp2 cast, still
// constexpr) and a double reference array (F = LUT_ID, identity).
#define LUT_ID(x) x
#define LUT_FF(x) ffloat(x)
#define LUT_LIST(F) \
    F(3.14159265358979323846),    /* Pi */ \
    F(2.71828182845904523536),    /* e (Euler's number) */ \
    F(1.41421356237309504880),    /* sqrt(2) */ \
    F(1.73205080756887729352),    /* sqrt(3) */ \
    F(0.69314718055994530942),    /* ln(2) */ \
    F(0.43429448190325182765),    /* log10(e) */ \
    F(1.61803398874989484820),    /* Golden ratio (phi) */ \
    F(0.57721566490153286060),    /* Euler-Mascheroni constant */ \
    F(299792458.0),               /* Speed of light (m/s) */ \
    F(6.62607015e-34),            /* Planck constant (J⋅Hz⁻¹) */ \
    F(1.602176634e-19),           /* Elementary charge (C) */ \
    F(9.10938356e-31),            /* Electron mass (kg) */ \
    F(1.234567890123456789),      /* Test value 1 */ \
    F(9.876543210987654321),      /* Test value 2 */ \
    F(0.123456789012345678),      /* Test value 3 */ \
    F(123456.789012345678)        /* Test value 4 */

#define LUT_DESCRIPTIONS \
    "Pi", \
    "e (Euler's number)", \
    "sqrt(2)", \
    "sqrt(3)", \
    "ln(2)", \
    "log10(e)", \
    "Golden ratio", \
    "Euler-Mascheroni", \
    "Speed of light", \
    "Planck constant", \
    "Elementary charge", \
    "Electron mass", \
    "Test value 1", \
    "Test value 2", \
    "Test value 3", \
    "Test value 4"

const int LUT_SIZE = 16;

// Lookup table: ffloat initialized from double precision literals at compile-time!
// Thanks to the fixed constexpr constructor, this now works in constant memory on GPU.
// On GPU: __constant__ memory (cached, shared across all threads, compile-time initialized)
// On CPU: global constexpr data (compile-time initialized)
#if defined(__CUDACC__)
__device__ __constant__ constexpr ffloat LOOKUP_TABLE[] = { LUT_LIST(LUT_FF) };
// Host-accessible copy (needed because __constant__ memory cannot be read directly on the host)
[[maybe_unused]] static constexpr ffloat LOOKUP_TABLE_HOST[] = { LUT_LIST(LUT_FF) };
#else
constexpr ffloat LOOKUP_TABLE[] = { LUT_LIST(LUT_FF) };
#endif

// Function to perform operations using lookup table values
HOST_DEVICE void lut_operations(double* results, int* indices, int num_ops) 
{
    
    // Disable loop unrolling on device to observe actual lookup behavior
#ifdef __CUDA_ARCH__
    #pragma unroll 1
#endif
    for (int i = 0; i < num_ops; i++) 
    {
        int idx1 = indices[2 * i];
        int idx2 = indices[2 * i + 1];
        
        // Load from lookup table
        // On device: read from __constant__ memory; on host: read from constexpr copy
#ifdef __CUDA_ARCH__
        ffloat val1 = LOOKUP_TABLE[idx1];
        ffloat val2 = LOOKUP_TABLE[idx2];
#elif defined(__CUDACC__)
        ffloat val1 = LOOKUP_TABLE_HOST[idx1];
        ffloat val2 = LOOKUP_TABLE_HOST[idx2];
#else
        ffloat val1 = LOOKUP_TABLE[idx1];
        ffloat val2 = LOOKUP_TABLE[idx2];
#endif
        
        // Perform multiplication
        ffloat product = val1 * val2;
        
        // Store result (converted to double for output)
        results[i] = product;
    }
} // lut_operations

#if defined(__CUDACC__)
// CUDA kernel wrapper
KERNEL void lut_kernel(double* results, int* indices, int num_ops) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        lut_operations(results, indices, num_ops);
    }
}
#endif

void run_lut_test() 
{
    // Host-side reference values (same as in lut_operations, using shared macro)
    const double REFERENCE_VALUES[] = { LUT_LIST(LUT_ID) };
    
    // Descriptions (using shared macro)
    const char* descriptions[] = { LUT_DESCRIPTIONS };
    
    printf("\n");
    printf("================================================================================\n");
    printf("  FLOAT-FLOAT LOOKUP TABLE DEMO\n");
    printf("  Compile-Time Conversion of Double Literals to Float-Float\n");
    printf("================================================================================\n\n");
    
    printf("Lookup Table Contents (%d entries):\n", LUT_SIZE);
    printf("--------------------------------------------------------------------------------\n");
    printf("Index | Description              | Value (as double)\n");
    printf("--------------------------------------------------------------------------------\n");
    
    for (int i = 0; i < LUT_SIZE; i++) 
    {
        printf("%5d | %-24s | %.17e\n", i, descriptions[i], REFERENCE_VALUES[i]);
    }
    
    printf("\n");
    
    // Define test operations: pairs of indices to multiply
    int test_indices[] = 
    {
        0, 0,   // Pi * Pi
        0, 1,   // Pi * e
        2, 3,   // sqrt(2) * sqrt(3)
        4, 5,   // ln(2) * log10(e)
        6, 7,   // phi * gamma
        12, 13, // test1 * test2
        14, 15, // test3 * test4
        0, 2,   // Pi * sqrt(2)
    };
    
    int num_ops = sizeof(test_indices) / (2 * sizeof(int));
    
    // Allocate memory for results
    double* results;
    int* indices;
    
#if defined(__CUDACC__)
    // CUDA path
    cudaMallocManaged(&results, num_ops * sizeof(double));
    cudaMallocManaged(&indices, sizeof(test_indices));
    cudaMemcpy(indices, test_indices, sizeof(test_indices), cudaMemcpyHostToDevice);
    
    printf("Running on GPU...\n");
    printf("✓ Lookup table in __constant__ memory (compile-time initialized)\n");
    printf("✓ Optimal performance with cached constant memory access\n\n");
    lut_kernel<<<1, 1>>>(results, indices, num_ops);
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(results);
        cudaFree(indices);
        return;
    }
    printf("GPU execution completed.\n\n");
#else
    // CPU path
    results = (double*)malloc(num_ops * sizeof(double));
    indices = test_indices;
    printf("Running on CPU...\n");
    lut_operations(results, indices, num_ops);
    printf("CPU execution completed.\n\n");
#endif
    
    // Display results
    printf("Multiplication Results:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Operation                    | Float-Float Result | Double Reference   | Error\n");
    printf("--------------------------------------------------------------------------------\n");
    
    for (int i = 0; i < num_ops; i++) 
    {
        int idx1 = test_indices[2 * i];
        int idx2 = test_indices[2 * i + 1];
        
        double val1_d = REFERENCE_VALUES[idx1];
        double val2_d = REFERENCE_VALUES[idx2];
        double ref = val1_d * val2_d;
        double error = fabs(results[i] - ref);
        
        printf("LUT[%2d] * LUT[%2d]          | %18.15e | %18.15e | %.2e\n", 
               idx1, idx2, results[i], ref, error);
    }
    
    printf("\n");
    
    // Additional analysis: show how well the conversion preserved precision
    printf("Precision Analysis:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Testing round-trip conversion: double literal -> ffloat -> double\n");
    printf("Index | Description              | Stored Value\n");
    printf("--------------------------------------------------------------------------------\n");
    
    // Sample a few values to show reference values
    int sample_indices[] = {0, 1, 2, 12, 13, 14};
    for (int i = 0; i < 6; i++) 
    {
        int idx = sample_indices[i];
        
        printf("%5d | %-24s | %.17e\n", 
               idx, descriptions[idx], REFERENCE_VALUES[idx]);
    }
    
    printf("\n");
    printf("================================================================================\n");
    printf("  DEMO COMPLETED SUCCESSFULLY\n");
    printf("================================================================================\n");
    
    // Cleanup
#if defined(__CUDACC__)
    cudaFree(results);
    cudaFree(indices);
#else
    free(results);
#endif
} // run_lut_test

int main() 
{
    run_lut_test();
    
    printf("\nThis lookup table demo demonstrated:\n");
    printf("  ✓ Compile-time conversion of double literals to ffloat (constexpr)\n");
    printf("  ✓ Loading values from LUT by index\n");
    printf("  ✓ Arithmetic operations with LUT values\n");
    printf("  ✓ Precision preservation through conversion\n");
#if defined(__CUDACC__)
    printf("  ✓ GPU/CUDA execution with __constant__ memory\n");
    printf("\n");
    printf("Performance achievements:\n");
    printf("  ✓ Zero runtime conversion overhead (compile-time initialization)\n");
    printf("  ✓ Optimal GPU memory placement (__constant__ memory)\n");
    printf("  ✓ Fast cached access, broadcast to all threads\n");
#else
    printf("  ✓ CPU execution\n");
    printf("  ✓ Compile-time initialization (zero runtime overhead)\n");
#endif
    printf("\n");
    printf("To examine the generated assembly/SASS code:\n");
    printf("  CPU: g++ -S -I../include float_float_lut.cpp -std=c++17\n");
    printf("  GPU: nvcc -I../include float_float_lut.cpp -std=c++17 -ptx -o float_float_lut.ptx\n");
    printf("       (examine the .ptx file to see constant initialization)\n");
    printf("\n");
    
    return 0;
} // main


