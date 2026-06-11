/*
    cast.cpp - Unit Test for Idempotency of Sequential Casts
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This unit test demonstrates the idempotency property of conversions between double and float-float 
    (fp32mp2) representations. The test verifies that sequential casts stabilize after the first 
    conversion: double -> float-float -> double -> float-float -> double

    Expected Behavior:
    -------------------------------------------------------------------------
    After the first conversion from double to float-float and back, the result should remain stable 
    through subsequent conversions. This demonstrates that the conversion process is idempotent - 
    applying it multiple times produces the same result as applying it once.

    Test Cases:
    -------------------------------------------------------------------------
    - Regular numbers
    - Very small numbers
    - Very large numbers
    - Values that may lose precision in conversion

    Usage:
    -------------------------------------------------------------------------
    Compile standalone on the host (CPU):
        g++ -I../include cast.cpp -std=c++17 -o cast_test.exe

    Or with CUDA (GPU), if available:
        nvcc -I../include cast.cpp -std=c++17 -o cast_test.exe
*/

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cfloat>
#include <inttypes.h>
#include <cstring>

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

// Number of test values
#define NUM_TESTS 10

// Structure to hold conversion results
struct ConversionResult 
{
    double original;
    double after_1st_conversion;
    double after_2nd_conversion;
    double after_3rd_conversion;
    int is_stable;  // 1 if stable, 0 otherwise
    int bits_lost;  // Number of bits that differ after 1st conversion
    double abs_error;  // Absolute error after 1st conversion
    double rel_error;  // Relative error after 1st conversion
};

// Find the position of the first unmatched bit (MSB to LSB)
// Returns 64 - position of first different bit
HOST_DEVICE int count_bits_lost(uint64_t diff) 
{
    if (diff == 0) 
    {
        return 0;  // No bits differ
    }
    
    // Find position of most significant bit that differs (counting from LSB = 0)
    int position = 0;
    
    // Manual implementation: find MSB position
    uint64_t temp = diff;
    while (temp > 1) 
    {
        temp >>= 1;
        position++;
    }

    // Return 64 - position (bits from MSB to first difference)
    return position+1;
} // count_bits_lost

// Perform sequential cast test on a single value
HOST_DEVICE void test_sequential_cast(double original, ConversionResult* result) 
{
    // Store original value
    result->original = original;
    
    // First conversion: double -> float-float -> double
    ffloat ff1 = ffloat(original);
    double d1 = static_cast<double>(ff1);
    result->after_1st_conversion = d1;
    
    // Second conversion: double -> float-float -> double
    ffloat ff2 = ffloat(d1);
    double d2 = static_cast<double>(ff2);
    result->after_2nd_conversion = d2;
    
    // Third conversion: double -> float-float -> double
    ffloat ff3 = ffloat(d2);
    double d3 = static_cast<double>(ff3);
    result->after_3rd_conversion = d3;
    
    // Check if stable (d1 == d2 == d3)
    result->is_stable = (d1 == d2) && (d2 == d3) ? 1 : 0;
    
    // Calculate position of first differing bit between original and after 1st conversion
    uint64_t orig_bits, conv_bits;
    memcpy(&orig_bits, &original, sizeof(double));
    memcpy(&conv_bits, &d1, sizeof(double));
    uint64_t diff = orig_bits ^ conv_bits;
    result->bits_lost = count_bits_lost(diff);
    
    // Calculate absolute and relative errors
    result->abs_error = fabs(original - d1);
    if (original != 0.0) {
        result->rel_error = result->abs_error / fabs(original);
    } else {
        result->rel_error = 0.0;
    }
} // test_sequential_cast

// Run all conversion tests
HOST_DEVICE void run_all_tests(ConversionResult* results) 
{
    // Test values covering various ranges with interesting mantissas
    double test_values[NUM_TESTS] = 
    {
        1234.567890123456777,
        0.1234567890123456777,        
        1.23456789012345e-10,
        1.2345678901234567891,
        -9.8765432109876543211,
        3.1415926535897932383,
        -1.98765432109876e-15,
        1.1111111111111111e10,
        -9.8765432109876543211e14,
        2.718281828459045235, 
    };
    
    for (int i = 0; i < NUM_TESTS; i++) {
        test_sequential_cast(test_values[i], &results[i]);
    }
} // run_all_tests

#if defined(__CUDACC__)
// CUDA kernel wrapper
KERNEL void cast_test_kernel(ConversionResult* results) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) 
    {
        run_all_tests(results);
    }
}
#endif

// Helper function to print a double in three formats on one line
void print_value_triple(const char* label, double value)
{
    // Get bit representation
    uint64_t bits;
    memcpy(&bits, &value, sizeof(double));
    
    // Print full mantissa without trailing zeros (except for pure zeros)
    // Use %a which automatically omits trailing zeros
    printf("  %-22s  %.17e  0x%016" PRIx64 "  %a\n", 
           label, value, bits, value);
}

void print_results(ConversionResult* results) 
{
    printf("\n");
    printf("================================================================================\n");
    printf("  FLOAT-FLOAT CAST IDEMPOTENCY TEST\n");
    printf("  Testing: double -> float-float -> double -> float-float -> double\n");
    printf("================================================================================\n\n");
    
    printf("Results:\n");
    printf("-------------------------------------------------------------------------------------------------------------------\n");
    printf("%-5s | %-22s | %-8s | %-10s | %-5s | %-10s | %-10s\n", 
           "Test", "Original Value", "Stable?", "Max Delta", "Loss", "Abs Error", "Rel Error");
    printf("-------------------------------------------------------------------------------------------------------------------\n");
    
    int all_stable = 1;
    
    for (int i = 0; i < NUM_TESTS; i++) {
        ConversionResult* r = &results[i];
        
        // Calculate maximum difference between conversions
        double delta1 = fabs(r->after_1st_conversion - r->after_2nd_conversion);
        double delta2 = fabs(r->after_2nd_conversion - r->after_3rd_conversion);
        double max_delta = (delta1 > delta2) ? delta1 : delta2;
        
        const char* stable_str = r->is_stable ? "YES" : "NO";
        
        printf("%-5d | %22.15e | %-8s | %-10.2e | %-5d | %-10.2e | %-10.2e\n", 
               i+1, r->original, stable_str, max_delta, r->bits_lost, 
               r->abs_error, r->rel_error);
        
        if (!r->is_stable) {
            all_stable = 0;
        }
    }
    
    printf("--------------------------------------------------------------------------------\n\n");
    
    // Print detailed view for any unstable conversions
    int found_unstable = 0;
    for (int i = 0; i < NUM_TESTS; i++) {
        if (!results[i].is_stable) {
            if (!found_unstable) {
                printf("Detailed view of UNSTABLE conversions:\n");
                printf("--------------------------------------------------------------------------------\n");
                found_unstable = 1;
            }
            
            ConversionResult* r = &results[i];
            printf("\n  %-22s  %-23s  %-18s  %s\n", 
                   "Stage", "Decimal", "Hex int64", "C99 hex");
            printf("  %s\n", "------------------------------------------------------------------------------");
            print_value_triple("Original:", r->original);
            print_value_triple("After 1st conversion:", r->after_1st_conversion);
            print_value_triple("After 2nd conversion:", r->after_2nd_conversion);
            print_value_triple("After 3rd conversion:", r->after_3rd_conversion);
            printf("  Bits lost (1st conv):   %d, ", r->bits_lost);
            printf("Abs err: %.15e, ", r->abs_error);
            printf("Rel err: %.15e, ", r->rel_error);
            printf("Delta (1st vs 2nd): %.2e, ", 
                   fabs(r->after_1st_conversion - r->after_2nd_conversion));
            printf("  Delta (2nd vs 3rd): %.2e\n", 
                   fabs(r->after_2nd_conversion - r->after_3rd_conversion));
            printf("\tConversions   DIFFER!!!\n");
        }
    }
    
    if (found_unstable) 
    {
        printf("--------------------------------------------------------------------------------\n");
    }
    
    // Print detailed view for some stable conversions
    printf("\nDetailed view of first 3 STABLE conversions:\n");
    printf("--------------------------------------------------------------------------------\n");
    int stable_count = 0;
    for (int i = 0; i < NUM_TESTS && stable_count < NUM_TESTS; i++) 
    {
        if (results[i].is_stable) 
        {
            ConversionResult* r = &results[i];
            printf("\n  %-22s  %-23s  %-18s  %s\n", 
                   "Stage", "Decimal", "Hex int64", "C99 hex");
            printf("  %s\n", "------------------------------------------------------------------------------");
            print_value_triple("Original:", r->original);
            print_value_triple("After 1st conversion:", r->after_1st_conversion);
            print_value_triple("After 2nd conversion:", r->after_2nd_conversion);
            print_value_triple("After 3rd conversion:", r->after_3rd_conversion);
            printf("  Bits lost (1st conv):   %d, ", r->bits_lost);
            printf("Abs err: %.8e, ", r->abs_error);
            printf("Rel err: %.8e\n", r->rel_error);
            printf("\tAll conversions   MATCH\n");
            stable_count++;
        }
    }

    // Summary
    printf("================================================================================\n");
    if (all_stable) 
    {
        printf("  + ALL TESTS PASSED: All conversions are IDEMPOTENT\n");
        printf("    After the first conversion, all values remain stable.\n");
    } else 
    {
        int stable_tests = 0;
        for (int i = 0; i < NUM_TESTS; i++) 
        {
            if (results[i].is_stable) stable_tests++;
        }
        printf("  - SOME TESTS FAILED: %d/%d conversions are stable\n", 
               stable_tests, NUM_TESTS);
        printf("    Some values do not stabilize after the first conversion.\n");
    }
    printf("================================================================================\n");
} // print_results

int main() 
{
    // Allocate memory for results
    ConversionResult* results;
    
#if defined(__CUDACC__)
    // CUDA path
    cudaMallocManaged(&results, NUM_TESTS * sizeof(ConversionResult));
    
    printf("Running cast idempotency test on GPU...\n");
    cast_test_kernel<<<1, 1>>>(results);
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(results);
        return 1;
    }
    printf("GPU execution completed.\n");
#else
    // CPU path
    results = (ConversionResult*)malloc(NUM_TESTS * sizeof(ConversionResult));
    printf("Running cast idempotency test on CPU...\n");
    run_all_tests(results);
    printf("CPU execution completed.\n");
#endif
    
    // Print results
    print_results(results);
    
    printf("\nThis demo tested the idempotency property of conversions:\n");
    printf("  • double -> float-float conversion\n");
    printf("  • float-float -> double conversion\n");
    printf("  • Repeated sequential conversions (3 cycles)\n");
    printf("  • Various input ranges (small, large, regular numbers)\n");
#if defined(__CUDACC__)
    printf("  • Executed on GPU using CUDA\n");
#else
    printf("  • Executed on CPU\n");
#endif
    
    // Cleanup
#if defined(__CUDACC__)
    cudaFree(results);
#else
    free(results);
#endif
    
    printf("\n");
    return 0;
} // main

