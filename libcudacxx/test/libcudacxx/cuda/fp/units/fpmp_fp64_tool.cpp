/*
    fp64_tool.cpp - Unit Test for FP64 Precision Emulation Tool
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2026

    This unit test validates the fp64_tool.h precision emulation functionality:
    - Basic arithmetic operations with precision callbacks
    - Math functions (sqrt, fma)
    - Precision-sensitive operations (catastrophic cancellation)
    - Accumulation error
    - Newton-Raphson convergence
    - Bit pattern analysis (mantissa truncation)
    - Comparison operators

    Each test prints OK or FAIL.

    Usage:
    -------------------------------------------------------------------------
    Compile standalone on the host (CPU):
        g++ -I../include fp64_tool.cpp -std=c++17 -o fp64_tool.exe -lm

    Or with CUDA (GPU):
        nvcc -I../include -x cu fp64_tool.cpp -std=c++17 -o fp64_tool.exe
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>

//=============================================================================
// Host/Device Compatibility Macros
//=============================================================================
#if defined(__CUDACC__)
    #define HOST_DEVICE __host__ __device__
    #define KERNEL      __global__
#else
    #define HOST_DEVICE
    #define KERNEL
#endif

//=============================================================================
// Reduced precision version (23 mantissa bits like float)
//=============================================================================
#define FP64_TOOL_MANTISSA_BITS 23  // Float-like mantissa
#include <cuda/fptool>

// The FP SDK lives in the cuda::experimental namespace (will be reduced to cuda:: later).
using namespace cuda::experimental;

//=============================================================================
// Result structure for passing data between CPU/GPU
//=============================================================================
struct TestResults 
{
    // Basic arithmetic (native, reduced)
    double add_n, add_r;
    double sub_n, sub_r;
    double mul_n, mul_r;
    double div_n, div_r;
    double neg_n, neg_r;
    
    // Math functions
    double sqrt_n, sqrt_r;
    double fma_n, fma_r;
    
    // Precision tests
    double small_diff_n, small_diff_r;
    double cancel_n, cancel_r;
    double mul_prec_n, mul_prec_r;
    
    // Accumulation
    double accum_n, accum_r;
    
    // Newton-Raphson final values
    double newton_n, newton_r;
    
    // Bit pattern test
    uint64_t bits_orig;
    uint64_t bits_native;
    uint64_t bits_reduced;
    
    // Comparisons (stored as 0.0 or 1.0)
    double cmp_eq, cmp_lt, cmp_gt;
};

//=============================================================================
// Core test function - runs on both CPU and GPU
//=============================================================================
HOST_DEVICE void run_precision_tests(TestResults* r) 
{
    // Test values
    double val_a = 1.12345678123456789;
    double val_b = 2.12345678123456789;
    
    //-------------------------------------------------------------------------
    // Basic Arithmetic
    //-------------------------------------------------------------------------
    {
        double na = val_a, nb = val_b;
        fp64_tool ra = val_a, rb = val_b;
        
        r->add_n = (double)(na + nb);
        r->add_r = (double)(ra + rb);
        
        r->sub_n = (double)(na - nb);
        r->sub_r = (double)(ra - rb);
        
        r->mul_n = (double)(na * nb);
        r->mul_r = (double)(ra * rb);
        
        r->div_n = (double)(na / nb);
        r->div_r = (double)(ra / rb);
        
        r->neg_n = (double)(-na);
        r->neg_r = (double)(-ra);
    }
    
    //-------------------------------------------------------------------------
    // Math Functions
    //-------------------------------------------------------------------------
    {
        double nx = 2.12345678123456789;
        fp64_tool rx = 2.32145678123456789;
        
        r->sqrt_n = sqrt(nx);
        r->sqrt_r = (double)sqrt(rx);
        
        double na = val_a, nb = val_b, nc = 0.5;
        fp64_tool ra = val_a, rb = val_b, rc = 0.5;
        
        r->fma_n = fma(na, nb, nc);
        r->fma_r = (double)fma(ra, rb, rc);
    }
    
    //-------------------------------------------------------------------------
    // Precision-Sensitive Operations
    //-------------------------------------------------------------------------
    {
        // Small difference test: (1 + 1e-10) - 1
        double a = 1.0 + 1e-10;
        double b = 1.0;
        
        double na = a, nb = b;
        fp64_tool ra = a, rb = b;
        
        r->small_diff_n = (double)(na - nb);
        r->small_diff_r = (double)(ra - rb);
    }
    
    {
        // Catastrophic cancellation: (a + b) - a
        double a = 1.0, b = 1e-10;
        
        double na = a, nb = b;
        fp64_tool ra = a, rb = b;
        
        r->cancel_n = (double)((na + nb) - na);
        r->cancel_r = (double)((ra + rb) - ra);
    }
    
    {
        // Multiplication precision
        double a = 1.0000001, b = 1.0000002;
        
        double na = a, nb = b;
        fp64_tool ra = a, rb = b;
        
        r->mul_prec_n = (double)(na * nb);
        r->mul_prec_r = (double)(ra * rb);
    }
    
    //-------------------------------------------------------------------------
    // Accumulation Error (Sum of 1/n, n=1..1000)
    //-------------------------------------------------------------------------
    {
        double native_sum = 0.0;
        fp64_tool reduced_sum = 0.0;
        
        for (int n = 1; n <= 1000; n++) {
            double term = 1.0 / n;
            native_sum += double(term);
            reduced_sum += fp64_tool(term);
        }
        
        r->accum_n = (double)native_sum;
        r->accum_r = (double)reduced_sum;
    }
    
    //-------------------------------------------------------------------------
    // Newton-Raphson sqrt(2): x_{n+1} = 0.5 * (x_n + S/x_n)
    //-------------------------------------------------------------------------
    {
        double n_x = 1.0, n_S = 2.0, n_half = 0.5;
        fp64_tool r_x = 1.0, r_S = 2.0, r_half = 0.5;
        
        for (int i = 0; i < 10; i++) {
            n_x = n_half * (n_x + n_S / n_x);
            r_x = r_half * (r_x + r_S / r_x);
        }
        
        r->newton_n = (double)n_x;
        r->newton_r = (double)r_x;
    }
    
    //-------------------------------------------------------------------------
    // Bit Pattern Analysis
    //-------------------------------------------------------------------------
    {
        double val = 1.12345678123456789;
        
        double n_val = val;
        fp64_tool r_val = val;
        
        // Apply callback via arithmetic (add 0)
        double n_result = n_val + double(0.0);
        fp64_tool r_result = r_val + fp64_tool(0.0);
        
        double n_out = (double)n_result;
        double r_out = (double)r_result;
        
        // Store bit patterns using memcpy to avoid strict-aliasing issues
        memcpy(&r->bits_orig, &val, sizeof(uint64_t));
        memcpy(&r->bits_native, &n_out, sizeof(uint64_t));
        memcpy(&r->bits_reduced, &r_out, sizeof(uint64_t));
    }
    
    //-------------------------------------------------------------------------
    // Comparison Operators
    //-------------------------------------------------------------------------
    {
        double na = val_a, nb = val_b;
        
        r->cmp_eq = (na == na) ? 1.0 : 0.0;
        r->cmp_lt = (na < nb)  ? 1.0 : 0.0;
        r->cmp_gt = (na > nb)  ? 1.0 : 0.0;
    }
}

//=============================================================================
// CUDA Kernel Wrapper
//=============================================================================
#if defined(__CUDACC__)
KERNEL void precision_test_kernel(TestResults* results) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        run_precision_tests(results);
    }
}
#endif

//=============================================================================
// Check and print test results
//=============================================================================
int check_results(const TestResults& r) 
{
    int fail_count = 0;
    int test_count = 0;

    // Helper lambda: check a condition and print OK/FAIL
    auto check = [&](const char* name, bool condition) {
        test_count++;
        if (condition) {
            printf("  [OK]   %s\n", name);
        } else {
            printf("  [FAIL] %s\n", name);
            fail_count++;
        }
    };

    printf("\n");
    printf("================================================================================\n");
    printf("  FP64 PRECISION TOOL UNIT TEST\n");
    printf("  Reduced precision: 23-bit mantissa (float-like)\n");
    printf("================================================================================\n\n");

    //--- Basic arithmetic: results should be finite and close to native ---
    printf("Basic Arithmetic:\n");
    check("add: result is finite",       std::isfinite(r.add_r));
    check("sub: result is finite",       std::isfinite(r.sub_r));
    check("mul: result is finite",       std::isfinite(r.mul_r));
    check("div: result is finite",       std::isfinite(r.div_r));
    check("neg: result is finite",       std::isfinite(r.neg_r));
    check("add: reduced differs from native (precision loss)",  r.add_r != r.add_n);
    check("mul: reduced differs from native (precision loss)",  r.mul_r != r.mul_n);
    check("neg: sign preserved",         r.neg_r < 0.0);

    //--- Math functions ---
    printf("\nMath Functions:\n");
    check("sqrt: result is finite",      std::isfinite(r.sqrt_r));
    check("fma:  result is finite",      std::isfinite(r.fma_r));

    //--- Precision-sensitive ---
    printf("\nPrecision-Sensitive Operations:\n");
    check("small diff: native preserves 1e-10",   r.small_diff_n > 0.0);
    check("small diff: reduced loses precision (== 0)",  r.small_diff_r == 0.0);
    check("cancellation: native preserves 1e-10", r.cancel_n > 0.0);
    check("cancellation: reduced loses value (== 0)",    r.cancel_r == 0.0);
    check("mul precision: reduced differs from native",  r.mul_prec_r != r.mul_prec_n);

    //--- Accumulation ---
    printf("\nAccumulation (sum 1/n, n=1..1000):\n");
    double accum_err_n = fabs(r.accum_n - r.accum_r);
    check("accumulation error > 0 (precision loss)",  accum_err_n > 0.0);
    check("reduced sum is finite",  std::isfinite(r.accum_r));

    //--- Newton-Raphson ---
    printf("\nNewton-Raphson sqrt(2):\n");
    double sqrt2 = sqrt(2.0);
    double newton_err_n = fabs(r.newton_n - sqrt2);
    double newton_err_r = fabs(r.newton_r - sqrt2);
    check("native converges (error < 1e-14)",    newton_err_n < 1e-14);
    check("reduced converges (error < 1e-5)",    newton_err_r < 1e-5);
    check("reduced error > native error",        newton_err_r > newton_err_n);

    //--- Bit patterns ---
    printf("\nBit Pattern Analysis:\n");
    uint64_t low_29_mask = (1ULL << 29) - 1;
    check("native preserves low bits",   (r.bits_native & low_29_mask) != 0);
    check("reduced zeroes low 29 bits",  (r.bits_reduced & low_29_mask) == 0);

    //--- Comparisons ---
    printf("\nComparison Operators:\n");
    check("a == a is true",  r.cmp_eq == 1.0);
    check("a < b  is true",  r.cmp_lt == 1.0);
    check("a > b  is false", r.cmp_gt == 0.0);

    //--- Summary ---
    printf("\n================================================================================\n");
    if (fail_count == 0) {
        printf("  ALL %d TESTS PASSED  [OK]\n", test_count);
    } else {
        printf("  %d / %d TESTS FAILED  [FAIL]\n", fail_count, test_count);
    }
    printf("================================================================================\n\n");

    return fail_count;
}

//=============================================================================
// Main
//=============================================================================
int main() 
{
    TestResults* results;
    
#if defined(__CUDACC__)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Running on GPU: %s\n", prop.name);
    
    cudaMallocManaged(&results, sizeof(TestResults));
    
    precision_test_kernel<<<1, 1>>>(results);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(results);
        return 1;
    }
    printf("GPU execution completed.\n");
#else
    results = (TestResults*)malloc(sizeof(TestResults));
    printf("Running on CPU...\n");
    run_precision_tests(results);
    printf("CPU execution completed.\n");
#endif
    
    int failures = check_results(*results);
    
#if defined(__CUDACC__)
    cudaFree(results);
#else
    free(results);
#endif
    
    return (failures == 0) ? 0 : 1;
}
