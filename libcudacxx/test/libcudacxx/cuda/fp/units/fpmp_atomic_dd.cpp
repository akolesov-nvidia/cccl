/*
    atomic_dd.cpp - Sanity Test for atomicAdd/atomicSub on fp64mp2 (Double-Double) Types
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This test verifies the correctness of the atomicAdd and atomicSub operations for double-double
    multi-precision types (fp64mp2). It requires CUDA compute capability >= 9.0 (Hopper architecture)
    for 128-bit atomicCAS support.

    Test 1 - Atomicity:
    -------------------------------------------------------------------------
    Multiple threads atomically add and subtract values. If the operations are truly atomic, 
    the final result should be zero (within rounding error). This test adds 1.0 to a shared 
    memory location from different threads using atomicAdd, then subtracts 1.0 using atomicSub. 
    If both are done atomically, there's guarantee that the result remains zero after every 
    thread finishes execution.

    Test 2 - Accuracy:
    -------------------------------------------------------------------------
    Multiple threads perform actual accumulation to verify that the atomic operations produce
    mathematically correct results, not just atomic ones:
    - Test 1: Adds small values (0.1) from all threads and compares with expected sum
    - Test 2: Adds and subtracts the same value (0.5) to verify cancellation
    - Test 3: Subtracts values (0.25) from all threads and compares with expected result

    Usage:
    -------------------------------------------------------------------------
        make UNIT=atomic_dd ARCH=90 run     # For Hopper GPUs
        make UNIT=atomic_dd ARCH=100 run    # For Blackwell GPUs
*/

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>

#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if defined(__CUDACC__)

#include <cuda_runtime.h>

// Type alias for the double-double multi-precision floating-point type
using dfloat = fp64mp2;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Helper kernel to detect compiled architecture
__global__ void get_compiled_arch_kernel(int* arch_out)
{
#ifdef __CUDA_ARCH__
    *arch_out = __CUDA_ARCH__;
#else
    *arch_out = 0;
#endif
}

int get_compiled_arch()
{
    int* d_arch;
    int h_arch = 0;
    cudaMalloc(&d_arch, sizeof(int));
    cudaMemset(d_arch, 0, sizeof(int));
    get_compiled_arch_kernel<<<1, 1>>>(d_arch);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arch);
    return h_arch;
}

// Kernel for testing atomicAdd and atomicSub
__global__ void test_atomicity_kernel_dd(dfloat *res)
{
    dfloat val(1.0);
    // Add 1.0 using atomicAdd, then subtract 1.0 using atomicSub
    // If both operations work correctly, these should cancel out
    atomicAdd(res, val);
    atomicSub(res, val);
} // test_atomicity_kernel_dd

// Kernel for testing accuracy of atomicAdd
__global__ void test_atomicAdd_accuracy_kernel_dd(dfloat *res, double value_to_add)
{
    dfloat val(value_to_add);
    atomicAdd(res, val);
} // test_atomicAdd_accuracy_kernel_dd

// Kernel for testing accuracy of atomicSub
__global__ void test_atomicSub_accuracy_kernel_dd(dfloat *res, double value_to_sub)
{
    dfloat val(value_to_sub);
    atomicSub(res, val);
} // test_atomicSub_accuracy_kernel_dd

// Test function for atomicity
bool test_atomicAdd_dd(int num_threads = 512, int num_blocks = 4)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing atomicAdd/atomicSub for fp64mp2 (double-double)" << std::endl;
    std::cout << "  Threads per block: " << num_threads << std::endl;
    std::cout << "  Number of blocks:  " << num_blocks << std::endl;
    std::cout << "  Total threads:     " << (num_threads * num_blocks) << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Allocate device memory
    // Note: cudaMalloc returns properly aligned memory (256+ bytes on modern GPUs)
    // With alignas(2*alignof(double)) on the type, 16-byte alignment is guaranteed for 128-bit atomics
    dfloat *d_res;
    
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(dfloat)));
    
    // Initialize device memory
    dfloat h_res(0.0);
    
    CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice));
    
    // Launch kernel
    test_atomicity_kernel_dd<<<num_blocks, num_threads>>>(d_res);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost));
    
    // Check results
    std::cout << "  Final result (hi):     " << std::scientific << std::setprecision(16) << h_res.hi() << std::endl;
    std::cout << "  Final result (lo):     " << std::scientific << std::setprecision(16) << h_res.lo() << std::endl;
    std::cout << "  Final result (double): " << std::scientific << std::setprecision(16) << static_cast<double>(h_res) << std::endl;
    
    // Verify the result is close to zero
    double result_double = static_cast<double>(h_res);
    double abs_error = fabs(result_double);
    
    // Expected error should be very small (ideally zero for exact cancellation)
    // Double-double has higher precision, so we use a tighter tolerance
    double tolerance = 1e-14;
    
    bool passed = abs_error < tolerance;
    
    if (passed) { std::cout << "  + TEST PASSED (error:  " << abs_error << ")" << std::endl; } 
    else        { std::cout << "  - TEST FAILED (error:  " << abs_error << " > tolerance: " << tolerance << ")" << std::endl; }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_res));
    
    return passed;
} // test_atomicAdd_dd

// Test function for accuracy of atomicAdd and atomicSub
bool test_atomic_accuracy_dd(int num_threads = 512, int num_blocks = 4)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing atomicAdd/atomicSub accuracy for fp64mp2 (double-double)" << std::endl;
    std::cout << "  Threads per block: " << num_threads << std::endl;
    std::cout << "  Number of blocks:  " << num_blocks << std::endl;
    std::cout << "  Total threads:     " << (num_threads * num_blocks) << std::endl;
    std::cout << "========================================" << std::endl;
    
    int total_threads = num_threads * num_blocks;
    
    // Test 1: Add small values
    {
        dfloat *d_res;
        CUDA_CHECK(cudaMalloc(&d_res, sizeof(dfloat)));
        
        dfloat h_res(0.0);
        CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice));
        
        double value_to_add = 0.1;
        test_atomicAdd_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value_to_add);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost));
        
        double result = static_cast<double>(h_res);
        double expected = value_to_add * total_threads;
        double error = fabs(result - expected);
        double rel_error = error / expected;
        
        std::cout << "\n  Test 1: Adding " << value_to_add << " from " << total_threads << " threads" << std::endl;
        std::cout << "    Expected:      " << std::scientific << std::setprecision(16) << expected << std::endl;
        std::cout << "    Computed:      " << std::scientific << std::setprecision(16) << result << std::endl;
        std::cout << "    Abs error:     " << std::scientific << std::setprecision(4) << error << std::endl;
        std::cout << "    Rel error:     " << std::scientific << std::setprecision(4) << rel_error << std::endl;
        
        CUDA_CHECK(cudaFree(d_res));
        
        // Double-double should achieve much better precision than float-float
        if (rel_error > 1e-14) {
            std::cout << "    - FAILED (accuracy check)" << std::endl;
            return false;
        }
        std::cout << "    + PASSED" << std::endl;
    }
    
    // Test 2: Add then subtract (combined accuracy test)
    {
        dfloat *d_res;
        CUDA_CHECK(cudaMalloc(&d_res, sizeof(dfloat)));
        
        dfloat h_res(100.0);
        CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice));
        
        double value = 0.5;
        test_atomicAdd_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        test_atomicSub_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost));
        
        double result = static_cast<double>(h_res);
        double expected = 100.0;
        double error = fabs(result - expected);
        double rel_error = error / expected;
        
        std::cout << "\n  Test 2: Adding and subtracting " << value << " from " << total_threads << " threads (start: 100.0)" << std::endl;
        std::cout << "    Expected:      " << std::scientific << std::setprecision(16) << expected << std::endl;
        std::cout << "    Computed:      " << std::scientific << std::setprecision(16) << result << std::endl;
        std::cout << "    Abs error:     " << std::scientific << std::setprecision(4) << error << std::endl;
        std::cout << "    Rel error:     " << std::scientific << std::setprecision(4) << rel_error << std::endl;
        
        CUDA_CHECK(cudaFree(d_res));
        
        if (rel_error > 1e-14) {
            std::cout << "    - FAILED (accuracy check)" << std::endl;
            return false;
        }
        std::cout << "    + PASSED" << std::endl;
    }
    
    // Test 3: Subtract values
    {
        dfloat *d_res;
        CUDA_CHECK(cudaMalloc(&d_res, sizeof(dfloat)));
        
        dfloat h_res(1000.0);
        CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice));
        
        double value_to_sub = 0.25;
        test_atomicSub_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value_to_sub);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost));
        
        double result = static_cast<double>(h_res);
        double expected = 1000.0 - (value_to_sub * total_threads);
        double error = fabs(result - expected);
        double rel_error = error / fabs(expected);
        
        std::cout << "\n  Test 3: Subtracting " << value_to_sub << " from " << total_threads << " threads (start: 1000.0)" << std::endl;
        std::cout << "    Expected:      " << std::scientific << std::setprecision(16) << expected << std::endl;
        std::cout << "    Computed:      " << std::scientific << std::setprecision(16) << result << std::endl;
        std::cout << "    Abs error:     " << std::scientific << std::setprecision(4) << error << std::endl;
        std::cout << "    Rel error:     " << std::scientific << std::setprecision(4) << rel_error << std::endl;
        
        CUDA_CHECK(cudaFree(d_res));
        
        if (rel_error > 1e-14) {
            std::cout << "    - FAILED (accuracy check)" << std::endl;
            return false;
        }
        std::cout << "    + PASSED" << std::endl;
    }
    
    std::cout << "\n  + ALL ACCURACY TESTS PASSED" << std::endl;
    return true;
} // test_atomic_accuracy_dd

int main(int argc, char** argv)
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "AtomicAdd/AtomicSub Sanity Test for fp64mp2 (double-double)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Get device properties
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Check compiled architecture
    int compiled_arch = get_compiled_arch();
    std::cout << "  Binary compiled for: sm_" << (compiled_arch / 10) << std::endl;
    
    // Check if compiled for sm_90+ (required for 128-bit atomics)
    if (compiled_arch < 900) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "SKIPPED: Double-double atomics require ARCH >= 90" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "\n  128-bit atomicCAS is only available on compute capability >= 9.0 (Hopper)" << std::endl;
        std::cout << "  This binary was compiled for sm_" << (compiled_arch / 10) << std::endl;
        std::cout << "\n  To run this test, recompile with:" << std::endl;
        std::cout << "    make UNIT=atomic_dd ARCH=90 run" << std::endl;
        std::cout << "\n" << std::string(60, '=') << std::endl << std::endl;
        return 0;  // Return success as skipping is not a failure
    }
    
    // Check if GPU supports 128-bit atomics
    if (prop.major < 9) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "SKIPPED: GPU does not support 128-bit atomics" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "\n  128-bit atomicCAS requires compute capability >= 9.0 (Hopper)" << std::endl;
        std::cout << "  This device has compute capability " << prop.major << "." << prop.minor << std::endl;
        std::cout << "\n" << std::string(60, '=') << std::endl << std::endl;
        return 0;  // Return success as skipping is not a failure
    }
    
    int num_threads = 512;
    int num_blocks  = 4;
    
    // Parse command line arguments
    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) num_blocks  = atoi(argv[2]);
    
    bool all_passed = true;
    
    // Atomicity tests
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "ATOMICITY TESTS (add + subtract should cancel out)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    all_passed &= test_atomicAdd_dd(num_threads, num_blocks);
    
    // Accuracy tests
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "ACCURACY TESTS (check arithmetic correctness)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    all_passed &= test_atomic_accuracy_dd(num_threads, num_blocks);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (all_passed) { std::cout << "+ ALL TESTS PASSED"  << std::endl; } 
    else            { std::cout << "- SOME TESTS FAILED" << std::endl; }
    std::cout << std::string(60, '=') << std::endl << std::endl;
    
    return all_passed ? 0 : 1;
} // main

#else // __CUDACC__

int main(int argc, char** argv)
{
    (void)argc; (void)argv;
    std::cout << "This test requires CUDA & fp64mp2 support" << std::endl;
    return 0;
} // main

#endif // __CUDACC__
