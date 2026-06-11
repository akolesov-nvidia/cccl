/*
    atomic_ff.cpp - Sanity Test for atomicAdd/atomicSub on fp32mp2 (Float-Float) Types
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This test verifies the correctness of the atomicAdd and atomicSub operations.

    Test 1 - Atomicity:
    -------------------------------------------------------------------------
    Multiple threads atomically add and subtract values. If the operations are truly atomic, 
    the final result should be zero (within rounding error). This test adds 1.0 to a shared 
    memory location from different threads using atomicAdd, then subtracts 1.0 using atomicSub. 
    If both are done atomically, there's guarantee that the result remains zero after every 
    thread finishes execution (provided there's no rounding error in accumulation).

    Test 2 - Accuracy:
    -------------------------------------------------------------------------
    Multiple threads perform actual accumulation to verify that the atomic operations produce
    mathematically correct results, not just atomic ones:
    - Test 1: Adds small values (0.1) from all threads and compares with expected sum
    - Test 2: Adds and subtracts the same value (0.5) to verify cancellation
    - Test 3: Subtracts values (0.25) from all threads and compares with expected result
*/

#include <cmath>
#include <iostream>
#include <iomanip>

#if defined(__CUDACC__)

#include <cuda_runtime.h>
#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the multi-precision floating-point type
using ffloat = fp32mp2;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel for testing atomicAdd and atomicSub
__global__ void test_atomicity_kernel(unsigned int *idx, ffloat *res)
{
    ffloat val(1.0f);
    // Count participating threads
    atomicAdd(idx, 1u);
    // Add 1.0 using atomicAdd, then subtract 1.0 using atomicSub
    // If both operations work correctly, these should cancel out
    atomicAdd(res, val);
    atomicSub(res, val);
} // test_atomicity_kernel

// Kernel for testing accuracy of atomicAdd
__global__ void test_atomicAdd_accuracy_kernel(ffloat *res, float value_to_add)
{
    ffloat val(value_to_add);
    atomicAdd(res, val);
} // test_atomicAdd_accuracy_kernel

// Kernel for testing accuracy of atomicSub
__global__ void test_atomicSub_accuracy_kernel(ffloat *res, float value_to_sub)
{
    ffloat val(value_to_sub);
    atomicSub(res, val);
} // test_atomicSub_accuracy_kernel
// Test function for atomicity
bool test_atomicAdd(int num_threads = 512, int num_blocks = 4)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing atomicAdd/atomicSub for fp32mp2 (fpmp2_accuracy::def)" << std::endl;
    std::cout << "  Threads per block: " << num_threads << std::endl;
    std::cout << "  Number of blocks:  " << num_blocks << std::endl;
    std::cout << "  Total threads:     " << (num_threads * num_blocks) << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Allocate device memory
    // Note: cudaMalloc returns properly aligned memory (256+ bytes on modern GPUs)
    // With alignas(2*alignof(float)) on the type, 8-byte alignment is guaranteed
    unsigned int *d_idx;
    ffloat *d_res;
    
    CUDA_CHECK(cudaMalloc(&d_idx, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(ffloat)));
    
    // Initialize device memory
    unsigned int h_idx = 0;
    ffloat h_res(0.0f);
    
    CUDA_CHECK(cudaMemcpy(d_idx, &h_idx, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(ffloat), cudaMemcpyHostToDevice));
    
    // Launch kernel
    test_atomicity_kernel<<<num_blocks, num_threads>>>(d_idx, d_res);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(&h_idx, d_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(ffloat), cudaMemcpyDeviceToHost));
    
    // Check results
    std::cout << "  Participating threads: " << h_idx << std::endl;
    std::cout << "  Final result (hi):     " << std::scientific << std::setprecision(10) << h_res.hi() << std::endl;
    std::cout << "  Final result (lo):     " << std::scientific << std::setprecision(10) << h_res.lo() << std::endl;
    std::cout << "  Final result (double): " << std::scientific << std::setprecision(10) << static_cast<double>(h_res) << std::endl;
    
    // Verify the result is close to zero
    double result_double = static_cast<double>(h_res);
    double abs_error = fabs(result_double);
    
    // Expected error should be very small (ideally zero for exact cancellation)
    double tolerance = 1e-6;  // Allow small numerical errors
    
    bool passed = abs_error < tolerance;
    
    if (passed) { std::cout << "  + TEST PASSED (error:  " << abs_error << ")" << std::endl; } 
    else        { std::cout << "  - TEST FAILED (error:  " << abs_error << " > tolerance: " << tolerance << ")" << std::endl; }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_res));
    
    return passed;
} // test_atomicAdd

// Test function for accuracy of atomicAdd and atomicSub
bool test_atomic_accuracy(int num_threads = 512, int num_blocks = 4)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing atomicAdd/atomicSub accuracy for fp32mp2 (fpmp2_accuracy::def)" << std::endl;
    std::cout << "  Threads per block: " << num_threads << std::endl;
    std::cout << "  Number of blocks:  " << num_blocks << std::endl;
    std::cout << "  Total threads:     " << (num_threads * num_blocks) << std::endl;
    std::cout << "========================================" << std::endl;
    
    int total_threads = num_threads * num_blocks;
    
    // Test 1: Add small values
    {
        ffloat *d_res;
        CUDA_CHECK(cudaMalloc(&d_res, sizeof(ffloat)));
        
        ffloat h_res(0.0f);
        CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(ffloat), cudaMemcpyHostToDevice));
        
        float value_to_add = 0.1f;
        test_atomicAdd_accuracy_kernel<<<num_blocks, num_threads>>>(d_res, value_to_add);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(ffloat), cudaMemcpyDeviceToHost));
        
        double result = static_cast<double>(h_res);
        double expected = value_to_add * total_threads;
        double error = fabs(result - expected);
        double rel_error = error / expected;
        
        std::cout << "\n  Test 1: Adding " << value_to_add << " from " << total_threads << " threads" << std::endl;
        std::cout << "    Expected:      " << std::scientific << std::setprecision(10) << expected << std::endl;
        std::cout << "    Computed:      " << std::scientific << std::setprecision(10) << result << std::endl;
        std::cout << "    Abs error:     " << std::scientific << std::setprecision(4) << error << std::endl;
        std::cout << "    Rel error:     " << std::scientific << std::setprecision(4) << rel_error << std::endl;
        
        CUDA_CHECK(cudaFree(d_res));
        
        if (rel_error > 1e-5) {
            std::cout << "    - FAILED (accuracy check)" << std::endl;
            return false;
        }
        std::cout << "    + PASSED" << std::endl;
    }
    
    // Test 2: Add then subtract (combined accuracy test)
    {
        ffloat *d_res;
        CUDA_CHECK(cudaMalloc(&d_res, sizeof(ffloat)));
        
        ffloat h_res(100.0f);
        CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(ffloat), cudaMemcpyHostToDevice));
        
        float value = 0.5f;
        test_atomicAdd_accuracy_kernel<<<num_blocks, num_threads>>>(d_res, value);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        test_atomicSub_accuracy_kernel<<<num_blocks, num_threads>>>(d_res, value);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(ffloat), cudaMemcpyDeviceToHost));
        
        double result = static_cast<double>(h_res);
        double expected = 100.0;
        double error = fabs(result - expected);
        double rel_error = error / expected;
        
        std::cout << "\n  Test 2: Adding and subtracting " << value << " from " << total_threads << " threads (start: 100.0)" << std::endl;
        std::cout << "    Expected:      " << std::scientific << std::setprecision(10) << expected << std::endl;
        std::cout << "    Computed:      " << std::scientific << std::setprecision(10) << result << std::endl;
        std::cout << "    Abs error:     " << std::scientific << std::setprecision(4) << error << std::endl;
        std::cout << "    Rel error:     " << std::scientific << std::setprecision(4) << rel_error << std::endl;
        
        CUDA_CHECK(cudaFree(d_res));
        
        if (rel_error > 1e-5) {
            std::cout << "    - FAILED (accuracy check)" << std::endl;
            return false;
        }
        std::cout << "    + PASSED" << std::endl;
    }
    
    // Test 3: Subtract values
    {
        ffloat *d_res;
        CUDA_CHECK(cudaMalloc(&d_res, sizeof(ffloat)));
        
        ffloat h_res(1000.0f);
        CUDA_CHECK(cudaMemcpy(d_res, &h_res, sizeof(ffloat), cudaMemcpyHostToDevice));
        
        float value_to_sub = 0.25f;
        test_atomicSub_accuracy_kernel<<<num_blocks, num_threads>>>(d_res, value_to_sub);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_res, d_res, sizeof(ffloat), cudaMemcpyDeviceToHost));
        
        double result = static_cast<double>(h_res);
        double expected = 1000.0 - (value_to_sub * total_threads);
        double error = fabs(result - expected);
        double rel_error = error / expected;
        
        std::cout << "\n  Test 3: Subtracting " << value_to_sub << " from " << total_threads << " threads (start: 1000.0)" << std::endl;
        std::cout << "    Expected:      " << std::scientific << std::setprecision(10) << expected << std::endl;
        std::cout << "    Computed:      " << std::scientific << std::setprecision(10) << result << std::endl;
        std::cout << "    Abs error:     " << std::scientific << std::setprecision(4) << error << std::endl;
        std::cout << "    Rel error:     " << std::scientific << std::setprecision(4) << rel_error << std::endl;
        
        CUDA_CHECK(cudaFree(d_res));
        
        if (rel_error > 1e-5) {
            std::cout << "    - FAILED (accuracy check)" << std::endl;
            return false;
        }
        std::cout << "    + PASSED" << std::endl;
    }
    
    std::cout << "\n  + ALL ACCURACY TESTS PASSED" << std::endl;
    return true;
} // test_atomic_accuracy

int main(int argc, char** argv)
{
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "AtomicAdd/AtomicSub Sanity Test for fpmp2_t" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Get device properties
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    int num_threads = 512;
    int num_blocks  = 4;
    
    // Parse command line arguments
    if (argc > 1) num_threads = atoi(argv[1]);
    if (argc > 2) num_blocks  = atoi(argv[2]);
    
    bool all_passed = true;
    
    // Atomicity tests
    std::cout << "\n" << std::string(65, '=') << std::endl;
    std::cout << "ATOMICITY TESTS (add 1.0 then subtract 1.0 to cancel out to 0.0)" << std::endl;
    std::cout << std::string(65, '=') << std::endl;
    all_passed &= test_atomicAdd(num_threads, num_blocks);
    
    // Accuracy tests
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "ACCURACY TESTS (check arithmetic correctness)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    all_passed &= test_atomic_accuracy(num_threads, num_blocks);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (all_passed) { std::cout << "+ ALL TESTS PASSED"  << std::endl; } 
    else            { std::cout << "- SOME TESTS FAILED" << std::endl; }
    std::cout << std::string(60, '=') << std::endl << std::endl;
    
    return all_passed ? 0 : 1;
} // main

#else // __CUDACC__

int main(int argc, char** argv)
{
    std::cout << "This example is only available on CUDA GPUs." << std::endl;
    return 0;
} // main

#endif // __CUDACC__