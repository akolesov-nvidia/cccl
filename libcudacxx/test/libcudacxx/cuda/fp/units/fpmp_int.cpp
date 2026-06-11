/*
    int.cpp - Unit Test for Integer Conversions with fp32mp2 (Float-Float)
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025

    This unit test verifies integer conversion correctness for fp32mp2 types.

    Test Cases:
    -------------------------------------------------------------------------
    1. Integer -> fp32mp2 conversions (should use round-toward-zero)
    2. fp32mp2 -> Integer conversions (should use round-toward-zero)
    3. Exactness: converting int->efloat->int should preserve the value
    4. Edge cases around float precision limits (2^24 for int32)

    Runs tests on both CPU and GPU for comparison.
*/

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <limits>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define HAS_CUDA 1
    #define CHECK_CUDA(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                        << cudaGetErrorString(err) << std::endl; \
                exit(EXIT_FAILURE); \
            } \
        } while(0)
#else // __CUDACC__
    #define HAS_CUDA 0
    #define __global__
    #define __device__
    #define __host__
#endif

#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the multi-precision floating-point type
using ffloat = fp32mp2;

// Structure to hold test results
struct TestResult 
{
    int32_t original_val;
    float hi;
    float lo;
    int32_t back_val;
    bool exact_match;
    bool sign_consistent;
};

struct UInt32TestResult 
{
    uint32_t original_val;
    float hi;
    float lo;
    uint32_t back_val;
    bool exact_match;
    bool all_positive;
};

struct TruncTestResult 
{
    double original_val;
    int32_t result;
    int32_t expected;
    bool correct;
};

struct NegativeLowTestResult 
{
    double original_val;
    float hi;
    float lo;
    int32_t result;
    int32_t expected;
    bool correct;
};

// ============================================================================
// CPU Tests
// ============================================================================

/**
 * @brief Print int32_t test result with exact match and sign consistency checks
 * @param r Test result containing original value, hi/lo parts, converted value, and validation flags
 */
void print_test_result(const TestResult& r) 
{
    std::cout << "\nOriginal int32: " << r.original_val << std::endl;
    std::cout << "  As ffloat: {hi: " << r.hi << ", lo: " << r.lo << "}" << std::endl;
    std::cout << "  Back to int32: " << r.back_val;
    
    if (r.exact_match) 
    {
        std::cout << " = PASS (exact)" << std::endl;
    } 
    else 
    {
        std::cout << " = FAIL (difference: " << (static_cast<int64_t>(r.back_val) - static_cast<int64_t>(r.original_val)) << ")" << std::endl;
    }
    
    if (r.sign_consistent) 
    {
        std::cout << "  = PASS: Sign consistent" << std::endl;
    } 
    else 
    {
        std::cout << "  = FAIL: Sign inconsistent" << std::endl;
    }
}

/**
 * @brief Test int32_t to fp32mp2 conversions on CPU
 * Tests edge cases around 2^24 precision limit and INT32_MAX/MIN
 * Validates: exact round-trip, sign consistency (lo part has same sign as input)
 * @return Number of errors detected (0 = all tests passed)
 */
int test_int32_cpu() 
{
    int err = 0;
    std::cout << "\n=== Testing int32_t conversions on CPU ===" << std::endl;
    
    int32_t test_values[] = 
    {
        0, 1, -1, 42, -42,
        16777215, 16777216, 16777217,  // Around 2^24
        -16777215, -16777216, -16777217,
        2147483647, -2147483648,  // INT32_MAX, INT32_MIN
    };
    
    for (int32_t val : test_values) 
    {
        ffloat x(val);
        int32_t back = static_cast<int32_t>(x);
        
        TestResult r;
        r.original_val = val;
        r.hi = x.hi();
        r.lo = x.lo();
        r.back_val = back;
        r.exact_match = (back == val);
        
        if (val > 0) 
        {
            r.sign_consistent = (x.lo() >= 0);
        } 
        else if (val < 0) 
        {
            r.sign_consistent = (x.lo() <= 0);
        } 
        else 
        {
            r.sign_consistent = true;
        }
        
        print_test_result(r);
        if (!r.exact_match) 
        {
            err += 1;
        }
        if (!r.sign_consistent) 
        {
            err += 1;
        }
    }

    return err;
}

/**
 * @brief Test uint32_t to fp32mp2 conversions on CPU
 * Tests edge cases around 2^24 precision limit and UINT32_MAX
 * Validates: exact round-trip, both hi/lo parts are non-negative
 * @return Number of errors detected (0 = all tests passed)
 */
int test_uint32_cpu() 
{
    int err = 0;
    std::cout << "\n=== Testing uint32_t conversions on CPU ===" << std::endl;
    
    uint32_t test_values[] = 
    {
        0u, 1u, 42u,
        16777215u, 16777216u, 16777217u,
        4294967295u,  // UINT32_MAX
    };
    
    for (uint32_t val : test_values) 
    {
        ffloat x(val);
        uint32_t back = static_cast<uint32_t>(x);
        
        std::cout << "\nOriginal uint32: " << val << std::endl;
        std::cout << "  As ffloat: {hi: " << x.hi() << ", lo: " << x.lo() << "}" << std::endl;
        std::cout << "  Back to uint32: " << back;
        
        if (back == val) 
        {
            std::cout << " = PASS (exact)" << std::endl;
        } 
        else 
        {
            std::cout << " = FAIL (difference: " << (static_cast<int64_t>(back) - static_cast<int64_t>(val)) << ")" << std::endl;
            err += 1;
        }
        
        if (x.hi() >= 0 && x.lo() >= 0) 
        {
            std::cout << "  = PASS: Both parts >= 0" << std::endl;
        } 
        else 
        {
            std::cout << "  = FAIL: Negative part found" << std::endl;
            err += 1;
        }
    }

    return err;
}

/**
 * @brief Test double->fp32mp2->int32_t conversion with negative low parts
 * Tests cases where double converts to {hi_int, -low} representation
 * Example: 19.9999999123... -> {20.0f, -8.766e-8f} -> should convert to 19, not 20
 * This validates that the conversion sums hi+lo before converting to int
 * @return Number of errors detected (0 = all tests passed)
 */
int test_negative_low_part_cpu()
{
    int err = 0;
    std::cout << "\n=== Testing double->ffloat->int with negative low parts (CPU) ===" << std::endl;
    
    struct TestCase 
    {
        double value;
        int32_t expected;
        const char* description;
    };
    
    TestCase test_cases[] = 
    {
        // Cases where double rounds to upper integer in hi, with negative lo
        {1.9999999123341809E+01, 19, "19.99999991... -> {20.0, -8.77e-8} -> 19"},
        {9.999999523162842E+00, 9, "9.99999952... -> {10.0, -4.77e-7} -> 9"},
        {1.9999998807907104E+00, 1, "1.99999988... -> {2.0, -1.19e-7} -> 1"},
        {9.9999999E+01, 99, "99.9999999 -> {100.0, -1.0e-6} -> 99"},
        {1.23E+02, 123, "123.0 (exact) -> {123.0, 0.0} -> 123"},
        {1.234567890123E+02, 123, "123.4567890... -> {123.0, 0.45678...} -> 123"},
        
        // Negative cases with negative low parts
        {-1.9999999123341809E+01, -19, "-19.99999991... -> {-20.0, +8.77e-8} -> -19"},
        {-9.999999523162842E+00, -9, "-9.99999952... -> {-10.0, +4.77e-7} -> -9"},
        
        // Edge case: very close to integer boundary
        {2.9999999E+00, 2, "2.9999999 -> {3.0, -1.0e-7} -> 2"},
        {-2.9999999E+00, -2, "-2.9999999 -> {-3.0, +1.0e-7} -> -2"},
    };
    
    for (const auto& tc : test_cases) 
    {
        ffloat x(tc.value);
        int32_t result = static_cast<int32_t>(x);
        
        std::cout << "\n" << tc.description << std::endl;
        std::cout << "  Double value: " << std::scientific << std::setprecision(16) << tc.value << std::fixed << std::endl;
        std::cout << "  As ffloat: {hi: " << std::setprecision(9) << x.hi() 
                  << ", lo: " << std::scientific << std::setprecision(3) << x.lo() << std::fixed << "}" << std::endl;
        std::cout << "  Sum (hi+lo): " << std::setprecision(16) << (x.hi() + x.lo()) << std::endl;
        std::cout << "  Result as int32: " << result;
        std::cout << ", Expected: " << tc.expected;
        
        if (result == tc.expected) 
        {
            std::cout << " = PASS" << std::endl;
        } 
        else 
        {
            std::cout << " = FAIL" << std::endl;
            err += 1;
        }
    }
    
    return err;
}

/**
 * @brief Test fp32mp2 to int32_t truncation behavior (round-toward-zero) on CPU
 * Verifies that conversion truncates correctly: 2.7->2, -2.7->-2, matching standard C++ cast behavior
 * @return Number of errors detected (0 = all tests passed)
 */
int test_truncation_cpu () 
{
    int err = 0;
    std::cout << "\n=== Testing float-to-int truncation on CPU (round-toward-zero) ===" << std::endl;
    
    struct TestCase 
    {
        double value;
        int32_t expected;
        const char* description;
    };
    
    TestCase test_cases[] = 
    {
        {2.7, 2, "Positive: 2.7 -> 2"},
        {2.3, 2, "Positive: 2.3 -> 2"},
        {-2.7, -2, "Negative: -2.7 -> -2"},
        {-2.3, -2, "Negative: -2.3 -> -2"},
        {0.9, 0, "Positive fraction: 0.9 -> 0"},
        {-0.9, 0, "Negative fraction: -0.9 -> 0"},
    };
    
    for (const auto& tc : test_cases) 
    {
        ffloat x(tc.value);
        int32_t result = static_cast<int32_t>(x);
        
        std::cout << "\n" << tc.description << std::endl;
        std::cout << "  Result: " << result;
        
        if (result == tc.expected) 
        {
            std::cout << " = PASS (exact)" << std::endl;
        } 
        else 
        {
            std::cout << " = FAIL (expected: " << tc.expected << ")" << std::endl;
            err += 1;
        }
    }

    return err;
}

// ============================================================================
// GPU Tests
// ============================================================================

#if HAS_CUDA
/**
 * @brief CUDA kernel: Test int32_t to fp32mp2 conversions on GPU
 * @param input_vals Array of int32_t test values
 * @param results Output array of test results with validation flags
 * @param num_tests Total number of test values
 */
__global__ void test_int32_kernel(const int32_t* input_vals, 
                                  TestResult* results, 
                                  int num_tests) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    int32_t val = input_vals[idx];
    ffloat x(val);
    int32_t back = static_cast<int32_t>(x);
    
    results[idx].original_val = val;
    results[idx].hi = x.hi();
    results[idx].lo = x.lo();
    results[idx].back_val = back;
    results[idx].exact_match = (back == val);
    
    if (val > 0) 
    {
        results[idx].sign_consistent = (x.lo() >= 0);
    } 
    else if (val < 0) 
    {
        results[idx].sign_consistent = (x.lo() <= 0);
    } 
    else 
    {
        results[idx].sign_consistent = true;
    }
}

/**
 * @brief CUDA kernel: Test uint32_t to fp32mp2 conversions on GPU
 * @param input_vals Array of uint32_t test values
 * @param results Output array of test results with validation flags
 * @param num_tests Total number of test values
 */
__global__ void test_uint32_kernel(const uint32_t* input_vals, 
                                   UInt32TestResult* results, 
                                   int num_tests) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    uint32_t val = input_vals[idx];
    ffloat x(val);
    uint32_t back = static_cast<uint32_t>(x);
    
    results[idx].original_val = val;
    results[idx].hi = x.hi();
    results[idx].lo = x.lo();
    results[idx].back_val = back;
    results[idx].exact_match = (back == val);
    results[idx].all_positive = (x.hi() >= 0 && x.lo() >= 0);
}

/**
 * @brief CUDA kernel: Test fp32mp2 to int32_t truncation (round-toward-zero) on GPU
 * @param input_vals Array of double test values to convert via fp32mp2
 * @param expected_vals Array of expected int32_t results after truncation
 * @param results Output array of test results with correctness flags
 * @param num_tests Total number of test values
 */
__global__ void test_truncation_kernel(const double* input_vals, 
                                       const int32_t* expected_vals, 
                                       TruncTestResult* results, 
                                       int num_tests) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    double val = input_vals[idx];
    ffloat x(val);
    int32_t result = static_cast<int32_t>(x);
    
    results[idx].original_val = val;
    results[idx].result = result;
    results[idx].expected = expected_vals[idx];
    results[idx].correct = (result == expected_vals[idx]);
}

/**
 * @brief CUDA kernel: Test double->fp32mp2->int32_t with negative low parts on GPU
 * @param input_vals Array of double test values
 * @param expected_vals Array of expected int32_t results
 * @param results Output array of test results with hi/lo parts and correctness flags
 * @param num_tests Total number of test values
 */
__global__ void test_negative_low_part_kernel(const double* input_vals, 
                                               const int32_t* expected_vals, 
                                               NegativeLowTestResult* results, 
                                               int num_tests) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    double val = input_vals[idx];
    ffloat x(val);
    int32_t result = static_cast<int32_t>(x);
    
    results[idx].original_val = val;
    results[idx].hi = x.hi();
    results[idx].lo = x.lo();
    results[idx].result = result;
    results[idx].expected = expected_vals[idx];
    results[idx].correct = (result == expected_vals[idx]);
}

/**
 * @brief Test int32_t to fp32mp2 conversions on GPU
 * Launches GPU kernel, transfers data to/from device, validates results
 * Tests edge cases around 2^24 precision limit and INT32_MAX/MIN
 * @return Number of errors detected (0 = all tests passed)
 */
int test_int32_gpu() 
{
    int err = 0;
    std::cout << "\n=== Testing int32_t conversions on GPU ===" << std::endl;
    
    int32_t test_values[] = 
    {
        0, 1, -1, 42, -42,
        16777215, 16777216, 16777217,
        -16777215, -16777216, -16777217,
        2147483647, -2147483648,
    };
    
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    int32_t* d_input;
    TestResult* d_results;
    TestResult* h_results = new TestResult[num_tests];
    
    CHECK_CUDA(cudaMalloc(&d_input, num_tests * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_results, num_tests * sizeof(TestResult)));
    CHECK_CUDA(cudaMemcpy(d_input, test_values, num_tests * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    int threads_per_block = 256;
    int blocks = (num_tests + threads_per_block - 1) / threads_per_block;
    test_int32_kernel<<<blocks, threads_per_block>>>(d_input, d_results, num_tests);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_results, d_results, num_tests * sizeof(TestResult), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < num_tests; i++) 
    {
        print_test_result(h_results[i]);
        if (!h_results[i].exact_match) {
            err += 1;
        }
        if (!h_results[i].sign_consistent) {
            err += 1;
        }
    }
    
    delete[] h_results;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_results));

    return err;
}

/**
 * @brief Test uint32_t to fp32mp2 conversions on GPU
 * Launches GPU kernel, transfers data to/from device, validates results
 * Tests edge cases around 2^24 precision limit and UINT32_MAX
 * @return Number of errors detected (0 = all tests passed)
 */
int test_uint32_gpu() 
{
    int err = 0;
    std::cout << "\n=== Testing uint32_t conversions on GPU ===" << std::endl;
    
    uint32_t test_values[] = 
    {
        0u, 1u, 42u,
        16777215u, 16777216u, 16777217u,
        4294967295u,
    };
    
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    uint32_t* d_input;
    UInt32TestResult* d_results;
    UInt32TestResult* h_results = new UInt32TestResult[num_tests];
    
    CHECK_CUDA(cudaMalloc(&d_input, num_tests * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_results, num_tests * sizeof(UInt32TestResult)));
    CHECK_CUDA(cudaMemcpy(d_input, test_values, num_tests * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    int threads_per_block = 256;
    int blocks = (num_tests + threads_per_block - 1) / threads_per_block;
    test_uint32_kernel<<<blocks, threads_per_block>>>(d_input, d_results, num_tests);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_results, d_results, num_tests * sizeof(UInt32TestResult), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < num_tests; i++) 
    {
        std::cout << "\nOriginal uint32: " << h_results[i].original_val << std::endl;
        std::cout << "  As ffloat: {hi: " << h_results[i].hi << ", lo: " << h_results[i].lo << "}" << std::endl;
        std::cout << "  Back to uint32: " << h_results[i].back_val;
        
        if (h_results[i].exact_match) 
        {
            std::cout << " = PASS (exact)" << std::endl;
        } 
        else 
        {
            std::cout << " = FAIL (difference: " << (static_cast<int64_t>(h_results[i].back_val) - static_cast<int64_t>(h_results[i].original_val)) << ")" << std::endl;
            err += 1;
        }
        
        if (h_results[i].all_positive) 
        {
            std::cout << "  = PASS: Both parts >= 0" << std::endl;
        } 
        else 
        {
            std::cout << "  = FAIL: Negative part found" << std::endl;
            err += 1;
        }
    }
    
    delete[] h_results;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_results));

    return err;
}

/**
 * @brief Test fp32mp2 to int32_t truncation behavior (round-toward-zero) on GPU
 * Launches GPU kernel, transfers data to/from device, validates results
 * Verifies that conversion truncates correctly, matching standard C++ cast behavior
 * @return Number of errors detected (0 = all tests passed)
 */
int test_truncation_gpu() 
{
    int err = 0;
    std::cout << "\n=== Testing float-to-int truncation on GPU (round-toward-zero) ===" << std::endl;
    
    double test_values[] = {2.7, 2.3, -2.7, -2.3, 0.9, -0.9};
    int32_t expected_vals[] = {2, 2, -2, -2, 0, 0};
    const char* descriptions[] = 
    {
        "Positive: 2.7 -> 2",
        "Positive: 2.3 -> 2",
        "Negative: -2.7 -> -2",
        "Negative: -2.3 -> -2",
        "Positive fraction: 0.9 -> 0",
        "Negative fraction: -0.9 -> 0"
    };
    
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    double* d_input;
    int32_t* d_expected;
    TruncTestResult* d_results;
    TruncTestResult* h_results = new TruncTestResult[num_tests];
    
    CHECK_CUDA(cudaMalloc(&d_input, num_tests * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_expected, num_tests * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_results, num_tests * sizeof(TruncTestResult)));
    CHECK_CUDA(cudaMemcpy(d_input, test_values, num_tests * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expected, expected_vals, num_tests * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    int threads_per_block = 256;
    int blocks = (num_tests + threads_per_block - 1) / threads_per_block;
    test_truncation_kernel<<<blocks, threads_per_block>>>(d_input, d_expected, d_results, num_tests);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_results, d_results, num_tests * sizeof(TruncTestResult), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < num_tests; i++) 
    {
        std::cout << "\n" << descriptions[i] << std::endl;
        std::cout << "  Result: " << h_results[i].result;
        
        if (h_results[i].correct) 
        {
            std::cout << " = PASS (exact)" << std::endl;
        } 
        else 
        {
            std::cout << " = FAIL (expected: " << h_results[i].expected << ")" << std::endl;
            err += 1;
        }
    }
    
    delete[] h_results;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_expected));
    CHECK_CUDA(cudaFree(d_results));

    return err;
}

/**
 * @brief Test double->fp32mp2->int32_t with negative low parts on GPU
 * Tests cases where double converts to {hi_int, -low} representation
 * Example: 19.9999999123... -> {20.0f, -8.766e-8f} -> should convert to 19, not 20
 * @return Number of errors detected (0 = all tests passed)
 */
int test_negative_low_part_gpu()
{
    int err = 0;
    std::cout << "\n=== Testing double->ffloat->int with negative low parts (GPU) ===" << std::endl;
    
    double test_values[] = {
        1.9999999123341809E+01,
        9.999999523162842E+00,
        1.9999998807907104E+00,
        9.9999999E+01,
        1.23E+02,
        1.234567890123E+02,
        -1.9999999123341809E+01,
        -9.999999523162842E+00,
        2.9999999E+00,
        -2.9999999E+00
    };
    
    int32_t expected_vals[] = {19, 9, 1, 99, 123, 123, -19, -9, 2, -2};
    
    const char* descriptions[] = {
        "19.99999991... -> {20.0, -8.77e-8} -> 19",
        "9.99999952... -> {10.0, -4.77e-7} -> 9",
        "1.99999988... -> {2.0, -1.19e-7} -> 1",
        "99.9999999 -> {100.0, -1.0e-6} -> 99",
        "123.0 (exact) -> {123.0, 0.0} -> 123",
        "123.4567890... -> {123.0, 0.45678...} -> 123",
        "-19.99999991... -> {-20.0, +8.77e-8} -> -19",
        "-9.99999952... -> {-10.0, +4.77e-7} -> -9",
        "2.9999999 -> {3.0, -1.0e-7} -> 2",
        "-2.9999999 -> {-3.0, +1.0e-7} -> -2"
    };
    
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    
    double* d_input;
    int32_t* d_expected;
    NegativeLowTestResult* d_results;
    NegativeLowTestResult* h_results = new NegativeLowTestResult[num_tests];
    
    CHECK_CUDA(cudaMalloc(&d_input, num_tests * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_expected, num_tests * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_results, num_tests * sizeof(NegativeLowTestResult)));
    CHECK_CUDA(cudaMemcpy(d_input, test_values, num_tests * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expected, expected_vals, num_tests * sizeof(int32_t), cudaMemcpyHostToDevice));
    
    int threads_per_block = 256;
    int blocks = (num_tests + threads_per_block - 1) / threads_per_block;
    test_negative_low_part_kernel<<<blocks, threads_per_block>>>(d_input, d_expected, d_results, num_tests);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_results, d_results, num_tests * sizeof(NegativeLowTestResult), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < num_tests; i++) 
    {
        std::cout << "\n" << descriptions[i] << std::endl;
        std::cout << "  Double value: " << std::scientific << std::setprecision(16) << h_results[i].original_val << std::fixed << std::endl;
        std::cout << "  As ffloat: {hi: " << std::setprecision(9) << h_results[i].hi 
                  << ", lo: " << std::scientific << std::setprecision(3) << h_results[i].lo << std::fixed << "}" << std::endl;
        std::cout << "  Sum (hi+lo): " << std::setprecision(16) << (h_results[i].hi + h_results[i].lo) << std::endl;
        std::cout << "  Result as int32: " << h_results[i].result;
        std::cout << ", Expected: " << h_results[i].expected;
        
        if (h_results[i].correct) 
        {
            std::cout << " = PASS" << std::endl;
        } 
        else 
        {
            std::cout << " = FAIL" << std::endl;
            err += 1;
        }
    }
    
    delete[] h_results;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_expected));
    CHECK_CUDA(cudaFree(d_results));

    return err;
}
#endif // HAS_CUDA

// ============================================================================
// Main
// ============================================================================

/**
 * @brief Main function: Runs comprehensive integer conversion tests for fp32mp2
 * 
 * Tests performed:
 * - int32_t ↔ fp32mp2 conversions (CPU and GPU)
 * - uint32_t ↔ fp32mp2 conversions (CPU and GPU)
 * - Truncation behavior verification (round-toward-zero semantics)
 * - Edge cases: precision limits (2^24), INT32_MAX/MIN, UINT32_MAX
 * - Validation: exact round-trip, sign consistency, non-negativity for unsigned
 * 
 * @return 0 if all tests pass, number of errors otherwise
 */
int main() 
{
    int err = 0;
    std::cout << std::fixed;
    std::cout << "==================================================" << std::endl;
    std::cout << "fp32mp2 Integer Conversion Tests" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // CPU Tests
    std::cout << "\n########## CPU Tests ##########" << std::endl;
    err += test_int32_cpu();
    err += test_uint32_cpu();
    err += test_truncation_cpu();
    err += test_negative_low_part_cpu();
    
#if HAS_CUDA
    // GPU Tests
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    
    if (device_count > 0) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        std::cout << "\n\n########## GPU Tests ##########" << std::endl;
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        
        err += test_int32_gpu();
        err += test_uint32_gpu();
        err += test_truncation_gpu();
        err += test_negative_low_part_gpu();
    } else {
        std::cout << "\n\nNo CUDA devices found, skipping GPU tests." << std::endl;
    }
#else
    std::cout << "\n\nCUDA not available, skipping GPU tests." << std::endl;
#endif
    
    std::cout << "\n==================================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    if (err == 0) {
        std::cout << "\n✓ Result: OK (all tests passed)" << std::endl;
    } else {
        std::cout << "\n✗ Result: FAIL (" << err << " error(s) detected)" << std::endl;
    }
    
    return err;
}
