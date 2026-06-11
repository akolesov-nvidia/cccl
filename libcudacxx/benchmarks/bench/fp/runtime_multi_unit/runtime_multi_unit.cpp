/*
    runtime_multi_unit.cpp - FP64 Tool Runtime Size Multi-Unit Test
    ======================================================================================================

    This benchmark tests that fp64_tool.hpp can be safely included in multiple compilation units
    within a single binary without causing linker errors or runtime issues.

    Test Structure:
    -------------------------------------------------------------------------
    - Main compilation unit (this file): Includes fp64_tool.hpp and calls test functions
    - Auxiliary compilation unit (runtime_multi_unit_aux.cpp): Also includes fp64_tool.hpp
    - Both units use runtime size functionality and call kernels
    - Verifies that there are no symbol conflicts or overdefinition errors

    Build Instructions:
    -------------------------------------------------------------------------
    Using the provided Makefile (recommended):
        make BENCH=runtime_multi_unit              # Build for GPU (CUDA)
        make TARGET=host BENCH=runtime_multi_unit  # Build for CPU only
        make BENCH=runtime_multi_unit run          # Build and run

    Manual compilation on CPU:
        g++ -std=c++17 -O2 -I../../include -c runtime_multi_unit.cpp -o runtime_multi_unit.o
        g++ -std=c++17 -O2 -I../../include -c runtime_multi_unit_aux.cpp -o runtime_multi_unit_aux.o
        g++ -std=c++17 -O2 runtime_multi_unit.o runtime_multi_unit_aux.o -o runtime_multi_unit.exe

    Manual compilation with CUDA:
        nvcc -std=c++17 -O2 -I../../include -dc -c runtime_multi_unit.cpp -o runtime_multi_unit.o
        nvcc -std=c++17 -O2 -I../../include -dc -c runtime_multi_unit_aux.cpp -o runtime_multi_unit_aux.o
        nvcc -std=c++17 -O2 -dlink runtime_multi_unit.o runtime_multi_unit_aux.o -o runtime_multi_unit.exe

    Configuration:
    -------------------------------------------------------------------------
    - Requires FP64_TOOL_RUNTIME_SIZE to be defined
    - Tests that setter functions work correctly across compilation units
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>

//=============================================================================
// Host/Device Compatibility Macros
//=============================================================================
#if defined(__CUDACC__)
    #include <cuda_runtime.h>
    
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                std::fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                           __FILE__, __LINE__, cudaGetErrorString(err)); \
                std::exit(EXIT_FAILURE); \
            } \
        } while(0)
#else
    #define CUDA_CHECK(call) (void)(call)
#endif

//=============================================================================
// Runtime size version (with FP64_TOOL_RUNTIME_SIZE) - FIRST COMPILATION UNIT
//=============================================================================
#define FP64_TOOL_RUNTIME_SIZE
#define FP64_TOOL_MANTISSA_BITS 52  // Start with full precision
#define FP64_TOOL_EXPONENT_BITS 11  // Full exponent range
#include <cuda/fptool>
using namespace cuda::experimental;  // fp64_tool_t and setters live here now

// Forward declaration of function from second compilation unit
extern double test_from_aux_unit(double a, double b);
extern void set_mantissa_from_aux_unit(int size);

//=============================================================================
// CUDA Kernel: Add two double precision values
//=============================================================================
#if defined(__CUDACC__)
__global__ void add_kernel_main(double a, double b, double* result) 
{
    fp64_tool_t x = a;
    fp64_tool_t y = b;
    fp64_tool_t sum = x + y;
    *result = (double)sum;
}
#endif

//=============================================================================
// Test function from main unit
//=============================================================================
double test_from_main_unit(double a, double b) 
{
#if defined(__CUDACC__)
    double result = 0.0;
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    
    add_kernel_main<<<1, 1>>>(a, b, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_result));
    return result;
#else
    fp64_tool_t x = a;
    fp64_tool_t y = b;
    return (double)(x + y);
#endif
}

//=============================================================================
// Main Function
//=============================================================================
int main(int argc, char** argv) 
{
    (void)argc;
    (void)argv;
    
    std::printf("FP64 Tool Runtime Size Multi-Unit Test\n");
    std::printf("======================================\n");
    std::printf("\n");
    std::printf("This test verifies that fp64_tool.hpp can be included in\n");
    std::printf("multiple compilation units without linker errors.\n");
    std::printf("\n");
    
    // Test values: 1 + 2^-52
    const double a = 1.0;
    const double b = 1.0 / (1ULL << 52);  // 2^-52
    const double expected_full = 1.0 + b;   // 1 + 2^-52
    const double expected_reduced = 1.0;    // With reduced mantissa, small value is lost
    
    bool all_passed = true;
    
    // Test 1: Main unit with full mantissa (52 bits)
    std::printf("Test 1: Main unit, full mantissa (52 bits)\n");
#if defined(__CUDACC__)
    fp64_tool_set_device_mantissa_size(52);
#else
    fp64_tool_set_host_mantissa_size(52);
#endif
    double result1 = test_from_main_unit(a, b);
    bool test1_ok = (result1 == expected_full);
    std::printf("  Expected: %.17e\n", expected_full);
    std::printf("  Got:      %.17e\n", result1);
    std::printf("  Status:   %s\n", test1_ok ? "PASS" : "FAIL");
    std::printf("\n");
    all_passed = all_passed && test1_ok;
    
    // Test 2: Auxiliary unit with full mantissa (52 bits)
    std::printf("Test 2: Auxiliary unit, full mantissa (52 bits)\n");
#if defined(__CUDACC__)
    fp64_tool_set_device_mantissa_size(52);
#else
    fp64_tool_set_host_mantissa_size(52);
#endif
    double result2 = test_from_aux_unit(a, b);
    bool test2_ok = (result2 == expected_full);
    std::printf("  Expected: %.17e\n", expected_full);
    std::printf("  Got:      %.17e\n", result2);
    std::printf("  Status:   %s\n", test2_ok ? "PASS" : "FAIL");
    std::printf("\n");
    all_passed = all_passed && test2_ok;
    
    // Test 3: Main unit with reduced mantissa (50 bits)
    std::printf("Test 3: Main unit, reduced mantissa (50 bits)\n");
#if defined(__CUDACC__)
    fp64_tool_set_device_mantissa_size(50);
#else
    fp64_tool_set_host_mantissa_size(50);
#endif
    double result3 = test_from_main_unit(a, b);
    bool test3_ok = (result3 == expected_reduced);
    std::printf("  Expected: %.17e\n", expected_reduced);
    std::printf("  Got:      %.17e\n", result3);
    std::printf("  Status:   %s\n", test3_ok ? "PASS" : "FAIL");
    std::printf("\n");
    all_passed = all_passed && test3_ok;
    
    // Test 4: Auxiliary unit with reduced mantissa (50 bits)
    std::printf("Test 4: Auxiliary unit, mantissa changed in main compilation unit should have no effect\n");
    double result4 = test_from_aux_unit(a, b);
    bool test4_ok = (result4 == expected_full);
    std::printf("  Expected: %.17e\n", expected_full);
    std::printf("  Got:      %.17e\n", result4);
    std::printf("  Status:   %s\n", test4_ok ? "PASS" : "FAIL");
    std::printf("\n");
    all_passed = all_passed && test4_ok;
    
    // Test 5: Auxiliary unit with reduced mantissa (50 bits)
    std::printf("Test 5: Auxiliary unit, reduced mantissa (50 bits)\n");
    set_mantissa_from_aux_unit(50);
    double result5 = test_from_aux_unit(a, b);
    bool test5_ok = (result5 == expected_reduced);
    std::printf("  Expected: %.17e\n", expected_reduced);
    std::printf("  Got:      %.17e\n", result5);
    std::printf("  Status:   %s\n", test5_ok ? "PASS" : "FAIL");
    std::printf("\n");
    all_passed = all_passed && test5_ok;
    
    // Summary
    std::printf("========================================\n");
    std::printf("Overall Test: %s\n", all_passed ? "PASS" : "FAIL");
    std::printf("========================================\n");
    std::printf("\n");
    
    if (all_passed) {
        std::printf("v No linker errors or symbol conflicts detected\n");
        std::printf("v Runtime size functionality works across compilation units\n");
    } else {
        std::printf("x Some tests failed\n");
    }
    std::printf("\n");
    
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
