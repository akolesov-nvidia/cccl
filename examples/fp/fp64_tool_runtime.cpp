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
    fp64_tool_runtime.cpp - FP64 Precision Emulation Tool Runtime Size Demo
    ======================================================================================================

    This example demonstrates the runtime size functionality of fp64_tool.h.
    It tests that precision can be changed at runtime using CCCL_FP64_TOOL_RUNTIME_SIZE.

    Test Case:
    -------------------------------------------------------------------------
    Adding 1 + 2^-52 (the smallest representable value in FP64, the ULP of 1.0):
    - With full mantissa (52 bits): Should produce 1 + 2^-52 (correct value)
    - With reduced mantissa (23 bits): Should produce 1.0 (small value rounded away)

    Build Instructions:
    -------------------------------------------------------------------------
    Using the provided Makefile (recommended):
        make EXAMPLE=fp64_tool_runtime              # Build for GPU (CUDA)
        make TARGET=host EXAMPLE=fp64_tool_runtime  # Build for CPU only
        make run EXAMPLE=fp64_tool_runtime          # Build and run

    Manual compilation on CPU:
        g++ -std=c++17 -O2 -I../include fp64_tool_runtime.cpp -o fp64_tool_runtime.exe -lm

    Manual compilation with CUDA:
        nvcc -std=c++17 -O2 -I../include -x cu fp64_tool_runtime.cpp -o fp64_tool_runtime.exe

    Configuration:
    -------------------------------------------------------------------------
    - Requires CCCL_FP64_TOOL_RUNTIME_SIZE to be defined
    - Initial mantissa bits: 52 (full precision)
    - Test mantissa bits: 23 (float-like precision)
*/

#include <cstdio>

#include <cuda/std/cstdlib>

//=============================================================================
// Runtime size version (with CCCL_FP64_TOOL_RUNTIME_SIZE)
//=============================================================================
#define CCCL_FP64_TOOL_RUNTIME_SIZE
#define CCCL_FP64_TOOL_MANTISSA_BITS 52  // Start with full precision
#define CCCL_FP64_TOOL_EXPONENT_BITS 11  // Full exponent range
#include <cuda/fptool>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

//=============================================================================
// Host/Device Compatibility Macros
//=============================================================================
#if _CCCL_CUDA_COMPILATION()
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
// CUDA Kernel: Add two double precision values
//=============================================================================
#if _CCCL_CUDA_COMPILATION()
__global__ void add_kernel(double a, double b, double* result) 
{
    fp64_tool x = a;
    fp64_tool y = b;
    fp64_tool sum = x - y;
    *result = (double)sum;
}
#endif

//=============================================================================
// Main Function
//=============================================================================
int main(int argc, char** argv) 
{
    (void)argc;
    (void)argv;
    
    std::printf("FP64 Tool Runtime Size Example\n");
    std::printf("================================\n");
    std::printf("\n");
    std::printf("Test: Adding 1 - 2^-52 (ULP of 1.0)\n");
    std::printf("  - With 52 mantissa bits: Should produce 1 - 2^-52\n");
    std::printf("  - With 50 mantissa bits: Should produce 1.0 (small value lost)\n");
    std::printf("\n");
    
    // 2^-52 is the smallest representable value in FP64 (the ULP of 1.0)
    const double a = 1.0;
    const double b = 1.0 / (1ULL << 52);  // 2^-52
    const double expected_full = 1.0 - b;   // 1 - 2^-52
    const double expected_reduced = 1.0;    // With reduced mantissa, small value is lost
    
    double result_full = 0.0;
    double result_reduced = 0.0;
    
#if _CCCL_CUDA_COMPILATION()
    // CUDA path: allocate device memory, launch kernels, copy back
    
    // Test 1: With full mantissa (52 bits)
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    
    add_kernel<<<1, 1>>>(a, b, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&result_full, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    
    // Test 2: With reduced mantissa (50 bits)
    fp64_tool_set_device_mantissa_size(50);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    add_kernel<<<1, 1>>>(a, b, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&result_reduced, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_result));
#else
    // CPU path: call directly
    fp64_tool x_full = a;
    fp64_tool y_full = b;
    result_full = (double)(x_full - y_full);
    
    fp64_tool_set_host_mantissa_size(50);
    fp64_tool x_reduced = a;
    fp64_tool y_reduced = b;
    result_reduced = (double)(x_reduced - y_reduced);
#endif
    
    // Display results
    std::printf("========================================\n");
    std::printf("Test Results\n");
    std::printf("========================================\n");
    std::printf("\n");
    
    std::printf("With Full Mantissa (52 bits):\n");
    std::printf("  Expected: %.17e\n", expected_full);
    std::printf("  Got:      %.17e\n", result_full);
    bool full_ok = (result_full == expected_full);
    std::printf("  Status:   %s\n", full_ok ? "PASS" : "FAIL");
    std::printf("\n");
    
    std::printf("With Reduced Mantissa (50 bits):\n");
    std::printf("  Expected: %.17e\n", expected_reduced);
    std::printf("  Got:      %.17e\n", result_reduced);
    bool reduced_ok = (result_reduced == expected_reduced);
    std::printf("  Status:   %s\n", reduced_ok ? "PASS" : "FAIL");
    std::printf("\n");
    
    std::printf("========================================\n");
    bool test_passed = full_ok && reduced_ok;
    std::printf("Overall Test: %s\n", test_passed ? "PASS" : "FAIL");
    std::printf("========================================\n");
    std::printf("\n");
    
    return test_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
