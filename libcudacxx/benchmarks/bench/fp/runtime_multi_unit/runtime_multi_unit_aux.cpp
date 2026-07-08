/*
    runtime_multi_unit_aux.cpp - FP64 Tool Runtime Size Multi-Unit Test (Auxiliary Unit)
    ======================================================================================================

    This is the second compilation unit that also includes fp64_tool.h to test for
    overdefinition issues when multiple units include the same header.

    This file is compiled separately and linked with runtime_multi_unit.cpp to verify
    that there are no symbol conflicts.
*/

#include <cstdio>
#include <cstdlib>
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
// Runtime size version (with CCCL_FP64_TOOL_RUNTIME_SIZE) - SECOND COMPILATION UNIT
//=============================================================================
#define CCCL_FP64_TOOL_RUNTIME_SIZE
#define CCCL_FP64_TOOL_MANTISSA_BITS 52  // Start with full precision
#define CCCL_FP64_TOOL_EXPONENT_BITS 11  // Full exponent range
#include <cuda/fptool>
using namespace cuda::experimental;  // fp64_tool and setters live here now

//=============================================================================
// CUDA Kernel: Add two double precision values (from auxiliary unit)
//=============================================================================
#if defined(__CUDACC__)
__global__ void add_kernel_aux(double a, double b, double* result) 
{
    fp64_tool x = a;
    fp64_tool y = b;
    fp64_tool sum = x + y;
    *result = (double)sum;
}
#endif

//=============================================================================
// Test functions from auxiliary unit
//=============================================================================
void set_mantissa_from_aux_unit(int size)
{
#if defined(__CUDACC__)
    fp64_tool_set_device_mantissa_size(size);
#else
    fp64_tool_set_host_mantissa_size(size);
#endif
}

double test_from_aux_unit(double a, double b) 
{
#if defined(__CUDACC__)
    double result = 0.0;
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    
    add_kernel_aux<<<1, 1>>>(a, b, d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_result));
    return result;
#else
    fp64_tool x = a;
    fp64_tool y = b;
    return (double)(x + y);
#endif
}
