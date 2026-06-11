/*
    reduce.cpp - Test for compatibility with cooperative_groups::reduce
    ======================================================================================================
    Author:  Thomas Grützmacher
    Date:    2026

    This test ensurs that multi-floating-point types can be used with the cooperative
   group reduce function
*/

#include <cinttypes>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <typeinfo>
#include <vector>

#if defined(__CUDACC__)

#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "         \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// Kernel for testing the reduce
template <int subwarp_size, typename mp_type>
__global__ void test_reduce_kernel(unsigned int seed, mp_type *res) {
    namespace cg             = cooperative_groups;
    auto          this_block = cg::this_thread_block();
    auto          subwarp    = cg::tiled_partition<subwarp_size>(this_block);
    const auto    thread_id  = this_block.thread_rank();
    const mp_type to_reduce  = seed + thread_id;

    mp_type result =
        cg::reduce(subwarp, to_reduce,
                   [](const mp_type &a, const mp_type &b) -> mp_type { return a + b; });
    if (subwarp.thread_rank() == 0) {
        res[thread_id / subwarp_size] = result;
    }
} // test_atomicity_kernel

// Test function for atomicity
template <int subwarp_size, typename mp_type>
bool test_reduce(int num_threads = subwarp_size) {
    constexpr int num_blocks = 1;
    // Ceil division
    const int num_subwarps   = (num_threads + subwarp_size - 1) / subwarp_size;
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing cg::reduce for " << typeid(mp_type).name()
              << " with size: " << sizeof(mp_type) << std::endl;
    std::cout << "  Threads per block: " << num_threads << std::endl;
    std::cout << "  Number of blocks:  " << num_blocks << std::endl;
    std::cout << "  Total threads:     " << (num_threads * num_blocks) << std::endl;
    std::cout << "  Subwarp size:      " << subwarp_size << std::endl;
    std::cout << "========================================" << std::endl;

    // Allocate device memory
    // Note: cudaMalloc returns properly aligned memory (256+ bytes on modern GPUs)
    // With alignas(2*alignof(FpType)) on the type, proper alignment is guaranteed
    mp_type *d_res;

    CUDA_CHECK(cudaMalloc(&d_res, num_subwarps * sizeof(mp_type)));

    // Initialize device memory
    unsigned int         seed = 10;
    std::vector<mp_type> h_res(num_subwarps);

    CUDA_CHECK(cudaMemset(d_res, 0, num_subwarps * sizeof(mp_type)));

    // Launch kernel
    test_reduce_kernel<subwarp_size><<<1, num_threads>>>(seed, d_res);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, num_subwarps * sizeof(mp_type),
                          cudaMemcpyDeviceToHost));

    bool passed = true;
    // Check results
    for (int i = 0; i < num_subwarps; ++i) {
        std::cout << "  Final result (hi):     " << std::scientific
                  << std::setprecision(10) << h_res[i].hi() << std::endl;
        std::cout << "  Final result (lo):     " << std::scientific
                  << std::setprecision(10) << h_res[i].lo() << std::endl;
        const auto to_double = static_cast<double>(h_res[i]);
        std::cout << "  Final result (double): " << std::scientific
                  << std::setprecision(10) << to_double << std::endl;
        std::int64_t expected =
            // sum of the lowest subwarp
            (2 * std::int64_t{seed} + subwarp_size - 1) * subwarp_size / 2
            // The sum of the next subwarp is size * size larger
            + i * subwarp_size * subwarp_size;
        if (std::abs(to_double - static_cast<double>(expected)) > 1e-4) {
            passed = false;
            std::cout << "  TEST FAILED! Expected: " << expected << std::endl;
        }
    }


    // Cleanup
    CUDA_CHECK(cudaFree(d_res));

    return passed;
} // test_reduce


int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Reduction tests" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Get device properties
    int            device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    bool all_passed = true;

    // Atomicity tests
    std::cout << "\n" << std::string(65, '=') << std::endl;
    std::cout << "REDUCTION TEST (add 1.0 then subtract 1.0 to cancel out to 0.0)"
              << std::endl;
    std::cout << std::string(65, '=') << std::endl;
    all_passed &= test_reduce<4, fp32mp2>();
    all_passed &= test_reduce<32, fp32mp2>(64);
    all_passed &= test_reduce<4, fp64mp2>();
    all_passed &= test_reduce<32, fp64mp2>(64);

    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (all_passed) {
        std::cout << "+ ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << "- SOME TESTS FAILED" << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl << std::endl;

    return all_passed ? 0 : 1;
} // main

#else // __CUDACC__

int main(int argc, char **argv) {
    std::cout << "This example is only available on CUDA GPUs." << std::endl;
    return 0;
} // main

#endif // __CUDACC__
