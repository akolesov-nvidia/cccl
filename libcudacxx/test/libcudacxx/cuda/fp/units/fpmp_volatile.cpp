/*
    volatile.cpp - Unit Test for Volatile Constructors/Copies and Trivial Copyability
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2026

    This test verifies that fpmp2_t types:
    1. Are trivially copyable (required for cooperative_groups, __shfl intrinsics, etc.)
    2. Correctly support volatile construction (copy from volatile object)
    3. Correctly support volatile assignment (assign to/from volatile object)
    4. Preserve values through volatile round-trips on both host and device
*/

#include <cmath>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <cxxabi.h>
#include <cstdlib>

#include <cuda/fpmp>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Demangle GCC/Clang typeid names into human-readable form
static std::string demangle(const char* mangled) {
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    std::string result = (status == 0 && demangled) ? demangled : mangled;
    std::free(demangled);
    return result;
}

// ============================================================
// Compile-time checks: trivially copyable
// ============================================================
static_assert(std::is_trivially_copyable<fp32mp2>::value,
              "fp32mp2 must be trivially copyable");
static_assert(std::is_trivially_copyable<fp32mp2_low>::value,
              "fp32mp2_low must be trivially copyable");
static_assert(std::is_trivially_copyable<fp32mp2_high>::value,
              "fp32mp2_high must be trivially copyable");

#if FPMP_FP64MP2_ENABLE == 1
static_assert(std::is_trivially_copyable<fp64mp2>::value,
              "fp64mp2 must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64mp2_low>::value,
              "fp64mp2_low must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64mp2_high>::value,
              "fp64mp2_high must be trivially copyable");
#endif

// ============================================================
// Host-side volatile tests
// ============================================================

template <typename mp_type>
bool test_volatile_host() {
    const double test_val     = 3.141592653589793;
    const double test_val2    = 2.718281828459045;
    const double tolerance    = 1e-6;
    bool         all_passed   = true;

    std::cout << "\n  --- Host volatile tests for " << demangle(typeid(mp_type).name())
              << " (size=" << sizeof(mp_type) << ") ---" << std::endl;

    // Test 1: Construct from volatile
    {
        volatile mp_type vol;
        // Assign through volatile assignment operator
        const mp_type tmp(test_val);
        vol = tmp;
        // Construct from volatile
        mp_type non_vol(vol);
        double  result = static_cast<double>(non_vol);
        double  error  = std::abs(result - test_val);
        bool    passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    Construct from volatile:   "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    // Test 2: Assign to volatile
    {
        mp_type          src(test_val);
        volatile mp_type vol;
        vol = src;
        // Read back via construct-from-volatile
        mp_type readback(vol);
        double  result = static_cast<double>(readback);
        double  error  = std::abs(result - test_val);
        bool    passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    Assign to volatile:        "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    // Test 3: Assign from volatile
    {
        volatile mp_type vol;
        const mp_type    tmp(test_val2);
        vol = tmp;
        mp_type dst;
        dst = vol;
        double result = static_cast<double>(dst);
        double error  = std::abs(result - test_val2);
        bool   passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    Assign from volatile:      "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    // Test 4: Volatile round-trip preserves hi/lo
    {
        mp_type          src(test_val);
        volatile mp_type vol;
        vol = src;
        mp_type dst(vol);
        bool    hi_match = (src.hi() == dst.hi());
        bool    lo_match = (src.lo() == dst.lo());
        bool    passed   = hi_match && lo_match;
        all_passed &= passed;
        std::cout << "    Volatile round-trip hi/lo: "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (hi " << (hi_match ? "OK" : "MISMATCH")
                  << ", lo " << (lo_match ? "OK" : "MISMATCH") << ")" << std::endl;
    }

    return all_passed;
}

// ============================================================
// Device-side volatile tests (CUDA)
// ============================================================
#if defined(__CUDACC__)

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

// Kernel: construct from volatile, assign to volatile, assign from volatile
template <typename mp_type>
__global__ void test_volatile_kernel(const mp_type input, mp_type *results) {
    // results[0] = construct from volatile
    // results[1] = assign to volatile then read back
    // results[2] = assign from volatile

    // Use raw aligned storage for __shared__ to avoid dynamic initialization warning
    // (fpmp2_t's default constructor counts as dynamic init for __shared__)
    __shared__ alignas(alignof(mp_type)) unsigned char shared_buf[sizeof(mp_type)];
    volatile mp_type& shared_vol = *reinterpret_cast<volatile mp_type*>(shared_buf);

    if (threadIdx.x == 0) {
        // Write to volatile via assignment
        shared_vol = input;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Test 1: Construct from volatile
        mp_type from_vol(shared_vol);
        results[0] = from_vol;

        // Test 2: Assign to volatile then read back
        // Use raw storage to avoid dynamic init warning for local volatile too
        alignas(alignof(mp_type)) unsigned char local_buf[sizeof(mp_type)];
        volatile mp_type& local_vol = *reinterpret_cast<volatile mp_type*>(local_buf);
        local_vol = input;
        mp_type readback(local_vol);
        results[1] = readback;

        // Test 3: Assign from volatile
        mp_type assigned;
        assigned   = shared_vol;
        results[2] = assigned;
    }
} // test_volatile_kernel

template <typename mp_type>
bool test_volatile_device() {
    const double test_val  = 3.141592653589793;
    const double tolerance = 1e-6;
    bool         all_passed = true;

    std::cout << "\n  --- Device volatile tests for " << demangle(typeid(mp_type).name())
              << " (size=" << sizeof(mp_type) << ") ---" << std::endl;

    constexpr int num_results = 3;
    mp_type      *d_results;
    mp_type       h_results[num_results];
    mp_type       h_input(test_val);

    CUDA_CHECK(cudaMalloc(&d_results, num_results * sizeof(mp_type)));
    CUDA_CHECK(cudaMemset(d_results, 0, num_results * sizeof(mp_type)));

    test_volatile_kernel<mp_type><<<1, 32>>>(h_input, d_results);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_results, d_results, num_results * sizeof(mp_type),
                          cudaMemcpyDeviceToHost));

    const char *test_names[] = {
        "Construct from volatile (device): ",
        "Assign to volatile (device):      ",
        "Assign from volatile (device):    ",
    };

    for (int i = 0; i < num_results; ++i) {
        double result = static_cast<double>(h_results[i]);
        double error  = std::abs(result - test_val);
        bool   passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    " << test_names[i]
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_results));
    return all_passed;
}

#endif // __CUDACC__

// ============================================================
// Main
// ============================================================
int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Volatile and Trivial Copyability Tests for fpmp2_t" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

#if defined(__CUDACC__)
    // Get device properties
    int            device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
#endif

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "COMPILE-TIME: trivially copyable static_assert checks passed" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    bool all_passed = true;

    // Host tests
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "HOST VOLATILE TESTS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    all_passed &= test_volatile_host<fp32mp2>();
    all_passed &= test_volatile_host<fp32mp2_low>();
    all_passed &= test_volatile_host<fp32mp2_high>();
#if FPMP_FP64MP2_ENABLE == 1
    all_passed &= test_volatile_host<fp64mp2>();
    all_passed &= test_volatile_host<fp64mp2_low>();
    all_passed &= test_volatile_host<fp64mp2_high>();
#endif

#if defined(__CUDACC__)
    // Device tests
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "DEVICE VOLATILE TESTS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    all_passed &= test_volatile_device<fp32mp2>();
    all_passed &= test_volatile_device<fp32mp2_low>();
    all_passed &= test_volatile_device<fp32mp2_high>();
#if FPMP_FP64MP2_ENABLE == 1
    all_passed &= test_volatile_device<fp64mp2>();
    all_passed &= test_volatile_device<fp64mp2_low>();
    all_passed &= test_volatile_device<fp64mp2_high>();
#endif
#endif // __CUDACC__

    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (all_passed) { std::cout << "+ ALL TESTS PASSED"  << std::endl; }
    else            { std::cout << "- SOME TESTS FAILED" << std::endl; }
    std::cout << std::string(60, '=') << std::endl << std::endl;

    return all_passed ? 0 : 1;
} // main
