/*
    fft.cpp - GPU FFT Benchmark for Float-Float and Native Double Types
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025-08-05

    This benchmark measures the accuracy and performance of FFT implementations using native double 
    and float-float types on CUDA GPUs. It supports configurable FFT size, batch size, and iteration 
    count via preprocessor macros or Makefile. The code verifies correctness against a host reference 
    and reports timing statistics for each type.

    Features:
    -------------------------------------------------------------------------
    - Supports multiple precision types: native double, ffloat, ffloat_fast, fp64emu (accurate/def/fast)
    - Configurable FFT size, batch size, and iteration counts
    - Comprehensive accuracy verification against host reference
    - Statistical timing analysis with outlier removal
    - Magnitude-based error checking for complex numbers
    - Template-based design for easy extension to new types
*/

 #define MAD_ENABLED  0
 #define DOT_ENABLED  0
 #define CMUL_ENABLED 1
 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <tuple>
#include <functional>
#include <string>

#include "timing_benchmarks.hpp"

#if !defined(NO_EMULATION)
    #include <cuda/fpmp>
    #include <cuda/fpmp_math>
    #include <cuda/fpemu>
    using namespace cuda::experimental;
#endif

// ---------------------------------------------------------------------------=
// Configuration Parameters
// ---------------------------------------------------------------------------=
// These parameters can be overridden via command line or Makefile
// Example: make SIZE=2048 BATCH=16 ITERATIONS=2000

// FFT size - must be a power of 2 for efficient implementation
#ifndef SIZE
  #define SIZE 1024
#endif

// Batch size - number of FFTs to process in parallel
#ifndef BATCH
  #define BATCH 8
#endif

// Number of iterations for timing measurements
#ifndef ITERATIONS
  #define ITERATIONS 1000
#endif

// Number of warm-up iterations to stabilize GPU performance
#ifndef WARMUP_ITERATIONS
  #define WARMUP_ITERATIONS 100
#endif

// Number of timing runs for statistical analysis
#ifndef TIMING_RUNS
  #define TIMING_RUNS 5
#endif

// Number of threads per CUDA block
#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK 256
#endif

// Tolerance for verification
#ifndef TOLERANCE
  #define TOLERANCE 5e-5
#endif

/**
 * CUDA error checking macro
 * 
 * Wraps CUDA calls and exits with error message if the call fails.
 * Usage: CUDA_OK(cudaMalloc(&ptr, size));
 */
 #define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      exit(1); \
    } \
  } while(0)

void __device__ __host__ __forceinline__ cmul_double(double xre, double xim, double yre, double yim, double& r_re, double& r_im)
{
    r_re = xre * yre - xim * yim;
    r_im = xre * yim + xim * yre;
    return;
}
  
template<typename T>
struct ComplexT 
{
    using value_type = T;  // Type alias for the underlying numeric type
    T real, imag;          // Real and imaginary components
    
    // Default constructor - initializes to zero
    __host__ __device__ __forceinline__ ComplexT() : real(0.0), imag(0.0) {}
    
    // Constructor with real and imaginary parts
    __host__ __device__ __forceinline__ ComplexT(T r, T i) : real(r), imag(i) {} 
    
    // Complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i
    __host__ __device__ __forceinline__ ComplexT operator+(const ComplexT& other) const { 
        return ComplexT(real + other.real, imag + other.imag); 
    }
    
    // Complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
    __host__ __device__ __forceinline__ ComplexT operator-(const ComplexT& other) const { 
        return ComplexT(real - other.real, imag - other.imag); 
    }
    
    // Complex multiplication: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
    __host__ __device__ __forceinline__ ComplexT operator*(const ComplexT& other) const 
    {
        // Use standard implementation for all types (double, ffloat, ffloat_fast)
        return ComplexT(real * other.real - imag * other.imag, 
                        real * other.imag + imag * other.real);
    }

    // Scalar multiplication: (a+bi) * k = (ak) + (bk)i
    __host__ __device__ __forceinline__ ComplexT operator*(T scalar) const { 
        return ComplexT(real * scalar, imag * scalar); 
    }
};


// ---------------------------------------------------------------------------=
// FFT Twiddle Factor Computation
// ---------------------------------------------------------------------------=
/**
 * @brief Pre-computes twiddle factors for FFT implementation
 * 
 * Twiddle factors are complex numbers of the form exp(-2πi*k/n) where k ranges
 * from 0 to n/2-1. These factors are used in the FFT butterfly operations to
 * combine frequency components. Pre-computing them avoids repeated trigonometric
 * calculations during the FFT.
 * 
 * @tparam T The complex number type (ComplexT<double>, ComplexT<ffloat>, etc.)
 * @param twiddles Output array to store the twiddle factors
 * @param n The size of the FFT (must be a power of 2)
 */
template<typename T>
__host__ __device__ void precompute_twiddles(T* twiddles, int n) 
{
    // Compute twiddle factors for k = 0 to n/2-1
    for (int i = 0; i < n/2; i++) 
    {
        // Calculate angle: -2π * k / n
        double angle = -2.0 * M_PI * i / n;
        
        // Twiddle factor: exp(-2πi*k/n) = cos(angle) + i*sin(angle)
        twiddles[i] = T(cos(angle), sin(angle));
    }
}

// ---------------------------------------------------------------------------=
// Device-Compatible Mathematical Functions
// ---------------------------------------------------------------------------=
/**
 * @brief Device-compatible log2 function for integer values
 * 
 * Computes the base-2 logarithm of an integer, which is equivalent to
 * finding the position of the highest set bit. This function works on
 * both host and device and is optimized for power-of-2 values.
 * 
 * @param x The input integer (must be a power of 2)
 * @return The base-2 logarithm of x
 */
__host__ __device__ int log2_int(int x) 
{
    int result = 0;
    while (x > 1) {
        x >>= 1;
        result++;
    }
    return result;
}

// ---------------------------------------------------------------------------=
// Bit Reversal for FFT
// ---------------------------------------------------------------------------=
/**
 * @brief Performs bit reversal of an integer for FFT implementation
 * 
 * The FFT algorithm requires data to be arranged in bit-reversed order before
 * performing the butterfly operations. This function reverses the bits of an
 * integer, e.g., 6 (110) becomes 3 (011) for 3-bit numbers.
 * 
 * @param x The input integer to reverse
 * @param bits The number of bits to reverse (log2 of FFT size)
 * @return The bit-reversed integer
 */
__host__ __device__ int bit_reverse(int x, int bits) 
{
    int result = 0;
    
    // Reverse bits one by one
    for (int i = 0; i < bits; i++) 
    {
        // Shift result left and add the least significant bit of x
        result = (result << 1) | (x & 1);
        
        // Shift x right to get the next bit
        x >>= 1;
    }
    return result;
}

// ---------------------------------------------------------------------------=
// Host-Side FFT Implementation
// ---------------------------------------------------------------------------=
/**
 * @brief Sequential FFT implementation on CPU for reference verification
 * 
 * This is the reference FFT implementation that computes twiddle factors
 * directly using sin and cos functions during the FFT computation.
 * No pre-computed twiddles are needed, making it the most straightforward
 * reference solution.
 * 
 * @param data Input/output vector of complex numbers (modified in-place)
 */
void fft_reference(std::vector<ComplexT<double>>& data) 
{
    int n = data.size();
    int bits = log2_int(n);
    
    // Phase 1: Bit-reversal permutation
    // Rearrange data so that indices are in bit-reversed order
    for (int i = 0; i < n; i++) 
    {
        int reversed = bit_reverse(i, bits);
        if (i < reversed) 
        {
            std::swap(data[i], data[reversed]);
        }
    }
    
    // Phase 2: Butterfly operations with direct twiddle computation
    // Perform FFT by computing twiddle factors directly using sin/cos
    for (int step = 1; step < n; step <<= 1) 
    {
        for (int group = 0; group < n; group += step << 1) 
        {
            for (int pair = group; pair < group + step; pair++) 
            {
                int match = pair + step;
                
                // Compute twiddle factor directly using sin/cos
                // w = exp(-2πi * (pair - group) / (step << 1))
                double angle = -2.0 * M_PI * (pair - group) / (step << 1);
                ComplexT<double> w(cos(angle), sin(angle));
                
                ComplexT<double> u = data[pair];
                ComplexT<double> t = w * data[match];
                
                data[pair]  = u + t;
                data[match] = u - t;
            }
        }
    }
}

// ---------------------------------------------------------------------------=
// Host-Side FFT Verification Utilities
// ---------------------------------------------------------------------------=
#include "fft_verify.hpp"

// ---------------------------------------------------------------------------=
// GPU FFT Kernel
// ---------------------------------------------------------------------------=
/**
 * @brief CUDA kernel for FFT computation with support for both parallel and sequential modes
 * 
 * This kernel implements the Cooley-Tukey FFT algorithm on GPU. It supports two execution modes:
 * - Parallel mode: Each thread processes one element, suitable for performance benchmarking
 * - Sequential mode: Single thread processes all elements, suitable for accuracy verification
 * 
 * The kernel handles batched FFTs where multiple FFTs are computed simultaneously.
 * The FFT size and bit count are computed internally from the total data size and batch count.
 * 
 * @tparam T The complex number type (ComplexT<double>, ComplexT<ffloat>, etc.)
 * @tparam is_sequential If true, runs in sequential mode for accuracy verification
 * @param data Input/output array of complex numbers (modified in-place)
 * @param twiddles Pre-computed twiddle factors
 * @param n FFT size
 * @param bits Number of bits for bit reversal
 */
template<typename T, bool is_sequential = false>
__global__ void fft_kernel(T* data, T* twiddles, int n, int bits) 
{
    // Determine batch index and offset for this thread block
    int batch = blockIdx.y;
    int batch_offset = batch * n;    
    
    // Determine thread index and processing range
    int idx, idx_end;
    
    if constexpr (is_sequential) { 
        // Sequential mode: single thread processes all elements
        idx = 0; 
        idx_end = n; 
    } else { 
        // Parallel mode: each thread processes one element
        idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx_end = idx+1; 
    }

    // Early exit conditions
    if constexpr (is_sequential)  { 
        if (threadIdx.x != 0) return; 
    } else { 
        if (idx >= n) return;
    }
    
    // Phase 1: Bit-reversal permutation
    // Rearrange data so that indices are in bit-reversed order
    for (int i = idx; i < idx_end; i++) 
    {
        int reversed = bit_reverse(i, bits);
        if (i < reversed) 
        {
            // Swap elements to achieve bit-reversed ordering
            T temp = data[batch_offset + i];
            data[batch_offset + i]        = data[batch_offset + reversed];
            data[batch_offset + reversed] = temp;
        }
    }

    // Phase 2: Butterfly operations
    // Perform FFT using pre-computed twiddle factors
    for (int step = 1; step < n; step <<= 1) 
    {
        for (int group = 0; group < n; group += step << 1) 
        {
            for (int pair = group; pair < group + step; pair++) 
            {
                int match = pair + step;
                bool is_butterfly = true;
                
                // In parallel mode, only process if this thread owns one of the elements
                if constexpr (!is_sequential) { 
                    is_butterfly = (idx == pair || idx == match); 
                }
                
                // Only process if this thread owns one of the elements in the butterfly
                if (is_butterfly) 
                {
                    int twiddle_idx = (pair - group) * (n / (step << 1));
                    T w = twiddles[twiddle_idx];
                    
                    T u = data[batch_offset + pair];
                    T t = w * data[batch_offset + match];
                    
                    data[batch_offset + pair]  = u + t;
                    data[batch_offset + match] = u - t;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------=
// Data Initialization Functions
// ---------------------------------------------------------------------------=
/**
 * @brief Template function for initializing batch data with different precision types
 * 
 * This function initializes complex data for FFT testing. It generates sinusoidal
 * test patterns with different phases for each batch to ensure diverse test cases.
 * The function handles both native double and float-float types through template specialization.
 * 
 * @tparam T The complex number type (ComplexT<double>, ComplexT<ffloat>, etc.)
 * @param data Output vector to be initialized
 * @param size Size of each FFT
 * @param batch_size Number of FFTs in the batch
 */
template<typename T>
void initialize_batch_data_template(std::vector<T>& data, int size, int batch_size) 
{
    for (int batch = 0; batch < batch_size; batch++) 
    {
        for (int i = 0; i < size; i++) 
        {
            int idx = batch * size + i;
            
            // Use template specialization to handle different types
            if constexpr (std::is_same_v<T, ComplexT<double>>) 
            {
                // Native double: use direct trigonometric functions
                data[idx] = ComplexT<double>(sin(i * 0.1 + batch * 0.5), cos(i * 0.1 + batch * 0.5));
            } 
            else 
            {
                // Float-float types: use the appropriate precision for phase calculation
                double phase = i * 0.1 + batch * 0.5;
                data[idx] = T(sin(phase), cos(phase));
            }
        }
    }
}

/**
 * @brief Initialize test data for single FFT verification
 * 
 * Creates a simple sinusoidal test pattern for accuracy verification.
 * This function is used to generate reference data for comparison.
 * 
 * @param data Output vector to be initialized with test pattern
 */
void initialize_data(std::vector<ComplexT<double>>& data) 
{
    for (size_t i = 0; i < data.size(); i++) 
    {
        // Generate sinusoidal test pattern: sin(i*0.1) + i*cos(i*0.1)
        data[i] = ComplexT<double>(sin(i * 0.1), cos(i * 0.1));
    }
}

// ---------------------------------------------------------------------------=
// Error Calculation and Verification
// ---------------------------------------------------------------------------=
/**
 * @brief Calculate L2 norm error between reference and computed results
 * 
 * Computes the Euclidean distance between two complex vectors, which is the
 * standard metric for measuring numerical accuracy in FFT implementations.
 * The L2 norm is calculated as sqrt(sum(|reference[i] - result[i]|²)).
 * 
 * @param reference Reference (correct) complex vector
 * @param result Computed result vector to compare against reference
 * @return L2 norm error, or -1.0 if vectors have different sizes
 */
double calculate_l2_error(const std::vector<ComplexT<double>>& reference, 
                          const std::vector<ComplexT<double>>& result) 
{
    if (reference.size() != result.size()) return -1.0;
    
    double sum_sq_error = 0.0;
    for (size_t i = 0; i < reference.size(); i++) 
    {
        // Calculate squared difference for each complex component
        double diff_real = reference[i].real - result[i].real;
        double diff_imag = reference[i].imag - result[i].imag;
        sum_sq_error += diff_real * diff_real + diff_imag * diff_imag;
    }
    return sqrt(sum_sq_error);
}

// Verify FFT results
/**
 * @brief Verify FFT results against reference using magnitude-based comparison
 * 
 * Compares computed FFT results against a reference implementation using
 * magnitude-based error checking. This approach is more mathematically sound
 * for complex numbers than component-wise comparison, as it measures the
 * actual distance between complex points.
 * 
 * @param name Name of the implementation being tested (for error reporting)
 * @param reference Reference (correct) complex vector
 * @param result Computed result vector to verify
 * @param tolerance Maximum allowed magnitude error (default: 1e-6)
 * @return true if all differences are within tolerance, false otherwise
 */
bool verify_results(const char* name, 
                    const std::vector<ComplexT<double>>& reference, 
                    const std::vector<ComplexT<double>>& result, 
                    double tolerance = TOLERANCE) 
{
    if (reference.size() != result.size()) return false;
    
    for (size_t i = 0; i < reference.size(); i++) 
    {
        // Calculate magnitude of the difference: sqrt(real*real + imag*imag)
        double diff_real = reference[i].real - result[i].real;
        double diff_imag = reference[i].imag - result[i].imag;
        double magnitude = sqrt(diff_real * diff_real + diff_imag * diff_imag);
        
        if (magnitude > tolerance) 
        {
            fprintf(stderr, "\t\t%s: Mismatch at index %zu: ref=(%f,%f), result=(%f,%f), magnitude=%e\n", 
                   name, i, reference[i].real, reference[i].imag, result[i].real, result[i].imag, magnitude);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------=
// Performance Benchmarking
// ---------------------------------------------------------------------------=
/**
 * @brief Unified benchmark function for FFT performance measurement
 * 
 * This template function measures the performance of FFT implementations for
 * different precision types. It includes warm-up runs to stabilize GPU performance,
 * multiple timing runs for statistical analysis, and outlier removal for robust
 * measurements.
 * 
 * @tparam T The complex number type to benchmark
 * @param size Size of each FFT
 * @param batch Number of FFTs to process in parallel
 * @param iterations Number of iterations for timing measurement
 * @param h_twiddles Pre-computed twiddle factors
 * @return Average execution time in milliseconds (excluding outliers)
 */
template<typename T>
float fft_timing(int size, int batch, int iterations, const std::vector<T>& h_twiddles) 
{
    // Initialize test data for benchmarking
    std::vector<T> h_data(size * batch);
    initialize_batch_data_template(h_data, size, batch);
    
    // Allocate GPU memory for data and twiddle factors
    T *d_data, *d_twiddles;
    cudaMalloc(&d_data, size * batch * sizeof(T));
    cudaMalloc(&d_twiddles, (size/2) * sizeof(T));
    
    // Copy data to GPU
    cudaMemcpy(d_data, h_data.data(), size * batch * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddles, h_twiddles.data(), (size/2) * sizeof(T), cudaMemcpyHostToDevice);
    
    // Calculate kernel launch configuration
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid_x = (size + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid_y = batch;
    
    dim3 grid(blocksPerGrid_x, blocksPerGrid_y);
    dim3 block(threadsPerBlock);
    
    // Warm-up phase to stabilize GPU performance
    for (int i = 0; i < WARMUP_ITERATIONS; i++) 
    {
        fft_kernel<<<grid, block>>>(d_data, d_twiddles, size, log2_int(size));
    }
    cudaDeviceSynchronize();
    
    // Multiple timing runs for statistical analysis
    std::vector<float> times(TIMING_RUNS);
    for (int run = 0; run < TIMING_RUNS; run++) 
    {
        Timer timer;
        timer.start();
        
        // Execute FFT kernel multiple times for accurate timing
        for (int iter = 0; iter < iterations; iter++) 
        {
            fft_kernel<<<grid, block>>>(d_data, d_twiddles, size, log2_int(size));
        }
        cudaDeviceSynchronize();
        
        times[run] = timer.stop();
        
        // Small delay between runs to avoid thermal throttling
        if (run < TIMING_RUNS - 1) { usleep(1000); }
    }
    
    // Statistical analysis: exclude outliers for robust measurement
    std::sort(times.begin(), times.end());
    float avg_time = 0.0f;
    int valid_runs = 0;
    
    // Use median and exclude outliers (first and last runs)
    for (int i = 1; i < TIMING_RUNS - 1; i++) 
    {
        avg_time += times[i];
        valid_runs++;
    }
    
    // Clean up GPU memory
    cudaFree(d_data);
    cudaFree(d_twiddles);
    return avg_time / valid_runs;
}

// ---------------------------------------------------------------------------=
// Results Reporting
// ---------------------------------------------------------------------------=
/**
 * @brief Print formatted benchmark results with performance metrics
 * 
 * Displays comprehensive benchmark results including accuracy verification,
 * execution time, GFLOPS calculation, and performance comparison against
 * native double precision. The GFLOPS calculation is based on the standard
 * FFT operation count: 5 * N * log2(N) operations per FFT.
 * 
 * @param name Name of the implementation being tested
 * @param size Size of each FFT
 * @param batch Number of FFTs processed in parallel
 * @param iterations Number of iterations used for timing
 * @param pass Whether the accuracy verification passed
 * @param l2_error L2 norm error compared to reference
 * @param time_ms Average execution time in milliseconds
 * @param native_time_ms Reference time for performance comparison
 */
void print_results (const char* name, 
                    int size, 
                    int batch, 
                    int iterations, 
                    bool pass, 
                    double l2_error, 
                    float time_ms, 
                    float native_time_ms) 
{
    // Calculate GFLOPS: 5 * N * log2(N) operations per FFT
    double gflops = 5.0 * size * log2_int(size) * batch * iterations / (time_ms * 1e-3) / 1e9;

    printf("%-22s: %s (%9.2e L2 err): %8.2f ms, %8.2f GFLOPS (%4.2gX vs native)\n", 
        name, pass?"OK":"FAIL", l2_error, time_ms, gflops, native_time_ms/time_ms);
}

// ---------------------------------------------------------------------------=
// Type Information and Testing Infrastructure
// ---------------------------------------------------------------------------=
/**
 * @brief Helper struct to hold type information for testing and benchmarking
 * 
 * This template struct encapsulates type-specific information needed for
 * testing and benchmarking different precision types. It provides a uniform
 * interface for handling native double and float-float types.
 * 
 * @tparam T The underlying numeric type (double, ffloat, ffloat_fast, etc.)
 */
template<typename T>
struct TypeInfo {
    using complex_type = ComplexT<T>;  // Type alias for complex numbers
    const char* name;                   // Human-readable name for reporting
    double tolerance;                   // Tolerance for accuracy verification
    
    TypeInfo(const char* n, double tol = TOLERANCE) : name(n), tolerance(tol) {}
};

/**
 * @brief Comprehensive testing and benchmarking function for a specific type
 * 
 * This template function performs both accuracy verification and performance
 * benchmarking for a given precision type. It handles type conversion,
 * GPU memory management, and returns comprehensive results including
 * verification status, L2 error, and execution time.
 * 
 * @tparam T The precision type to test (double, ffloat, ffloat_fast, etc.)
 * @param type_info Type information including name for reporting
 * @param test_data Input test data in native double precision
 * @param reference Reference result for accuracy verification
 * @param native_twiddles Twiddle factors in native double precision
 * @param size Size of each FFT
 * @param batch Number of FFTs to process in parallel
 * @param iterations Number of iterations for timing
 * @return Tuple containing (verification_passed, l2_error, execution_time)
 */
template<typename T>
std::tuple<bool, double, float> 
test_and_benchmark_type( const TypeInfo<T>& type_info,
                         const std::vector<ComplexT<double>>& test_data,
                         const std::vector<ComplexT<double>>& reference,
                         const std::vector<ComplexT<double>>& native_twiddles,
                         int size, int batch, int iterations) 
{
    // Convert twiddles to target type for GPU computation
    std::vector<ComplexT<T>> twiddles(size/2);
    for (int i = 0; i < size/2; i++) {
        twiddles[i] = ComplexT<T>(native_twiddles[i].real, native_twiddles[i].imag);
    }
    
    // Convert test data to target type for GPU computation
    std::vector<ComplexT<T>> test_typed(size);
    for (int i = 0; i < size; i++) {
        test_typed[i] = ComplexT<T>(test_data[i].real, test_data[i].imag);
    }
    
    // Allocate GPU memory for computation
    ComplexT<T> *d_test, *d_twiddles;
    cudaMalloc(&d_test, size * sizeof(ComplexT<T>));
    cudaMalloc(&d_twiddles, (size/2) * sizeof(ComplexT<T>));
    
    // Copy data to GPU for computation
    cudaMemcpy(d_test, test_typed.data(), size * sizeof(ComplexT<T>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddles, twiddles.data(), (size/2) * sizeof(ComplexT<T>), cudaMemcpyHostToDevice);
    
    // Run FFT using sequential kernel for accuracy verification
    fft_kernel<ComplexT<T>, true><<<1, 1>>>(d_test, d_twiddles, size, log2_int(size));
    cudaDeviceSynchronize();
    
    // Copy result back to host for analysis
    cudaMemcpy(test_typed.data(), d_test, size * sizeof(ComplexT<T>), cudaMemcpyDeviceToHost);
    cudaFree(d_test);
    cudaFree(d_twiddles);
    
    // Convert result back to native double for comparison with reference
    std::vector<ComplexT<double>> result(size);
    for (int i = 0; i < size; i++) {
        result[i] = ComplexT<double>(test_typed[i].real, test_typed[i].imag);
    }
    
    // Calculate accuracy metrics and verify results
    double l2_error = calculate_l2_error(reference, result);
    bool passed = verify_results(type_info.name, reference, result, type_info.tolerance);
    
    // Run performance benchmark
    float time = fft_timing<ComplexT<T>>(size, batch, iterations, twiddles);
    
    return std::make_tuple(passed, l2_error, time);
}

/**
 * @brief Process all precision types for comprehensive testing and benchmarking
 * 
 * This function orchestrates the testing and benchmarking of all supported
 * precision types (native double, ffloat, ffloat_fast, fp64emu accurate/def/fast). It provides a unified
 * interface for testing multiple types and returns results in a consistent
 * format for easy comparison and reporting.
 * 
 * @param test_data Input test data in native double precision
 * @param reference Reference result for accuracy verification
 * @param native_twiddles Twiddle factors in native double precision
 * @param size Size of each FFT
 * @param batch Number of FFTs to process in parallel
 * @param iterations Number of iterations for timing
 * @return Vector of tuples containing results for each type: (passed, l2_error, time, name)
 */
std::vector<std::tuple<bool, double, float, const char*>> 
process_all_types( const std::vector<ComplexT<double>>& test_data,
                   const std::vector<ComplexT<double>>& reference,
                   const std::vector<ComplexT<double>>& native_twiddles,
                   int size, int batch, int iterations) 
{
#if !defined(NO_EMULATION)
    // Define float-float type aliases for clarity
    using type_ffloat      = fp32mp2;       // default (mid) accuracy
    using type_ffloat_fast = fp32mp2_low;

    // Define fp64emu type aliases for different accuracies
    // (fp64emu == high, the library default; fp64emu_mid / fp64emu_low are explicit)
    using type_fpemu_accurate = fp64emu;
    using type_fpemu_def      = fp64emu_mid;
    using type_fpemu_fast     = fp64emu_low;
#endif

    // Define display names for each precision type
    const char* type_native_name      = "Native Double";
#if !defined(NO_EMULATION)
    const char* type_ffloat_name      = "FP32MP2 def";
    const char* type_ffloat_fast_name = "FP32MP2 fast";
    const char* type_fpemu_acc_name   = "FPEMU accurate";
    const char* type_fpemu_def_name   = "FPEMU def";
    const char* type_fpemu_fast_name  = "FPEMU fast";
#endif

    // Tolerance for fp64emu fast method (fp32-comparable precision, ~7 significant digits;
    // accumulated error in FFT butterfly operations grows with problem size)
    const double tolerance_fpemu_fast = 1e-1;

    std::vector<std::tuple<bool, double, float, const char*>> results;
    
    // Test and benchmark each precision type individually
    auto native_result = test_and_benchmark_type(TypeInfo<double>(type_native_name), test_data, reference, native_twiddles, size, batch, iterations);
#if !defined(NO_EMULATION)
    auto ffloat_result      = test_and_benchmark_type(TypeInfo<type_ffloat>     (type_ffloat_name),      test_data, reference, native_twiddles, size, batch, iterations);
    auto ffloat_fast_result = test_and_benchmark_type(TypeInfo<type_ffloat_fast>(type_ffloat_fast_name), test_data, reference, native_twiddles, size, batch, iterations);
    auto fpemu_acc_result   = test_and_benchmark_type(TypeInfo<type_fpemu_accurate>(type_fpemu_acc_name),  test_data, reference, native_twiddles, size, batch, iterations);
    auto fpemu_def_result   = test_and_benchmark_type(TypeInfo<type_fpemu_def>     (type_fpemu_def_name),  test_data, reference, native_twiddles, size, batch, iterations);
    auto fpemu_fast_result  = test_and_benchmark_type(TypeInfo<type_fpemu_fast>    (type_fpemu_fast_name, tolerance_fpemu_fast), test_data, reference, native_twiddles, size, batch, iterations);
#endif
    
    // Collect results in a consistent format for reporting
    results.push_back(std::make_tuple(std::get<0>(native_result), std::get<1>(native_result), std::get<2>(native_result), type_native_name));
#if !defined(NO_EMULATION)
    results.push_back(std::make_tuple(std::get<0>(ffloat_result),      std::get<1>(ffloat_result),      std::get<2>(ffloat_result),      type_ffloat_name));
    results.push_back(std::make_tuple(std::get<0>(ffloat_fast_result), std::get<1>(ffloat_fast_result), std::get<2>(ffloat_fast_result), type_ffloat_fast_name));
    results.push_back(std::make_tuple(std::get<0>(fpemu_acc_result),   std::get<1>(fpemu_acc_result),   std::get<2>(fpemu_acc_result),   type_fpemu_acc_name));
    results.push_back(std::make_tuple(std::get<0>(fpemu_def_result),   std::get<1>(fpemu_def_result),   std::get<2>(fpemu_def_result),   type_fpemu_def_name));
    results.push_back(std::make_tuple(std::get<0>(fpemu_fast_result),  std::get<1>(fpemu_fast_result),  std::get<2>(fpemu_fast_result),  type_fpemu_fast_name));
#endif
    return results;
}

// ---------------------------------------------------------------------------=
// Main Function
// ---------------------------------------------------------------------------=
/**
 * @brief Main entry point for FFT benchmark application
 * 
 * This function orchestrates the complete FFT benchmarking workflow:
 * 1. Display configuration parameters
 * 2. Initialize test data and compute reference results
 * 3. Test and benchmark all precision types
 * 4. Report comprehensive results and performance analysis
 * 
 * The benchmark compares native double precision against float-float variants
 * (ffloat, ffloat_fast) and fp64emu variants (accurate, def, fast)
 * for both accuracy and performance metrics.
 * 
 */
int main(void) 
{
    // Display benchmark configuration
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0)); // Use device 0
    printf("\nGPU: %s\n", prop.name);

    printf("FFT Size: %d\n", SIZE);
    printf("Batch Size: %d\n", BATCH);
    printf("Iterations: %d\n", ITERATIONS);
    printf("Warm-up Iterations: %d\n", WARMUP_ITERATIONS);
    printf("Threads Per Block: %d\n", THREADS_PER_BLOCK);
    printf("Tolerance: %e\n", TOLERANCE);
    printf("\n");

    // Initialize data structures
    std::vector<ComplexT<double>> test_data(SIZE);
    std::vector<ComplexT<double>> reference(SIZE);
    std::vector<ComplexT<double>> native_twiddles(SIZE/2);

    // Generate test data for benchmarking
    initialize_data(test_data);
    
    // Pre-compute twiddle factors for GPU benchmarking
    precompute_twiddles(native_twiddles.data(), SIZE);
    
    // Create reference implementation for accuracy verification
    reference = test_data;
    
    // Compute reference result using reference implementation
    fft_reference(reference);
    
    // Additional comprehensive correctness verification
    auto native_fft_func = [&](std::vector<ComplexT<double>>& data) {
        fft_reference(data);
    };
    
    auto native_verification = comprehensive_fft_verification<double>(native_fft_func, SIZE);
        
    printf("\n--- Measure performance and accuracy of FFT implementations with different precision types ---\n");
    // Test and benchmark all precision types (native double, ffloat, ffloat_fast, fp64emu accurate/def/fast)
    auto results = process_all_types(test_data, reference, native_twiddles, SIZE, BATCH, ITERATIONS);

    // Extract native time for performance comparison
    float native_time = std::get<2>(results[0]); // First result is native double
    
    // Display comprehensive benchmark results
    printf("\nProcessing %d FFTs of size %d in batches of %d\n", SIZE * BATCH, SIZE, BATCH);
    printf("Each benchmark runs %d iterations with %d warm-up iterations\n", ITERATIONS, WARMUP_ITERATIONS);
    printf("Multiple timing runs with statistical analysis for stability\n\n");
    
    // Print results for each precision type
    for (const auto& result : results) {
        print_results(std::get<3>(result), SIZE, BATCH, ITERATIONS, std::get<0>(result), std::get<1>(result), std::get<2>(result), native_time);
    }

    return 0;
} 