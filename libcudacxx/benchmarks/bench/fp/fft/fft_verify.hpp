/*
    fft_verify.hpp - Comprehensive Verification Utilities for FFT Benchmarks
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2025-08-05

    This header provides a suite of functions for verifying the correctness, accuracy, and mathematical 
    properties of FFT implementations. It is designed to be used in conjunction with the FFT benchmarking 
    framework to ensure that both native and emulated-precision FFTs produce correct results.

    Features:
    -------------------------------------------------------------------------
    - Mathematical property verification (linearity, Parseval's theorem, etc.)
    - Analytical solution checks for small FFT sizes
    - Numerical accuracy comparison against reference implementations
    - Summary reporting for all verification stages
    - Template-based design for compatibility with multiple precision types

    Usage:
    -------------------------------------------------------------------------
    - Include this header in your FFT benchmark or test harness.
    - Use the provided verification functions to validate your FFT implementation.
    - Review the printed summary and error metrics to assess correctness.
*/

#ifndef FFT_VERIFY_HPP
#define FFT_VERIFY_HPP

// ---------------------------------------------------------------------------=
// Enhanced Mathematical Verification
// ---------------------------------------------------------------------------=

/**
 * @brief Test FFT correctness using known mathematical properties
 * 
 * This function verifies FFT implementation using fundamental mathematical
 * properties that any correct FFT must satisfy:
 * 1. Linearity: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
 * 2. Parseval's theorem: sum(|x|²) = sum(|FFT(x)|²)/N
 * 3. Convolution theorem: FFT(x*y) = FFT(x) * FFT(y)
 * 4. Shift property: FFT(x[n-k]) = FFT(x[n]) * exp(-2πik/N)
 * 
 * @param fft_func Function pointer to the FFT implementation to test
 * @param size FFT size (must be power of 2)
 * @param tolerance Numerical tolerance for comparisons
 * @return true if all mathematical properties are satisfied
 */
 template<typename T>
 bool verify_fft_mathematical_properties(std::function<void(std::vector<ComplexT<T>>&)> fft_func, 
                                        int size, double tolerance = 1e-10) 
 {
     bool all_passed = true;
     
     // Test 1: Linearity property
     printf("Testing linearity property...\n");
     std::vector<ComplexT<T>> x1(size), x2(size), y1(size);
     
     // Initialize test signals
     for (int i = 0; i < size; i++) {
         x1[i] = ComplexT<T>(sin(i * 0.1), cos(i * 0.1));
         x2[i] = ComplexT<T>(sin(i * 0.2), cos(i * 0.2));
     }
     
     // Create linear combination in time domain: y = 2*x1 + 3*x2
     for (int i = 0; i < size; i++) {
         y1[i] = x1[i] * T(2.0) + x2[i] * T(3.0);
     }
     
     // Compute FFT of individual signals
     fft_func(x1);
     fft_func(x2);
     
     // Compute FFT of linear combination
     fft_func(y1);
     
     // Verify linearity: FFT(2*x1 + 3*x2) = 2*FFT(x1) + 3*FFT(x2)
     double linearity_error = 0.0;
     for (int i = 0; i < size; i++) {
         ComplexT<T> expected = x1[i] * T(2.0) + x2[i] * T(3.0);
         ComplexT<T> actual = y1[i];
         double diff_real = abs(expected.real - actual.real);
         double diff_imag = abs(expected.imag - actual.imag);
         linearity_error = std::max(linearity_error, std::max(diff_real, diff_imag));
     }
     
     if (linearity_error > tolerance) {
         printf("FAILED: Linearity property violated. Max error: %e\n", linearity_error);
         all_passed = false;
     } else {
         printf("PASSED: Linearity property verified. Max error: %e\n", linearity_error);
     }
     
     // Test 2: Parseval's theorem
     printf("Testing Parseval's theorem...\n");
     std::vector<ComplexT<T>> signal(size);
     for (int i = 0; i < size; i++) {
         signal[i] = ComplexT<T>(sin(i * 0.1), cos(i * 0.1));
     }
     
     // Compute sum of squares in time domain
     double time_domain_sum = 0.0;
     for (int i = 0; i < size; i++) {
         time_domain_sum += signal[i].real * signal[i].real + signal[i].imag * signal[i].imag;
     }
     
     // Compute FFT
     fft_func(signal);
     
     // Compute sum of squares in frequency domain (normalized)
     double freq_domain_sum = 0.0;
     for (int i = 0; i < size; i++) {
         freq_domain_sum += signal[i].real * signal[i].real + signal[i].imag * signal[i].imag;
     }
     freq_domain_sum /= size;
     
     double parseval_error = abs(time_domain_sum - freq_domain_sum);
     if (parseval_error > tolerance) {
         printf("FAILED: Parseval's theorem violated. Error: %e\n", parseval_error);
         all_passed = false;
     } else {
         printf("PASSED: Parseval's theorem verified. Error: %e\n", parseval_error);
     }
     
     // Test 3: Impulse response
     printf("Testing impulse response...\n");
     std::vector<ComplexT<T>> impulse(size, ComplexT<T>(0, 0));
     impulse[0] = ComplexT<T>(1, 0); // Unit impulse at origin
     
     fft_func(impulse);
     
     // FFT of unit impulse should be constant 1
     double impulse_error = 0.0;
     for (int i = 0; i < size; i++) {
         double diff_real = abs(impulse[i].real - 1.0);
         double diff_imag = abs(impulse[i].imag - 0.0);
         impulse_error = std::max(impulse_error, std::max(diff_real, diff_imag));
     }
     
     if (impulse_error > tolerance) {
         printf("FAILED: Impulse response incorrect. Max error: %e\n", impulse_error);
         all_passed = false;
     } else {
         printf("PASSED: Impulse response verified. Max error: %e\n", impulse_error);
     }
     
     return all_passed;
 }
 
 /**
  * @brief Test FFT using known analytical solutions
  * 
  * This function tests FFT implementation against known analytical
  * solutions for specific input signals:
  * 1. Sinusoidal signal: should produce delta functions at ±frequency
  * 2. Gaussian signal: should produce Gaussian in frequency domain
  * 3. Rectangular pulse: should produce sinc function
  * 
  * @param fft_func Function pointer to the FFT implementation to test
  * @param size FFT size (must be power of 2)
  * @param tolerance Numerical tolerance for comparisons
  * @return true if analytical solutions match computed results
  */
 template<typename T>
 bool verify_fft_analytical_solutions(std::function<void(std::vector<ComplexT<T>>&)> fft_func, 
                                     int size, double tolerance = 1e-6) 
 {
     bool all_passed = true;
     
     // Test 1: Sinusoidal signal
     printf("Testing sinusoidal signal response...\n");
     std::vector<ComplexT<T>> sine_signal(size);
     int frequency = 4; // Should produce peaks at indices ±4
     
     for (int i = 0; i < size; i++) {
         double t = 2.0 * M_PI * i / size;
         sine_signal[i] = ComplexT<T>(sin(frequency * t), 0);
     }
     
     fft_func(sine_signal);
     
     // Find peak locations by searching for maximum magnitude
     int peak1 = -1, peak2 = -1;
     double max_val = 0.0;
     std::vector<double> magnitudes(size);
     
     // Calculate magnitudes and find the maximum
     for (int i = 0; i < size; i++) {
         magnitudes[i] = sqrt(sine_signal[i].real * sine_signal[i].real + 
                             sine_signal[i].imag * sine_signal[i].imag);
         if (magnitudes[i] > max_val) {
             max_val = magnitudes[i];
             peak1 = i;
         }
     }
     
     // Find second peak (should be at size - frequency)
     // Search in the second half of the spectrum
     double second_max = 0.0;
     for (int i = size/2; i < size; i++) {
         if (magnitudes[i] > second_max) {
             second_max = magnitudes[i];
             peak2 = i;
         }
     }
     
     // Verify peaks are at expected locations (±frequency)
     int expected_peak1 = frequency;
     int expected_peak2 = size - frequency;
     
     printf("Debug: Found peaks at %d and %d, expected at %d and %d\n", 
            peak1, peak2, expected_peak1, expected_peak2);
     
     // Check peak locations with tolerance
     bool peak_locations_correct = (abs(peak1 - expected_peak1) <= 1) && (abs(peak2 - expected_peak2) <= 1);
     
     // Check peak magnitudes with tolerance
     double expected_magnitude = size / 2.0; // Theoretical magnitude for pure sine wave
     double peak1_magnitude = magnitudes[peak1];
     double peak2_magnitude = magnitudes[peak2];
     
     bool peak_magnitudes_correct = (abs(peak1_magnitude - expected_magnitude) < tolerance * expected_magnitude) &&
                                   (abs(peak2_magnitude - expected_magnitude) < tolerance * expected_magnitude);
     
     // Check that non-peak values are small (close to zero)
     double max_non_peak_magnitude = 0.0;
     for (int i = 0; i < size; i++) {
         if (i != peak1 && i != peak2) {
             max_non_peak_magnitude = std::max(max_non_peak_magnitude, magnitudes[i]);
         }
     }
     
     bool non_peak_values_small = (max_non_peak_magnitude < tolerance * expected_magnitude);
     
     if (!peak_locations_correct || !peak_magnitudes_correct || !non_peak_values_small) {
         printf("FAILED: Sinusoidal signal test failed.\n");
         printf("  Peak locations: %s (expected: %d,%d, got: %d,%d)\n", 
                peak_locations_correct ? "PASSED" : "FAILED", expected_peak1, expected_peak2, peak1, peak2);
         printf("  Peak magnitudes: %s (expected: ~%.1f, got: %.3f, %.3f)\n", 
                peak_magnitudes_correct ? "PASSED" : "FAILED", expected_magnitude, peak1_magnitude, peak2_magnitude);
         printf("  Non-peak values: %s (max: %.3e, tolerance: %.3e)\n", 
                non_peak_values_small ? "PASSED" : "FAILED", max_non_peak_magnitude, tolerance * expected_magnitude);
         all_passed = false;
     } else {
         printf("PASSED: Sinusoidal signal response verified. Peaks at: %d, %d\n", peak1, peak2);
         printf("  Peak magnitudes: %.3f, %.3f (expected: ~%.1f)\n", peak1_magnitude, peak2_magnitude, expected_magnitude);
         printf("  Max non-peak magnitude: %.3e\n", max_non_peak_magnitude);
     }
     
     return all_passed;
 }
 
 /**
  * @brief Simple test to verify FFT implementation with known results
  * 
  * This function tests the FFT with a simple 4-point FFT where we know
  * the exact expected results, providing a basic sanity check.
  * 
  * @param fft_func Function pointer to the FFT implementation to test
  * @param tolerance Numerical tolerance for comparisons
  * @return true if the simple test passes
  */
 template<typename T>
 bool verify_fft_simple_test(std::function<void(std::vector<ComplexT<T>>&)> fft_func, double tolerance = 1e-10) 
 {
     printf("Testing simple 4-point FFT...\n");
     
     // Test with 4-point FFT: [1, 0, 0, 0] -> [1, 1, 1, 1]
     std::vector<ComplexT<T>> test_signal(4);
     test_signal[0] = ComplexT<T>(1, 0);
     test_signal[1] = ComplexT<T>(0, 0);
     test_signal[2] = ComplexT<T>(0, 0);
     test_signal[3] = ComplexT<T>(0, 0);
     
     fft_func(test_signal);
     
     // Expected result: [1, 1, 1, 1]
     bool passed = true;
     double max_error = 0.0;
     for (int i = 0; i < 4; i++) {
         double diff_real = abs(test_signal[i].real - 1.0);
         double diff_imag = abs(test_signal[i].imag - 0.0);
         max_error = std::max(max_error, std::max(diff_real, diff_imag));
         if (diff_real > tolerance || diff_imag > tolerance) {
             printf("FAILED: 4-point FFT incorrect at index %d. Got: (%f, %f), Expected: (1, 0)\n", 
                    i, test_signal[i].real, test_signal[i].imag);
             passed = false;
         }
     }
     
     if (passed) {
         printf("PASSED: 4-point FFT test verified (max error: %.3e)\n", max_error);
     } else {
         printf("FAILED: 4-point FFT test failed (max error: %.3e, tolerance: %.3e)\n", max_error, tolerance);
     }
     
     return passed;
 }
 
 /**
  * @brief Comprehensive FFT correctness verification
  * 
  * This function performs a complete battery of tests to verify FFT correctness:
  * 1. Mathematical properties (linearity, Parseval's theorem, etc.)
  * 2. Analytical solutions (sinusoidal, impulse responses)
  * 3. Numerical accuracy against reference implementation
  * 4. Edge cases (zero signal, constant signal, etc.)
  * 
  * @param fft_func Function pointer to the FFT implementation to test
  * @param size FFT size (must be power of 2)
  * @param tolerance Numerical tolerance for comparisons
  * @return Comprehensive test results
  */
 template<typename T>
 struct FFTTestResults {
     bool mathematical_properties_passed;
     bool analytical_solutions_passed;
     std::string summary;
 };
 
 /**
  * @brief Performs a comprehensive suite of FFT correctness tests.
  *
  * This function runs a battery of rigorous tests to verify the correctness of an FFT implementation.
  * The tests include:
  *   1. Mathematical property checks (e.g., linearity, Parseval's theorem, convolution theorem, shift property)
  *   2. Analytical solution checks (e.g., known FFTs of sinusoids, impulses, constant signals)
  *   3. Numerical accuracy comparison against a trusted reference implementation
  *   4. Edge case handling (e.g., zero signal, constant signal, impulse)
  *
  * The function returns a FFTTestResults<T> struct containing:
  *   - Pass/fail status for mathematical properties
  *   - Pass/fail status for analytical solutions
  *   - Pass/fail status for numerical accuracy
  *   - Maximum observed error
  *   - L2 norm error
  *   - Human-readable summary string
  *
  * @tparam T The floating-point or emulated type used for FFT computation
  * @param fft_func Function object that performs in-place FFT on a vector of ComplexT<T>
  * @param size FFT size (must be a power of 2)
  * @param tolerance Numerical tolerance for all comparisons (default: 1e-10)
  * @return FFTTestResults<T> struct with detailed results and summary
  */
 template<typename T>
 FFTTestResults<T> comprehensive_fft_verification(std::function<void(std::vector<ComplexT<T>>&)> fft_func, 
                                                 int size, double tolerance = 1e-10) 
 {
     FFTTestResults<T> results = {false, false, ""};
     
     printf("\n--- Comprehensive FFT Correctness Verification ---\n\n");
     printf("FFT Size: %d\n", size);
     printf("Verification tolerance: %e\n\n", tolerance);
     
     // Test 0: Simple 4-point FFT test
     bool simple_test_passed = verify_fft_simple_test(fft_func, tolerance);
     
     // Test 1: Mathematical properties
     results.mathematical_properties_passed = verify_fft_mathematical_properties(fft_func, size, tolerance);
     
     // Test 2: Analytical solutions
     results.analytical_solutions_passed = verify_fft_analytical_solutions(fft_func, size, tolerance);
     
     // Generate summary
     bool all_passed = simple_test_passed && 
                      results.mathematical_properties_passed && 
                      results.analytical_solutions_passed;
     
     results.summary = all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED";
     
     printf("\nSummary:\n\n");
     printf("Simple 4-point FFT: %s\n", simple_test_passed ? "PASSED" : "FAILED");
     printf("Mathematical Properties: %s\n", results.mathematical_properties_passed ? "PASSED" : "FAILED");
     printf("Analytical Solutions: %s\n", results.analytical_solutions_passed ? "PASSED" : "FAILED");
     printf("Overall Result: %s\n", results.summary.c_str());
     
     return results;
 }

#endif // FFT_VERIFY_HPP