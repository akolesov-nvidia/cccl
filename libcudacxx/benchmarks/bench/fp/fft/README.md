FFT Benchmark
============

This directory contains a GPU (CUDA) benchmark for measuring the performance of FFT (Fast Fourier Transform) operations using fp32mp2-based complex arithmetic. It compares native double precision with float-float (fp32mp2) using different implementation methods.

Table of Contents
----------------
1. [Overview](#overview)
2. [FFT Algorithm](#fft-algorithm)
3. [Benchmark Features](#benchmark-features)
4. [Running the Benchmark](#running-the-benchmark)
5. [Results Interpretation](#results-interpretation)
6. [Performance Analysis](#performance-analysis)

Overview
--------
The FFT benchmark evaluates the performance impact of using float-float (fp32mp2) multi-precision floating point operations in compute-intensive FFT workloads. FFT is a fundamental algorithm used in many scientific and engineering applications, making it an excellent test case for evaluating multi-precision floating point performance.

The benchmark implements:
- Radix-2 FFT algorithm with bit reversal
- Complex number arithmetic using both native double and float-float types
- Performance comparison across different float-float implementation methods
- GFLOPS calculation and speedup analysis

FFT Algorithm
-------------
The benchmark implements a Cooley-Tukey radix-2 FFT algorithm with the following characteristics:

1. **Bit Reversal**: Reorders input data for in-place computation
2. **Butterfly Operations**: Performs complex arithmetic operations
3. **Twiddle Factors**: Uses trigonometric functions for phase calculations
4. **In-Place Computation**: Modifies data in-place to minimize memory usage

The algorithm complexity is O(N log N) where N is the FFT size.

Benchmark Features
-----------------

### 1. Multiple Precision Modes
- **Native Double**: Standard IEEE-754 double precision
- **ffloat (fp32mp2)**: Float-float precision with default method (classic Dekker-based)
- **ffloat_fast (fp32mp2)**: Float-float precision with fast method (optimized for speed)

### 2. Configurable Parameters
- **FFT Size**: Power-of-2 sizes (default: 1024 when built via this directory Makefile)
- **Batch Size**: Number of FFTs to process in parallel (default: 8)
- **Iterations**: Number of benchmark iterations (default: 100 when built via this directory Makefile)
- **Warm-up Iterations**: GPU stabilization runs (default: 100)

### 3. Performance Metrics
- **Execution Time**: Measured in milliseconds with statistical analysis
- **GFLOPS**: Floating point operations per second (accounts for batch processing)
- **Speedup**: Performance relative to native double precision
- **Throughput**: Total FFTs processed per second

### 4. Verification
- Host-side reference implementation
- Result validation with configurable tolerance
- Error detection and reporting

### 5. Batch Processing
The benchmark supports batch processing to improve GPU utilization:
- **Parallel FFTs**: Multiple FFTs processed simultaneously
- **Memory Efficiency**: Better memory bandwidth utilization
- **GPU Utilization**: Improved occupancy and throughput
- **Scalable Performance**: Performance scales with batch size

### 6. Timing Methodology
For stable and reliable timing results:
- **Multiple Runs**: 5 timing runs per benchmark
- **Warm-up Phase**: 100 iterations to stabilize GPU state
- **Statistical Analysis**: Excludes outliers for reliable averages
- **Longer Workloads**: Increased iterations for accurate measurements
- **Delays Between Runs**: 1ms delays to prevent interference

Running the Benchmark
--------------------

### Build Options
| Option | Description | Default |
|--------|-------------|---------|
| `LINKAGE` | Linkage mode (`inline`, `static`, `lto`) | `inline` |
| `VERBOSE` | Verbose mode (`0`, `1`, `2`, `3`); aliases: `V`, `VERB` | `0` |
| `SIZE` | FFT size (power of 2) | `1024` |
| `BATCH` | Batch size | `8` |
| `ITERATIONS` | Number of iterations | `100` |

### Build Commands
```bash
# Build and run with default parameters
make

# Run with different FFT sizes
make SIZE=2048 rerun
make SIZE=4096 rerun

# Run with batch processing
make BATCH=4 rerun

# Verbose build
make VERBOSE=2 rerun
```

### Example Output
```
FFT Benchmark
=============
FFT Size: 1024
Batch Size: 8
Iterations: 100
Warm-up Iterations: 100
Target: CUDA Device

Verifying FFT implementation...
FFT verification passed!

Running benchmarks (this may take a while for stable timing)...
Processing FFTs of size 1024 in batches of 8
Each benchmark runs 100 iterations with 100 warm-up iterations
Multiple timing runs with statistical analysis for stability
Implementation      :     Time ms,  GFLOPS
--------------------------------------------------------
Running Native Double benchmark...
Native Double      :   245.67 ms,  182.34 GFLOPS
Running ffloat (fp32mp2) benchmark...
ffloat (fp32mp2)   :   278.45 ms,  164.32 GFLOPS
Running ffloat_fast (fp32mp2) benchmark...
ffloat_fast        :   256.23 ms,  178.56 GFLOPS

Speedup vs Native Double:
ffloat (fp32mp2): 0.88x
ffloat_fast (fp32mp2): 0.96x

Performance Analysis:
Total FFTs processed: 32768000
Total operations: 1.18e+12
Average time per FFT: 0.0075 ms
Timing methodology: 5 runs, excluding outliers for stability

Benchmark completed successfully!
```

Results Interpretation
---------------------

### Performance Metrics

1. **Execution Time**: Lower is better
   - Measured in milliseconds
   - Includes memory transfers and kernel execution
   - Averaged over multiple iterations

2. **GFLOPS**: Higher is better
   - Calculated as: 5 × N × log₂(N) × iterations / (time × 10⁻³) / 10⁹
   - Represents floating point operations per second
   - Accounts for FFT algorithm complexity

3. **Speedup**: Relative to native double precision
   - Values > 1.0 indicate faster performance
   - Values < 1.0 indicate slower performance
   - Useful for comparing different precision modes

### Expected Results

1. **Native Double**: Baseline performance
2. **ffloat (fp32mp2)**: Good precision with moderate performance overhead
3. **ffloat_fast (fp32mp2)**: Better performance with fast arithmetic methods

### Factors Affecting Performance

1. **FFT Size**: Larger sizes increase computation and memory requirements
2. **Batch Size**: Multiple FFTs can improve GPU utilization
3. **Memory Bandwidth**: FFT is memory-intensive
4. **Arithmetic Intensity**: Complex arithmetic operations per memory access
5. **Precision Requirements**: Higher precision may require more operations

Performance Analysis
-------------------

### Memory Access Patterns
- **Coalesced Access**: FFT benefits from memory coalescing
- **Strided Access**: Bit reversal can cause strided memory access
- **Shared Memory**: Could be optimized with shared memory usage

### Computational Characteristics
- **Complex Arithmetic**: Addition, multiplication, and trigonometric functions
- **Branch Divergence**: Minimal due to regular access patterns
- **Instruction Throughput**: Limited by arithmetic operations

### Optimization Opportunities
1. **Shared Memory**: Cache twiddle factors in shared memory
2. **Memory Coalescing**: Optimize bit reversal for better memory access
3. **Loop Unrolling**: Unroll FFT stages for better instruction-level parallelism
4. **Specialized Kernels**: Different kernels for different FFT sizes

### Accuracy vs Performance Trade-offs
- **ffloat (default method)**: Uses classic Dekker-based split for high accuracy
- **ffloat_fast (fast method)**: Optimized arithmetic with faster operations

The benchmark helps users understand these trade-offs and choose the appropriate method for their specific application requirements. 