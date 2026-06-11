# CG (Conjugate Gradient) Benchmark

A high-performance CUDA benchmark implementing the Conjugate Gradient method for solving large, sparse, symmetric positive-definite linear systems. This benchmark is designed to evaluate GPU performance with both native double-precision and emulated floating-point arithmetic.

## Overview

The Conjugate Gradient method is an iterative algorithm for solving systems of linear equations Ax = b, where A is a symmetric positive-definite matrix. This implementation uses a 2D Laplacian operator (5-point stencil) as the coefficient matrix, which is common in finite difference methods for partial differential equations.

### Problem Formulation
- **Matrix A**: 2D Laplacian operator on N×N grid
- **Right-hand side b**: Computed as A * x_true where x_true = 1
- **Solution x**: Iteratively computed using CG method
- **Convergence**: Measured by relative residual ||r||/||b||

## Features

### Core Functionality
- **Conjugate Gradient Solver**: Complete CG implementation with residual recomputation
- **2D Laplacian Operator**: 5-point stencil sparse matrix-vector multiplication
- **Flexible Data Types**: Support for both native `double` and custom `fp64emu_t` types
- **Emulation Support**: Built-in support for fp64emu emulation system

### Performance Features
- **CUDA Optimized**: Fully GPU-accelerated with optimized kernels
- **Configurable Threading**: Adjustable block size and grid dimensions
- **Memory Efficient**: Minimal memory footprint with in-place operations
- **Timing Measurement**: Precise CUDA event-based timing

### Benchmarking Capabilities
- **Convergence Analysis**: Residual tracking and error analysis
- **Performance Metrics**: GFLOP/s calculation and iteration counting
- **Parameter Sweeping**: Easy testing of different problem sizes and configurations
- **Hardware Validation**: GPU device information and memory usage reporting

## Algorithm Details

### Conjugate Gradient Method
```
1. Initialize: x = 0, r = b, p = r
2. For each iteration:
   a. Compute Ap = A * p
   b. Compute α = (r·r) / (p·Ap)
   c. Update x = x + α * p
   d. Update r = r - α * Ap
   e. Check convergence: ||r||/||b|| < tolerance
   f. Compute β = (r_new·r_new) / (r·r)
   g. Update p = r + β * p
3. Output: solution x, iterations, performance metrics
```

### CUDA Kernel Operations
- **`fill_kernel`**: Vector initialization with constant values
- **`axpby_kernel`**: Vector operation y = a*x + b*y
- **`axpy_kernel`**: Vector operation y += a*x
- **`copy_kernel`**: Vector copy operation
- **`spmv2d_laplacian`**: Sparse matrix-vector multiplication
- **`dot_kernel`**: Dot product with shared memory reduction

## Installation and Build

### Prerequisites
- **CUDA Toolkit**: Version 11.0 or higher
- **NVIDIA GPU**: Compute capability 6.0 or higher
- **Make**: Standard build system
- **Compiler**: nvcc (CUDA compiler)

### Build Instructions

#### Using the Makefile (Recommended)
```bash
# Navigate to the benchmark directory
cd benchmarks/cg

# Build with default settings
make

# Build with custom parameters
make SIZE=1024 MAX_ITER=1000 TOLERANCE=1e-6 rerun

# Clean build artifacts
make clean
```

#### Manual Build
```bash
# Compile with nvcc
nvcc -O3 -arch=sm_86 -o cg cg.cu

# Run the benchmark
./cg
```

### Build Configuration

The Makefile supports several configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIZE` | 2048 | Grid size N (creates N×N problem) |
| `MAX_ITER` | 3000 | Maximum CG iterations |
| `TOLERANCE` | 1e-4 | Convergence tolerance |
| `THREADS_PER_BLOCK` | 256 | CUDA threads per block |
| `GRID_MULTIPLIER` | 1.0 | Grid size multiplier for parallelism |

## Usage

### Command Line Interface
```bash
./cg [N] [maxIters] [tolerance] [threadsPerBlock] [gridMultiplier]
```

### Parameter Examples

#### Basic Usage
```bash
# Use all defaults (N=2048, maxIters=3000, tol=1e-4, threads=256)
./cg

# Custom problem size
./cg 1024

# Custom problem size and iterations
./cg 1024 1000

# Custom problem size, iterations, and tolerance
./cg 1024 1000 1e-6

# Full customization including CUDA parameters
./cg 1024 1000 1e-6 128 2.0
```

#### Performance Tuning
```bash
# Test different block sizes
./cg 1024 1000 1e-6 32    # 32 threads per block
./cg 1024 1000 1e-6 64    # 64 threads per block
./cg 1024 1000 1e-6 128   # 128 threads per block
./cg 1024 1000 1e-6 256   # 256 threads per block (default)
./cg 1024 1000 1e-6 512   # 512 threads per block
./cg 1024 1000 1e-6 1024  # 1024 threads per block

# Test different grid configurations
./cg 1024 1000 1e-6 256 0.5   # Half the optimal grid size
./cg 1024 1000 1e-6 256 1.0   # Optimal grid size (default)
./cg 1024 1000 1e-6 256 2.0   # Double the optimal grid size
```

### Emulation Mode

#### Enable fp64emu Emulation (Default)
```bash
# Uses fp64emu_t type for all computations
make rerun
```

#### Disable Emulation (Native Double)
```bash
# Uses native double type for faster execution
make NO_EMULATION=1 rerun
```

#### Accuracy Selection
```bash
# Use default accuracy (def)
make rerun

# Use high accuracy (full mantissa precision, full IEEE-754 range)
make ACCURACY=high rerun

# Use low accuracy (relaxed precision, relaxed special values/denormals)
make ACCURACY=low rerun
```

## Output and Results

### Benchmark Output
```
GPU: NVIDIA GeForce RTX 4090
CG FP64 benchmark: N=2048 (n=4194304), maxIters=3000, tol=1.000e-04
CUDA config: threadsPerBlock=256, numBlocks=16384, gridMultiplier=1.00
Grid optimization: optimal=16384, actual=16384 (multiplier=1.00

Results:
  Final rel residual: 9.876e-05 (tol=1.000e-04): OK
  Rel error vs x*=1 : 2.345e-06
  Time              : 1155.480 ms
  Perf              : 120.45 GFLOP/s
```

### Performance Metrics
- **GFLOP/s**: Floating-point operations per second
- **Convergence**: Final relative residual vs. tolerance
- **Accuracy**: Relative error compared to true solution
- **Timing**: Total execution time in milliseconds
- **Iterations**: Number of CG iterations performed

### Memory Usage
The benchmark allocates the following GPU memory:
- **Vectors**: 6 vectors of size N² (x, b, r, p, Ap, xtrue)
- **Block Sums**: Array for dot product reduction
- **Total**: Approximately 67 MB for N=2048 with double precision

## Performance Analysis

### Expected Performance
- **Modern GPUs (RTX 4000 series)**: 100-200 GFLOP/s
- **Data Center GPUs (A100, H100)**: 200-500 GFLOP/s
- **Emulation Overhead**: 1.5x-3x slower than native double

### Optimization Guidelines

#### Block Size Selection
- **32-64**: Good for very small problems, minimal memory usage
- **128**: Sweet spot for many GPUs, good balance
- **256**: Default, good for most problems
- **512-1024**: Good for larger problems on newer GPUs

#### Grid Size Tuning
- **Grid Multiplier < 1.0**: Reduces parallelism, may improve cache efficiency
- **Grid Multiplier = 1.0**: Optimal parallelism (default)
- **Grid Multiplier > 1.0**: Increases parallelism, may improve occupancy

#### Problem Size Scaling
- **Small (N < 1024)**: Limited by kernel launch overhead
- **Medium (1024 ≤ N < 4096)**: Good balance of parallelism and memory
- **Large (N ≥ 4096)**: Memory bandwidth limited, good for stress testing

## Troubleshooting

### Common Issues

#### Build Errors
```bash
# CUDA version compatibility
error: identifier "clockRate" is undefined
# Solution: Use CUDA 13.0+ or update the code for older versions

# Memory allocation failures
CUDA error: out of memory
# Solution: Reduce problem size N or use smaller data types
```

#### Runtime Errors
```bash
# Non-positive pAp error
Non-positive pAp at iter X: value
# Solution: Check problem setup, may indicate numerical instability

# Convergence failures
Final rel residual: X.XXXe-XX (tol=Y.YYYe-YY): FAIL
# Solution: Increase maxIters or adjust tolerance
```

#### Performance Issues
```bash
# Low GFLOP/s
# Solutions:
# - Check GPU utilization (nvidia-smi)
# - Verify block size is optimal for your GPU
# - Ensure problem size is large enough to hide launch overhead
```

### Debug Mode
```bash
# Enable verbose output
make VERBOSE=2 rerun

# Check GPU configuration
./cg -h
```

## File Structure

```
benchmarks/cg/
├── cg.cu              # Main benchmark source code
├── Makefile           # Build configuration and targets
├── README.md          # This documentation file
└── fpemu_type.h      # Emulated floating-point type definitions
```

## Dependencies

### Required Headers
- `cuda_runtime.h`: CUDA runtime API
- `fpemu_type.h`: Emulated floating-point types
- Standard C++ headers: `<cstdio>`, `<cstdlib>`, `<cmath>`, `<algorithm>`

### External Libraries
- **CUDA Runtime**: NVIDIA CUDA runtime library
- **fpemu Library**: Optional emulation library (see main project Makefile, `libcufp`)

## Contributing

### Code Style
- **Naming**: Use descriptive names for functions and variables
- **Documentation**: Document all public functions with Doxygen-style comments
- **Error Handling**: Use CUDA_OK macro for all CUDA calls
- **Templates**: Use template parameters for data type flexibility

### Testing
- **Validation**: Verify results against known solutions
- **Performance**: Test on multiple GPU architectures
- **Edge Cases**: Test with extreme parameter values
- **Memory**: Verify memory usage across problem sizes

## License

This benchmark is part of the CUDA FP64 project. See the main project license for details.

## References

1. **Conjugate Gradient Method**: Hestenes, M.R., Stiefel, E. (1952)
2. **CUDA Programming Guide**: NVIDIA Corporation
3. **Numerical Linear Algebra**: Trefethen, L.N., Bau, D. (1997)
4. **Finite Difference Methods**: LeVeque, R.J. (2007)

## Contact

For questions, issues, or contributions:
- **Author**: Andrei Kolesov
- **Project**: CUDA Multi FP Benchmark Suite
- **Date**: 2025-01-15

---

*This benchmark provides a robust foundation for evaluating GPU performance with iterative linear solvers and can be easily extended for other sparse matrix types and numerical methods.*
