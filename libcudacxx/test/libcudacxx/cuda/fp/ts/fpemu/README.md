FPEMU Test Suite (`ts/fpemu/`)
==============================

The FPEMU test suite is a comprehensive testing framework for validating the FPEMU library's functionality, performance, and accuracy. It provides tools for testing various mathematical operations with different computation methods and rounding modes.

Run from the repository root with `make tsemu`, or from this directory with `make run` / `make rerun`.

Table of Contents
----------------
1. [Overview](#overview)
2. [Test Components](#test-components)
3. [Build Configuration](#build-configuration)
4. [Running Tests](#running-tests)
5. [Test Parameters](#test-parameters)
6. [Output and Analysis](#output-and-analysis)
7. [Troubleshooting](#troubleshooting)

Overview
--------
The test suite supports testing of various floating-point operations including:
- Basic arithmetic (add, sub, mul, div)
- Fused multiply-add (fma)
- Square root
- And more...

Each operation can be tested with different:
- Computation methods (accurate, def, fast)
- Rounding modes (rn, rz, ru, rd)

Test Components
--------------
The test suite consists of several key components:

1. Core Files:
   - `ts.hpp`: Main test suite definitions
   - `ts.cpp`: Test suite implementation
   - `ts_functions.hpp`: Function implementations
   - `ts_run.hpp`: Test execution logic
   - `ts_analyze.hpp`: Result analysis
   - `ts_datasets.hpp`: Test data generation
   - `ts_fixed.hpp`: Fixed-point arithmetic support

2. Test Types:
   - Accuracy testing
   - Performance benchmarking
   - Edge case validation
   - Range testing

Dataset Generators
----------------
The test suite provides several dataset generators for comprehensive testing:

1. Basic Datasets:
   - `zero`: Generates zero values (32 samples)
   - `inf`: Generates infinity values (32 samples)
   - `nan`: Generates NaN values (1024 samples)
   - `denorm`: Generates denormal numbers
   - `finite`: Generates finite numbers only
   - `normal`: Generates normal numbers (excluding denormals)

2. Statistical Distributions:
   - `uniform`: Generates uniformly distributed numbers across the full range
   - `gauss`: Generates Gaussian (normal) distribution with configurable mean and standard deviation
   - `mixed`: Combines multiple distributions (50% Gaussian, 35% uniform, 15% key points)

3. Special Test Cases:
   - `key`: Uses predefined key points (+INF, NaN, max normal, min denormal, 0, etc...) for testing
   - `fixed`: Uses tables with operation-specific fixed 'problem' input values, or specified by make parameters A1, A2, A3
   - `hard`: Generates hard-to-round cases for specific operations

Each dataset can be configured with:
- Custom length via `ACCURACY_LEN`
- Custom distribution parameters (mean, stddev)
- Custom input values for fixed tests
- NaN payload handling

Example usage:
```bash
# Test with Gaussian distribution
make FUNC=dadd DATASET=gauss MEAN=0 STDDEV=1

# Test with fixed inputs
make FUNC=dmul DATASET=fixed A1=1.5 A2=2.5

# Test with mixed distribution
make FUNC=fma DATASET=mixed ACCURACY_LEN=1000
```

Build Configuration
-----------------

### Makefile Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TARGET` | Target device | `device` |
| `FUNC` | Function to test | - |
| `TYPE` | Type of emulation functions | `fpemu` |
| `ROUNDING` | Rounding mode | `rn` |
| `ACCURACY` | Accuracy level (high, mid, low; def==high) | `def` |
| `ARCH` | CUDA architecture | `86` |
| `LINKAGE` | Linkage mode | `inline` |
| `THREADS` | Threads per block | - |
| `REPEATS` | Test repetitions | - |
| `ITERATIONS` | Iterations per test | - |

### Advanced Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ACCURACY_LEN` | Accuracy array length | - |
| `LATENCY_LEN` | Latency timing array length | - |
| `THROUGHPUT_LEN` | Throughput timing array length | - |
| `NAN_PAYLOAD` | Use NaN payload | `0` |
| `MEAN` | Input mean | `0` |
| `STDDEV` | Input standard deviation | `1` |
| `USE_BUILTINS` | Use builtins for compound functions | `1` |
| `PRINT_LIMIT` | Print limit | `10` |
| `PRINT` | Log output level | `fail` |
| `TIMING` | Run timing tests | `0` |
| `UNPACKED` | Used unpacked add/mul | `0` |
| `DEBUG` | Debug mode to enable custom printouts | `n` |
| `CONSOLE` | console type (`stdout`, `stderr`, `file`, `null`) | `stdout` |
| `ASM` | Generate assembly outputs for tested functions | `n` |
| `A1`, `A2`, `A3` | First, second and third input arguments for testing | - |
| `OUT` | Output directory for test results | `_out` |
| `VERBOSE` | Verbose mode (`0`=silent, `1`=minimal, `2`=full) | `0` |

Running Tests
------------

### Basic Usage

1. Test all functions:
```bash
make
```

2. Test specific function:
```bash
make FUNC=dadd
```

3. Test with specific parameters:
```bash
make FUNC=fma ROUNDING=rn ACCURACY=high
```

### Build Targets

| Target | Description |
|--------|-------------|
| `all` | Build and run all tests |
| `run` | Run existing tests |
| `rerun` | Rebuild and run tests |
| `clean` | Clean build artifacts |

### Common Examples

1. Test addition on host:
```bash
make TARGET=host FUNC=dadd
```

2. Test with custom parameters:
```bash
make FUNC=fma ROUNDING=rn ACCURACY=low THREADS=256
```

3. Debug mode:
```bash
make DEBUG=1 FUNC=dadd
```

### Advanced Examples

1. Test with specific input values:
```bash
make FUNC=dadd A1=1.5 A2=2.5
```

2. Test with custom accuracy settings:
```bash
make FUNC=dmul ACCURACY=high ROUNDING=rn ACCURACY_LEN=1000
```

3. Performance testing with timing:
```bash
make FUNC=fma TIMING=1 THREADS=256 REPEATS=1000
```

4. Test with custom input distribution:
```bash
make FUNC=ddiv MEAN=0.5 STDDEV=0.1 ACCURACY=low
```

5. Verbose output with custom print settings:
```bash
make FUNC=dadd VERBOSE=2 PRINT=all PRINT_LIMIT=20
```

6. Test with NaN handling:
```bash
make FUNC=dsqrt NAN_PAYLOAD=1 ACCURACY=high
```

7. Test with custom output directory:
```bash
make FUNC=dadd OUT=test_results_001
```

8. Combined parameters example:
```bash
make FUNC=fma ROUNDING=rn ACCURACY=high THREADS=512 REPEATS=100 TIMING=1 VERBOSE=1
```

Output and Analysis
-----------------

The test suite generates several types of output:

1. Accuracy Reports:
   - Bit-level accuracy comparison
   - Error statistics
   - Edge case results

2. Performance Metrics:
   - Latency measurements
   - Throughput analysis
   - Resource utilization

3. Log Files:
   - Test results
   - Error reports
   - Performance data

Timing Methods
-------------
The test suite provides two timing methods for performance evaluation:

1. Throughput Measurement:
   - Measures operations per second per streaming multiprocessor (SM)
   - Uses large input arrays (default: 64K elements on GPU, 128K on CPU)
   - Multiple threads per block (default: 512)
   - Multiple iterations to find minimum execution time
   - Formula: `throughput = (repeats * array_size) / (time * num_SMs * clock_rate)`

2. Latency Measurement:
   - Measures cycles per operation
   - Uses small input arrays (default: 512 elements on GPU, 1024 on CPU)
   - Single thread per block
   - Multiple iterations to find minimum execution time
   - Formula: `latency = (time * clock_rate) / (repeats * array_size)`

Implementation Details:
- Uses CUDA events for GPU timing
- Uses high-resolution timers for CPU timing
- Prevents compiler optimizations by using results
- Configurable number of repeats and iterations
- Supports both device and host execution

Optimization Prevention:
- Results are stored in output arrays to prevent dead code elimination
- Input values are modified between iterations using bit manipulation
- Results are used to update input values to prevent loop unrolling
- Memory operations are preserved to maintain realistic timing
- Thread synchronization is enforced to prevent instruction reordering

Example timing output:
```
| ev/sm/ck=1.234e6 / 2.345e6 (0.53x), ck/ev=123 / 234 (1.90x)
```
Where:
- `ev/sm/ck`: Events (operations) per SM per clock cycle
- `ck/ev`: Clock cycles per event (operation)
- Numbers in parentheses show speedup/slowdown ratios

Troubleshooting
--------------

Common Issues and Solutions:

1. Build Failures:
   - Ensure CUDA toolkit is properly installed
   - Check architecture compatibility
   - Verify compiler version

2. Test Failures:
   - Check input parameters
   - Verify computation method settings
   - Review error logs

3. Performance Issues:
   - Adjust thread count
   - Check memory usage
   - Verify architecture settings
