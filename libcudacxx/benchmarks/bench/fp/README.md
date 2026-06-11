FPMP Benchmarks
================

This directory contains benchmarks for evaluating FPMP performance and resource utilization across different kernels and data types. Each subdirectory contains its own Makefile and (where applicable) additional documentation.

Table of Contents
----------------
1. [Overview](#overview)
2. [Common Parameters](#common-parameters)
3. [Running Benchmarks](#running-benchmarks)
4. [Results Analysis](#results-analysis)

Overview
--------
The benchmarks in this directory help evaluate FPMP by measuring:
- Performance characteristics (throughput, latency)
- Resource utilization (registers, memory)
- Hardware compatibility
- Optimization opportunities

Common Parameters
---------------

### Build Options (wrapper Makefile: `benchmarks/Makefile`)
| Option | Description | Default |
|--------|-------------|---------|
| `TARGET` | Target device (`device`, `host`) | `device` |
| `LINKAGE` | Linkage mode (`inline`, `static`, `lto`) | `inline` |
| `OUT` | Output directory for results | `_out` |
| `VERBOSE` | Verbose mode (`0`=silent, `1`=minimal, `2`=full, `3`=very verbose); aliases: `V`, `VERB` | `0` |
| `BENCHMARK` | Benchmark(s) to run (space-separated), e.g. `fft` | all |

### Benchmark-specific options
Each benchmark may define additional parameters in its own Makefile/README (for example, `mixed/` supports a `TYPE` selector). See the corresponding subdirectory documentation.

Running Benchmarks
----------------

### Common Commands
```bash
# Build and run all benchmarks
make rerun

# Run existing builds (no clean)
make run

# Run a single benchmark
make BENCHMARK=fft rerun

# Clean build artifacts
make clean
```

### Directory-Specific Commands
Each benchmark directory has its own specific commands and parameters. Please refer to the individual README files:
- `demo/README.md` - Basic float-float operations demo
- `acc/README.md` - ACC vs ADD: Optimized accumulate vs full addition
- `atomic/README.md` - Atomic operations under contention
- `fft/README.md` - FFT benchmarks
- `mc/README.md` - Monte Carlo European option pricing across six FP types
- `mixed/README.md` - Mixed kernels/data-type sweep benchmarks
- `fortran_sum/README.md` - Recursive-summation accuracy & throughput in Fortran (host + CUDA Fortran + OpenACC), comparing `real(real32)`, `real(real64)`, `type(fp32mp2)`, and `type(fp64mp2)`
- `hpc_exp_error/README.md` - Composite-`exp()` error & GPU-throughput study (`compose` vs `expsum` kernels, `__float128` reference, alpha-exponent sweep). Skipped from default `make rerun`; run explicitly with `make BENCHMARK=hpc_exp_error rerun`. Long build/run time, intended for FP64-throttled GPUs (Ada / L40S / RTX-class).

Results Analysis
--------------

### Performance Analysis
1. Throughput Analysis:
   - Operations per second
   - Bandwidth utilization
   - Instruction throughput
   - Memory throughput

2. Latency Analysis:
   - Operation latency
   - Memory access latency
   - Instruction latency
   - Synchronization overhead

3. Resource Analysis:
   - Register usage
   - Shared memory usage
   - Thread block occupancy
   - Cache utilization

### Accuracy Analysis
If a benchmark includes correctness/accuracy validation, it will document the methodology in its own README.

### Optimization Analysis
1. Code Analysis:
   - Instruction count
   - Register usage
   - Memory access patterns
   - Resource utilization

2. Performance Analysis:
   - Bottleneck identification
   - Optimization opportunities
   - Resource constraints
   - Scaling behavior

