# Mixed Precision Throughput Benchmark

This benchmark compares the throughput performance of various floating-point data types for three fundamental arithmetic operations: **ADD**, **MUL**, and **DIV**.

## Data Types Compared

| Type | Description | Mantissa | Size |
|------|-------------|----------|------|
| `float` | Native IEEE 754 single precision | 24-bit | 4 bytes |
| `float-float` | Double-float emulation (fp32mp2) | up to ~46 effective bits | 8 bytes |
| `double` | Native IEEE 754 double precision | 53-bit | 8 bytes |
| `efp64_cr` | Correctly rounded fp64 emulation | 53-bit | 8 bytes |
| `efp64_ha` | High-accuracy fp64 emulation (1-2 ULP) | ~53-bit | 8 bytes |
| `double-double` | Double-double emulation (fp64mp2) | up to ~104 effective bits | 16 bytes |
| `fp128` | IEEE 754 quad precision | 113-bit | 16 bytes |

### Library Dependencies

- **float-float, double-double**: `cuda_multi_fp` library (`fpmp.h`)
- **efp64_cr, efp64_ha**: `cuda_fp64` library (`efp64.hpp`)

## Performance Metrics

### Primary: Operations per SM per GPU Cycle

```
Ops/SM/Cycle = total_ops / (time_sec × clock_rate_Hz × num_SMs)
```

This metric normalizes performance across different GPUs and provides a hardware-independent measure of arithmetic throughput.

### Secondary: GFLOPS

```
GFLOPS = total_ops / (time_ms × 1e6)
```

Absolute performance measure in billions of floating-point operations per second.

## Building and Running

### Quick Start

```bash
# Build and run with default types (fp32, fp32-fp32, fp64, fp64-fp64)
make

# Run all data types
make TYPE=all run

# Or explicitly
make build
make run
```

### Configuration Options

| Parameter | Default (`make`) | Description |
|-----------|------------------|-------------|
| `TYPE` | fp32 fp32-fp32 fp64 fp64-fp64 | Data types to benchmark (see below) |
| `TARGET` | device | `device` (CUDA) or `host` (CPU) |
| `ARCH` | 86 | CUDA compute capability (86=Ampere, 89=Ada, 100=Blackwell) |
| `NUM_ITER` | 10 | Number of timing iterations (`NUM_ITERATIONS` in `mixed.cpp`) |
| `THREADS` | 256 | Threads per CUDA block; passed as `-DTHREADS_PER_BLOCK=$(THREADS)` |
| `NUM_BLOCKS` | 8192 | Number of CUDA blocks |
| `REPS` | 4096 | Operations per thread in the inner benchmark loop |
| `UNROLL` | 64 | Loop unroll factor (must divide `REPS`) |
| `OUT` | `_out` | Build output directory (under `benchmarks/mixed/`) |
| `LINKAGE` | inline | Library linkage mode (`inline`, `static`, `lto`) |
| `VERBOSE` | 1 | Verbosity level (0=silent, 1=minimal, 2=full, 3=very verbose); aliases: `V`, `VERB` |

### Data Type Selection (TYPE parameter)

| Type Name | Aliases | Description |
|-----------|---------|-------------|
| `fp32` | - | Native float |
| `fp32-fp32` | `fp32mp2` | Float-float (fp32mp2; up to ~46 effective mantissa bits) |
| `fp64` | - | Native double |
| `fp64cr` | `fp64-cr` | Emulated FP64, correctly rounded |
| `fp64ha` | `fp64-ha` | Emulated FP64, high accuracy (1-2 ULP) |
| `fp64-fp64` | `fp64mp2` | Double-double (fp64mp2; up to ~104 effective mantissa bits) |
| `fp128` | - | Quad precision (113-bit mantissa) |
| `all` | - | All types above |

### Examples

```bash
# Run on RTX 40xx (Ada Lovelace)
make ARCH=89 run

# Run on Blackwell with full fp128 support
make ARCH=100 run

# Benchmark all data types
make TYPE=all run

# Benchmark only native types
make TYPE="fp32 fp64" run

# Benchmark only emulated FP64 types
make TYPE="fp64cr fp64ha" run

# Compare float-float vs double-double
make TYPE="fp32-fp32 fp64-fp64" run

# Lighter run than default `REPS` / `NUM_BLOCKS` (see table above)
make REPS=1024 NUM_BLOCKS=512 THREADS=128 run

# Rebuild with full verbose output
make VERBOSE=3 rebuild

# Clean and rebuild
make rerun
```

## Output Format

### Detailed Results Table

```
Data Type      | Op     |     Time (ms) |       Ops/SM/Cycle |       GFLOPS
---------------+--------+--------------+--------------------+--------------
float          | ADD    |       0.0176 |          42.377943 |     15254.79
float          | MUL    |       0.0175 |          42.587042 |     15330.06
float          | DIV    |       0.1200 |           6.213473 |      2236.66
---------------+--------+--------------+--------------------+--------------
float-float    | ADD    |       0.0688 |          10.836882 |      3900.95
...
```

### Summary Table (Relative Performance)

```
Data Type      |          ADD |          MUL |          DIV
---------------+--------------+--------------+--------------
float          |        1.00x |        1.00x |        1.00x
float-float    |        3.91x |        3.57x |        0.66x
double         |       28.08x |       28.22x |       29.56x
efp64_cr       |       40.49x |       35.70x |       51.63x
efp64_ha       |       16.39x |       10.70x |       51.16x
double-double  |      212.70x |      147.96x |       20.90x
fp128          |      125.55x |      251.44x |       32.95x
```

## Interpreting Results

- **Ops/SM/Cycle**: Higher is better. Indicates how many operations each SM completes per clock cycle.
- **GFLOPS**: Higher is better. Absolute performance measure.
- **Relative (Nx)**: Lower is better. Shows slowdown compared to native float (1.00x = same speed).

## Use Cases

### Consumer GPUs (RTX Series)

On consumer GPUs where FP64 performance is severely limited (typically 1/32 of FP32):
- **float-float** provides ~48-bit precision with much better performance than native `double`
- **efp64_ha** provides fp64-compatible precision at higher throughput than native `double`

### Data Center GPUs (A100, H100, B100)

On data center GPUs with full FP64 support:
- Native `double` typically outperforms emulated types
- **double-double** provides ~106-bit precision when needed
- **fp128** provides full quad precision (113-bit mantissa)

### Precision vs Performance Trade-offs

| Need | Recommended Type |
|------|------------------|
| FP32 speed, ~48-bit precision | float-float |
| FP64 compatibility, faster than native | efp64_ha |
| FP64 correctly rounded | efp64_cr or native double |
| Extended precision (~106-bit) | double-double |
| Quad precision (113-bit) | fp128 |

## Technical Notes

### Anti-Optimization Measures

The benchmark uses shared **`add_perf_kernel` / `mul_perf_kernel` / `div_perf_kernel`** templates (per numeric `Type`) with:

1. **Unrolled accumulators** (`aa[]`, `bb[]` for division): each lane keeps its own value so work is not folded away.
2. **Per-lane operand spread**: `a + 0.0001 * j` (and `b` the same for div) so unrolled lanes are not bitwise-identical.
3. **Optional sink**: a `never` flag and conditional write to `result[]` so results stay live when enabled; timing runs use `never == 0`.
4. **Operand layout**: division uses `aa[j] / bb[j]` with separate `bb[]` divisors to avoid a single shared reciprocal dominating the timed region.

### FP128 Support

FP128 (quad precision) requires:
- Enable via `TYPE=fp128` or `TYPE=all`
- Blackwell architecture (sm_100+) recommended for optimal performance
- Operands are built from `double` / thread id and cast to `fp128_t` in the shared kernels (same pattern as other types)

## Files

- `mixed.cpp` - Main benchmark source code
- `Makefile` - Build system
- `README.md` - This documentation