# Double-Precision Gaussian RNG Performance Benchmark

Measures throughput and statistical correctness of double-precision Gaussian
random number generation on GPU, comparing three methods side-by-side in a
single run.

## Methods

| Method | Implementation | Notes |
|--------|---------------|-------|
| `curand_normal_double` | `curand_normal_double(&state)` | cuRAND built-in Box-Muller |
| `uniform + normcdfinv` | `normcdfinv(curand_uniform_double(&state))` | CUDA math intrinsic |
| `fpmp icdf(uint64)` | Two `curand()` → `uint64` → `icdf()` → `double` | fpmp fp32mp2 polynomial (~46-bit mantissa) |

All methods use the same `curandStatePhilox4_32_10_t` RNG state, so timing
differences reflect the Gaussian transform cost alone.

The `fpmp icdf` method combines two 32-bit `curand()` outputs into a 64-bit
uniform integer to match the entropy that `curand_normal_double` uses
internally.  The ICDF produces an `fp32mp2` result which is cast to `double`.

## Quick start

```bash
make            # build and run
make REPS=256   # higher rigor (longer kernels, more stable timing)
```

## Build parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ARCH` | `86` | CUDA SM architecture |
| `NUM_SAMPLES` | `16M` | Samples per launch |
| `REPS` | `128` | Inner repetitions per kernel (controls rigor) |
| `NUM_ITER` | `10` | Timing iterations for averaging |
| `THREADS` | `256` | CUDA threads per block |
| `NUM_BLOCKS` | `2048` | CUDA grid blocks |
| `LINKAGE` | `inline` | `inline`, `static`, or `lto` |
| `VERBOSE` | `1` | Verbosity level (`0`..`3`) |

## Statistical validation

Each launch generates `NUM_SAMPLES * REPS` Gaussian doubles.  Per-thread
partial sums and sums-of-squares are reduced on host to compute the overall
mean and standard deviation.  The benchmark checks:

- `|mean| < 5 / sqrt(N)`
- `|stddev - 1| < 5 / sqrt(2N)`

where `N = NUM_SAMPLES * REPS`.

## Example output

```
Double-Precision Gaussian RNG Performance Benchmark
===================================================
Target: GPU (NVIDIA GeForce RTX 3090)

Samples: 16777216  REPS: 128
Threads: 524288  Timing iterations: 10

Effective samples per launch: 2147483648

Methods:
  curand_normal_double  - cuRAND built-in Box-Muller
  uniform + normcdfinv  - cuRAND uniform + CUDA normcdfinv
  fpmp icdf(uint64)     - two curand() -> uint64 -> fpmp ICDF -> double
                          (fp32mp2 precision, ~46-bit mantissa)

Running benchmarks...

Method                    Time(ms)          Samples/sec          Mean        StdDev  Check
------------------------  ----------  ------------------  ------------  ------------  -----
curand_normal_double          x.xxx   x'xxx'xxx'xxx'xxx    x.xxxxe-xx  0.99xxxxxxxx  OK
uniform + normcdfinv          x.xxx   x'xxx'xxx'xxx'xxx    x.xxxxe-xx  0.99xxxxxxxx  OK
fpmp icdf(uint64)             x.xxx   x'xxx'xxx'xxx'xxx    x.xxxxe-xx  0.99xxxxxxxx  OK

Speedups vs curand_normal_double:
  ------------------------  ------
  uniform + normcdfinv       x.xxx
  fpmp icdf(uint64)          x.xxx

Result: ALL PASSED (tolerance: mean<x.xxe-xx, |stddev-1|<x.xxe-xx)
```

## Build targets

| Target | Description |
|--------|-------------|
| `all` | Build and run (default) |
| `build` | Build only |
| `run` | Run (builds if needed) |
| `rebuild` | Clean and build |
| `rerun` | Clean, build, and run |
| `clean` | Remove generated files |
| `help` | Show parameter reference |
