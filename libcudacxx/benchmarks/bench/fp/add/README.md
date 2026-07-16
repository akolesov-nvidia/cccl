# FP64 Add Emulation — Accuracy & Performance Test

This test validates the correctness and performance of standalone integer-only FP64 addition emulation functions (`__internal_fp64_add_int32`, `__internal_fp64_add_fp64`, `__internal_int32_to_fp64`). Results are compared against native double-precision addition.

## Overview

The function under test adds an `int32_t` value to a `double` accumulator using **only integer operations** (no floating-point arithmetic). The algorithm is based on `__internal_fp64emu_mid_dadd` with HA accuracy (9 extra precision bits).

### Test Categories

| Category | Description |
|----------|-------------|
| BASIC | Simple cases: zero, one, negative, identity, small fractions |
| BOUNDARY | INT32_MIN/MAX, large/small doubles, near-cancellation, subnormals |
| RANDOM | 64K Gaussian-distributed doubles with uniform random int32 values |
| RANDOM_SMALL | 64K small-range doubles with small int32 values |
| ACCUMULATE | 16K repeated accumulations starting from zero |
| ACCUM_FRAC | 16K repeated accumulations starting from fractional value |

### Accuracy Metrics

Results are reported in **ULP** (Units in the Last Place) distance from native double addition:
- **Exact**: bit-identical result
- **<=1 ULP**: within 1 unit in the last place
- **<=2 ULP**: within 2 units in the last place (HA accuracy target)
- **Fail**: exceeds 2 ULP tolerance

## Usage

```bash
# Build and run (default: host/CPU)
make

# Run with more random tests
make RANDOM_LEN=1048576 run

# Run with different seed
make SEED=123 rerun

# Build for CUDA device
make TARGET=device ARCH=80 rebuild

# Clean up
make clean
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET` | host | Target platform (host/device) |
| `ARCH` | 86 | CUDA architecture (for device target) |
| `RANDOM_LEN` | 65536 | Number of random test pairs |
| `ACCUM_LEN` | 16384 | Number of accumulation steps |
| `SEED` | 42 | Random number generator seed |
| `OUT` | _out | Output directory |
| `VERBOSE` | 1 | Verbosity level (0-3) |

## Expected Output

```
================================================================================
  STANDALONE FP64 EMULATION — ACCURACY & PERFORMANCE TEST
  Integer-Only Double-Precision Addition Emulation
================================================================================

Configuration:
  Random test length:       65536
  Accumulation length:      16384
  RNG seed:                 42
  Reference:                native double-precision addition

================================================================================
  TEST RESULTS (vs native double addition)
================================================================================
  Category       |   Total |   Exact          |  <=1 ULP        |  <=2 ULP        | Fail | Max ULP | Max RelErr
  ---------------+---------+------------------+-----------------+-----------------+------+---------+-----------
  BASIC          |      22 |      22 (100.0%) |     22 (100.0%) |     22 (100.0%) |    0 |    0 |   0.00e+00
  BOUNDARY       |      27 |      27 (100.0%) |     27 (100.0%) |     27 (100.0%) |    0 |    0 |   0.00e+00
  RANDOM         |   65536 |   65000 ( 99.2%) |  65536 (100.0%) |  65536 (100.0%) |    0 |    1 |   1.00e-16
  ...
================================================================================

RESULT: PASS - all tests within 2 ULP of reference
```
