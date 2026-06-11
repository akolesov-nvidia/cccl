FPMP Unit Tests
================

This directory contains unit-test style programs for validating FPMP functionality on NVIDIA GPUs (CUDA) and CPUs (host).

Table of Contents
----------------
1. [Overview](#overview)
2. [Example Implementation](#example-implementation)
3. [Building Unit Tests](#building-unit-tests)
4. [Running Unit Tests](#running-unit-tests)

Overview
--------
These programs validate:
- Core arithmetic (+, -, *, /) and comparisons
- Conversions between FPMP types and scalar types
- Atomic operations (CUDA only)
- Compile-time constant tables and bit-level properties

Unit Programs
-------------

### `fpmp_api.cpp`
Basic API smoke-test: construction, operators, and a few representative operations.

### `fpmp_atomic_ff.cpp`
CUDA-only atomic operations for `fp32mp2` (`atomicAdd`, `atomicSub`) under contention.

### `fpmp_atomic_dd.cpp`
CUDA-only atomic operations for `fp64mp2` (`atomicAdd`, `atomicSub`) where supported by the target architecture.

### `fpmp_cast.cpp`
Type conversion and round-trip stability checks.

### `fpmp_int.cpp`
Integer conversion edge cases and boundary behavior.

### `fpmp_lut.cpp`
Compile-time lookup table generation and constant-data initialization utilities.

### `fpmp_copyable.cpp`
Trivial copyability check (`static_assert`) and volatile value round-trip verification.

### `fpmp_volatile.cpp`
Comprehensive volatile constructor/copy/assignment tests on both host and device, verifying value preservation through volatile round-trips for all FPMP type aliases.

### `fpmp_reduce.cpp`
CUDA-only compatibility test with `cooperative_groups::reduce`, confirming that trivially copyable FPMP types work with warp-level shuffle intrinsics.

### `fpmp_fp64_tool.cpp`
Unit tests for the FP64 precision emulation tool (`fp64_tool.hpp`).

### `fpmp_limits.cpp`
`cuda::std::numeric_limits<>` specialization checks for `fp32mp2`/`fp64mp2`: compile-time
(`static_assert`) validation of every reported characteristic and the exact `min/max/lowest/epsilon`
components, plus host/device runtime checks (`(1 + eps) != 1`, `infinity() > max()`, NaN inequality).

Building Unit Tests
------------------

### Build Options
| Option | Description | Default |
|--------|-------------|---------|
| `TARGET` | Target device (`device`, `host`) | `device` |
| `UNIT` | Unit(s) to build (e.g., `fpmp_api`, `fpmp_atomic_ff`, `fpmp_cast`, ...) | all *.cpp files |
| `LINKAGE` | Linkage mode (`inline`, `static`, `lto`) | `inline` |
| `OUT` | Output directory for results | `_out` |
| `VERBOSE` | Verbose mode (`0`=silent, `1`=minimal, `2`=full, `3`=very verbose); aliases: `V`, `VERB` | `0` |
| `ARCH` | CUDA compute capability (e.g., `86` for SM 8.6) | `86` |
| `EXTRA_FLAGS` | Extra compiler/linker flags | - |

### Build Commands
```bash
# Build all unit programs
make

# Build a single unit program
make UNIT=fpmp_cast

# Build for host
make TARGET=host

# Build with verbose output
make VERBOSE=2
```

Running Unit Tests
------------------

```bash
# Build and run all units (writes logs to OUT/*.log)
make run

# Run a single unit
make run UNIT=fpmp_api

# Run on host
make run TARGET=host

# Rebuild and run
make rerun UNIT=fpmp_atomic_ff
```
