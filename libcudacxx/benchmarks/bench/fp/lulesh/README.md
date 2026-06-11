# LULESH 2.0 Benchmark — Multi-Precision Floating-Point

LULESH (Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics) is a
proxy application for hydrodynamics codes, ported to CUDA with support for
multiple floating-point precision types via the cuda_multi_fp library.

More information about LULESH: https://codesign.llnl.gov/lulesh.php

## Supported precision types

| `DATATYPE` | Type | Precision | Exponent range |
|------------|------|-----------|----------------|
| `double` | `double` | ~15 digits | full IEEE-754 fp64 |
| `float` | `float` | ~7 digits | full IEEE-754 fp32 |
| `fpemu` | `__nv_fp64emu_t` | ~15 digits (emulated double) | full IEEE-754 fp64 |
| `fpmp` | `__nv_fpmp2_t<float>` | ~14 digits (float-float) | same as fp32 |

## Prerequisites

- NVIDIA CUDA Toolkit
- cuda_multi_fp library (parent project)

## Building and running

```bash
cd benchmarks/lulesh

# Build and run with default settings (fpemu, LTO, size=16)
make rerun

# Build a specific precision type
make DATATYPE=fpmp

# Run with custom problem size and iteration count
make run SIZE=20 ITERATIONS=100

# Build with verbose output showing compile times
make build V=2
```

## Comparing all precision types

The `compare` target builds and runs LULESH for all four precision types
(double, float, fpemu, fpmp) and prints a summary table with compile time,
binary size, runtime performance, and accuracy:

```bash
make compare SIZE=10 V=2
```

Example output:

```
==========================================
 Summary
==========================================
Type         Compile(s)     Binary       Run(s)     FOM(z/s)     MaxRelDiff
----------+------------+----------+------------+--------------+------------
double             23.4       624K         0.05    10230.179   1.784689e-14
float              23.3       620K         0.03    18728.696   4.164693e-05
fpemu              87.1       3.3M         0.21    2435.0215   7.181761e-14
fpmp               29.6       1.1M         0.08    6400.5735   4.475459e-12
==========================================
```

Additional options for `compare`:

```bash
# Compare with a larger problem
make compare SIZE=20

# Compare with static linkage
make compare LINKAGE=static

# Compare a specific accuracy
make compare ACCURACY=accurate
```

## Makefile parameters

| Parameter | Description | Default | Values |
|-----------|-------------|---------|--------|
| `DATATYPE` | Precision type | `fpemu` | `double`, `float`, `fpemu`, `fpmp` |
| `TARGET` | Target device | `device` | `device`, `host` |
| `ARCH` | CUDA SM architecture | `86` | `70`, `75`, `80`, `86`, `89`, `90` |
| `LINKAGE` | Library linkage mode | `lto` | `inline`, `static`, `lto` |
| `ACCURACY` | Computation accuracy | `def` | `def`, `accurate`, `fast` |
| `SIZE` | Problem size (NxNxN elements) | `16` | any integer |
| `ITERATIONS` | Number of time steps | `500` | any integer |
| `V` | Verbose level | `0` | `0` (silent), `1` (minimal), `2` (full) |
| `OUT` | Output directory | `_out` | custom path |
| `EXTRA_FLAGS` | Extra compiler/linker flags | | |
| `ABI` | Custom ABI (x,y) | | e.g. `ABI=5,5` |

## Build targets

| Target | Description |
|--------|-------------|
| `all` / `build` | Build the benchmark (prints compile time with `V=2`) |
| `run` | Build and run |
| `rerun` | Clean, rebuild, and run |
| `compare` | Build and run all 4 precision types, print summary table |
| `clean` | Remove generated files |
| `help` | Show available parameters and targets |

## File structure

- `lulesh.cu` — main computational kernels
- `fpemu_type.h` — precision type selection based on `DATATYPE`
- `allocator.cu/h` — pooled GPU memory allocator
- `util.h` — utility functions
- `sm_utils.inl` — warp shuffle helpers
- `vector.h` — thrust vector wrappers

## CI

This benchmark is excluded from the default `make ci` target (long compile
time). Run it explicitly with `make -C benchmarks/lulesh run` or
`make -C benchmarks/lulesh compare`.

## Known issues

See [docs/README_fpmp.md](../../docs/README_fpmp.md) for details on a float-float
overflow issue and the NaN-safe comparison fix applied to the LULESH code.

