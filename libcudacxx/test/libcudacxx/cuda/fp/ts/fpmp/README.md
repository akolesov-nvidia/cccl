FPMP Test Suite (`ts/fpmp/`) — Work in Progress
================================================

This directory contains an internal test suite for evaluating FPMP operations on GPU (CUDA) and CPU (host).
It is **work in progress**: parameter names, output formats, and coverage may change.

Run from the repository root with `make tsmp`, or from this directory with `make run` / `make rerun`.

Overview
--------

The suite builds test executables for combinations of:
- **Function** (`FUNC`): `add`, `sub`, `mul`, `div`, `sqrt`, `rsqrt`, `exp`, `fma`, …
- **Type** (`TYPE`): `fp32mp2`, `fp64mp2`
- **Accuracy** (`ACCURACY`): `def`, `low`, `high`

Each executable reports:
- **Performance**: throughput and latency (see `ts_performance.hpp`)
- **Accuracy**: relative error and correct-bits estimates over multiple input-generation modes (see `ts_accuracy.hpp`)

Key files
---------

- `ts.cpp`: main driver (prints performance + accuracy summaries, CSV logging)
- `ts.hpp`: shared macros/config plumbing
- `ts_types.hpp`: type definitions and configuration
- `ts_functions.hpp`: function tags / dispatch helpers
- `ts_performance.hpp`: throughput/latency measurement helpers
- `ts_accuracy.hpp`: accuracy measurement and error logging
- `ts_dataset.hpp`: input generation strategies (work, normal, special, pattern)
- `ts_print.hpp`: printing and formatting utilities
- `ts_utils.hpp`: CUDA utilities and type traits

Build and run
-------------

From this directory:

```bash
# Build + run default set (device)
make run

# Clean + rebuild + run
make rerun

# Host mode (no nvcc)
make TARGET=host rerun

# Narrow selection
make FUNC="add mul" TYPE=fp32mp2 ACCURACY=def run
```

Makefile parameters
-------------------

Core selectors:

- `SET`: `def` (default) or `all`
- `FUNC`: space-separated list, or `def` / `all`
- `TYPE`: space-separated list, or `def` / `all` (values: `fp32mp2`, `fp64mp2`)
- `ACCURACY`: space-separated list, or `def` / `all` (values: `def`, `low`, `high`)

Common build options:

- `TARGET`: `device` (default) or `host`
- `ARCH`: CUDA architecture (default: `86`)
- `LINKAGE`: `inline` (default), `static`, `lto`
- `OUT`: output directory (default: `_out`)
- `VERBOSE`: `0..3` (aliases: `V`, `VERB`)
- `EXTRA_FLAGS`: extra compiler flags (often inherited/exported from the root Makefile)

Test knobs:

- `RIGOR`: accuracy rigor level, log2 of samples (`TS_ACCURACY_RIGOR`); e.g., `24`=16M, `32`=4G samples
- `RUN_ACCURACY`: enable accuracy tests (`y`=enable (default), `n`=disable for timing-only runs)
- `CLASSIFY`: enable accuracy classification (`y`=enable (default), `n`=disable for faster host runs)
- `TIMING`: enable timing tests (`y`=enable (default), `n`=disable for accuracy-only runs)
- `PRINT`: what to print (`ok`, `fail`, `warn`, `all`, `none`)
- `DATASET`: datasets to run (`work`, `normal`, `special`, `pattern`)
- `A1`, `A2`, `A3`, `A4`: fixed input arguments for debugging (enables `__FIXED_INPUTS__` mode: 1 sample, no special table)
- `CONSOLE`: console stream (`stdout`, `stderr`, `file`, `null`)

Output
------

- Console output shows accuracy tables (relative error, correct bits) and performance tables (GFLOPS, ev/clk/SM, clk/eval)
- When warnings or errors occur, a per-class accuracy classification summary is printed
- CSV log file is generated in `OUT` directory (default: `_out/YYYY-MM-DD.csv`) with per-test metrics for further analysis

Accuracy Classification
-----------------------

When accuracy issues (warnings or errors) are detected, the test suite classifies them into categories:

| Class | Description |
|-------|-------------|
| `normal` | Within warning threshold |
| `output_special` | Output is INF or NaN |
| `input_special` | Input is INF or NaN |
| `output_denormal` | Output is denormal |
| `input_denormal` | Input is denormal |
| `output_near_denormal` | Output is in smallest normal binade |
| `input_near_denormal` | Input is in smallest normal binade |
| `output_near_inf` | Output is in largest normal binade |
| `input_near_inf` | Input is in largest normal binade |
| `cancellation`        | Result much smaller than inputs (cancellation) |
| `error` | Exceeds error threshold without mitigating factors |

Example output:
```
================================================================================
Accuracy Classification Summary: mul <def> [pattern]
================================================================================
Class                |        Count |   Max RelErr |   Avg RelErr |  Bits
---------------------+--------------+--------------+--------------+------
normal               |   3023131864 |     1.00e-13 |     2.13e-15 |    43
output_denormal      |      3569762 |     2.00e-01 |     8.73e-07 |     2
cancellation         |    151600806 |     8.84e-08 |     1.98e-09 |    23
---------------------+--------------+--------------+--------------+------
Warning classes with issues: 2
Error class count:          0
================================================================================
```

Notes (WIP)
-----------

- The accuracy dataset modes and rigor are controlled in `ts_dataset.hpp` (see `TS_ACCURACY_*` macros).
- For `fp32mp2` on GPU, the pattern dataset automatically uses 32-bit rigor for exhaustive 32-bit integer coverage (unless `RIGOR >= 32` is explicitly set).
- Use `RUN_ACCURACY=n TIMING=y` for timing-only benchmarks, or `RUN_ACCURACY=y TIMING=n` for accuracy-only tests.