# Double/fp32mp2 Conversion Throughput Benchmark

Measures the throughput of conversions between `double` and `fp32mp2`
(float-float) in both directions, comparing standard (FP64-based) and
optimized (integer-only) conversion paths:

### double → fp32mp2 (`FPMP_OPTIMIZED_DOUBLE_TO_FPMP`)

| Path | Implementation | FP64 ops |
|------|----------------|----------|
| Standard (`=0`, default) | `hi = (float)x; lo = (float)(x - (double)hi)` | 2 (cast + subtract) |
| Optimized (`=1`) | Integer bit manipulation (bit_cast, shifts, masks, `__clz`) | 0 |

### fp32mp2 → double (`FPMP_OPTIMIZED_FPMP_TO_DOUBLE`)

| Path | Implementation | FP64 ops |
|------|----------------|----------|
| Standard (`=0`, default) | `(double)hi + (double)lo` | 3 (2x F2D + DADD) |
| Optimized (`=1`) | Integer float-to-double promote + software double-add | 0 |

Standard paths use FP64 arithmetic, bottlenecked on GPUs with limited FP64
throughput (1:64 ratio on GeForce/L40).  Optimized paths avoid FP64 entirely,
using only INT32 and FP32 units.

All four kernel variants are compiled into the same binary via separate
compilation and compared in a single run.

## Method

Each direction has its own DAXPY-like kernel with `DEPTH` parallel conversion
chains per thread.  Each chain per iteration:

**double → fp32mp2 (d2ff):**

1. Converts a `double` value to `fp32mp2` (the operation under test)
2. Multiplies by a `fp32mp2` scalar (light FP32 arithmetic)
3. Accumulates into a `fp32mp2` sum (light FP32 arithmetic)
4. Feeds result bits back into the `double` input (dependency chain)
5. Evolves the `double` value via multiply + add (2 FP64 ops)

**fp32mp2 → double (ff2d):**

1. Converts an `fp32mp2` value to `double` (the operation under test)
2. Multiplies by a `double` scalar (light FP64 arithmetic)
3. Accumulates into a `double` sum (light FP64 arithmetic)
4. Feeds result bits back into the `fp32mp2` input (dependency chain)

Step 4 uses the `ts_update_bits` pattern (XOR LSB of result into input) to
create a true data dependency between consecutive iterations, preventing the
compiler from reordering or eliminating the conversion.

The `DEPTH` parameter controls register pressure: each chain keeps its own
`double val`, `fp32mp2 acc`, and `fp32mp2 scale` live across the loop.
At high `DEPTH` (e.g., 64), the kernel approaches the 255-register limit
and begins spilling to local memory.

## Quick start

```bash
# GPU (default: DEPTH=16)
make

# CPU
make T=host rerun

# Test register pressure
make DEPTH=64 rerun

# Show register usage and spills
make info

# High register pressure info
make info DEPTH=64
```

## Build parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET` | `device` | `device` (GPU) or `host` (CPU) |
| `ARCH` | `86` | CUDA SM architecture |
| `DEPTH` | `8` | Parallel conversion chains per thread (controls register pressure) |
| `REPS` | `4096` (GPU) / `1024` (CPU) | Loop iterations per chain |
| `UNROLL` | `8` | Loop unroll factor |
| `NUM_ITER` | `10` | Timing iterations for averaging |
| `THREADS` | `256` | CUDA threads per block |
| `NUM_BLOCKS` | `2048` | CUDA grid blocks |
| `LINKAGE` | `inline` | `inline`, `static`, or `lto` |
| `VERBOSE` | `1` | Verbosity level (`0`..`3`) |

## Files

| File | Description |
|------|-------------|
| `d2ff.cpp` | Host driver: runs all four variants, prints comparison tables |
| `d2ff_std.cpp` | Standard d2ff TU (`FPMP_OPTIMIZED_DOUBLE_TO_FPMP=0`) |
| `d2ff_opt.cpp` | Optimized d2ff TU (`FPMP_OPTIMIZED_DOUBLE_TO_FPMP=1`) |
| `d2ff_kernel.hpp` | d2ff kernel + wrapper, parameterized by macros |
| `ff2d_std.cpp` | Standard ff2d TU (`FPMP_OPTIMIZED_FPMP_TO_DOUBLE=0`) |
| `ff2d_opt.cpp` | Optimized ff2d TU (`FPMP_OPTIMIZED_FPMP_TO_DOUBLE=1`) |
| `ff2d_kernel.hpp` | ff2d kernel + wrapper, parameterized by macros |
| `d2ff_common.hpp` | Shared result struct, parameter defaults, extern declarations |
| `Makefile` | Build system |

## Example output

```
Double/fp32mp2 Conversion Benchmark
===================================
Target: GPU (NVIDIA L40)
Threads: 524288  REPS: 4096  DEPTH: 16  UNROLL: 8  Timing iterations: 10
Conversions per iteration: 34359738368

double -> fp32mp2:
  Conversion                              Time (ms)    Gconv/s  Speedup
  -------------------------------------- ---------- ---------- --------
  Standard (FP64 ops)                       xxx.xxx      xxx.x      ref
  Optimized (INT32 bit ops, no FP64)         xx.xxx      xxx.x    x.xxx
  Sanity: std=xxxx.xxxxxxxxxx  opt=xxxx.xxxxxxxxxx  OK (exact)

fp32mp2 -> double:
  Conversion                              Time (ms)    Gconv/s  Speedup
  -------------------------------------- ---------- ---------- --------
  Standard (FP64 ops)                       xxx.xxx      xxx.x      ref
  Optimized (INT32 bit ops, no FP64)         xx.xxx      xxx.x    x.xxx
  Sanity: std=xxxx.xxxxxxxxxx  opt=xxxx.xxxxxxxxxx  OK (exact)
```

## `make info` output

The `info` target recompiles all four kernels with `--resource-usage` and
reports register counts, spill bytes, object sizes, and compile times for
both directions (d2ff and ff2d).

## Results on NVIDIA L40S (double → fp32mp2)

GPU: NVIDIA L40 (Ada Lovelace, FP64:FP32 = 1:64), `sm_86`, CUDA 12.x.

### DEPTH=16 (default)

| Conversion | Time (ms) | Gconv/s | Speedup |
|---|---|---|---|
| Standard (FP64 cast) | 284.064 | 121.0 | ref |
| Optimized (INT32 bit ops) | 98.438 | 349.0 | **2.89x** |

Both paths produce identical output (`114688.4466643515`).

Kernel resource usage (`make info DEPTH=16`):

| Metric | Standard (FP64) | Optimized (INT32) |
|---|---|---|
| Registers | 92 | 110 (+18) |
| Spill stores | 0 | 0 |
| Spill loads | 0 | 0 |
| Object size | 100,216 bytes | 292,632 bytes (2.9x) |
| Compile time | 2,591 ms | 3,282 ms (1.3x) |

The optimized path uses 18 more registers for the integer bit manipulation
temporaries, but remains well below the 255-register limit.  No spills in
either path.

### DEPTH=64 (high register pressure)

| Conversion | Time (ms) | Gconv/s | Speedup |
|---|---|---|---|
| Standard (FP64 cast) | 1,150.592 | 119.5 | ref |
| Optimized (INT32 bit ops) | 779.562 | 176.3 | **1.48x** |

Both paths produce identical output (`1087905.1896390691`).

Kernel resource usage (`make info DEPTH=64`):

| Metric | Standard (FP64) | Optimized (INT32) |
|---|---|---|
| Registers | 255 (max) | 255 (max) |
| Stack frame | 624 bytes | 248 bytes (2.5x less) |
| Spill stores | 1,648 bytes | 716 bytes (2.3x less) |
| Spill loads | 1,556 bytes | 668 bytes (2.3x less) |
| Object size | 400,296 bytes | 589,160 bytes (1.5x) |
| Compile time | 3,997 ms | 5,325 ms (1.3x) |

Both kernels hit the 255-register max.  The optimized path produces
**2.3x fewer spills** despite generating more instructions (larger object).
The standard path's FP64 subtraction has very high latency on the L40's 1:64
FP64 pipeline, forcing the compiler to keep many values live while waiting
for results.  The optimized path's integer ops complete quickly, so
temporaries die fast and registers get recycled sooner.  The speedup drops
from 2.89x to 1.48x due to both paths spilling, but the optimized path
still wins.

### Summary

The optimized conversion wins on two fronts:
1. **Throughput**: 2.89x faster at DEPTH=16 (no spills), 1.48x at DEPTH=64
   (both spilling), by avoiding the slow FP64 pipeline
2. **Register pressure**: 2.3x fewer spills under high pressure (despite more
   instructions), because integer ops have shorter latency and free registers
   sooner

## Build targets

| Target | Description |
|--------|-------------|
| `all` | Build and run (default) |
| `build` | Build only |
| `run` | Run (builds if needed) |
| `rebuild` | Clean and build |
| `rerun` | Clean, build, and run |
| `info` | Show kernel register usage, spills, object sizes, compile times |
| `clean` | Remove generated files |
| `help` | Show parameter reference |

## Notes

- Requires C++20 (`-std=c++20`) for the double → fp32mp2 direction to
  enable the constructor path that respects `FPMP_OPTIMIZED_DOUBLE_TO_FPMP`.
  With C++17, the `fpmp2_t(double)` constructor always uses the
  cast-based path regardless of the flag.
- The fp32mp2 → double direction (`operator double()`) works with any C++
  standard since it calls `__nv_fpmp2_to_double` directly.
- On CPU (host), both paths produce similar throughput since x86 CPUs have
  full FP64 support.  The optimization targets GPUs with limited FP64
  throughput (1:64 FP64:FP32 ratio on consumer/workstation GPUs).
