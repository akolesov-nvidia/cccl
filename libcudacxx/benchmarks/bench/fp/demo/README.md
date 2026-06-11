# Float-Float Arithmetic Benchmark (`demo/`)

Performance benchmark comparing **float-float (double-float) arithmetic** to native **double precision** on CPU and GPU.

## Quick Start

```bash
# Run on CPU
make -j TARGET=host run

# Run on GPU (requires CUDA)
make -j rerun

# Custom parameters
make -j TARGET=host NUM_ITER=20 rerun
```

## What's Inside

| File | Description |
|------|-------------|
| `demo.cpp` | Main benchmark source |
| `Makefile` | Build system |
| `README.md` | This file |

## Operations Tested

- **ADD**, **SUB**, **MUL**, **DIV**, **SQRT**, **RSQRT**, **FMA**
- **EXP**, **LOG**, **ERF**, **ERFC**, **SIN**, **COS**, **NORMCDFINV**
- **BOYS_F0** (Boys function F_0 — quantum chemistry special function)
- **ACC** (optimized single-component accumulation)
- Uses Gaussian random numbers
- Reports throughput (GFLOPS) and speedup ratios

## Sample Output

```
================================================================================
  BENCHMARK RESULTS
================================================================================
Operation    | FF Time(ms)| Dbl Time(ms)| FF GFLOPS | Dbl GFLOPS| Speedup
-------------+------------+-------------+-----------+-----------+----------
ADD          |      0.092 |       0.497 |   2915.95 |    540.06 |    +5.40x
SUB          |      0.092 |       0.497 |   2916.36 |    540.07 |    +5.40x
MUL          |      0.063 |       0.497 |   4230.05 |    540.09 |    +7.83x
DIV          |      0.120 |       3.567 |   2236.72 |     75.26 |   +29.72x
SQRT         |      0.343 |       3.722 |    781.59 |     72.12 |   +10.84x
FMA          |      0.140 |       0.497 |   3846.57 |   1080.13 |    +3.56x
================================================================================
```

## Configuration

```bash
TARGET=host              # CPU execution
TARGET=device            # GPU execution (default)
NUM_ITER=10              # Timing iterations (default)
ARCH=86                  # CUDA architecture (default: RTX 30xx/A100)
THREADS=256              # Threads per block (default)
UNROLL=64                # Unroll factor of internal loop
```

## Help

```bash
make help
```


