# PCM Off-Diagonal Matrix-Vector Product Benchmark

Benchmarks the `accumulate_A_offdiagonal` kernel from Polarizable Continuum
Model (PCM) quantum chemistry code, comparing native `double` against
`fp32mp2` (double-float) arithmetic on NVIDIA GPUs.

| Type | Description | Mantissa |
|------|-------------|----------|
| `double` | IEEE-754 double precision | 53 bits |
| `fp32mp2` | Double-float (pair of floats) | ~46 bits |

## Background

The Polarizable Continuum Model discretises a molecular cavity surface into
`npoint` points.  The off-diagonal part of the cavity matrix **A** is applied
to a charge vector **q** as:

$$\text{lhs}[i] \mathrel{+}= \sum_{j \neq i} A(i,j)\;\text{rhs}[j]$$

where each matrix element is

$$A(i,j) = \frac{\operatorname{erf}\!\bigl(\zeta_{ij}\,r_{ij}\bigr)}{r_{ij}},
\qquad
\zeta_{ij} = \frac{\zeta_i\,\zeta_j}{\sqrt{\zeta_i^2 + \zeta_j^2}}$$

The kernel is \(O(n^2)\) and dominates the PCM self-consistent field (PCG)
iteration.  It is compute-bound, making it a good target for reduced-precision
arithmetic.

### Kernel internals

Per point-pair the kernel performs:

- 3D distance computation (\(r^2 = \Delta x^2 + \Delta y^2 + \Delta z^2\))
- Combined zeta via `rsqrt`
- `erf(zeta * r)` evaluation
- Multiply-accumulate into the local sum

A type-dispatched `fast_rsqrt` is used: hardware approximate + Newton-Raphson
refinement for `double`, and the fpmp library `rsqrt` for `fp32mp2`.

## Quick start

```bash
# GPU (default, npoint=10000)
make

# Sweep problem sizes
make NPOINT=8000 rerun
make NPOINT=20000 rerun
make NPOINT=30000 rerun
```

This benchmark is GPU-only.  Building with `TARGET=host` produces an
executable that prints a warning and exits.

## Build parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ARCH` | `86` | CUDA SM architecture |
| `NPOINT` | `10000` | Number of cavity surface points (8 000–30 000 typical) |
| `THREADS` | `256` | CUDA threads per block |
| `NUM_BLOCKS` | `512` | CUDA grid blocks |
| `NUM_ITER` | `10` | Timing iterations for averaging |
| `LINKAGE` | `inline` | `inline`, `static`, or `lto` |
| `FMAD_FLAG` | `true` | `-fmad=` flag passed to nvcc |
| `VERBOSE` | `1` | Verbosity level (`0`..`3`) |

The problem size can also be overridden at runtime:

```bash
./pcm.exe [npoint] [threadsPerBlock] [numBlocks]
```

## Input data

All inputs are randomly generated on the host with a fixed seed (`42`) for
reproducibility:

| Array | Shape | Range | Physical meaning |
|-------|-------|-------|------------------|
| `xyzz` | `npoint × 4` | xyz ∈ \[−10, 10\], ζ ∈ \[0.5, 5.0\] | Point coordinates (Bohr) + Gaussian exponent |
| `rhs` | `npoint` | \[−1, 1\] | Charge-like vector |

Data is generated in `double` and converted to the target type (`T`) via
`static_cast` before upload to the GPU.

## Output

The benchmark prints a performance table and an accuracy check:

```
PCM Off-Diagonal Matrix-Vector Product Benchmark
=================================================
GPU: NVIDIA L40

npoint         = 10000
threadsPerBlock= 256
numBlocks      = 512
timing iters   = 10
pairs          = 99990000

Running double precision...
Running fp32mp2 precision...

=== Performance ===

Type                Time(ms)       GFLOP/s   Speedup          Checksum
----------------  ----------  ------------  --------  ----------------
double               123.456         16.19     1.00x    1.234568e+03
fp32mp2             45.678         43.78     2.70x    1.234567e+03

=== Accuracy ===

  Checksum (double)  = 1.2345678901e+03
  Checksum (fp32mp2) = 1.2345671234e+03
  |rel error|        = 6.21e-07
  Status             = OK
```

### Metrics

- **Time (ms)** — average kernel execution time over `NUM_ITER` launches.
- **GFLOP/s** — estimated from ~20 FLOPs per point-pair × `npoint × (npoint − 1)` pairs.
- **Speedup** — ratio of `double` time to `fp32mp2` time.
- **Checksum** — sum of all `lhs` elements (cast to `double`), used to verify
  both types compute the same result.
- **|rel error|** — relative difference between the two checksums.  Values
  below \(10^{-6}\) are reported as OK.

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

## File structure

```
benchmarks/pcm/
├── pcm.cu        # Benchmark source (kernel + harness)
├── Makefile       # Build configuration
└── README.md      # This file
```

## Dependencies

- **CUDA Toolkit** ≥ 13.0 (provides `double4_32a` in `vector_types.h`)
- **fpmp headers**: `fpmp.h`, `fpmp_math.h` (for `fp32mp2`, `erf`, `rsqrt`)
