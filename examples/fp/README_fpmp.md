FPMP Examples
=============

Examples demonstrating **FPMP** — multi-precision floating-point arithmetic
using pairs of IEEE-754 floats — on NVIDIA GPUs and CPUs.

For build/run instructions and the wider examples landscape, see the
[examples README](README.md).

For Fortran usage of FPMP (host with `gfortran`, device with `nvfortran`,
including ready-to-use bindings under `include/fortran/`), see the
[examples/fortran README](fortran/README.md).

Available Examples
------------------

### fp32mp2.cpp — Float-Float Precision Demo

Demonstrates float-float (double-float) extended precision arithmetic:

- **Precision**: up to ~46 effective mantissa bits (between IEEE-754 float and double; accuracy dependent)
- **Memory**: 8 bytes (2 × float)
- **Use case**: More precision than float using FP32 hardware, especially where FP64 is scarce/slow

Features demonstrated:
- Construction and assignment from various types (double, float, int)
- Basic arithmetic: addition, subtraction, multiplication, division
- Advanced operations: sqrt, rsqrt, fma
- Fast arithmetic with `fp32mp2_low` and renormalization for dot products
- Comparison operators
- Compound assignment operators (+=, *=)
- Accuracy comparison with IEEE-754 double precision

### fp32mp2_math.cpp — Dedicated fp32mp2 Math Functions Demo

Minimal example exercising every general-purpose dedicated `fp32mp2` math function provided by `fpmp_math.h` — no accuracy checks, just a single call per function with a printed result.  Covers:

- `exp`, `log`, `pow` (Exponential / Logarithmic)
- `cbrt`, `rcbrt` (Power)
- `sin`, `cos`, `sincos` (Trigonometric)
- `tanh` (Hyperbolic)
- `erf`, `erfc` (Error functions)
- `boys_f0` (Special function — zeroth-order Boys, quantum chemistry)
- `normcdfinv` (Inverse normal CDF)
- `floor`, `ceil`, `round`, `trunc` (Rounding)
- `fabs`, `fmin`, `fmax`, `min`, `max` (Absolute / Min / Max)

All `fp32mp2` math implementations use pure float-float arithmetic — no fp64 ops — making them well suited to GPUs where fp64 throughput is limited.

`icdf` (integer uniform → Gaussian, also dedicated for `fp32mp2`) is a specialized Gaussian-sampling helper and is intentionally omitted from this overview example.

### fp64mp2.cpp — Double-Double Precision Demo

Demonstrates double-double (quad-like) extended precision arithmetic:

- **Precision**: up to ~104 effective mantissa bits (between IEEE-754 double and binary128; accuracy dependent)
- **Memory**: 16 bytes (2 × double)
- **Use case**: Scientific computing requiring very high precision

Features demonstrated:
- Construction and assignment from various types
- Basic arithmetic: addition, subtraction, multiplication, division
- Advanced operations: sqrt, rsqrt, fma
- High-precision computation demonstration
- Comparison with double precision reference

**Note**: `fp64mp2` is enabled by default. To disable it (build float-only), define `FPMP_FP64MP2_ENABLE=0` before including FPMP headers.

### fp64mp2_math.cpp — fp64mp2 Math Functions Demo

Minimal example exercising the same math-function API on `fp64mp2`.  Unlike `fp32mp2`, most `fp64mp2` math functions are **not** dedicated double-double implementations but delegate to higher-precision fallbacks (`__float128` via libquadmath when `FPMP_FP128_MATH_FALLBACK=1`, otherwise system fp64), so this example primarily exercises the public API.  Covers the same functions as `fp32mp2_math.cpp` except `icdf` (which is float-only).

### fp32mp2_thread.cpp / fp64mp2_thread.cpp — Thread Cooperation Primitives Demo

Two device-only examples demonstrating CUDA thread-cooperation intrinsics overloaded for `fp32mp2` and `fp64mp2`:

- **Atomics**: `atomicAdd`, `atomicSub` — CAS-based, used inside a grid-wide reduction kernel.  fp32mp2 uses 64-bit `atomicCAS` (any modern GPU); fp64mp2 uses 128-bit `atomicCAS` and requires **compute capability ≥ 9.0** (Hopper).
- **Warp shuffles**: `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync` — used to build a warp-level butterfly reduction (down-shuffle) and a lane-0 broadcast (sync-shuffle).  Available on sm_70+ (Volta).

Both examples print a clear skip message when built on the host (`TARGET=host`) or, in the fp64mp2 case, when run on a pre-Hopper GPU.

For exhaustive correctness tests, see `units/fpmp_atomic_ff.cpp`, `units/fpmp_atomic_dd.cpp`, and `units/fpmp_shfl.cpp`.
