# Black-Scholes Option Pricing Benchmark

Prices N random European options using the analytical Black-Scholes formula
and compares throughput and accuracy across up to eight floating-point data types:

| Type | Description | Mantissa |
|------|-------------|----------|
| `float` | IEEE-754 single precision | 23 bits |
| `fp32mp2_low` | Double-float, fast method | ~46 bits |
| `fp32mp2` | Double-float, default method | ~46 bits |
| `double` | IEEE-754 double precision | 53 bits |
| `fp64emu_mid` | Emulated fp64 via fp32 arithmetic (default) | 53 bits |
| `fp64emu` | Emulated fp64 via fp32 arithmetic (accurate) | 53 bits |
| `fp64mp2` | Double-double (pair of doubles) | ~104 bits |
| `fp128_t` | IEEE-754 quad precision | ~113 bits |

The `fp128_t` type is optional and requires `FP128=1` (plus `ARCH=100+` for GPU).

Every type goes through the same templated code path (`norm_cdf`, `d1`/`d2`,
discounting) so that timing differences reflect arithmetic cost alone.

## Method

The standard Black-Scholes formula for European options:

$$C = S\,e^{-qT}\,N(d_1) - K\,e^{-rT}\,N(d_2)$$

where $N(x) = \tfrac{1}{2}\mathrm{erfc}(-x/\sqrt{2})$ and:

$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

Each option has randomized parameters: S in [10, 200], K in [10, 200],
T in [0.05, 5.0], sigma in [0.05, 1.0], with fixed r=0.02, q=0.01.

### Reference and accuracy

By default (`FP128_REF=1`), accuracy is measured against a quad-precision
reference computed on the host using `libquadmath` on x86 (`erfcq`, `sqrtq`,
`logq`, `expq`) or 128-bit `long double` where that is the platform quad type.
Per-option relative errors are accumulated in quad precision to reduce
measurement noise.  When `FP128=1` is also enabled, benchmark partial-sum
reduction uses the same quad type.

Set `FP128_REF=0` to fall back to a double-precision reference.

### Device-side `erf(fp128_t)` and `erfc(fp128_t)`

CUDA does not provide `__nv_fp128_erf` or `__nv_fp128_erfc`.  On device,
`erf(fp128_t)` is
implemented via the non-alternating series (DLMF 7.6.2):

$$\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}}\,e^{-x^2}\sum_{n=0}^{N}\frac{2^n\,x^{2n+1}}{(2n+1)!!}$$

All terms are positive, avoiding catastrophic cancellation.  The series uses
early termination and delegates `exp(-x^2)` to `__nv_fp128_exp`.  The device
`erfc(fp128_t)` overload is provided so the benchmark can evaluate the
normal CDF in the cancellation-resistant `0.5*erfc(-x/sqrt(2))` form.

## Quick start

```bash
# GPU (default)
make

# CPU
make T=host

# Include fp128 type (GPU, Blackwell+)
make ARCH=100 FP128=1 rerun

# Include fp128 type (CPU)
make T=host FP128=1 rerun

# Higher rigor
make REPS=256 rerun
```

## Build parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET` | `device` | `device` (GPU) or `host` (CPU) |
| `ARCH` | `86` | CUDA SM architecture |
| `NUM_OPTIONS` | `1M` (GPU) / `32K` (CPU) | Options per launch |
| `REPS` | `128` (GPU) / `2` (CPU) | Inner repetitions per kernel (controls rigor) |
| `NUM_ITER` | `10` | Timing iterations for averaging |
| `THREADS` | `256` | CUDA threads per block |
| `NUM_BLOCKS` | `2048` | CUDA grid blocks |
| `LINKAGE` | `inline` | `inline`, `static`, or `lto` |
| `VERBOSE` | `1` | Verbosity level (`0`..`3`) |
| `FP128` | `0` | Include `fp128_t` as benchmarked type (`0` or `1`, requires `ARCH=100+` for GPU) |
| `FP128_REF` | `1` | Use quad-precision reference prices (`0` for double reference) |

## Files

| File | Description |
|------|-------------|
| `bs.cpp` | Main benchmark source (templated BS formula, kernels, output) |
| `bs_ref.cpp` | Quad-precision reference price and accuracy (compiled by g++) |
| `bs_fp128_math.hpp` | `fp128_t` math overloads for device (custom `erf`/`erfc`) and host (`libquadmath` where available) |
| `Makefile` | Build system |

## Example output

```
Black-Scholes Option Pricing Benchmark
======================================
Target: CPU (host)
Types:  float(4) fp32mp2_fast(8) fp32mp2(8) double(8) fp64emu(8) fp64emu_acc(8) fp64mp2(16) fp128(16)

Options: 32768  REPS: 2  r=0.0200  q=0.0100  Call
Threads: 524288  Timing iterations: 10

RNG seed: 123456789

Reference mean price (float128): 44.9363010160

Accuracy vs float128 reference:

Type                        Mean Price     Max|RelErr|     Avg|RelErr|       MAPE(%)  Quality                 Status
------------------------  ------------  --------------  --------------  ------------  ----------------------  ------
float                        44.936301  7.97033934e-04  1.52227620e-06      1.52e-04  good                    PASS
fp32mp2_low               44.936301  1.03800794e-10  1.41998462e-13      1.42e-11  excellent               PASS
fp32mp2                    44.936301  6.79704077e-11  1.06318345e-13      1.06e-11  excellent               PASS
double                       44.936301  1.53239258e-12  2.89141060e-15      2.89e-13  near machine eps        PASS
fp64emu_mid                    44.936301  1.79018517e-12  4.08490699e-15      4.08e-13  near machine eps        PASS
fp64emu           44.936301  1.51375272e-12  2.89717834e-15      2.90e-13  near machine eps        PASS
fp64mp2                    44.936301  1.00508079e-12  2.31989305e-15      2.32e-13  near machine eps        PASS
fp128_t                      44.936301  3.81024633e-15  1.85916099e-16      1.86e-14  near machine eps        PASS

MAPE (Mean Absolute Percentage Error) quality scale:
  < 1e-11  near machine eps      at the limit of floating-point representation
  < 1e-7   excellent             well-optimized double-precision algorithm
  < 1e-3   good                  typical double-precision math library
  < 1e-1   rough approximation   reduced precision or single-precision path
  >= 1e-1  very inaccurate       significant precision loss

Performance:

Type                        Time(ms)         Options/sec
------------------------  ----------  ------------------
float                          3.155          20'772'016
fp32mp2_low                24.186           2'709'717
fp32mp2                     26.324           2'489'624
double                         3.524          18'598'759
fp64emu_mid                     17.260           3'796'942
fp64emu            24.827           2'639'653
fp64mp2                    105.659             620'256
fp128_t                      252.276             259'778

Speedups vs double:
  ------------------------  ------
  float                      1.12x
  fp32mp2_low             0.15x
  fp32mp2                  0.13x
  fp64emu_mid                  0.20x
  fp64emu         0.14x
  fp64mp2                  0.03x
  fp128_t                    0.01x

Result: COMPLETED
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
