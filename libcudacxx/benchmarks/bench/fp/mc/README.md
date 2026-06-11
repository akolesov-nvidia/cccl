# Monte Carlo European Option Pricing Benchmark

Compares pricing accuracy and performance of a Monte Carlo European option
pricer across multiple arithmetic types:

| Type | Description | Mantissa |
|------|-------------|----------|
| `float` | IEEE-754 single precision | 24 bits |
| `fp32mp2` | Double-float (pair of floats) | ~46 bits |
| `double` | IEEE-754 double precision | 53 bits |
| `fp64emu_mid` | Emulated fp64 via fp32 arithmetic (default) | 53 bits |
| `fp64emu` | Emulated fp64 via fp32 arithmetic (accurate) | 53 bits |
| `fp64emu_unpacked_mid` | Emulated fp64 via fp32 arithmetic (unpacked) | 53 bits |
| `fp64mp2` | Double-double (pair of doubles) | ~104 bits |

`fp64emu_unpacked_mid` is optional and controlled by the `MC_FPEMU_UNPACKED`
compile-time flag (default: off).

Every type is benchmarked in a single run with two RNG modes:

- **pregen** — pre-generated normal variates (shared across types), isolating
  pure arithmetic cost.
- **inline** — normals generated inside the kernel (cuRAND Philox on GPU,
  `std::mt19937_64` on CPU), measuring arithmetic + RNG cost together.
- **inline + icdf** — (GPU only) uses FPMP `icdf(uint64_t)` instead of cuRAND
  Box-Muller.  Two `curand()` calls are combined into a 64-bit uniform integer
  to match the entropy that `curand_normal_double` uses.  The icdf produces
  fp32mp2 precision (~46-bit mantissa); the `double` variant casts from fp32mp2.
  Shown for `fp32mp2` and `double` only.

The output is organized into four sections:

1. **Performance: pregen** — MC price, time, paths/sec, speedup vs `double`
   using pre-generated normals (isolates arithmetic cost).
2. **Performance: inline RNG** — same metrics with inline RNG (arithmetic + RNG cost).
3. **Accuracy: inline RNG** — `|error vs BS analytical|` across four parameter
   sets (ATM, OTM, deep OTM, extreme) showing how MC accuracy varies with
   option moneyness and volatility.
4. **Arithmetic precision** — pregen MC prices compared against `double` as
   reference, isolating pure floating-point arithmetic error from MC statistical noise.

## Method

Each Monte Carlo path simulates a Geometric Brownian Motion terminal price:

$$S_T = S_0 \cdot \exp\!\bigl((r - q - \tfrac{1}{2}\sigma^2)\,T + \sigma\sqrt{T}\,Z\bigr)$$

where \(Z \sim \mathcal{N}(0,1)\).  The discounted payoff
\(\,e^{-rT}\max(S_T - K, 0)\,\) (call) or \(\,e^{-rT}\max(K - S_T, 0)\,\)
(put) is accumulated across all paths and averaged to produce the MC price.

**Antithetic variates** use both \(Z\) and \(-Z\) per draw, halving variance
at the cost of two `exp` evaluations per draw.

Results are validated against the **Black-Scholes analytical price**, computed
by default in `__float128` (quad precision) via `libquadmath` for maximum
reference accuracy (~34 decimal digits).  See `mc_host.cpp`.  Disable with
`FP128_REF=0` to fall back to a `double`-precision reference.

## Quick start

```bash
# GPU (default)
make

# CPU
make T=host
```

## Build parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET` | `device` | `device` (GPU) or `host` (CPU) |
| `ARCH` | `86` | CUDA SM architecture |
| `NUM_PATHS` | `16M` (GPU) / `32K` (CPU) | Total Monte Carlo paths |
| `REPS` | `128` (GPU) / `16` (CPU) | Inner repetitions per kernel (controls rigor) |
| `NUM_ITER` | `10` | Timing iterations for averaging |
| `THREADS` | `256` | CUDA threads per block |
| `NUM_BLOCKS` | `2048` | CUDA grid blocks |
| `STRIKE` | `100.0` | Strike price for performance tables |
| `SIGMA` | `0.2` | Volatility for performance tables |
| `ICDF` | `0` | Gaussian method for inline RNG: `0`=Box-Muller, `1`=ICDF |
| `FP128_REF` | `1` | Use `__float128` BS analytical reference (`1`=on, `0`=off) |
| `LINKAGE` | `inline` | `inline`, `static`, or `lto` |
| `VERBOSE` | `1` | Verbosity level (`0`..`3`) |

### Controlling benchmark rigor

The `REPS` parameter is the main knob for measurement stability.  Each kernel
launch repeats the full path computation `REPS` times, multiplying GPU/CPU
work without additional memory.  The MC price is unchanged (divided by
effective path count `NUM_PATHS * REPS`).

```bash
make REPS=64              # lighter (faster)
make REPS=256             # heavier (more stable timing)
make REPS=512             # production-grade
```

### Gaussian method (`ICDF`)

The `ICDF` parameter selects how uniform variates are transformed to Gaussian
in the inline RNG path:

| `ICDF` | CUDA device function | Formula |
|--------|---------------------|---------|
| `0` | `curand_normal_double` | Box-Muller (cuRAND built-in) |
| `1` | `normcdfinv` | \(\Phi^{-1}(u)\) |
| `2` | `erfinv` | \(\sqrt{2}\,\mathrm{erfinv}(2u-1)\) |
| `3` | `erfcinv` | \(-\sqrt{2}\,\mathrm{erfcinv}(2u)\) |
| `4` | `fast_erfinv` | \(\sqrt{2}\,\mathrm{erfinv}(2u-1)\) — branchless, GPU only |

`ICDF=4` uses a **branchless** custom `fast_erfinv` that always evaluates both
the central and tail polynomials (Mike Giles coefficients) and selects the
result with a predicated move instead of a branch.  This eliminates warp
divergence at the cost of extra arithmetic per thread.  Compare with `ICDF=2`
(standard `erfinv` with branches) to measure the divergence overhead.

On host, all ICDF modes use Acklam's rational approximation of \(\Phi^{-1}\).

```bash
make ICDF=0 rerun   # Box-Muller (default)
make ICDF=1 rerun   # normcdfinv
make ICDF=2 rerun   # erfinv (branching)
make ICDF=3 rerun   # erfcinv
make ICDF=4 rerun   # fast_erfinv (branchless)
```

## Example output

```
Monte Carlo European Option Pricing Benchmark
==============================================
Target: GPU (NVIDIA L40)

Option: S0=100.00 K=100.00 r=0.0500 q=0.0000 sigma=0.2000 T=1.0000 Call
Paths: 16777216  Antithetic: yes  REPS: 128
RNG (pregen): std::mt19937_64 normal distribution
RNG (inline): cuRAND Philox4x32 Box-Muller
Threads: 524288  Timing iterations: 10

Param sets (S0=100, r=0.0500, q=0.0000, T=1.0, Call):
  ATM          K=100    sigma=0.2   BS=10.4505835722
  OTM          K=130    sigma=0.2   BS=1.6395929156
  deep OTM     K=200    sigma=0.4   BS=1.2402554599
  extreme      K=500    sigma=1.0   BS=5.0879707325

Black-Scholes analytical: 10.4505835722 (float128 reference)

Running 16777216 paths (8388608 draws) x 128 reps per type...

=== Performance: pregen (K=100, sigma=0.2) ===
Pre-generated normals, measures arithmetic cost only.

Type                            MC Price    Time(ms)           Paths/sec  Speedup vs double
------------------------  --------------  ----------  ------------------  --------
float                      10.4538237159       1.588   1'352'406'734'458   47.44x
fp32mp2                  10.4538232077      16.693     128'645'940'615    4.51x
double                     10.4538232077      75.336      28'505'404'691    1.00x
fp64emu_mid                  10.4538232077      56.329      38'124'267'814    1.34x
fp64emu         10.4538232077      56.448      38'043'760'121    1.33x
fp64mp2                  10.4538232077     249.438       8'609'304'686    0.30x

=== Performance: inline RNG (K=100, sigma=0.2) ===
RNG inside kernel, measures arithmetic + RNG cost.
(icdf) = fpmp icdf(uint64) instead of Box-Muller.

Type                            MC Price    Time(ms)           Paths/sec  Speedup vs double
------------------------  --------------  ----------  ------------------  --------
float                      10.4506602809       2.881     745'424'603'954   48.42x
fp32mp2                  10.4505654614      72.055      29'803'214'034    1.94x
fp32mp2 (icdf)           10.4509709201      51.094      42'030'022'513    2.73x
double                     10.4505654614     139.483      15'395'988'337    1.00x
double (icdf)              10.4509709201      80.896      26'546'251'161    1.72x
fp64emu_mid                  10.4505654614     120.702      17'791'644'613    1.16x
fp64emu         10.4505654614     120.743      17'785'627'868    1.16x
fp64mp2                  10.4505654614     313.908       6'841'122'005    0.44x

=== Accuracy: inline RNG (|error vs float128 BS|) ===
Each type uses its own RNG sequence.

  Type                               ATM           OTM      deep OTM       extreme
  ------------------------  ------------  ------------  ------------  ------------
  float                         7.67e-05      1.53e-04      9.57e-05      2.72e-03
  fp32mp2                     1.81e-05      6.74e-05      4.34e-06      1.14e-04
  fp32mp2 (icdf)              3.87e-04      2.26e-04      3.33e-05      5.12e-04
  double                        1.81e-05      6.74e-05      4.34e-06      1.14e-04
  double (icdf)                 3.87e-04      2.26e-04      3.33e-05      5.12e-04
  fp64emu_mid                     1.81e-05      6.74e-05      4.34e-06      1.14e-04
  fp64emu            1.81e-05      6.74e-05      4.34e-06      1.14e-04
  fp64mp2                     1.81e-05      6.74e-05      4.34e-06      1.14e-04

=== Arithmetic precision (pregen, vs double) ===
Same normals, same payoff — isolates pure arithmetic error.

  float                     MC Price    10.4538237159   |rel err vs double| = 4.86e-08
  fp32mp2                 MC Price    10.4538232077   |rel err vs double| = 8.50e-16
  fp64emu_mid                 MC Price    10.4538232077   |rel err vs double| = 9.28e-14
  fp64emu        MC Price    10.4538232077   |rel err vs double| = 0.00e+00
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
