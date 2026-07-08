# Composite-`exp()` error & GPU-throughput study

This benchmark studies how a "long composition of `exp()`" workload
behaves under three working types — `float`, `double`, and the
double-float `fpmp2<float, *>` family — and across
two semantically equivalent rewrites of the same expression.

The questions the benchmark answers:

1. How does numerical error propagate as chain length `N` grows? (alpha
   sweep with a `__float128` reference oracle.)
2. How fast is each variant on the **intended deployment hardware**?
   `fp32mp2` is a software-emulated double-float built on FP32, so its
   value proposition is on **FP64-throttled GPUs** (Ada / L40S / RTX-class,
   FP64 = 1/64 FP32). On full-FP64 datacenter GPUs (A100 / H100, FP64 =
   1/2 FP32) `cuda_multi_fp` is not the right tool and any timings on
   them would be misleading marketing — they are intentionally not in
   this report.

## Quick start

This directory is self-contained: it depends only on `nvcc`, a recent
`g++` (with `libquadmath` for the `__float128` reference oracle), and
the cuda_multi_fp headers. cuda_multi_fp is **header-only** for the
templates used here (`fpmp.h`, `fpmp_math.h`); nothing needs to be
linked from it.

Point the build at your cuda_multi_fp checkout and go:

```bash
make CUDA_MULTI_FP_DIR=/path/to/cuda_multi_fp rerun                  # GPU
make CUDA_MULTI_FP_DIR=/path/to/cuda_multi_fp TARGET=host rerun      # CPU
make CUDA_MULTI_FP_DIR=/path/to/cuda_multi_fp NUM_OPTIONS=200000 CHAIN_LEN=2048 rerun
make help                                                             # all knobs
```

Two equivalent ways to point at the headers:

* `CUDA_MULTI_FP_DIR=<root of cuda_multi_fp>` — preferred, appends `/include`
  automatically.
* `FPMP_INC=<path that already contains fpmp.h>` — lower-level direct
  override.

When both are unset the Makefile falls back to `../../include` relative to
its own location (the layout when the benchmark sits inside a cuda_multi_fp
checkout). If the resolved path doesn't actually contain `fpmp.h` the
build aborts immediately with a one-line hint listing the override knobs.

Output goes to `_out/`. Running the executable prints three sections in
this order:

* **Per-variant alpha sweep** (per `(dtype, kernel)`: meanRelErr at
  multiple chain depths plus the fitted log-log slope `alpha`).
* **Accuracy table** (8 rows, all stats vs the `__float128` reference:
  `maxAbsErr`, `meanAbsErr`, `rmsAbsErr`, `maxRelErr`, `meanRelErr`).
* **Performance / speedup table** (8 rows; per-kernel `vs_fp64`
  speedup vs native FP64 on the same kernel, plus the `alpha` and
  propagation conclusion).

## What is being computed

For an input matrix `xs` of shape `numSamples × chainLen` we evaluate
two semantically identical functions per chain:

| kernel    | formula                            | per-step cost                                         |
| --------- | ---------------------------------- | ----------------------------------------------------- |
| `compose` | `prod_{i=0..N-1} exp(xs[i])`       | 1 `exp()` call + 1 multiply, on a serial chain in `p` |
| `expsum`  | `exp(sum_{i=0..N-1} xs[i])`        | 1 add per step, then a single `exp()` at the end     |

The default workload is `numSamples = 10000`, `chainLen = 10000`,
`xs[i]` IID-uniform in `[-1e-2, -1e-4]` (all-negative — so the
`expsum` partial sum monotonically decreases and there is no
catastrophic-cancellation in the summation).

The reference oracle is an inlined `__float128` (`libquadmath`)
implementation of `expsum` (compiled with `g++` because `nvc++`
doesn't accept the GCC-only quad-precision flags).

## The three working types

| `WorkingType` | concrete C++ type                          | mantissa  | notes                                       |
| ------------- | ------------------------------------------ | --------- | ------------------------------------------- |
| `Float`       | `float`                                    | ~24 bits  | reference for the FP32:FP64 throughput ratio |
| `Double`      | `double`                                   | ~53 bits  | native baseline                             |
| `Fpmp2`       | `fpmp2<float, fpmp2_accuracy::def>`      | ~46 bits  | 2-Sum-style add with low-part renormalize   |

`fp32mp2` represents a value as `(hi, lo)` where `hi` is the FP32
approximation and `lo` is the unnormalized residual. The `add` is a
2-Sum-style step that leaves `(hi, lo)` canonicalised — preserving the
unbiased per-step rounding noise that makes the random-walk regime
work (see "Theory" below).

## Theory: alpha-exponent regimes

For a chain of length `N` with per-step rounding noise `ε`, the
relative error of the final result grows as `~ ε · N^α`. Three
regimes show up empirically and each has a clean physical
interpretation:

| α       | regime                | mechanism                                                                                                |
| ------- | --------------------- | -------------------------------------------------------------------------------------------------------- |
| ~0.5    | `~O(√N)` random-walk  | per-step rounding errors are independent and unbiased (random sign); CLT gives `√N · ε`                   |
| ~1.0    | linear in `N`         | per-step rounding errors are biased / coherent; they all push the answer the same way                     |
| ~1.5    | super-linear          | per-step *absolute* error grows along the chain (e.g. summation magnitude grows linearly), so later steps are noisier than earlier ones |
| ~2.0    | quadratic-ish         | both the bias and the per-step magnitude grow with `N`                                                   |

A single ratio sums it up: at `N = 10000` going from `α = 0.5` to
`α = 1.5` is a `100×` accuracy hit; at `N = 1e6` it is `1000×`.

### Why `compose` lands in the random-walk regime

Each step of `compose` is `p ← p · exp(xs[i])`. The relative error of
one factor is `~1 ulp` with random sign. The product's relative
error is the sum of `N` independent unbiased `~ulp`-scale terms, so
the central limit theorem gives `O(√N · ulp)`. Each step contributes
in the *correct currency* (relative) and is unbiased.

### Why `expsum` lands in the super-linear regime

Each step of `expsum` is `s ← s + xs[i]`. The summand has
*absolute* error `~|s|·ε`, but `|s|` grows linearly with `N` because
all `xs[i]` carry the same sign on the default inputs. So later steps
contribute progressively *larger* absolute error to `s`, giving
`abs_err(s) ~ ε · N^1.5`. The trailing `exp(s)` then transmits the
absolute error in `s` directly to the relative error of the result
(`d(exp(s))/exp(s) = ds`), so the final relative error inherits the
`α ≈ 1.5` slope.

## Empirical accuracy (alpha sweep)

The alpha sweep is *almost* GPU-independent — error propagation is
a function of the working type, the kernel, and the per-step accuracy
of the math library. For `double` and `fp32mp2` the host (`std::exp`)
and device (`exp`) paths agree on alpha to within fit noise; for
`float` they don't, see the note after the table. Numbers below are
the GPU values from the default config (`numSamples = 10000`,
`chainLen = 10000`, `sweepNumDepths = 20`, all-negative inputs):

| variant                 | maxRelErr at N = 10000 | meanRelErr at N = 10000 | fitted α | conclusion   |
| ----------------------- | ---------------------- | ----------------------- | -------- | ------------ |
| `float` / `compose`     | 2.16e-05               | 7.43e-06                | **0.79** | linear       |
| `float` / `expsum`      | 7.74e-05               | 1.49e-05                | **1.47** | super-linear |
| `double` / `compose`    | 2.24e-14               | 4.56e-15                | **0.51** | sqrt(N)      |
| `double` / `expsum`     | 1.29e-13               | 2.78e-14                | **1.48** | super-linear |
| `fp32mp2` / `compose`   | 6.77e-13               | 1.25e-13                | **0.51** | sqrt(N)      |
| `fp32mp2` / `expsum`    | 3.07e-12               | 6.62e-13                | **1.48** | super-linear |

Two main accuracy takeaways for the all-negative input regime:

1. **`compose` beats `expsum` by `~6×` at `N = 10000` and the gap
   *grows* with depth.** This is purely the alpha-exponent gap
   (`0.5` vs `1.5`); the two kernels are mathematically identical
   in real arithmetic. At `N = 1e5` the gap would be `~50×`; at
   `N = 1e6` it would be `~380×`.
2. **`fp32mp2` gives `double` a hard time on absolute error budget.**
   At `N = 10000` `fp32mp2 / expsum` has `meanRelErr ≈ 6.6e-13`,
   which is `~24×` worse than `double / expsum`'s `2.8e-14` — but
   crucially the *propagation rate* `α` is identical, so the
   relationship is just an `eps_T / eps_double` floor that `fp32mp2`
   pays at every depth. With `eps_double ≈ 2e-16` and
   `eps_fp32mp2 ≈ 1e-14` (ulp of the lo-component when paired with
   a unit-magnitude hi-component), this `~50×` floor is the expected
   one.

### Why `float / compose` shows α ≈ 0.78 on GPU vs ≈ 0.52 on host

Same source code, same `__float128` reference, **different per-call
accuracy of `exp(float)`**:

* **glibc `std::exp(float)` (host)** is conformant to ~1 ulp, and
  on modern glibc is effectively correctly rounded for typical
  arguments. Per-step rounding errors are unbiased → CLT applies →
  `meanRelErr ~ √N · ε_f` → α ≈ 0.5. Pure random walk.
* **CUDA `expf()` (device)** is specified at ≤2 ulp and carries a
  small *directional* bias inside certain argument bands of its
  polynomial approximation. On uniform-random negative inputs that
  bias has a non-zero mean, so the per-step errors are no longer
  zero-mean independent — they pick up a coherent component that
  accumulates linearly. The result is α drifting out of √N
  (α = 0.5) toward linear (α = 1.0); empirically it lands around
  α ≈ 0.78.

`double` and `fp32mp2` rows are unaffected because at FP64 ulp scale
(`ε_d ≈ 2e-16`) any `expf`-style directional bias is smaller than the
random component and gets washed out — both host and device show
α ≈ 0.5 for `compose`, α ≈ 1.5 for `expsum`.

`expsum` rows on `float` also stay at α ≈ 1.5 on both targets: the
dominant error source is the linear growth of `|s|` along the sum
(see the "expsum lands in super-linear" subsection above), which
swamps the `expf`-bias contribution.

## Empirical performance — Ada-class GPU, default config

`fp32mp2` is software-emulated FP32 arithmetic, so its value
proposition is on hardware where FP64 is heavily throttled. On
Ada-class GPUs (L40S / RTX-6000 Ada / RTX-4090) the FP64 issue rate is
**1/64** of FP32 — exactly the regime `cuda_multi_fp` is built for.

Default config (`numSamples = 10000`, `chainLen = 10000`, all-negative
inputs):

```
dtype     kernel   time[ms]   Gexps/s    vs_fp64
float     compose      8.5    94.432    27.351×
float     expsum       7.8   102.404     1.950×
double    compose    231.7     3.453     1.000×*
double    expsum      15.2    52.527     1.000×*
fp32mp2   compose     38.6    20.702     5.996×
fp32mp2   expsum      10.0    80.004     1.523×
```

What the numbers say:

* **`double / compose` collapses to `3.5 Gexps/s`** — Ada's FP64 unit
  is 64× slower than FP32, and `compose` is FP64-arithmetic-bound. This
  is exactly the workload regime where `cuda_multi_fp` is intended to
  help.
* **`fp32mp2 / compose` is `6.0×` faster than `double / compose`.**
  The throughput ceiling for this op-count ratio is
  `64 / 7 ≈ 9×` (Ada FP32:FP64 ratio divided by fp32mp2's
  ~7× higher static FP-op count); the observed `6×` is that ceiling
  minus a latency-bound drag at the relatively small default
  `numSamples`.
* **`float / compose` is `27×` faster than `double / compose`** —
  cleanly tracks Ada's `FP32:FP64 = 64:1` ratio, with the gap from
  the theoretical 64× absorbed by the per-`exp()` polynomial latency.
* **`expsum` is bandwidth-/latency-bound rather than compute-bound**:
  all dtypes (float, double, fp32mp2) land within ~2× of one another.
  Read traffic dominates; the working precision barely matters.

The two kernels stress the GPU differently:

* `compose` is **compute-bound**: per chain step it issues
  `~7 FP32 ops (fp32mp2)` vs `~1 FP64 op (double)` of arithmetic.
  Whichever has the higher `FP_issue_rate / op_count` ratio wins —
  on Ada that's overwhelmingly fp32mp2 and float.
* `expsum` is **memory- and latency-bound**: very little arithmetic
  per element loaded, so dtype choice is a wash.

## Verification: no FP64 ops in the `fp32mp2` SASS hot loop

Disassembling `_out/hpc_exp_error_gpu.o` with `cuobjdump --dump-sass`
and grepping `fp32mp2` kernels for `D*` arithmetic opcodes returns
**only 4 `DADD` instances total across all four `fp32mp2` kernels** —
one per kernel, each immediately preceded by two `F2F.F64.F32`
conversions and followed by `STG.E.64`. They are the result
write-back that converts the accumulated `fp32mp2` into a `double`
for the host output buffer; the inner-loop arithmetic is entirely
`FFMA / FMUL / FADD`. So `fp32mp2` genuinely runs on FP32 hardware
end-to-end.

## Build details

The benchmark ships as eight source files plus a `Makefile` and this
`README.md`. There are no other dependencies on the surrounding tree:

```
hpc_exp_error.h               - public types & function declarations
hpc_exp_error_kernels.hpp     - templated compose / expsum kernels
hpc_exp_error_host.cpp        - CPU benchmark + shared reporting
hpc_exp_error_ref.cpp         - __float128 reference oracle (g++ only)
hpc_exp_error_gpu.cu          - CUDA implementation (nvcc only)
main_exp_error.cpp            - CLI driver
Makefile                      - build & run rules
README.md                     - this file
```

External dependencies:

* **cuda_multi_fp headers** — `fpmp.h`, `fpmp_math.h`. Header-only;
  pointed at via `CUDA_MULTI_FP_DIR` or `FPMP_INC` (see Quick start).
* **`g++`** — recent enough to provide `__float128` and `libquadmath`
  (any GCC ≥ 7 on x86_64 Linux). Used for both host TUs and for the
  `__float128` reference.
* **`nvcc`** — for the GPU TU. Architecture is auto-detected from
  `nvidia-smi --query-gpu=compute_cap` and defaults to `sm_86` if
  probing fails. Override with `make ARCH=89` for Ada / `ARCH=90` for
  Hopper, etc.
* **`libquadmath`** — linked at the end. Present in standard GCC
  distributions on x86_64 Linux.

Compiler-specific notes:

* `hpc_exp_error_ref.cpp` is always built with `g++`. `nvc++` rejects
  the GCC-only `-fext-numeric-literals` flag, so the reference TU is
  pinned to `g++`.
* When the user sets `CXX=nvc++`, the linker driver is switched to
  `nvc++ -cuda` so `nvc++`-private runtime symbols resolve correctly.

Performance-measurement knobs are exposed as Makefile variables:
`NUM_OPTIONS`, `CHAIN_LEN`, `REPS`, `NUM_ITER`, `THREADS`,
`NUM_BLOCKS`, `SWEEP_NUM_DEPTHS`. See `make help` for the
authoritative list.

## Recommendation

For the workload this benchmark targets (all-negative
`xs ∈ [-1e-2, -1e-4]`, chain length 10K–100K) on FP64-throttled
GPUs (Ada / L40S / RTX-class):

1. **Use `compose`, not `expsum`** — it gives `O(√N)` error
   propagation vs `O(N^1.5)`, a `~6×` accuracy win at `N = 10000`
   that grows with depth. The arithmetic cost is moot on Ada because
   the dominant baseline (`double / compose`) is the slow path you'd
   already be replacing.
2. **`fp32mp2 / compose` is the recommended replacement for
   `double / compose`.** It delivers a ~6× speedup with a `~50×`
   absolute-error floor relative to FP64 — fine whenever the
   downstream tolerance is `~1e-12` rather than `~1e-14`. This is
   the regime `cuda_multi_fp` is built for.

Note on hardware scope: this benchmark intentionally **does not**
report numbers on full-FP64 datacenter GPUs (A100 / H100). On those
parts FP64 is only 2× slower than FP32, so the ~7× higher static
op-count of fp32mp2 makes it net-slower than native `double` for
`compose`. `cuda_multi_fp` is not a recommended migration target on
that hardware class — use native FP64 instead.
