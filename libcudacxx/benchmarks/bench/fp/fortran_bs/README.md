fortran_bs - Black-Scholes option pricing accuracy & throughput in Fortran
==========================================================================

Compares **accuracy** (error vs the host fp64 reference price) and
**throughput** (options per second) of GPU European call pricing across
two arithmetic types and two execution paths that share the same input
distribution, the same reference, and the same FPMP device LTO library:

| Path     | Compiler            | Source                       | What it shows                                |
|----------|---------------------|------------------------------|----------------------------------------------|
| `device` | nvfortran `-cuda`   | `fortran_bs_device.cuf`      | CUDA Fortran `attributes(global)` kernels    |
| `acc`    | nvfortran `-acc=gpu`| `fortran_bs_acc.f90`         | OpenACC `!$acc parallel loop` kernels        |

| Type             | Mantissa bits | Hardware lane | Notes                |
|------------------|---------------|---------------|----------------------|
| `real(real64)`   |  52           | FP64          | native double        |
| `type(fp32mp2)`  | ~46           | FP32 (×2)     | FPMP float-float     |

Why Black-Scholes
-----------------

Recursive summation (`benchmarks/fortran_sum`) is a memory-bound test —
once the input array fits in L2 the kernel is bottlenecked on L2 read
bandwidth and the per-type compute differences are mostly hidden.

Black-Scholes is the **opposite** end of the spectrum: each option uses
4 transcendentals (`sqrt`, `log`, `exp`, `erfc`) plus a dozen
arithmetic ops, working only on a handful of scalars per option, so
the kernel is firmly **compute-bound**.  This makes it an honest test
of the relative arithmetic cost of `real(real64)` (native fp64) vs
`type(fp32mp2)` (FPMP's float-float) on the GPU — particularly on
consumer / RTX-class hardware where fp64 throughput is 1/32 of fp32
and fp32mp2 can run noticeably faster than native fp64 while keeping
~46-bit mantissa accuracy.

This benchmark mirrors the C++ `benchmarks/bs/` benchmark but is
scoped down to the two precision paths the user actually compares for
Fortran work.  Both rows go through the same templated `bs_price`
formula:

```fortran
sqrtT  = sqrt(T)
vsqrtT = sigma * sqrtT
d1     = (log(S/K) + (r - q + 0.5*sigma*sigma)*T) / vsqrtT
d2     = d1 - vsqrtT
disc_r = exp(-r*T);  disc_q = exp(-q*T)
N(x)   = 0.5 * erfc(-x * sqrt(0.5))
price  = S*disc_q*N(d1) - K*disc_r*N(d2)
```

`N(x)` is evaluated through `erfc` rather than the textbook
`0.5 * (1 + erf(x * sqrt(0.5)))` to avoid catastrophic cancellation
in the deep-OTM tails where one of `Nd1` / `Nd2` is close to zero —
`erfc` returns those small tail values directly, no `1 + (something
close to -1)` subtraction.  Matches the C++ `benchmarks/bs/`
benchmark which uses the same cancellation-resistant `norm_cdf`.

So the speedup column is an apples-to-apples arithmetic comparison:
same algorithm, same transcendentals, only the underlying scalar type
changes.

Workload
--------

Each of the `N = NUM_OPTIONS` options (default 1 048 576) has
randomized parameters:

| Variable | Range          | Notes                       |
|----------|----------------|-----------------------------|
| `S`      | [10, 200]      | spot price                  |
| `K`      | [10, 200]      | strike price                |
| `T`      | [0.05, 5.0]    | time to maturity (years)    |
| `sigma`  | [0.05, 1.0]    | volatility                  |

The risk-free rate `r = 0.02` and dividend yield `q = 0.01` are fixed
globally.  All options are CALLS (matching `bs.cpp`'s default).

Inputs are generated on the host with a 4-stream golden-ratio
low-discrepancy sequence (one stream per parameter, all using different
irrationals so the four input arrays decorrelate):

```fortran
real(real64), parameter :: PHI_S  = 0.6180339887498948_real64  ! (sqrt(5)-1)/2
real(real64), parameter :: PHI_K  = 0.7548776662466928_real64  ! (sqrt(17)-3)/2
real(real64), parameter :: PHI_T  = 0.8392867552141612_real64  ! plastic const - 1
real(real64), parameter :: PHI_V  = 0.5436890126920764_real64  ! sqrt(2)/2 - sqrt(2)/8

do i = 1, N
  S(i)     = S_LO + modulo(i * PHI_S, 1.0) * (S_HI - S_LO)
  K(i)     = K_LO + modulo(i * PHI_K, 1.0) * (K_HI - K_LO)
  Tmat(i)  = T_LO + modulo(i * PHI_T, 1.0) * (T_HI - T_LO)
  V(i)     = V_LO + modulo(i * PHI_V, 1.0) * (V_HI - V_LO)
end do
```

The deterministic LDS means the benchmark is fully reproducible — no
RNG seed to set, identical inputs from run to run.

Natural-arithmetic OpenACC kernel — pure-OpenACC build
------------------------------------------------------

The OpenACC fp32mp2 kernel in `fortran_bs_acc.f90` is written with
**end-user arithmetic**:

```fortran
sqrtT  = sqrt(Ti)
d1     = (log(Si / Ki) + (R_FF - Q_FF + HALF_FF * sigi * sigi) * Ti) / vsqrtT
price  = Si * disc_q * Nd1 - Ki * disc_r * Nd2
```

i.e. exactly the same `+`/`-`/`*`/`/`/`sqrt`/`log`/`exp`/`erfc` syntax
you would write for `real(real64)`.  No raw bind(C) subroutine plumbing
in the kernel body.  Getting this to work on nvfortran 25.11 took
**two** discipline rules in the build, both forced on us by codegen
bugs in the compiler (see the standalone reproducers under
`../../../nvfortran_bug_temp/`):

1. **Pure-OpenACC build, no `-cuda`.**  The entire OpenACC build is
   compiled with `-acc=gpu` only — no `-cuda` on either nvfortran
   invocation, no `use cudafor` in the source.  When `-cuda` was on
   (either as a bindings TU flag or on the main program TU, for
   `use cudafor` timing support), nvfortran emitted FPMP procedures
   returning `type(fp32mp2)` **by value** in a codegen state that
   crashed at runtime with `CUDA_ERROR_ILLEGAL_ADDRESS` on the first
   such call inside an `!$acc parallel loop`.  Compiling the whole
   OpenACC stack pure-OpenACC sidesteps that codegen path entirely.

2. **Never store a `type(fp32mp2)` expression directly into a global
   array element.**  Even under pure-OpenACC compilation, nvfortran
   25.11 miscompiles the "by-value derived-type return whose result
   slot is an indexed-global-array store" ABI path: every `%lo` field
   in the returned struct gets the `%hi` source value, producing a
   deterministic 2x error on one component of the result.  The
   workaround is to stage every `type(fp32mp2)` intermediate through
   a private scalar declared in the `!$acc parallel loop private(...)`
   clause, then issue ONE intrinsic same-type copy from the private
   scalar into the global array:

   ```fortran
   ! BUG (silent miscompile):
   prices_ff(i) = Si * disc_q * Nd1 - Ki * disc_r * Nd2

   ! OK (private scalar -> array intrinsic copy):
   price        = Si * disc_q * Nd1 - Ki * disc_r * Nd2   ! into private
   prices_ff(i) = price                                    ! private -> array
   ```

   `fortran_bs_acc.f90`'s `bs_kernel_fp32mp2` follows this discipline:
   every named intermediate (`Si`, `Ki`, `Ti`, `sigi`, `sqrtT`,
   `vsqrtT`, `d1`, `d2`, `disc_r`, `disc_q`, `Nd1`, `Nd2`, `price`,
   and the per-iteration copies of the BS model constants) is in the
   `private(...)` clause, and `prices_ff(i) = price` is the only
   store into the global prices array.  See
   `../../../nvfortran_bug_temp/expert_repro/`'s `simple_loop.f90`
   FAIL vs `simple_loop_via_priv.f90` PASS contrast for the cleanest
   demonstration of why the discipline matters in the minimal case.

   **Important caveat:** the private-scalar discipline is _necessary
   but not sufficient_ on nvfortran 25.11.  Even with every
   `type(fp32mp2)` intermediate staged through a private scalar as
   above, `bs_kernel_fp32mp2` still **segfaults at runtime** on the
   first OpenACC launch.  The kernel chains many derived-type
   operations (`+`, `-`, `*`, `/`, `sqrt`, `log`, `exp`, `erfc` over
   ~15 named intermediates) and at least one additional codegen path
   is broken inside that chain.  We have not yet isolated that
   second failure mode into a minimal reproducer, so the fortran_bs
   OpenACC build remains opt-in (see "Quick start" above; pass
   `WITH_ACC=1` to include it in `make all`, or `make acc`
   explicitly to observe the crash).  The bug-report package in
   `../../../nvfortran_bug_temp/bugreport_nvfortran/` documents the
   isolated by-value-into-array bug (which IS reproducible
   minimally) and references this kernel as the broader-surface
   counter-example.

Going pure-OpenACC means we lose access to `cudafor`'s `cudaEvent`
timer.  The benchmark uses Fortran's intrinsic `system_clock`
instead, with `!$acc wait(0)` before the closing clock read so the
timer captures real GPU execution time, not CPU dispatch latency.

Device LTO (`-gpu=cc<arch>,lto`) is still in play across both TUs,
so the OpenACC binary executes the same FPMP arithmetic PTX as the
CUDA Fortran row — the FPMP primitives get inlined into the kernel
body at link time.

The Makefile compiles the unified `fpmp_api.f90` into a separate
`fpmp_api_acc.o` (with `-acc=gpu -gpu=cc<arch>,lto`, NO `-cuda`, so the
`!@cuf` directives stay dormant) and into a parallel `mod_acc/`
module-output directory so it does not collide with the
`fortran_bs_device.cuf` build's own `fpmp_api_dev.o` and `mod_dev/`
artifacts (where the same `fpmp_api.f90` is compiled with `-cuda`, so
`!@cuf` activates and every wrapper gets `attributes(device)`).

For context: the companion `fortran_sum_acc.f90` benchmark still
passes `-cuda -acc=gpu` together and works fine, because its kernels
only ever call `call fpmp_acc(...)` (subroutine, out parameter) and
never trigger the by-value-return ABI path.  This benchmark cannot
do that — the whole point of `fortran_bs_acc` is to demonstrate
end-user natural-arithmetic syntax on `type(fp32mp2)` inside
`!$acc parallel loop`, so it pays the pure-OpenACC tax.


Where the `double -> fp32mp2` conversion lives
---------------------------------------------

`fp32mp2` was introduced specifically to **replace fp64 in hot loops**
on fp64-limited hardware.  Nobody pays a per-option `double ->
fp32mp2` conversion at run time in practice, and that conversion is
relatively expensive (~5-10 cheap fp32 ops on the GPU, much more on
the CPU), so this benchmark keeps it strictly **outside** the timed
window.

The two execution paths differ on **how** they do the conversion:

* CUDA Fortran (`fortran_bs_device.cuf`) launches a one-shot untimed
  device kernel (`convert_inputs_kernel`) that calls `fp32mp2(S(i))`
  inside `attributes(global)` code.  This goes through the standard
  FPMP runtime (`__fp32mp2_from_double`) and is fully inlined via
  `-gpu=lto,cc<arch>` device LTO across `fpmp_api.cuf` and `fpmp_lib.o`.
* OpenACC (`fortran_bs_acc.f90`) does the split with **pure host
  arithmetic** — see the `dbl_to_ff` helper:

  ```fortran
  hi = real(x, c_float)                         ! leading fp32
  r%hi = hi
  r%lo = real(x - real(hi, c_double), c_float)  ! residual
  ```

  No FPMP call, no `fp32mp2()` constructor, no defined assignment.
  Both the `fp32mp2()` constructor and the `type(fp32mp2) =
  real(real64)` defined assignment ARE host-callable through
  `fpmp_api`, but they each dispatch into the FPMP runtime
  (`__fp32mp2_from_double`), which is overkill for what is really
  just a textbook double-float field split.  Doing the decomposition
  with two fp32 casts on the host runs once at startup and is
  identical numerically to what the FPMP runtime would compute —
  `hi + lo` reconstructs the original double exactly in fp64.

The five fp32mp2 model constants (`0.5`, `1.0`, `r=0.02`, `q=0.01`,
`sqrt(0.5)`) follow the same pattern:

* CUDA Fortran initialises them in a 1-thread
  `init_fp32mp2_consts_kernel` and stashes them in module-level
  `device` storage.
* OpenACC stores them as a 5-element host-resident `BS_CONSTS_FF(:)`
  array (populated by the same `dbl_to_ff` helper) and `copyin`'s the
  array into the data region with the other inputs.  The array form
  is deliberate: nvfortran's OpenACC `copyin` / `present` clauses do
  not reliably map **scalar** derived-type variables to device memory
  (the kernel ends up dereferencing a host pointer and crashes with
  a segfault).  A small derived-type **array** uses the same code
  path that `fortran_sum_acc.f90`'s `partials_ff` array uses and
  works correctly.

Reading the prices back uses the host-callable `fpmp_to_double`
generic from `fpmp_api`:

```fortran
pi = fpmp_to_double(prices_ff(ii))
```

All of this happens outside the timed window, so conversion costs do
not contaminate the throughput numbers.

Either way, once the data region is open the timed kernels never call
into the FPMP runtime for any `double -> fp32mp2` cast — they read
the pre-converted fp32mp2 inputs and pre-converted fp32mp2 constants
directly.  This is the same discipline that `fortran_sum_acc.f90`
uses for its fp32 input array (no `double -> fp32mp2` in the inner
loop), just applied to a more compute-heavy workload where 4-5
conversions per option *would* otherwise show up as measurable
overhead.

Reference and accuracy
----------------------

A per-option reference price array is computed **once on the host in
fp64** using the Fortran intrinsic `sqrt`, `log`, `exp`, `erfc` (which
on nvfortran / gfortran resolve to `libm`
`sqrt`/`log`/`exp`/`erfc`).  Computing `N(x)` through `erfc` rather
than `0.5 * (1 + erf(...))` avoids catastrophic cancellation in the
deep-OTM tails where the textbook `1 + erf(...)` subtraction loses
most of its significant digits — the reference and every GPU row
then agree to within their underlying intrinsic precision even on
options where `Nd1` / `Nd2` is close to zero.  This reference is
used to compute max / average relative error for every GPU row.

The fp64 row's `Avg|RelErr|` is essentially the host-vs-device fp64
mismatch noise floor — a few ULP per option, modulo FMA contraction,
averaging to ~1e-7.

The `Max|RelErr|` column, on the other hand, is dominated by the
handful of deep-OTM options where the BS call price is near zero
(price ~ 1e-9 to 1e-10 dollars) — for those, even sub-ULP absolute
differences between the host and device `erfc` implementations
divide out as a 1e-2 to 1e-3 *relative* error.  This is exactly the
same shape the C++ `benchmarks/bs/` benchmark reports for fp64 vs an
fp128 reference (`Max|RelErr| ~ 4e-3`), so don't read the max-rel-err
column as "the kernel is broken" — it is the deep-OTM tail of the
LDS, not arithmetic precision.  The `Avg|RelErr|` column is the
honest summary number.

> Note: this is a **fp64** reference, not the `__float128` reference
> used by the C++ `bs/` benchmark.  fp32mp2 at ~46 bits is well below
> fp64 at ~53 bits, so the host fp64 reference is plenty accurate to
> resolve fp32mp2's error.  An `fp64mp2` device reference is also
> available via the FPMP bindings if you want sub-ULP precision; see
> "Adding more types" below.

Files
-----

| File                          | Purpose                                            |
|-------------------------------|----------------------------------------------------|
| `fortran_bs_device.cuf`       | CUDA Fortran benchmark, two kernels (fp64, fp32mp2)|
| `fortran_bs_acc.f90`          | OpenACC benchmark, two kernels (fp64, fp32mp2)     |
| `Makefile`                    | Build/run both paths with CI-friendly auto-skip    |


Quick start
-----------

> **OpenACC `acc` target is opt-in.**  `make` (i.e. `make all`) builds and
> runs only the CUDA Fortran `device` path by default.  On nvfortran 25.11
> the OpenACC fp32mp2 kernel still segfaults at runtime even after the
> private-scalar discipline (see "Natural-arithmetic OpenACC kernel"
> below) — the discipline is necessary but not sufficient for this
> compute-intensive kernel.  The minimal isolated reproducer is in
> `../../../nvfortran_bug_temp/`.  To re-include the OpenACC path in
> the default chain once the compiler is fixed, pass `WITH_ACC=1`; or
> invoke `make acc` explicitly to observe the current crash (the
> binary builds successfully and segfaults on the first OpenACC
> kernel launch).

```bash
# Default: build and run CUDA Fortran `device` only.
make

# Run a single path explicitly:
make device ARCH=86                    # CUDA Fortran (nvfortran -cuda)
make acc    ARCH=86                    # OpenACC      (nvfortran -acc=gpu)

# Run BOTH device + acc in the default chain (re-enables the OpenACC row):
make WITH_ACC=1

# Larger problem size (more options per launch):
make NUM_OPTIONS=4194304 rerun         # 4M options

# Stretch the GPU timing window.  Defaults are per-target -- `device`
# runs at 500 ms per kernel (per-launch overhead is ~10 us so the
# launch count stays in the thousands), `acc` runs at 100 ms per kernel
# (per-`!$acc parallel` overhead is ~1 ms so 100 ms keeps total
# wallclock reasonable).  Override either, or both at once with
# MIN_BATCH_MS:
make device MIN_BATCH_MS_DEV=1000      # 1.0 s per CUDA Fortran window
make acc    MIN_BATCH_MS_ACC=300       # 0.3 s per OpenACC window
make        MIN_BATCH_MS=500           # set both to 0.5 s in one shot

# Disable auto-tuning and use a fixed batch count instead:
make device KERNEL_LAUNCHES=64
make acc    KERNEL_LAUNCHES=64
```

If `nvfortran` is missing, both `device` and `acc` targets become no-op
stubs that print a warning and exit with success.  This keeps the
benchmark CI-safe on environments without the NVIDIA HPC SDK.


Parameters
----------

| Variable      | Default      | Description                                         |
|---------------|--------------|-----------------------------------------------------|
| `ARCH`        | autodetect   | CUDA `sm_xx` (falls back to 86)                     |
| `NUM_OPTIONS` | 1048576      | options per kernel launch                           |
| `N_REPS`      | 5            | timing repetitions; fastest run is reported         |
| `BLOCK_SIZE`  | 256          | CUDA threads per block; also OpenACC vector_length  |
| `NUM_BLOCKS`  | 2048         | CUDA grid size                                      |
| `MIN_BATCH_MS_DEV` | 500     | CUDA Fortran `device` auto-tune target (ms): each per-kernel timed window runs ≥ this long |
| `MIN_BATCH_MS_ACC` | 100     | OpenACC `acc` auto-tune target (ms): same meaning as `MIN_BATCH_MS_DEV` but for the OpenACC path.  Kept shorter because OpenACC has ~1 ms per-`!$acc parallel` overhead |
| `MIN_BATCH_MS` | _unset_     | Convenience override: when set on the command line, overrides BOTH `MIN_BATCH_MS_DEV` and `MIN_BATCH_MS_ACC` at once |
| `KERNEL_LAUNCHES` | 0        | GPU launches per timed window; **0 = auto-tune to `MIN_BATCH_MS_{DEV,ACC}`**, >0 = fixed override |
| `V` / `VERBOSE` / `VERB` | 0 | verbosity level (0 = silent, 1 = minimal, 2 = full, 3 = very verbose; `V=3` also enables nvfortran `-Minfo=accel,inline` diagnostics for the `acc` build) |
| `OUT`         | `_out`       | build output directory                              |


How it works
------------

### One kernel skeleton per execution path, two kernels per skeleton

`fortran_bs_device.cuf` declares two CUDA Fortran kernels:

| Kernel                  | Type                | Reads          | Writes         |
|-------------------------|---------------------|----------------|----------------|
| `bs_kernel_fp64`        | `real(real64)`      | fp64 inputs    | fp64 prices    |
| `bs_kernel_fp32mp2`     | `type(fp32mp2)`     | fp32mp2 inputs | fp32mp2 prices |

Each kernel takes a grid-stride loop over the option array: one CUDA
thread per option-slice, `stride = NUM_BLOCKS * BLOCK_SIZE`.  Per
option, the thread computes the BS call price, writes it to its slot
in the typed `prices` array, and accumulates a per-thread partial sum
(folded on the host into a mean for the sanity row of the printed
table).

`fortran_bs_acc.f90` declares the same pair of kernels but using
`!$acc parallel loop gang vector vector_length(BLK)` with one OpenACC
thread per option `i` — there is no grid-stride, just N parallel
iterations.  BS is compute-heavy enough per option that the
launch-overhead amortisation comes from MIN_BATCH_MS batching, not
from cramming many options into each thread.


### Per-launch cudaEvent timing (auto-tuned per kernel)

Each kernel is timed by a `cudaEvent` pair around a *batch* of
back-to-back launches.  The per-launch time the table reports is
`elapsed / launches`.

The number of launches in each timed window is **auto-tuned per kernel**
so the window runs for at least the per-target `MIN_BATCH_MS` (default
`500` ms for `device`, `100` ms for `acc`).  This is what you want for
reliable performance numbers — without it, fast launches would be
entirely dominated by `cudaEventRecord` / `cudaEventSynchronize`
overhead.

The auto-tuner does:

1. A **batch of `CALIB_LAUNCHES`** (default `32`) warm-up launches:
   fully populates caches, JIT-compiles, and gets the kernel into a
   steady-state launch cadence.
2. A **multi-launch calibration**: `CALIB_LAUNCHES` back-to-back
   timed launches, with `baseline_s = elapsed_ms · 10⁻³ /
   CALIB_LAUNCHES`.
3. `launches = ceil(MIN_BATCH_MS_{DEV,ACC} · 10⁻³ / baseline_s)`,
   clamped to `[1, MAX_KERNEL_LAUNCHES]`.
4. `N_REPS` timed windows of `launches` launches each; the fastest
   per-launch time wins.

The resolved launch count is shown in the **Launches** column of the
output so you can see what the tuner picked.

**Disabling auto-tuning.**  Set `KERNEL_LAUNCHES` to a positive integer
on the make command line — that fixed batch count is then used for
every kernel.


### OpenACC timing wait

The OpenACC path needs an extra `!$acc wait(0)` just before the
closing `system_clock(clk_stop)`.  Without it, OpenACC's async
launches return to the host immediately and the clock reads while
the GPU is still executing the batch, so the elapsed-time
measurement is CPU-dispatch latency instead of GPU-execution time
(which on the small calibration batch under-reports per-launch time
by 30-100× and tricks the auto-tuner into picking 100-1000× too
many launches).  The wait drains async queue 0 so `clk_stop`
captures the moment GPU work actually finishes.


Output table
------------

```
======================================================================================
  FPMP CUDA Fortran Benchmark — Black-Scholes Option Pricing (device)
======================================================================================
  N            = 1048576 options
  blocks       = 2048
  threads/blk  = 256
  repetitions  = 5
  launches/rep = auto-tune to ≥ 500 ms per timed window (per-kernel)
  r, q         =  0.02,  0.01
  inputs       = deterministic LDS (S, K ∈ [10,200]; T ∈ [0.05,5]; σ ∈ [0.05,1])
  is_call      = .true.
  reference    = host fp64 (intrinsic sqrt/log/exp/erfc)
  ref. mean    =  4.500000000000000E+01
--------------------------------------------------------------------------------------
  Type           Max|RelErr|  Avg|RelErr|  Throughput        Time     Speedup  Launches
                                           (Mopts/s)         (s)     vs fp64
  -------------  -----------  -----------  ----------  ----------  ----------  --------
  real(real64)    X.XXXE-XX    X.XXXE-XX   X.XXXE+XX   X.XXXE-XX     1.000        XX
  type(fp32mp2)   X.XXXE-XX    X.XXXE-XX   X.XXXE+XX   X.XXXE-XX     X.XXX        XX
======================================================================================
```

Columns:

| Column       | Format | Meaning                                                       |
|--------------|--------|---------------------------------------------------------------|
| `Type`       | string | scalar arithmetic type used in `bs_price`                     |
| `Max|RelErr|`| ES11.3 | max `|price - ref| / |ref|` across all options                |
| `Avg|RelErr|`| ES11.3 | mean `|price - ref| / |ref|` across all options               |
| `Throughput` | ES10.3 | `N / per-launch-time` (millions of options per second)        |
| `Time`       | ES10.3 | per-launch wallclock time (seconds, `elapsed / launches`)     |
| `Speedup`    | F10.3  | `time_fp64 / time_this_row` (`1.000` on the `real(real64)` row by construction) |
| `Launches`   | I8     | resolved per-window batch count from the auto-tuner           |


Expected behaviour
------------------

On a consumer / RTX-class GPU (1/32 fp64 throughput ratio):

- `Avg|RelErr|` for `real(real64)` is the honest host-vs-device fp64
  mismatch noise floor — around 1e-7, dominated by a few ULPs per
  option averaged across the LDS.
- `Avg|RelErr|` for `type(fp32mp2)` is similar in magnitude (~1e-6 to
  1e-7) for this workload — fp32mp2's ~46-bit mantissa is plenty for
  BS, and the average error is set by the same "near-zero deep-OTM"
  tail of the LDS, not by fp32mp2's truncation.
- `Max|RelErr|` for both rows lands in the 1e-3 to 1e-2 range; this
  is **not** an arithmetic error, it is a handful of deep-OTM options
  where the BS call price is near zero and any ULP-level difference
  divides out as a large relative number.  Matches the shape of the
  C++ `benchmarks/bs/` benchmark (which reports `Max|RelErr| ~ 4e-3`
  for `double` vs an fp128 reference).
- `Speedup` for `type(fp32mp2)` vs `real(real64)` is the
  interesting number — it tells you how much you save on this
  particular hardware by switching to FPMP's float-float arithmetic
  for a transcendental-heavy workload.  Typical RTX-class speedup
  for fp32mp2 is 5-10× on this benchmark.

On hardware with **1:1 fp64 / fp32 throughput** (datacenter A100/H100,
or any GPU with full fp64 hardware), `type(fp32mp2)` is expected to be
slower than `real(real64)` for BS: fp32mp2 needs ~10-12 fp32 ops per
arithmetic op and proportionally more for the transcendentals, while
fp64 is a single hardware instruction.  The benchmark surfaces
exactly this trade-off; the speedup column on those GPUs is `< 1` on
the fp32mp2 row.


Adding more types
-----------------

The two kernels live in `fortran_bs_device.cuf` /
`fortran_bs_acc.f90` and follow a regular shape — copy one and
substitute the type to add an `fp64mp2` row (host reference would
then naturally upgrade to `fp64mp2` for sub-ULP precision) or a
`real(real32)` row (single-precision sanity check; expect Avg|RelErr|
~1e-6).  The Makefile builds the FPMP library with `fp64mp2`
support (always enabled), so no library rebuild is needed.

To match `bs.cpp`'s coverage end-to-end (float, fp32mp2_fast,
fp32mp2, double, fp64emu, fp64emu_accurate, fp64mp2, fp128), the
Fortran side would also need bindings for `fp64emu` (not currently
exposed through `fpmp_api.cuf`) and a `__float128` story (no portable
Fortran equivalent — this is why `fortran_bs` deliberately stops at
fp64 + fp32mp2).
