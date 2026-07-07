fortran_sum - Recursive-summation accuracy & throughput in Fortran
========================================================

Compares **accuracy** (error vs the high-precision reference sum) and
**throughput** (elements per second) of recursive summation across four
accumulator types and three execution paths that all share the same
input array, the same reference, and the same FPMP device LTO library:

| Path     | Compiler            | Source                       | What it shows                                  |
|----------|---------------------|------------------------------|------------------------------------------------|
| `host`   | gfortran/nvfortran  | `fortran_sum_host.f90`       | host CPU; Mode 1 (serial) + Mode 2 (blocked)   |
| `device` | nvfortran `-cuda`   | `fortran_sum_device.cuf`     | CUDA Fortran `attributes(global)` kernels      |
| `acc`    | nvfortran `-acc=gpu`| `fortran_sum_acc.f90`        | OpenACC `!$acc parallel loop` kernels          |

| Accumulator      | Mantissa bits | Hardware lane | Notes                         |
|------------------|---------------|---------------|-------------------------------|
| `real(real32)`   |  23           | FP32          | native single                 |
| `real(real64)`   |  52           | FP64          | native double                 |
| `type(fp32mp2)`  | ~46           | FP32 (×2)     | FPMP float-float              |
| `type(fp64mp2)`  | ~106          | FP64 (×2)     | FPMP double-double            |

The point of having both a CUDA Fortran (`device`) and an OpenACC (`acc`)
GPU path - rather than pinning the Fortran story to CUDA Fortran only -
is that **OpenACC is the natural offload path for many existing Fortran
codes**.  Users do not have to rewrite their kernels as
`attributes(global)` subroutines; they can keep their host loops and
just decorate them with `!$acc parallel loop`, calling FPMP arithmetic
through the same `fpmp_api` module they already use on the host.
The two GPU paths share the same FPMP library build
(`make/lib.mk TARGET=device LINKAGE=lto`, the default), so the speedup
columns are directly comparable, modulo "OpenACC kernel skeleton vs
hand-tuned CUDA Fortran kernel" overhead.

Workload
--------

The input array `x(:)` holds `N` deterministic pseudo-random `real(real32)`
values in `[0, 2^SCALE_BITS)`, generated with a golden-ratio low-discrepancy
sequence:

```fortran
real(real64), parameter :: PHI = 0.6180339887498948_real64    ! φ - 1
do i = 1, N
  u    = modulo(real(i, real64) * PHI, 1.0_real64)
  x(i) = real(u * 2.0_real64**SCALE_BITS, real32)
end do
```

Each `x(i)` therefore exercises the full 23-bit fp32 mantissa (no integer
inputs, no special structure), so summing N of them probes **rounding at
every accumulator's precision boundary** - not just fp32 magnitude
saturation.

All four kernels read from this **shared fp32 input array**.  Each
accumulator does its own native cast/conversion in the inner loop:
`real(x(i), real64)` for `real(real64)`, the mixed-type `acc + x(i)`
overload for `fp32mp2` / `fp64mp2`, and the native `acc + x(i)` for
`real(real32)`.  Sharing one input array also keeps the bandwidth profile
identical across kernels - the throughput differences then reflect the
per-kernel compute cost, not differing memory pressure.

Reference
---------

The reference sum is computed in `fp64mp2` (~106-bit mantissa).  This is
effectively exact relative to every other accumulator in the table, so by
construction the `fp64mp2` row reports zero error.

- The host benchmark sums into an `fp64mp2` accumulator before timing the
  other three types.
- The device benchmark launches the `fp64mp2` reduction kernel once before
  the timing loop and uses its result as the reference.

This benchmark is the productionised version of a customer-supplied minimal
example (`accuracy_sum.f90`).  It also exercises two ergonomic additions to
the Fortran bindings:

- **Defined `assignment(=)`** - `acc = 0.0_c_float` initialises an FPMP
  value directly from a `real(c_float)` / `real(c_double)` / integer scalar
  without an explicit `fp32mp2(...)` / `fp64mp2(...)` constructor call.
- **Mixed-type arithmetic** - `+ - * /` accept a `real(c_float)` or
  `real(c_double)` scalar on either side of an `fp32mp2` / `fp64mp2`
  operand, so the inner loop is the same one line for every accumulator
  type:

  ```fortran
  type(fp32mp2) :: acc
  real(real32)  :: x(:)
  acc = 0.0_c_float
  do i = 1, size(x)
    acc = acc + x(i)         ! no per-element cast, no temporary
  end do
  ```


Files
-----

| File                          | Purpose                                                |
|-------------------------------|--------------------------------------------------------|
| `fortran_sum_host.f90`        | Host CPU benchmark, four accumulator types             |
| `fortran_sum_device.cuf`      | GPU CUDA Fortran benchmark, four kernels               |
| `fortran_sum_acc.f90`         | GPU OpenACC benchmark, four kernels                    |
| `Makefile`                    | Build/run all three paths with CI-friendly auto-skip   |


Quick start
-----------

```bash
# Build and run host + CUDA Fortran device + OpenACC benchmarks
# (with autodetected ARCH):
make

# Run a single path:
make host                              # gfortran/nvfortran
make device ARCH=86                    # CUDA Fortran (nvfortran -cuda)
make acc    ARCH=86                    # OpenACC      (nvfortran -acc=gpu)

# Larger problem size:
make N_ELEMENTS=100000000 rerun

# Push partial sums closer to fp64's precision boundary:
make SCALE_BITS=30 rerun

# Tune the device / OpenACC launch geometry (also re-uses these for host Mode 2):
make NUM_BLOCKS=2048 BLOCK_SIZE=512 rerun

# Stretch the GPU timing window.  Defaults are per-target - `device`
# runs at 500 ms per kernel (per-launch overhead is ~10 µs so the launch
# count stays in the thousands), `acc` runs at 100 ms per kernel (per-
# `!$acc parallel` overhead is ~1 ms so 100 ms keeps total wallclock
# reasonable).  Override either, or both at once with MIN_BATCH_MS:
make device MIN_BATCH_MS_DEV=1000      # 1.0 s per CUDA Fortran window
make acc    MIN_BATCH_MS_ACC=300       # 0.3 s per OpenACC window
make        MIN_BATCH_MS=500           # set both to 0.5 s in one shot

# Disable auto-tuning and use a fixed batch count instead:
make device KERNEL_LAUNCHES=2048
make acc    KERNEL_LAUNCHES=2048
```

If `gfortran` (or your override `FC=...`) is missing, the host targets
become no-op stubs that print a warning and exit with success.  Likewise,
if `nvfortran` is missing, the `device` and `acc` targets become no-op
stubs.  This keeps the benchmark CI-safe on environments without Fortran
tooling.


Parameters
----------

| Variable      | Default      | Description                                          |
|---------------|--------------|------------------------------------------------------|
| `ARCH`        | autodetect   | CUDA `sm_xx` (falls back to 86)                      |
| `FC`          | `gfortran`   | host Fortran compiler                                |
| `N_ELEMENTS`  | 10000000     | array length                                         |
| `N_REPS`      | 5            | timing repetitions; fastest run is reported          |
| `SCALE_BITS`  | 0            | inputs in `[0, 1)`; controls partial-sum magnitude   |
| `BLOCK_SIZE`  | 256          | CUDA threads per block; **also used by host Mode 2** |
| `NUM_BLOCKS`  | 1024         | CUDA grid size; **also used by host Mode 2**         |
| `MIN_BATCH_MS_DEV` | 500     | CUDA Fortran `device` auto-tune target (ms): each per-kernel timed window runs ≥ this long.  The device path's per-launch overhead is ~10 µs so even a 0.5 s window stays in the thousands of launches and the four kernels still complete in well under a minute. |
| `MIN_BATCH_MS_ACC` | 100     | OpenACC `acc` auto-tune target (ms): same meaning as `MIN_BATCH_MS_DEV`, but for the OpenACC path.  Kept shorter because OpenACC has ~1 ms per-`!$acc parallel` overhead. |
| `MIN_BATCH_MS` | _unset_     | Convenience override: when set on the command line, overrides BOTH `MIN_BATCH_MS_DEV` and `MIN_BATCH_MS_ACC` at once.  Existing scripts that pass `MIN_BATCH_MS=X` continue to work unchanged. |
| `KERNEL_LAUNCHES` | 0        | GPU launches per timed window; **0 = auto-tune to `MIN_BATCH_MS_{DEV,ACC}`**, >0 = fixed override (applies to both `device` and `acc`) |
| `FP2MP`       | 0            | build the lib with `CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=1` (integer-only conversion) |
| `MP2FP`       | 0            | build the lib with `CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE=1` (integer-only conversion) |
| `HOST_STRICT_FP` | auto      | compiler-specific flags pinning strict FP order      |
| `V` / `VERBOSE` / `VERB` | 0 | verbosity level (0 = silent, 1 = minimal, 2 = full, 3 = very verbose).  `V=3` also enables nvfortran `-Minfo=accel,inline` diagnostics for the `acc` build (kernel schedule + every cross-procedure inline decision; ~750 lines per build). |
| `OUT`         | `_out`       | build output directory                               |


How it works
------------

### Host - two modes per run

**Mode 1: serial accumulation.**  A simple `do i = 1, N` loop over the
array, one variant per accumulator type.  `cpu_time` brackets each loop,
and the fastest of `N_REPS` runs is reported.  This is the natural
Fortran loop and the canonical "serial summation accuracy" picture: each
accumulator does `N` rounding-prone adds with a monotonically-growing
partial, so each format's mantissa width separates clearly in the table.

**Mode 2: device-style blocked reduction.**  The host runs the *same*
algorithm the GPU kernels run, in pure Fortran:

1. **Per-"thread" grid-stride accumulation** - `NUM_BLOCKS · BLOCK_SIZE`
   independent register accumulators each chew through `~N /
   (NUM_BLOCKS · BLOCK_SIZE)` elements.
2. **Per-block tree reduction** - `BLOCK_SIZE` per-thread partials are
   collapsed by a `log₂(BLOCK_SIZE)` shared-style tree (just an array on
   the host) into one block partial of the accumulator type.
3. **Final fp64 sum** - `NUM_BLOCKS` block partials are converted to
   `real(real64)` and summed on the host.

Because each accumulator's chain length collapses from `N` to about
`N / (NUM_BLOCKS · BLOCK_SIZE) + log₂(BLOCK_SIZE)` (`~46` with the
defaults), every type above `fp32` sits comfortably below the final
fp64 ulp at the answer's magnitude - exactly the pattern the CUDA
benchmark exhibits, demonstrated to be **purely algorithmic** and not
GPU-specific.

Mode 2 uses its own reference: the **blocked fp64mp2 sum** (computed
once outside the timing loop).  This mirrors the device benchmark's
choice and removes the small (~`ulp_fp64(reference)`) cross-algorithm
offset that would otherwise show up as a precision floor for every
accumulator above `fp32`.  The header line
`|mode2_ref - serial fp64mp2 ref|` reports that algorithmic offset
explicitly.

### Device

Each kernel performs the same two-level reduction as host Mode 2, just
in parallel on the GPU:

1. **Thread-level**: every thread strides over the array and accumulates
   a register-resident value of its own type.
2. **Block-level**: a shared-memory tree reduction collapses the
   per-thread register accumulators down to one block partial.  The
   partial is converted to `real(real64)` (via `fpmp_to_double` for FPMP
   types) and written to a per-block global array of `NUM_BLOCKS` doubles.

After the kernel finishes the host sums the (small) array of fp64
partial sums.  The final-stage error from this fp64 summation is
negligible (O(`NUM_BLOCKS · ulp(reference)`)) compared to the
kernel-internal accumulation error being measured.

The reference is the result of the `fp64mp2` reduction kernel itself
(launched once before the timing loop), so the `fp64mp2` row reports
zero error by construction - same convention as host Mode 2.

#### Timing batching (auto-tuned per kernel)

Each kernel is timed by a `cudaEvent` pair around a *batch* of
back-to-back launches.  The per-launch time the table reports is
`elapsed / launches`.

The number of launches in each timed window is **auto-tuned per kernel**
so the window runs for at least the per-target `MIN_BATCH_MS` (default
`500` ms for `device`, `100` ms for `acc` - see the parameters table
above).  This is what you want for reliable performance numbers -
without it, fast kernels (e.g. native fp32 over 10⁷ inputs is ~15 µs)
would be entirely dominated by `cudaEventRecord` /
`cudaEventSynchronize` overhead.

The auto-tuner does:

1. A **batch of `CALIB_LAUNCHES`** (default `32`) warm-up launches:
   fully populates caches, JIT-compiles, and gets the kernel into a
   steady-state launch cadence.  A single warm-up launch is *not*
   enough on the FPMP rows - their first-after-warmup launch still
   measures ~2× steady-state on consumer GPUs, which biases the
   baseline high and makes the auto-tuner pick a batch that under-runs
   the `MIN_BATCH_MS` target by roughly the same factor.
2. A **multi-launch calibration**: `CALIB_LAUNCHES` back-to-back
   timed launches, with `baseline_s = elapsed_ms · 10⁻³ /
   CALIB_LAUNCHES`.  Averaging across the batch gets the baseline
   within a few percent of steady-state for every kernel.
3. `launches = ceil(MIN_BATCH_MS_{DEV,ACC} · 10⁻³ / baseline_s)`,
   clamped to `[1, MAX_KERNEL_LAUNCHES]`.
4. `N_REPS` timed windows of `launches` launches each; the fastest
   per-launch time wins.

`CALIB_LAUNCHES` is exposed as a compile-time `-D` only (no Makefile
knob - the `32` default is fine for the workloads we ship); raise it
if you want even tighter auto-tune stability at the cost of slightly
longer startup, or drop it to `1` to recover the old single-shot
behaviour.

The resolved launch count is shown in the **Launches** column of the
output (right-aligned `I8`) so you can see what the tuner picked.

**Disabling auto-tuning.**  Set `KERNEL_LAUNCHES` to a positive integer
on the make command line - that fixed batch count is then used for
every kernel.  Use this when you want a strictly identical workload per
kernel for cross-comparison; the trade-off is that the slowest kernel
will dictate the runtime and the fastest one may be timer-noise-bound.

**Tightening / loosening the window.**  Per-target:
`make device MIN_BATCH_MS_DEV=1000` runs each CUDA Fortran rep for ≥ 1 s;
`make acc MIN_BATCH_MS_ACC=300` runs each OpenACC rep for ≥ 0.3 s.
Or pass `MIN_BATCH_MS=500` to set both at once.  Drop the target value
to `10` for quick smoke runs.

> Note on cache residency: with the default `N=10⁷` the fp32 input
> array is `40 MB`, which fits inside the L2 cache of recent NVIDIA
> consumer GPUs (sm_89 / RTX 4090: 72 MB L2; RTX A6000 Ada: 96 MB L2).
> After the warm-up launch the array stays L2-resident, so subsequent
> launches are L2-bandwidth bound.  This is intentional - it surfaces
> the per-kernel compute differences cleanly (this is where the
> ~2.7× fp32mp2-over-fp64 advantage on consumer GPUs shows up).  For a
> DRAM-bound profile, push `N` past the L2 footprint
> (`N_ELEMENTS=$((40 * 1024 * 1024))` ≈ 160 MB at fp32).

### OpenACC

The OpenACC path (`make acc`) runs the same workload, the same reference,
and the same four accumulator types, but expresses each reduction kernel
as an `!$acc parallel loop` region instead of an `attributes(global)`
subroutine.  This is the path most existing Fortran codes can adopt
without a kernel rewrite - drop in `!$acc` directives and link against
the same FPMP device LTO library.  The bindings make this work without
any extra effort: every wrapper and every raw `bind(C)` interface in
`fpmp_api.f90` carries an `!$acc routine seq` directive, so calls
into `+`, `*`, `sqrt`, `fpmp_to_double`, the defined-assignment, … all
flow through cleanly when they appear inside an OpenACC compute region.

#### Uniform per-thread-partials algorithm

All four rows share **one kernel skeleton**: a per-thread partials loop
with no in-kernel cross-thread combine.  Each OpenACC thread accumulates
a strided slice of `x(:)` into a register, writes one partial to a
per-type device array, and the host does the final `NTH -> 1` fold in
fp64 after the timed window.  Only the inner accumulator TYPE and the
inner-loop primitive change between rows:

```fortran
type(fp32mp2) :: local
!$acc parallel loop gang vector vector_length(BLK) present(x, partials_ff) private(local)
do gid = 1, NTH                          ! NTH = NUM_BLOCKS * BLOCK_SIZE
  local%hi = 0.0_c_float
  local%lo = 0.0_c_float
  !$acc loop seq
  do i = gid, N, NTH                     ! grid-stride load
    call fpmp_acc(x(i), local)           ! single-component accumulate
  end do
  partials_ff(gid) = local
end do
```

Expected behaviour
------------------

### Accuracy

With the default workload (`N = 10⁷`, `SCALE_BITS = 0` → max input
≈ `1`):

#### Host Mode 1 (serial)

- `real(real32)`  - relative error ≈ `2⁻²³` ≈ `1.4·10⁻⁷`, the inherent
                    precision of single precision.  Each add of an
                    `~10⁷`-magnitude value to an `~10¹⁴` partial loses
                    one ulp.
- `real(real64)`  - error ≈ 0.  The 52-bit mantissa has ~30 bits of
                    headroom over the addend at this magnitude; the LDS
                    inputs cancel rounding errors almost perfectly.
- `type(fp32mp2)` - relative error ≈ `10⁻¹⁰` - about **1500× better than
                    fp32**.  This is exactly its `~46`-bit accumulator at
                    work.
- `type(fp64mp2)` - error = 0 (this is the serial reference).

To bring `real(real64)` itself into the visibly-non-zero regime in
Mode 1, push `SCALE_BITS` higher: at `SCALE_BITS = 30` the partial sum
approaches `2⁵²`, where fp64 also begins to lose ulps per add.
`fp32mp2` will then trail fp64 (it has fewer bits), and `fp64mp2` will
remain the only exact accumulator.

#### Host Mode 2 (blocked) - matches the device output

Mode 2 collapses each accumulator's effective add count from `10⁷` to
`~46`, so every type above `fp32` produces an answer whose error is
below the final fp64 ulp at the answer's magnitude.  The result is the
same pattern you see on the GPU:

- `real(real32)`  - relative error ≈ `2.5·10⁻⁹`, ~50× better than
                    serial: per-thread accumulators only see ~`38`
                    elements, so the 23-bit fp32 mantissa stays inside
                    its precise range until the last few tree levels.
                    Still visibly non-zero because fp32's per-block ulp
                    (`~2·10⁴`) survives the final fp64 sum.
- `real(real64)`, `type(fp32mp2)`, `type(fp64mp2)` - all report **zero
                    error**.  Their per-block errors (`~10⁻⁵`,
                    `~10⁻³`, `~10⁻²⁰`) are well below `ulp_fp64(1.7·10¹⁴) ≈ 2`,
                    so all three round to the same fp64 bit-pattern
                    as the blocked fp64mp2 reference.

This is exactly the GPU benchmark's accuracy table.  Comparing Mode 1
and Mode 2 side-by-side directly shows that **the GPU's "all zero"
pattern for fp64/fp32mp2/fp64mp2 is purely a consequence of the
parallel-reduction algorithm**, not anything GPU-specific.

The Mode 2 header also prints `|mode2_ref - serial fp64mp2 ref|` (~`0.03`
at this scale) - the small but exactly-`ulp_fp64`-class offset between
the two algorithms' answers, so the "zero" results in Mode 2 cannot be
mistaken for an absolute precision claim.

#### Strict-IEEE pinning on the host

The host benchmark **must** be built with strict left-to-right FP
accumulation for the SP saturation to be visible.  Some compilers
(notably `nvfortran -O3` and `-fast`) reassociate reductions by default,
splitting the loop into independent SIMD-lane partials that never
saturate - making single precision look bit-exact and erasing the very
behaviour the benchmark is meant to demonstrate.

The Makefile auto-picks the right strict-IEEE flag per compiler:

| Compiler   | `HOST_STRICT_FP`                        |
|------------|-----------------------------------------|
| `gfortran` | `-fno-fast-math -fno-associative-math`  |
| `nvfortran`| `-Kieee -Mnofma`                        |
| `ifort`/`ifx` | `-fp-model precise -no-fma`          |

Override with `HOST_STRICT_FP=...` (or pass empty to opt out and let the
compiler reassociate, which is useful for showing the throughput
upper-bound of fast-math SP accumulation).

`make help` prints the resolved flag for the current `FC`.

#### Device-side accuracy

The device benchmark runs a **parallel reduction**: a grid-stride loop
collapses the array into one register accumulator per thread, then a
shared-memory tree reduces inside each block, then a small per-block
`real(real64)` array is summed on the host.  This dramatically shortens
the **per-accumulator rounding chain**:

| Phase                                | Adds per accumulator        | Format used    |
|--------------------------------------|-----------------------------|----------------|
| Grid-stride per-thread accumulation  | `N / (NUM_BLOCKS·BLOCK_SIZE)` | accumulator type |
| Shared-memory tree reduction         | `log₂(BLOCK_SIZE)`          | accumulator type |
| Host final sum over block partials   | `NUM_BLOCKS`                | `real(real64)` |

With the defaults (`N=10⁷`, `BLOCK_SIZE=256`, `NUM_BLOCKS=1024`) each
accumulator only sees `~38 + 8 = 46` rounding-prone adds, vs `10⁷` on the
serial host loop.  At the same time the per-block partial peaks at
`~N · mean(x) / NUM_BLOCKS ≈ 1.7·10¹¹` instead of the full `1.7·10¹⁴`,
so the per-add ulp inside each accumulator is much smaller than what the
host loop sees.

The combined effect: every accumulator above `fp32` typically produces a
result whose error is **below the final-stage `ulp_fp64(reference)`** (≈
`2` at the default scale).  All three (`fp64`, `fp32mp2`, `fp64mp2`)
therefore round to the same fp64 bit-pattern as the reference and report
zero error.  Only `fp32`'s per-block ulp (`~2·10⁴`) survives the final
rounding, so its error remains visible (and is also visibly smaller than
on the host because of the shorter chain).

This is the canonical reason `fp32mp2` is interesting on consumer GPUs:
when parallel reduction is available, the `~46`-bit `fp32mp2` accumulator
closes the precision gap to native `fp64` essentially for free, while
running in FP32 lanes.

To recover the serial-style precision pyramid on the device - useful for
a head-to-head accuracy story where every type's accumulator chain is the
full length of the array:

```bash
make device NUM_BLOCKS=1 BLOCK_SIZE=1     # one thread, fully serial; very slow
```

Or push `SCALE_BITS` high enough that even the per-block partials stress
fp64 (e.g. `SCALE_BITS=30+`).

### Throughput

On a consumer GPU (RTX 30xx / 40xx / Ada / Hopper consumer), expect:

- `real(real32)`  - fastest, FP32-bound.
- `real(real64)`  - slowest of the natives; consumer FP64 is ~1/32 of FP32.
- `type(fp32mp2)` - uses FP32 lanes; typically **~2.5–3× faster than
                    native fp64** on consumer GPUs while delivering
                    near-fp64 accuracy.
- `type(fp64mp2)` - uses FP64 lanes ~12× per add/mul; slower than native
                    fp64 but provides ~106-bit precision.

On datacenter GPUs (A100/H100) where FP64 is at parity with FP32,
`type(fp32mp2)` is closer to native fp64 in throughput while still offering
the FP32-multiplier accuracy improvement over native single.


Output format
-------------

```
======================================================================================
  FPMP Host Fortran Benchmark - Accuracy Sum
======================================================================================
  N            = 10000000
  repetitions  = 5
  scale        = 2^25 =  3.355E+07
  inputs       = pseudo-random fp32 in [0, scale) (golden-ratio LDS)
  reference    = sum(x) in fp64mp2 (~106-bit mantissa)
  reference    =  1.677721493741411E+14
--------------------------------------------------------------------------------------
  Mode 1: serial accumulation (acc = acc + x(i)) over the full N
          one accumulator, N rounding-prone adds
--------------------------------------------------------------------------------------
  Accumulator       |Error|     Rel.Err  Throughput        Time     Speedup
                                         (Melem/s)         (s)   vs real64
  -------------  ----------  ----------  ----------  ----------  ----------
  real(real32)    2.293E+07   1.367E-07   1.377E+03   7.260E-03       1.030
  real(real64)    0.000E+00   0.000E+00   1.337E+03   7.481E-03       1.000
  type(fp32mp2)   1.587E+04   9.457E-11   4.283E+01   2.335E-01       0.032
  type(fp64mp2)   0.000E+00   0.000E+00   4.052E+01   2.468E-01       0.030
--------------------------------------------------------------------------------------
  Mode 2: device-style blocked reduction (NUM_BLOCKS=1024, BLOCK_SIZE=256, threads=262144)
          per-thread grid-stride + per-block tree + fp64 final sum
          ~38 adds per "thread", log2(BLOCK_SIZE) tree adds, NUM_BLOCKS final fp64 adds
          mode2 reference (blocked fp64mp2)  =  1.677721493741410E+14
          |mode2_ref - serial fp64mp2 ref|   =  3.125E-02
--------------------------------------------------------------------------------------
  Accumulator       |Error|     Rel.Err  Throughput        Time     Speedup
                                         (Melem/s)         (s)   vs real64
  -------------  ----------  ----------  ----------  ----------  ----------
  real(real32)    4.186E+05   2.495E-09   5.567E+02   1.796E-02       1.448
  real(real64)    0.000E+00   0.000E+00   3.845E+02   2.601E-02       1.000
  type(fp32mp2)   0.000E+00   0.000E+00   4.230E+01   2.364E-01       0.110
  type(fp64mp2)   0.000E+00   0.000E+00   4.015E+01   2.491E-01       0.104
======================================================================================
```

The device and OpenACC outputs add a right-aligned `Launches` column
showing the auto-tuned per-window batch size:

```
======================================================================================
  FPMP CUDA Fortran Benchmark - Accuracy Sum (device)
======================================================================================
  N            = 10000000
  blocks       = 1024
  threads/blk  = 256
  repetitions  = 5
  launches/rep = auto-tune to ≥ 100 ms per timed window (per-kernel)
  scale        = 2^25 =  3.355E+07
  inputs       = pseudo-random fp32 in [0, scale) (golden-ratio LDS)
  reference    = fp64mp2 reduction (~106-bit mantissa)
  reference    =  1.677721493741410E+14
--------------------------------------------------------------------------------------
  Accumulator       |Error|     Rel.Err  Throughput        Time     Speedup  Launches
                                         (Melem/s)         (s)   vs real64
  -------------  ----------  ----------  ----------  ----------  ----------  --------
  real(real32)    4.186E+05   2.495E-09   7.069E+05   1.415E-05       2.897      5745
  real(real64)    0.000E+00   0.000E+00   2.440E+05   4.099E-05       1.000      2283
  type(fp32mp2)   0.000E+00   0.000E+00   6.756E+05   1.480E-05       2.770      5745
  type(fp64mp2)   0.000E+00   0.000E+00   4.341E+04   2.304E-04       0.178       431
======================================================================================
```

The OpenACC table has the same column layout - same auto-tune,
same cudaEvent timing - with the expected OpenACC vs CUDA Fortran tax
on the FPMP rows (kernel-internal local-memory spill at the `bind(C)`
wrapper boundary; see the "OpenACC timing" section above):

```
======================================================================================
  FPMP OpenACC Fortran Benchmark - Accuracy Sum
======================================================================================
  N            = 10000000
  threads      = 262144 (NUM_BLOCKS * BLOCK_SIZE; one partial per thread)
  repetitions  = 5
  launches/rep = auto-tune to ≥ 100 ms per timed window (per-kernel)
  scale        = 2^0 =  1.000E+00
  inputs       = pseudo-random fp32 in [0, scale) (golden-ratio LDS)
  reference    = fp64mp2 OpenACC reduction (~106-bit mantissa)
  reference    =  4.999999683324780E+06
--------------------------------------------------------------------------------------
  Accumulator       |Error|     Rel.Err  Throughput        Time     Speedup  Launches
                                         (Melem/s)         (s)   vs real64
  -------------  ----------  ----------  ----------  ----------  ----------  --------
  ...
======================================================================================
```

Numeric columns are uniform `ES10.3` so every header / data byte lands
on the same column boundary; the `Speedup vs real64` column uses
`F10.3` so the typical O(0.1)..O(10) ratios read cleanly without
exponent noise.  Units (`Melem/s`, `s`, `vs real64`) live in the
header sub-line, not next to each value.  The device and OpenACC
`Time` columns are per-launch times - the cudaEvent window covers
`Launches` back-to-back launches, then is divided through.

The `Speedup vs real64` column is `time_real64 / time_this_row`, so it
is `1.000` on the `real(real64)` row by construction.  Values above
`1.000` are faster than native fp64; below `1.000` are slower.

The `Launches` column makes the auto-tuner's choice transparent: faster
kernels get more launches per window so every measurement runs for at
least the per-target auto-tune target - `MIN_BATCH_MS_DEV` (default
`500` ms) for the CUDA Fortran table, `MIN_BATCH_MS_ACC` (default
`100` ms) for the OpenACC table - well above launch-overhead noise.
Set `KERNEL_LAUNCHES=<positive int>` on the command line to disable
auto-tuning and use a fixed batch count for every kernel.
