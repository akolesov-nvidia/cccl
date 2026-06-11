! ============================================================================
!  FPMP OpenACC Fortran Benchmark — Accuracy Sum
!
!  Purpose
!  -------
!  Companion to `fortran_sum_host.f90` (host CPU) and `fortran_sum_device.cuf`
!  (CUDA Fortran device): same accuracy-and-throughput experiment, same
!  four accumulator types, same shared single-precision input array, but
!  with the GPU reduction kernels expressed as `!$acc parallel loop`
!  regions instead of `attributes(global)` kernels.  The point is that
!  **OpenACC is the natural offload path for many existing Fortran codes**:
!  users do not have to rewrite kernels as CUDA Fortran subroutines, they
!  can decorate their existing host loops with `!$acc parallel loop` and
!  call FPMP arithmetic through the same `fpmp_api` module they
!  already use on the host.
!
!    Accumulator           Hardware lane     Notes
!    -------------------   ---------------   --------------------------------
!    real(real32)           FP32             native single
!    real(real64)           FP64             native double
!    type(fp32mp2)          FP32 (×2)        float-float; ~46-bit mantissa
!    type(fp64mp2)          FP64 (×2)        double-double; ~106-bit mantissa
!
!  Algorithm: per-thread partials (uniform across all four rows)
!  -------------------------------------------------------------
!  All four kernels share one skeleton: NTH = NUM_BLOCKS * BLOCK_SIZE
!  OpenACC threads each accumulate a strided slice of the input into a
!  thread-private register, then write one partial accumulator into a
!  device array; the host folds NTH partials in fp64 after the timed
!  loop.  Only the inner accumulator TYPE changes between rows, so the
!  `Speedup vs real64` column reflects pure inner-arithmetic cost rather
!  than reduction-pattern asymmetry.
!
!  The benchmark deliberately does NOT use `!$acc parallel loop
!  reduction(+:acc)` for the native rows even though §2.5.15 permits it
!  on intrinsic types.  The FPMP rows cannot use it (the spec forbids
!  `reduction()` on user-defined types — `NVFORTRAN-S-0155`).  Mixing
!  `reduction(+:acc)` for the natives with per-thread partials for the
!  FPMP types puts the four rows on different algorithms, and on hardware
!  where nvfortran's reduction-clause lowering is slow at this size the
!  resulting table can show nonsensical orderings — e.g. fp32mp2 faster
!  than fp32, which is impossible since fp32mp2 runs on the same FP32
!  lanes and uses ~5-10 fp32 ops per add vs 1 op for native.  The
!  uniform per-thread-partials algorithm here matches the CUDA Fortran
!  benchmark in `fortran_sum_device.cuf`, so the two benchmarks' speedup
!  columns are directly comparable, modulo "OpenACC kernel skeleton vs
!  hand-tuned CUDA Fortran kernel" overhead.
!
!  Reference
!  ---------
!  The reference sum is the result of the fp64mp2 OpenACC kernel
!  (~106-bit mantissa).  Each other kernel's error is reported relative to
!  this value; the fp64mp2 row therefore reports zero error by construction.
!
!  Per-launch timing: each timed `system_clock` window covers a batch
!  of back-to-back launches of the same kernel so even a sub-millisecond
!  kernel produces a measurement well above timer noise.  The reported
!  time is `elapsed / launches` per launch, and the throughput is
!  computed from that per-launch time.  This matches the CUDA Fortran
!  benchmark's batched-timing methodology so the two GPU paths are
!  directly comparable; the only difference is the timer source
!  (`system_clock` here, `cudaEvent*` there), which is irrelevant to
!  the measurement because we bracket the GPU work with `!$acc wait(0)`
!  -- see below.  The previous `cpu_time + !$acc wait` strategy
!  measured a single launch per rep, which on nvfortran's OpenACC
!  runtime is dominated by ~1 ms of per-`!$acc parallel`
!  setup/teardown overhead -- that overhead is constant per launch,
!  so light kernels (fp32, fp64, fp32mp2) reported wildly inflated
!  times while only fp64mp2 (whose ~250 us of real GPU work amortises
!  the overhead) looked plausible, producing nonsensical orderings
!  like "fp64mp2 is the fastest".  Batching the timed window across
!  many launches removes that overhead from the measurement.
!
!  The kernels use `async(0)` so successive launches inside a timed
!  window queue back-to-back on CUDA stream 0.  Crucially, we issue an
!  explicit `!$acc wait(0)` between the launch loop and the closing
!  `call system_clock(c_stop)` -- without it, OpenACC's async launches
!  return to the host immediately and the second clock read fires
!  before the GPU finishes the batch, so the measured elapsed time
!  reflects CPU-dispatch latency instead of GPU-execution time (which
!  on small calibration batches under-reports per-launch time by
!  30-100x and makes the auto-tuner pick 100-1000x too many launches).
!  With the wait, the `system_clock` interval brackets the whole queue
!  and measures GPU-only time, independent of the CPU dispatch latency
!  between launches.
!
!  Why not `cudaEvent*` like the CUDA Fortran companion?  This
!  benchmark is intentionally PURE OpenACC: `use cudafor` would force
!  `-cuda` at compile time, which activates the `!@cuf
!  attributes(device) &` directives in the unified `fpmp_api.f90` --
!  every FPMP wrapper then ends up with both `attributes(device)` and
!  `!$acc routine seq` (the "dual-attribute state").  In that state
!  nvfortran refuses to resolve the host-side `fpmp_to_double` generic
!  against host data after `!$acc update self` (NVFORTRAN-S-0155),
!  because none of the specific procedures has a host-callable form
!  anymore.  `system_clock` is a Fortran intrinsic and needs no CUDA
!  Fortran support, so the OpenACC path keeps `!@cuf` dormant.
!
!  Auto-tuning: by default (KERNEL_LAUNCHES=0) the number of launches
!  per timed window is chosen PER KERNEL so each batch runs for at least
!  MIN_BATCH_MS milliseconds.  Fast kernels get thousands of launches
!  per window, slow ones get only a few hundred.  Set KERNEL_LAUNCHES
!  to a positive value to disable auto-tuning and use a fixed batch
!  count.
!
!  Compile-time controls (override on the make command line):
!      -DN_ELEMENTS=<int>      default: 10000000
!      -DN_REPS=<int>          default: 5
!      -DSCALE_BITS=<int>      default: 0  (max input ≈ 2^0)
!      -DBLOCK_SIZE=<int>      default: 256
!      -DNUM_BLOCKS=<int>      default: 1024
!      -DMIN_BATCH_MS=<int>    default: 100 (≥ 0.1 s per timed window)
!      -DKERNEL_LAUNCHES=<int> default: 0   (0 = auto-tune to MIN_BATCH_MS;
!                                            >0 = fixed override)
!      -DMAX_KERNEL_LAUNCHES=<int> default: 1000000 (auto-tune safety cap)
!      -DCALIB_LAUNCHES=<int>  default: 32  (warm-up + calibration batch
!                                            size; raise for more stable
!                                            auto-tune at the cost of
!                                            slightly longer startup)
! ============================================================================

#ifndef N_ELEMENTS
#define N_ELEMENTS 10000000
#endif

#ifndef N_REPS
#define N_REPS 5
#endif

#ifndef SCALE_BITS
#define SCALE_BITS 0
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 1024
#endif

#ifndef MIN_BATCH_MS
#define MIN_BATCH_MS 100
#endif

#ifndef KERNEL_LAUNCHES
#define KERNEL_LAUNCHES 0
#endif

#ifndef MAX_KERNEL_LAUNCHES
#define MAX_KERNEL_LAUNCHES 1000000
#endif

#ifndef CALIB_LAUNCHES
#define CALIB_LAUNCHES 32
#endif


program fortran_sum_acc
  use, intrinsic :: iso_fortran_env
  use, intrinsic :: iso_c_binding
  use fpmp_api
  implicit none

  ! NOTE: this benchmark is intentionally PURE OpenACC -- it does not
  ! `use cudafor` and is not compiled with `-cuda`.  Mixing the two on the
  ! OpenACC offload path activates the `!@cuf attributes(device) &`
  ! directives inside the unified `fpmp_api.f90`, which gives every
  ! FPMP wrapper *both* `attributes(device)` and `!$acc routine seq`
  ! (the "dual-attribute state").  In that state nvfortran refuses to
  ! resolve the host-side `fpmp_to_double` generic against host data
  ! (NVFORTRAN-S-0155 "Could not resolve generic procedure"), because
  ! none of the specific procedures has a host-callable form anymore.
  ! Timing therefore goes through the Fortran intrinsic `system_clock`
  ! (int64 counter, nanosecond resolution on Linux) instead of
  ! `cudaEvent*`, and `-cuda` only appears at the final link step where
  ! it is needed to pull `libcudart` in for the FPMP runtime objects.

  integer, parameter :: N        = N_ELEMENTS
  integer, parameter :: REPS     = N_REPS
  integer, parameter :: NB       = NUM_BLOCKS
  integer, parameter :: BLK      = BLOCK_SIZE
  integer, parameter :: NTH      = NB * BLK              ! one OpenACC thread per slot
  integer, parameter :: KL_FIXED = KERNEL_LAUNCHES       ! 0 = auto
  integer, parameter :: KL_MAX   = MAX_KERNEL_LAUNCHES
  integer, parameter :: KL_CALIB = CALIB_LAUNCHES        ! warm-up + calibration batch
  real(real64), parameter :: TARGET_BATCH_S = real(MIN_BATCH_MS, real64) * 1.0e-3_real64

  ! Golden-ratio LDS multiplier (φ - 1).
  real(real64), parameter :: PHI   = 0.6180339887498948_real64
  real(real64), parameter :: SCALE = 2.0_real64**SCALE_BITS

  real(real32),  allocatable :: x(:)
  ! One per-thread partials buffer per accumulator type.  All four rows
  ! use the same per-thread-partials skeleton so the kernels differ only
  ! in their inner accumulator TYPE; that keeps the speedup column an
  ! apples-to-apples arithmetic comparison.
  real(real32),  allocatable :: partials_f32(:)
  real(real64),  allocatable :: partials_f64(:)
  type(fp32mp2), allocatable :: partials_ff(:)
  type(fp64mp2), allocatable :: partials_dd(:)

  real(real64) :: ref_sum, u
  integer      :: i

  ! Timing state.  `system_clock` with an int64 count gives nanosecond
  ! resolution on Linux (`c_rate` is the ticks-per-second the kernel
  ! reports for `CLOCK_MONOTONIC`).  We compute elapsed seconds as
  !     real(c_stop - c_start, real64) / real(c_rate, real64)
  ! and drive both the calibration baseline and per-launch min directly
  ! from seconds.
  integer(int64) :: c_start, c_stop, c_rate
  real(real64)   :: elapsed_s

  ! Per-accumulator results, indexed by `kind + 1`
  ! (1 = fp32, 2 = fp64, 3 = fp32mp2, 4 = fp64mp2).  run_one() fills these
  ! and the table is printed in one pass at the end; that lets every row
  ! reference dt_arr(2) (the real64 row) for the "Speedup vs real64"
  ! column without forcing the real64 kernel to run first.
  real(real64) :: got_arr(4), dt_arr(4)
  integer      :: kl_arr(4)

  allocate(x(N), &
           partials_f32(NTH), partials_f64(NTH), &
           partials_ff(NTH),  partials_dd(NTH))

  do i = 1, N
    u    = modulo(real(i, real64) * PHI, 1.0_real64)
    x(i) = real(u * SCALE, real32)
  end do

  ! Probe the system_clock tick rate once.  `c_rate` is invariant across
  ! calls on Linux so we cache it here and reuse it for every elapsed-time
  ! computation in `run_one`.
  call system_clock(c_start, c_rate)

  ! Single data region: input lives on the device for the whole run;
  ! every accumulator type has its own per-thread partials buffer as
  ! kernel-private scratch (NTH * sizeof(type) bytes each, ~9 MB
  ! combined at the default geometry — 1+2+2+4 MB for fp32/fp64/fp32mp2/
  ! fp64mp2, negligible next to the 40 MB input array).  The host folds
  ! the partials in fp64 once after each row's timed loop completes.
  !$acc data copyin(x) create(partials_f32, partials_f64, partials_ff, partials_dd)

  ! ---- Reference run: one fp64mp2 reduction, untimed --------------------
  call reduce_fp64mp2_kernel()
  !$acc wait(0)
  !$acc update self(partials_dd)
  ref_sum = 0.0_real64
  do i = 1, NTH
    ref_sum = ref_sum + fpmp_to_double(partials_dd(i))
  end do

  print '(A)',         '======================================================================================'
  print '(A)',         '  FPMP OpenACC Fortran Benchmark — Accuracy Sum'
  print '(A)',         '======================================================================================'
  print '(A,I0)',      '  N            = ', N
  print '(A,I0,A)',    '  threads      = ', NTH, &
                       ' (NUM_BLOCKS * BLOCK_SIZE; one partial per thread)'
  print '(A,I0)',      '  repetitions  = ', REPS
  if (KL_FIXED > 0) then
    print '(A,I0,A)',  '  launches/rep = ', KL_FIXED, ' (fixed; per-launch time = elapsed / launches)'
  else
    print '(A,I0,A)',  '  launches/rep = auto-tune to ≥ ', MIN_BATCH_MS, &
                       ' ms per timed window (per-kernel)'
  end if
  print '(A,I0,A,ES10.3)', '  scale        = 2^', SCALE_BITS, ' = ', SCALE
  print '(A)',         '  inputs       = pseudo-random fp32 in [0, scale) (golden-ratio LDS)'
  print '(A)',         '  reference    = fp64mp2 OpenACC reduction (~106-bit mantissa)'
  print '(A,ES22.15)', '  reference    = ', ref_sum
  print '(A)',         '--------------------------------------------------------------------------------------'
  print '(A)',         '  Accumulator       |Error|     Rel.Err  Throughput        Time     Speedup  Launches'
  print '(A)',         '                                         (Melem/s)         (s)   vs real64'
  print '(A)',         '  -------------  ----------  ----------  ----------  ----------  ----------  --------'

  call run_one('real(real32) ', 0)
  call run_one('real(real64) ', 1)
  call run_one('type(fp32mp2)', 2)
  call run_one('type(fp64mp2)', 3)

  call print_table()

  !$acc end data

  print '(A)',         '======================================================================================'

  deallocate(x, partials_f32, partials_f64, partials_ff, partials_dd)

contains

  ! --------------------------------------------------------------------------
  ! Reduction kernels: per-thread partials, uniform shape across all four
  ! accumulator types.  Each OpenACC thread runs a grid-stride loop over
  ! its slice of `x(:)` and writes one partial accumulator to the
  ! corresponding `partials_*` array.  The (small) NTH -> 1 final fold
  ! happens on the host in fp64, outside the timed window in each run_*.
  !
  ! Every kernel uses `async(0)` so successive launches in a timed window
  ! queue back-to-back on CUDA stream 0 without per-launch CPU sync;
  ! `!$acc wait(0)` before the closing `system_clock` drains queue 0 so
  ! the elapsed-time interval brackets pure GPU time across the batch.
  !
  ! We deliberately do NOT use `!$acc parallel loop reduction(+:acc)` for
  ! the native rows: that would put fp32/fp64 on a different kernel
  ! skeleton than the FPMP rows (forced to per-thread-partials because
  ! OpenACC §2.5.15 forbids `reduction()` on user-defined types) and
  ! make the speedup column compare two different algorithms rather
  ! than two different accumulator types.
  !
  ! Why the explicit `gang vector vector_length(BLK)` + `loop seq`:
  ! with default scheduling nvfortran auto-vectorizes the inner sum
  ! loop on the *native* rows (it sees a textbook scalar reduction
  ! pattern) but cannot do the same on the FPMP rows (where the `+` is
  ! an opaque function call into the FPMP runtime).  That asymmetry
  ! makes the native kernels much slower than the FPMP ones at this
  ! geometry and produces nonsensical orderings in the speedup column
  ! (e.g. fp32mp2 faster than fp32).  Forcing `gang vector` on the
  ! outer loop (one OpenACC thread per gid) and `seq` on the inner
  ! loop (each thread accumulates sequentially) gives every kernel the
  ! same schedule the FPMP rows naturally get and restores apples-to-
  ! apples timing.  vector_length(BLK) matches the CUDA Fortran
  ! benchmark's BLOCK_SIZE so the two benchmarks run the same
  ! 1024 gangs * 256 threads = 262144-thread geometry.
  !
  ! Inner-loop surface: every kernel uses the natural Fortran form
  ! `local = local + x(i)`.  That is the same surface the sibling host
  ! benchmark (`fortran_sum_host.f90`, *"the inner loop is `acc = acc +
  ! x(i)` for every accumulator type, with no explicit casts"*) and the
  ! sibling CUDA Fortran benchmark (`fortran_sum_device.cuf`, lines
  ! 145/177/209/241) use, so the three backends are an apples-to-apples
  ! like-for-like comparison.  The bindings provide a
  ! `<fp*mp2> + real(c_float)` mixed-type overload, so user-visible code
  ! stays cast-free in every row, including the fp64mp2 row (the
  ! fp32->fp64 promotion happens inside the overload, one hardware
  ! instruction).
  !
  ! Note that on nvfortran 25.11 `-acc=gpu` this natural-`+` surface
  ! exposes the OpenACC backend to two known codegen bugs filed
  ! upstream:
  !   issues/nvfortran_openacc_indexed_global_array_store_marshalling/
  !   issues/nvfortran_openacc_chained_derived_type_return/
  ! and `-Minfo=accel,inline` also reports that the operator wrapper
  ! does not inline through into the kernel body (which a working
  ! CUDA Fortran build does via `attributes(device)` LTO), so the
  ! OpenACC row is currently slower than its CUDA Fortran twin even
  ! when results happen to come out correct.  The whole point of
  ! keeping this benchmark in the tree is to track exactly that gap:
  ! once nvfortran ships a fix the OpenACC row should match the
  ! CUDA Fortran row.  This is why `benchmarks/fortran_sum/Makefile`
  ! defaults `WITH_ACC=0` -- the benchmark is opt-in until the
  ! filed bugs are fixed.
  ! --------------------------------------------------------------------------

  subroutine reduce_fp32_kernel()
    real(real32) :: local
    integer      :: gid, i
    !$acc parallel loop gang vector vector_length(BLK) async(0) present(x, partials_f32) private(local)
    do gid = 1, NTH
      local = 0.0_real32
      !$acc loop seq
      do i = gid, N, NTH
        local = local + x(i)
      end do
      partials_f32(gid) = local
    end do
  end subroutine reduce_fp32_kernel

  subroutine reduce_fp64_kernel()
    real(real64) :: local
    integer      :: gid, i
    !$acc parallel loop gang vector vector_length(BLK) async(0) present(x, partials_f64) private(local)
    do gid = 1, NTH
      local = 0.0_real64
      !$acc loop seq
      do i = gid, N, NTH
        local = local + real(x(i), real64)
      end do
      partials_f64(gid) = local
    end do
  end subroutine reduce_fp64_kernel

  subroutine reduce_fp32mp2_kernel()
    type(fp32mp2) :: local
    integer       :: gid, i
    !$acc parallel loop gang vector vector_length(BLK) async(0) present(x, partials_ff) private(local)
    do gid = 1, NTH
      local%hi = 0.0_c_float
      local%lo = 0.0_c_float
      !$acc loop seq
      do i = gid, N, NTH
        ! Mixed-type operator(+) -- same surface as the host and device
        ! benchmarks.  x(i) is real(real32) ≡ real(c_float); the bindings
        ! provide a `fp32mp2 + real(c_float)` overload, so no explicit
        ! cast or intermediate fp32mp2 temporary is required in the user
        ! code.
        local = local + x(i)
      end do
      partials_ff(gid) = local
    end do
  end subroutine reduce_fp32mp2_kernel

  subroutine reduce_fp64mp2_kernel()
    type(fp64mp2) :: local
    integer       :: gid, i
    !$acc parallel loop gang vector vector_length(BLK) async(0) present(x, partials_dd) private(local)
    do gid = 1, NTH
      local%hi = 0.0_c_double
      local%lo = 0.0_c_double
      !$acc loop seq
      do i = gid, N, NTH
        ! Mixed-type operator(+) -- same surface as the host and device
        ! benchmarks.  The bindings provide a `fp64mp2 + real(c_float)`
        ! overload, so the fp32->fp64 promotion happens implicitly
        ! inside the overload (one hardware instruction) and the user
        ! code stays cast-free across all four accumulator rows.
        local = local + x(i)
      end do
      partials_dd(gid) = local
    end do
  end subroutine reduce_fp64mp2_kernel

  ! --------------------------------------------------------------------------
  ! Driver: launch the chosen kernel REPS times with `system_clock`
  ! batched timing, auto-tuned exactly like fortran_sum_device.cuf.
  ! Each timed window covers `kl` back-to-back async(0) launches of the
  ! same kernel; `system_clock` brackets pure GPU time across the
  ! batch and the reported per-launch time is `elapsed_s / kl`.
  !
  ! IMPORTANT -- the explicit `!$acc wait(0)` between the launch loop
  ! and the closing `call system_clock(c_stop)` is *not* cosmetic.
  ! OpenACC `async(0)` launches return to the host immediately, so a
  ! `system_clock` read taken right after the loop fires while the
  ! GPU is still processing the queue; the measured elapsed time is
  ! then the *CPU dispatch latency*, not the GPU execution time --
  ! which on small calibration batches can be 30-100x shorter than the
  ! actual kernel time, tricking the auto-tuner into picking
  ! 100-1000x too many launches.  The wait drains async queue 0 so
  ! the second clock read happens after the GPU work actually
  ! finishes.  (The long timed windows happened to "self-correct" via
  ! runtime backpressure once `kl` was large enough to fill the CUDA
  ! stream queue; the wait makes the timing correct for any `kl`.)
  !
  ! `kl` is auto-tuned per kernel by default (KERNEL_LAUNCHES=0): a
  ! `CALIB_LAUNCHES`-launch warm-up settles caches / kernel resident
  ! state, then a `CALIB_LAUNCHES`-launch calibration batch is timed
  ! and divided through to get a steady-state per-launch baseline.
  ! That baseline drives `kl = ceil(MIN_BATCH_MS / baseline_s)`,
  ! clamped to [1, MAX_KERNEL_LAUNCHES], so each timed window runs for
  ! at least MIN_BATCH_MS milliseconds.  This amortises both (a) the
  ! `system_clock` timer noise floor and (b) the ~1 ms of OpenACC-runtime
  ! per-`!$acc parallel` overhead (the latter is what made the previous
  ! single-shot `cpu_time + !$acc wait` strategy unusable on this
  ! workload — see file header).
  !
  ! A multi-launch calibration is used (vs the older single-shot
  ! warm-up + single-launch baseline) so the first-launch transients
  ! don't trick the auto-tuner into a too-small `kl` -- on the CUDA
  ! Fortran companion benchmark the single-shot baseline under-shot
  ! the MIN_BATCH_MS target by ~2x on the FPMP rows.
  !
  ! Setting KERNEL_LAUNCHES to a positive value disables auto-tuning
  ! and uses that fixed batch count for every kernel.
  ! --------------------------------------------------------------------------

  subroutine run_one(label, kind)
    ! `label` is unused inside this routine (the table printer owns the
    ! row labels now), but is retained as a dummy argument so the call
    ! sites in the main program stay self-documenting:
    !     call run_one('real(real32) ', 0)
    !     call run_one('real(real64) ', 1)
    !     ...
    character(len=*), intent(in) :: label
    integer,          intent(in) :: kind   ! 0=r32, 1=r64, 2=fp32mp2, 3=fp64mp2

    integer      :: rep, j, kl
    real(real64) :: total, dt_per_launch, baseline_s, dt_min

    ! ---- 1. Pick the per-window launch count ----
    if (KL_FIXED > 0) then
      kl = KL_FIXED
    else
      ! Warm-up: a batch of KL_CALIB async(0) launches.  A single
      ! warm-up launch is not enough on this workload -- the first
      ! few launches still pay setup overheads that bias the baseline.
      do j = 1, KL_CALIB
        call dispatch_kernel(kind)
      end do
      !$acc wait(0)

      ! Multi-launch calibration: KL_CALIB back-to-back launches, with
      ! `baseline_s = elapsed / KL_CALIB`.  Averages out timer noise
      ! and first-launch transients, yielding a baseline within a few
      ! percent of steady-state for every kernel.  The `!$acc wait(0)`
      ! between the launches and the second clock read is required: it
      ! drains async queue 0 so the stop count is taken *after* the GPU
      ! finishes, not as soon as the CPU has enqueued the work.  See
      ! the comment block above for the full rationale.
      call system_clock(c_start)
      do j = 1, KL_CALIB
        call dispatch_kernel(kind)
      end do
      !$acc wait(0)
      call system_clock(c_stop)
      elapsed_s  = real(c_stop - c_start, real64) / real(c_rate, real64)

      baseline_s = max(elapsed_s / real(KL_CALIB, real64), 1.0e-9_real64)
      kl         = int(ceiling(TARGET_BATCH_S / baseline_s))
      kl         = max(1, min(KL_MAX, kl))
    end if

    ! ---- 2. Time `REPS` windows, each batching `kl` async(0) launches ----
    ! Same `!$acc wait(0)` discipline as the calibration above: drain
    ! async queue 0 before the second clock read so `system_clock` brackets
    ! real GPU time and not just the CPU-dispatch latency.
    dt_min = huge(dt_min)
    do rep = 1, REPS
      call system_clock(c_start)
      do j = 1, kl
        call dispatch_kernel(kind)
      end do
      !$acc wait(0)
      call system_clock(c_stop)
      elapsed_s     = real(c_stop - c_start, real64) / real(c_rate, real64)
      dt_per_launch = elapsed_s / real(kl, real64)
      dt_min        = min(dt_min, dt_per_launch)
    end do

    ! ---- 3. Read back partials for this kind and fold on host in fp64 ----
    !$acc wait(0)
    total = 0.0_real64
    select case (kind)
    case (0)
      !$acc update self(partials_f32)
      do j = 1, NTH
        total = total + real(partials_f32(j), real64)
      end do
    case (1)
      !$acc update self(partials_f64)
      do j = 1, NTH
        total = total + partials_f64(j)
      end do
    case (2)
      !$acc update self(partials_ff)
      do j = 1, NTH
        total = total + fpmp_to_double(partials_ff(j))
      end do
    case (3)
      !$acc update self(partials_dd)
      do j = 1, NTH
        total = total + fpmp_to_double(partials_dd(j))
      end do
    end select

    got_arr(kind + 1) = total
    dt_arr(kind + 1)  = dt_min
    kl_arr(kind + 1)  = kl
  end subroutine run_one

  ! Dispatch one launch of the chosen accumulator kernel.  Pulled into its
  ! own routine so calibration and timing share the exact same launch path.
  subroutine dispatch_kernel(kind)
    integer, intent(in) :: kind
    select case (kind)
    case (0); call reduce_fp32_kernel()
    case (1); call reduce_fp64_kernel()
    case (2); call reduce_fp32mp2_kernel()
    case (3); call reduce_fp64mp2_kernel()
    end select
  end subroutine dispatch_kernel

  ! Print all four rows in one pass, including the "Speedup vs real64"
  ! column (time_real64 / time_this_row) and the auto-tuned per-window
  ! launch count.  Splitting print from time collection lets every row
  ! reference dt_arr(2) (the real64 row) even though run_one() for the
  ! real32 kernel completes first.
  subroutine print_table()
    call print_row('real(real32) ', got_arr(1), dt_arr(1), kl_arr(1))
    call print_row('real(real64) ', got_arr(2), dt_arr(2), kl_arr(2))
    call print_row('type(fp32mp2)', got_arr(3), dt_arr(3), kl_arr(3))
    call print_row('type(fp64mp2)', got_arr(4), dt_arr(4), kl_arr(4))
  end subroutine print_table

  subroutine print_row(label, got, dt, kl)
    character(len=*), intent(in) :: label
    real(real64),     intent(in) :: got, dt
    integer,          intent(in) :: kl
    real(real64) :: err, rel, melem_per_s, speedup
    err         = abs(got - ref_sum)
    rel         = err / max(abs(ref_sum), 1.0e-300_real64)
    melem_per_s = real(N, real64) / max(dt, 1.0e-12_real64) * 1.0e-6_real64
    speedup     = dt_arr(2) / max(dt, 1.0e-12_real64)
    print '(2X,A,2X,ES10.3,2X,ES10.3,2X,ES10.3,2X,ES10.3,2X,F10.3,2X,I8)', &
      label, err, rel, melem_per_s, dt, speedup, kl
  end subroutine print_row

end program fortran_sum_acc
