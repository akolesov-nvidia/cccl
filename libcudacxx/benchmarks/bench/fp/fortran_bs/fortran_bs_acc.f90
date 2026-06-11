! ============================================================================
!  FPMP OpenACC Fortran Benchmark — Black-Scholes Option Pricing
!
!  Purpose
!  -------
!  OpenACC counterpart of `fortran_bs_device.cuf`.  Same workload, same
!  reference, same auto-tuned batched timing strategy.  Compares the two
!  arithmetic types the user actually compares for Fortran work:
!
!    Type                  Hardware lane     Notes
!    -------------------   ---------------   ------------------------------
!    real(real64)           FP64             native double (baseline)
!    type(fp32mp2)          FP32 (×2)        float-float; ~46-bit mantissa
!
!  Both rows go through the same OpenACC kernel skeleton (`!$acc parallel
!  loop gang vector vector_length(BLK)`) with one option per OpenACC
!  thread; the only difference between the rows is the scalar TYPE of
!  every variable in the per-option `bs_price` computation.  This keeps
!  the speedup column an apples-to-apples arithmetic comparison.
!
!  End-user arithmetic in the OpenACC region
!  -----------------------------------------
!  The fp32mp2 kernel body is written with NATURAL Fortran syntax:
!
!      sqrtT  = sqrt(Ti)
!      d1     = (log(Si / Ki) + (R_FF - Q_FF + HALF_FF * sigi * sigi) * Ti) / vsqrtT
!      Nd1    = HALF_FF * erfc(-d1 * SQRT_1_2_FF)
!      price  = Si * disc_q * Nd1 - Ki * disc_r * Nd2
!      prices_ff(i) = price
!
!  i.e. exactly the same `+`/`-`/`*`/`/`/`sqrt`/`log`/`exp`/`erfc` syntax a
!  user would write for `real(real64)`.  N(x) is evaluated through erfc
!  rather than via `0.5*(1+erf(...))` to avoid catastrophic cancellation
!  in the deep-OTM tails (matches the C++ `benchmarks/bs/` benchmark).
!  The fp32mp2 overloads come from `use fpmp_api` (no
!  `attributes(device)`, just `!$acc routine seq` on every wrapper) and
!  the **entire** OpenACC build is compiled pure-OpenACC -- `-acc=gpu`
!  only, no `-cuda` on either nvfortran invocation.  See Makefile for
!  the build commands.
!
!  Two nvfortran 25.11 codegen issues are sidestepped here, and a third
!  one is STILL ACTIVE (causes the OpenACC build of this benchmark to
!  segfault at runtime; see Makefile `WITH_ACC` opt-in):
!
!    1.  Mixed `-cuda -acc=gpu` ABI:  earlier iterations that passed
!        `-cuda` (for `use cudafor` timing) caused nvfortran to emit
!        FPMP procedures in an ABI state that crashed at runtime with
!        `CUDA_ERROR_ILLEGAL_ADDRESS` on the first FPMP function-call
!        inside an `!$acc parallel loop`.  Pure-OpenACC mode (no
!        `-cuda` on any compile step) bypasses that codegen path
!        entirely.
!
!    2.  By-value derived-type return into global-array slot:  if a
!        function returning `type(fp32mp2)` by value writes its
!        result DIRECTLY into an indexed array slot inside an OpenACC
!        compute region -- e.g.
!
!            prices_ff(i) = fp32mp2(7.0d0)                    ! BUG
!            prices_ff(i) = bs_price_fp32mp2(Si, Ki, ...)     ! BUG
!            prices_ff(i) = Si * disc_q * Nd1 - Ki * disc_r * Nd2   ! BUG
!
!        nvfortran's marshalling of the returned struct corrupts the
!        result (every `%lo` field reads back as the `%hi` source
!        value, producing a deterministic 2x error on one component
!        of the returned struct).  The discipline in the kernel below
!        is to stage every derived-type-valued expression through a
!        named PRIVATE scalar (declared in the `!$acc parallel loop
!        private(...)` clause) before the final array store:
!
!            price        = Si * disc_q * Nd1 - Ki * disc_r * Nd2   ! OK (private)
!            prices_ff(i) = price                                    ! OK (priv->arr copy)
!
!        Standalone reproducer of (2):  ../../nvfortran_bug_temp/
!        expert_repro/.  See its `simple_loop_via_priv.f90` PASS vs
!        `simple_loop.f90` FAIL contrast for the cleanest demonstration.
!
!    3.  *** NOT YET ISOLATED *** -- some additional codegen path in
!        the chained-derived-type-operation kernel below.  Even with
!        the discipline from (2) applied EVERYWHERE (every named
!        intermediate listed in `private(...)`; every `+`/`-`/`*`/
!        `/`/`sqrt`/`log`/`exp`/`erfc` result assigned to a private
!        scalar; the ONLY global-array store is the final
!        `prices_ff(i) = price` intrinsic copy from a private
!        scalar), `bs_kernel_fp32mp2` still SEGFAULTS at runtime on
!        the first OpenACC kernel launch on nvfortran 25.11.  The
!        binary builds cleanly with no diagnostic; the crash happens
!        on dispatch.  We have not yet isolated this third failure
!        mode into a minimal reproducer; the kernel chains ~15
!        named `type(fp32mp2)` intermediates through 4
!        transcendentals (`sqrt`, `log`, `exp`, `erfc`) plus
!        operator-overloaded arithmetic, and one of those codegen
!        paths is still broken inside the chain.  This is the
!        reason the OpenACC build of fortran_bs is gated behind
!        `WITH_ACC=1` in the Makefile -- the binary is reproducibly
!        buildable but not runnable on affected nvfortran versions.
!
!  Workload
!  --------
!  Each of the N options has randomized parameters:
!     S, K  in [10, 200]   (spot, strike)
!     T     in [0.05, 5.0] (time to maturity)
!     sigma in [0.05, 1.0] (volatility)
!  The risk-free rate r and dividend yield q are fixed globally to
!  r = 0.02, q = 0.01.  All options are CALLS (matching bs.cpp's default).
!
!  Inputs are deterministic pseudo-random fp64 values derived from a
!  golden-ratio low-discrepancy sequence on each parameter so the
!  benchmark is fully reproducible without an RNG seed.
!
!  Reference
!  ---------
!  A per-option reference price array is computed ONCE on the host in
!  fp64 using the Fortran intrinsic `sqrt`, `log`, `exp`, and `erfc`.
!  Every device row's max / average relative error is reported against
!  this reference.
!
!  Per-launch timing
!  -----------------
!  Each timed window covers a batch of back-to-back launches of the
!  same kernel.  We use Fortran's intrinsic `system_clock` (not
!  cudaEvent) so the whole OpenACC build is pure-OpenACC — no
!  `use cudafor`, no `-cuda` on either nvfortran invocation.  This
!  matters: when `-cuda` was on either the bindings TU or the main
!  program TU, nvfortran emitted FPMP procedures (any function
!  returning `type(fp32mp2)` by value) in an ABI that crashed at
!  runtime inside `!$acc parallel loop`.  Going pure-OpenACC bypasses
!  that codegen path entirely.  Crucially, `!$acc wait(0)` is issued
!  before the closing `system_clock` so the timer captures real GPU
!  execution time rather than the CPU-dispatch latency on the async
!  `!$acc parallel loop async(0)` launches.
!
!  Auto-tuning: by default (KERNEL_LAUNCHES=0) the number of launches
!  per timed window is chosen PER KERNEL so each batch runs for at
!  least MIN_BATCH_MS milliseconds.  Set KERNEL_LAUNCHES to a positive
!  value to disable auto-tuning and use a fixed batch count.
!
!  Compile-time controls (override on the make command line):
!      -DNUM_OPTIONS=<int>     default: 1048576 (1M options)
!      -DN_REPS=<int>          default: 5
!      -DBLOCK_SIZE=<int>      default: 256
!      -DNUM_BLOCKS=<int>      default: 2048
!      -DMIN_BATCH_MS=<int>    default: 100 (≥ 0.1 s per timed window)
!      -DKERNEL_LAUNCHES=<int> default: 0   (0 = auto-tune to MIN_BATCH_MS;
!                                            >0 = fixed override)
!      -DMAX_KERNEL_LAUNCHES=<int> default: 1000000 (auto-tune safety cap)
!      -DCALIB_LAUNCHES=<int>  default: 32  (warm-up + calibration batch
!                                            size; raise for more stable
!                                            auto-tune at the cost of
!                                            slightly longer startup)
! ============================================================================

#ifndef NUM_OPTIONS
#define NUM_OPTIONS 1048576
#endif

#ifndef N_REPS
#define N_REPS 5
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 2048
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


program fortran_bs_acc
  use, intrinsic :: iso_fortran_env
  use, intrinsic :: iso_c_binding
  use fpmp_api
  implicit none

  integer, parameter :: N        = NUM_OPTIONS
  integer, parameter :: REPS     = N_REPS
  integer, parameter :: NB       = NUM_BLOCKS
  integer, parameter :: BLK      = BLOCK_SIZE
  integer, parameter :: NTH      = NB * BLK             ! one OpenACC thread per slot
  integer, parameter :: KL_FIXED = KERNEL_LAUNCHES      ! 0 = auto
  integer, parameter :: KL_MAX   = MAX_KERNEL_LAUNCHES
  integer, parameter :: KL_CALIB = CALIB_LAUNCHES       ! warm-up + calibration batch
  real(real64), parameter :: TARGET_BATCH_S = real(MIN_BATCH_MS, real64) * 1.0e-3_real64

  ! Fixed BS model parameters.
  real(real64), parameter :: R_D    = 0.02_real64
  real(real64), parameter :: Q_D    = 0.01_real64
  real(real64), parameter :: HALF_D = 0.5_real64
  ! 1/sqrt(2) to fp64 precision; used inside N(x) = 0.5*erfc(-x*sqrt(1/2)).
  real(real64), parameter :: SQRT_1_2_D = 0.7071067811865475244_real64

  ! Named slots for the fp32mp2 constants array `BS_CONSTS_FF` (see below).
  ! We pack the four model constants into a fixed-size derived-type ARRAY
  ! instead of four scalar variables because nvfortran's OpenACC `copyin`
  ! / `present` clauses do not reliably map SCALAR derived-types to device
  ! memory (the kernel ends up dereferencing a host pointer and crashes
  ! with a segfault).  A small derived-type ARRAY, on the other hand, is
  ! the same code path that `fortran_sum_acc.f90` uses for `partials_ff`
  ! and works correctly.
  integer, parameter :: BS_HALF      = 1
  integer, parameter :: BS_R         = 2
  integer, parameter :: BS_Q         = 3
  integer, parameter :: BS_SQRT_1_2  = 4
  integer, parameter :: BS_NCONSTS   = 4

  ! Golden-ratio LDS multipliers (φ - 1) — four different irrationals so
  ! the four input arrays decorrelate across the LDS.
  real(real64), parameter :: PHI_S  = 0.6180339887498948_real64
  real(real64), parameter :: PHI_K  = 0.7548776662466928_real64
  real(real64), parameter :: PHI_T  = 0.8392867552141612_real64
  real(real64), parameter :: PHI_V  = 0.5436890126920764_real64

  ! Parameter ranges (match bs.cpp).
  real(real64), parameter :: S_LO = 10.0_real64,  S_HI = 200.0_real64
  real(real64), parameter :: K_LO = 10.0_real64,  K_HI = 200.0_real64
  real(real64), parameter :: T_LO = 0.05_real64,  T_HI = 5.0_real64
  real(real64), parameter :: V_LO = 0.05_real64,  V_HI = 1.0_real64

  ! Host arrays.  Inputs live on the host in fp64; per-option reference
  ! prices are computed once at startup via the host intrinsics.  The
  ! fp32mp2 mirrors of the inputs (`*_ff`) are populated on the host
  ! via pure-arithmetic `(hi, lo)` field split (see `dbl_to_ff` below)
  ! and then `copyin`'d to the device.  The fp32mp2 prices array is
  ! device-only scratch (`create`'d, never read on host until after the
  ! timed window, where it is brought back via `!$acc update self`).
  real(real64),  allocatable :: S(:), K(:), Tmat(:), V(:), ref(:)
  real(real64),  allocatable :: prices_fp64(:)
  type(fp32mp2), allocatable :: prices_ff(:)
  type(fp32mp2), allocatable :: S_ff(:), K_ff(:), T_ff(:), V_ff(:)

  ! fp32mp2 model constants -- a small 5-element array populated on the
  ! host (via the same pure-arithmetic `dbl_to_ff` helper) and `copyin`'d
  ! to the device with the rest of the data.  Packed into a derived-type
  ! ARRAY rather than five scalar variables because nvfortran's OpenACC
  ! `copyin` / `present` clauses do not reliably map SCALAR derived
  ! types to device memory.  Array mapping is the same code path
  ! `fortran_sum_acc.f90` uses for `partials_ff` and works correctly.
  type(fp32mp2) :: BS_CONSTS_FF(BS_NCONSTS)

  integer         :: i
  real(real64)    :: u, ref_mean
  integer(int64)  :: clk_rate, clk_start, clk_stop
  real(real64)    :: elapsed_s
  real(real64)    :: dt_min

  ! Per-type results, indexed by `kind + 1` (1 = fp64, 2 = fp32mp2).
  real(real64) :: mean_arr(2), max_re_arr(2), avg_re_arr(2), dt_arr(2)
  integer      :: kl_arr(2)

  allocate(S(N), K(N), Tmat(N), V(N), ref(N))
  allocate(prices_fp64(N), prices_ff(N))
  allocate(S_ff(N), K_ff(N), T_ff(N), V_ff(N))

  ! ---- Generate inputs on the host (deterministic LDS) ----
  do i = 1, N
    u       = modulo(real(i, real64) * PHI_S, 1.0_real64)
    S(i)    = S_LO + u * (S_HI - S_LO)
    u       = modulo(real(i, real64) * PHI_K, 1.0_real64)
    K(i)    = K_LO + u * (K_HI - K_LO)
    u       = modulo(real(i, real64) * PHI_T, 1.0_real64)
    Tmat(i) = T_LO + u * (T_HI - T_LO)
    u       = modulo(real(i, real64) * PHI_V, 1.0_real64)
    V(i)    = V_LO + u * (V_HI - V_LO)
  end do

  ! ---- Build the host fp64 reference (untimed) ----
  ref_mean = 0.0_real64
  do i = 1, N
    ref(i)   = bs_price_host_fp64(S(i), K(i), Tmat(i), V(i))
    ref_mean = ref_mean + ref(i)
  end do
  ref_mean = ref_mean / real(N, real64)

  ! ---- Pre-convert inputs to fp32mp2 on the host (untimed, no FPMP call) ----
  ! `fpmp_api` does expose a host-callable defined assignment
  ! `fp32mp2 = real(real64)` and a `fp32mp2()` constructor, but we
  ! still avoid them: every call dispatches into the FPMP runtime
  ! (`__nv_fp32mp2_from_double`) which is overkill for what is really
  ! just a textbook double-float field split.  Doing the split inline
  ! with pure arithmetic skips the runtime call entirely:
  !
  !      hi = (float)  x          ! leading 24-bit fp32 approximation
  !      lo = (float) (x - hi)    ! residual; |lo| <= 0.5 ulp(hi)
  !
  ! which is the textbook double-float decomposition and matches exactly
  ! what `__nv_fp32mp2_from_double` would compute, but with zero FPMP
  ! calls.  See `dbl_to_ff` below.
  do i = 1, N
    call dbl_to_ff(S(i),    S_ff(i))
    call dbl_to_ff(K(i),    K_ff(i))
    call dbl_to_ff(Tmat(i), T_ff(i))
    call dbl_to_ff(V(i),    V_ff(i))
  end do

  ! ---- Pre-compute fp32mp2 model constants on the host (untimed) ----
  ! Packed into a 4-element ARRAY (not four scalars) because nvfortran's
  ! OpenACC `copyin` / `present` does not reliably map scalar derived
  ! types to device memory (the kernel ends up dereferencing a host
  ! pointer and segfaulting).  Array mapping is the same code path that
  ! `fortran_sum_acc.f90`'s `partials_ff` array uses and works correctly.
  call dbl_to_ff(HALF_D,     BS_CONSTS_FF(BS_HALF))
  call dbl_to_ff(R_D,        BS_CONSTS_FF(BS_R))
  call dbl_to_ff(Q_D,        BS_CONSTS_FF(BS_Q))
  call dbl_to_ff(SQRT_1_2_D, BS_CONSTS_FF(BS_SQRT_1_2))

  call system_clock(count_rate=clk_rate)

  ! Data region covers the whole run.  Every device-resident array is
  ! `copyin`'d straight from its already-populated host counterpart; we
  ! never call into the FPMP runtime from inside the OpenACC region for
  ! initialisation.  Per-type prices arrays are device-only scratch.
  !$acc data copyin(S, K, Tmat, V, S_ff, K_ff, T_ff, V_ff, BS_CONSTS_FF) &
  !$acc&     create(prices_fp64, prices_ff)

  print '(A)',         '======================================================================================'
  print '(A)',         '  FPMP OpenACC Fortran Benchmark — Black-Scholes Option Pricing'
  print '(A)',         '======================================================================================'
  print '(A,I0,A)',    '  N            = ', N, ' options'
  print '(A,I0,A)',    '  threads      = ', NTH, &
                       ' (NUM_BLOCKS * BLOCK_SIZE; one OpenACC thread per option-slice)'
  print '(A,I0)',      '  repetitions  = ', REPS
  if (KL_FIXED > 0) then
    print '(A,I0,A)',  '  launches/rep = ', KL_FIXED, ' (fixed; per-launch time = elapsed / launches)'
  else
    print '(A,I0,A)',  '  launches/rep = auto-tune to ≥ ', MIN_BATCH_MS, &
                       ' ms per timed window (per-kernel)'
  end if
  print '(A,F5.2,A,F5.2)', '  r, q         = ', R_D, ', ', Q_D
  print '(A)',         '  inputs       = deterministic LDS (S, K ∈ [10,200]; T ∈ [0.05,5]; σ ∈ [0.05,1])'
  print '(A)',         '  is_call      = .true.'
  print '(A)',         '  reference    = host fp64 (intrinsic sqrt/log/exp/erfc)'
  print '(A,ES22.15)', '  ref. mean    = ', ref_mean
  print '(A)',         '--------------------------------------------------------------------------------------'
  print '(A)',         '  Type           Max|RelErr|  Avg|RelErr|  Throughput        Time     Speedup  Launches'
  print '(A)',         '                                           (Mopts/s)         (s)     vs fp64'
  print '(A)',         '  -------------  -----------  -----------  ----------  ----------  ----------  --------'

  call run_one(0)   ! fp64
  call run_one(1)   ! fp32mp2

  call print_table()

  !$acc end data

  print '(A)', '======================================================================================'

  deallocate(S, K, Tmat, V, ref)
  deallocate(prices_fp64, prices_ff)
  deallocate(S_ff, K_ff, T_ff, V_ff)

contains

  ! Reference per-option BS call price computed on the host in fp64
  ! using the Fortran intrinsic `sqrt`, `log`, `exp`, `erfc`.  Standalone
  ! helper so the kernel-side code can stay focused on the OpenACC path.
  function bs_price_host_fp64(S, K, T, sigma) result(price)
    real(real64), intent(in) :: S, K, T, sigma
    real(real64) :: price
    real(real64) :: sqrtT, vsqrtT, d1, d2, disc_r, disc_q, Nd1, Nd2
    sqrtT  = sqrt(T)
    vsqrtT = sigma * sqrtT
    d1     = (log(S / K) + (R_D - Q_D + HALF_D * sigma * sigma) * T) / vsqrtT
    d2     = d1 - vsqrtT
    disc_r = exp(-R_D * T)
    disc_q = exp(-Q_D * T)
    Nd1    = HALF_D * erfc(-d1 * SQRT_1_2_D)
    Nd2    = HALF_D * erfc(-d2 * SQRT_1_2_D)
    price  = S * disc_q * Nd1 - K * disc_r * Nd2
  end function bs_price_host_fp64

  ! --------------------------------------------------------------------------
  !  OpenACC kernels — one per arithmetic type.
  !
  !  Both kernels use a single-pass loop over the options: one OpenACC
  !  thread per `i` so the loop body is a pure independent computation.
  !  We deliberately do NOT do the `fortran_sum_acc.f90`-style grid-stride
  !  ("NTH threads, each strides through N/NTH options") pattern here
  !  because BS is much more compute-heavy than recursive summation —
  !  the per-option work amortises the launch overhead by itself, so we
  !  pick the cleaner one-thread-per-option scheduling that maps
  !  one-to-one onto the C++ bs.cpp kernel.
  !
  !  Why the explicit `gang vector vector_length(BLK)`: with default
  !  scheduling nvfortran sometimes vectorises the inner per-option
  !  computation differently between the native (fp64) and FPMP
  !  (fp32mp2) rows -- the FPMP `+`/`*`/transcendentals are opaque
  !  function calls into the FPMP runtime, while the fp64 path is just
  !  straight machine ops the compiler can fuse freely.  Forcing
  !  `gang vector` gives every kernel the same one-thread-per-option
  !  schedule and matches CUDA Fortran companion benchmark's
  !  NUM_BLOCKS*BLOCK_SIZE geometry.
  !
  !  Every kernel uses `async(0)` so successive launches in a timed
  !  window queue back-to-back on the OpenACC default async queue;
  !  the `!$acc wait(0)` before the closing `system_clock` (see
  !  run_one below) drains the queue so the timer captures real GPU
  !  time, not CPU dispatch latency.
  !
  !  No reduction is needed: each thread writes its own `prices(i)`
  !  slot independently.  The host folds the prices array into a mean
  !  for the printed sanity row outside the timed window.
  ! --------------------------------------------------------------------------

  subroutine bs_kernel_fp64()
    real(real64) :: Si, Ki, Ti, sigi, price
    real(real64) :: sqrtT, vsqrtT, d1, d2, disc_r, disc_q, Nd1, Nd2
    integer      :: i
    !$acc parallel loop gang vector vector_length(BLK) async(0) &
    !$acc&     present(S, K, Tmat, V, prices_fp64) &
    !$acc&     private(Si, Ki, Ti, sigi, price, sqrtT, vsqrtT, d1, d2, disc_r, disc_q, Nd1, Nd2)
    do i = 1, N
      Si   = S(i)
      Ki   = K(i)
      Ti   = Tmat(i)
      sigi = V(i)
      sqrtT  = sqrt(Ti)
      vsqrtT = sigi * sqrtT
      d1     = (log(Si / Ki) + (R_D - Q_D + HALF_D * sigi * sigi) * Ti) / vsqrtT
      d2     = d1 - vsqrtT
      disc_r = exp(-R_D * Ti)
      disc_q = exp(-Q_D * Ti)
      Nd1    = HALF_D * erfc(-d1 * SQRT_1_2_D)
      Nd2    = HALF_D * erfc(-d2 * SQRT_1_2_D)
      price  = Si * disc_q * Nd1 - Ki * disc_r * Nd2
      prices_fp64(i) = price
    end do
  end subroutine bs_kernel_fp64

  ! Black-Scholes kernel for `type(fp32mp2)`.  Written with end-user
  ! arithmetic — `+`, `-`, `*`, `/`, `sqrt`, `log`, `exp`, `erfc` — on
  ! `type(fp32mp2)` variables.  The overloads come from `fpmp_api`
  ! (no `attributes(device)`, just `!$acc routine seq` on every
  ! wrapper) compiled as a pure-OpenACC translation unit (i.e. no
  ! `-cuda` on the bindings nvfortran invocation -- see Makefile).
  !
  ! ABI discipline (see file header, issue 2): every derived-type
  ! intermediate is named in the `private(...)` clause and every
  ! `fp32mp2` expression is assigned to a private scalar BEFORE the
  ! result ever lands in `prices_ff`.  The only store into the global
  ! `prices_ff` array is `prices_ff(i) = price` — an intrinsic
  ! same-type copy from a private scalar, NOT a by-value function
  ! return.  Do not "inline" any compound `fp32mp2` expression into
  ! the `prices_ff(i) = ...` RHS or it will silently miscompile under
  ! nvfortran 25.11 (every `%lo` will read back as the `%hi` source
  ! value).
  subroutine bs_kernel_fp32mp2()
    type(fp32mp2) :: Si, Ki, Ti, sigi, price
    type(fp32mp2) :: sqrtT, vsqrtT, d1, d2, disc_r, disc_q, Nd1, Nd2
    type(fp32mp2) :: HALF_FF, R_FF, Q_FF, SQRT_1_2_FF
    integer       :: i
    !$acc parallel loop gang vector vector_length(BLK) async(0) &
    !$acc&     present(S_ff, K_ff, T_ff, V_ff, prices_ff, BS_CONSTS_FF) &
    !$acc&     private(Si, Ki, Ti, sigi, price, sqrtT, vsqrtT, d1, d2, disc_r, disc_q, Nd1, Nd2, &
    !$acc&             HALF_FF, R_FF, Q_FF, SQRT_1_2_FF)
    do i = 1, N
      HALF_FF     = BS_CONSTS_FF(BS_HALF)
      R_FF        = BS_CONSTS_FF(BS_R)
      Q_FF        = BS_CONSTS_FF(BS_Q)
      SQRT_1_2_FF = BS_CONSTS_FF(BS_SQRT_1_2)
      Si   = S_ff(i)
      Ki   = K_ff(i)
      Ti   = T_ff(i)
      sigi = V_ff(i)
      sqrtT  = sqrt(Ti)
      vsqrtT = sigi * sqrtT
      d1     = (log(Si / Ki) + (R_FF - Q_FF + HALF_FF * sigi * sigi) * Ti) / vsqrtT
      d2     = d1 - vsqrtT
      disc_r = exp(-R_FF * Ti)
      disc_q = exp(-Q_FF * Ti)
      Nd1    = HALF_FF * erfc(-d1 * SQRT_1_2_FF)
      Nd2    = HALF_FF * erfc(-d2 * SQRT_1_2_FF)
      price  = Si * disc_q * Nd1 - Ki * disc_r * Nd2
      prices_ff(i) = price
    end do
  end subroutine bs_kernel_fp32mp2

  ! --------------------------------------------------------------------------
  !  Host-side `double -> fp32mp2` conversion via pure arithmetic.  We
  !  could call `fp32mp2(x)` or use defined assignment instead -- both
  !  ARE host-callable in `fpmp_api` -- but the textbook
  !  double-float decomposition is cheaper than the FPMP runtime call:
  !
  !      hi = (float)  x          ! leading 24-bit fp32 approximation
  !      lo = (float) (x - hi)    ! residual; |lo| <= 0.5 ulp(hi)
  !
  !  which gives back the original double when summed in fp64 -- the
  !  same value the FPMP runtime computes, just inline and without any
  !  function-call hop.
  ! --------------------------------------------------------------------------
  subroutine dbl_to_ff(x, r)
    real(real64), intent(in)   :: x
    type(fp32mp2), intent(out) :: r
    real(c_float) :: hi
    hi   = real(x, c_float)
    r%hi = hi
    r%lo = real(x - real(hi, real64), c_float)
  end subroutine dbl_to_ff

  ! --------------------------------------------------------------------------
  ! Driver: launch the chosen kernel REPS times with batched
  ! `system_clock`-based timing.  Pure-OpenACC build (no
  ! `use cudafor`, no `-cuda`) so we use Fortran's intrinsic clock
  ! rather than `cudaEvent`.  Key points:
  !   * the explicit `!$acc wait(0)` before the closing
  !     `system_clock(clk_stop)` is required: without it, the clock
  !     stops while the GPU is still executing the queued async
  !     kernels and ends up measuring CPU dispatch latency instead
  !     of GPU execution time.
  !   * the multi-launch calibration (single-shot under-shoots the
  !     auto-tune target on short-kernel rows).
  !   * the `MAX_KERNEL_LAUNCHES` safety cap.
  ! --------------------------------------------------------------------------

  subroutine run_one(kind)
    integer, intent(in) :: kind   ! 0 = fp64, 1 = fp32mp2

    integer      :: rep, j, kl
    real(real64) :: dt_per_launch, baseline_s

    ! ---- 1. Pick the per-window launch count ----
    if (KL_FIXED > 0) then
      kl = KL_FIXED
    else
      ! Warm-up: KL_CALIB launches followed by an explicit wait so the
      ! GPU is fully idle before the calibration window begins.
      do j = 1, KL_CALIB
        call dispatch_kernel(kind)
      end do
      !$acc wait(0)

      ! Multi-launch calibration.  `!$acc wait(0)` before
      ! `system_clock(clk_stop)` is required so the timer captures GPU
      ! execution time rather than CPU dispatch latency on the async
      ! `!$acc parallel loop async(0)` launches.
      call system_clock(clk_start)
      do j = 1, KL_CALIB
        call dispatch_kernel(kind)
      end do
      !$acc wait(0)
      call system_clock(clk_stop)
      elapsed_s = real(clk_stop - clk_start, real64) / real(clk_rate, real64)

      baseline_s = max(elapsed_s / real(KL_CALIB, real64), 1.0e-9_real64)
      kl         = int(ceiling(TARGET_BATCH_S / baseline_s))
      kl         = max(1, min(KL_MAX, kl))
    end if

    ! ---- 2. Time REPS windows of `kl` back-to-back launches each ----
    ! Same wait-before-stop discipline as the calibration above.
    dt_min = huge(dt_min)
    do rep = 1, REPS
      call system_clock(clk_start)
      do j = 1, kl
        call dispatch_kernel(kind)
      end do
      !$acc wait(0)
      call system_clock(clk_stop)
      elapsed_s = real(clk_stop - clk_start, real64) / real(clk_rate, real64)
      dt_per_launch = elapsed_s / real(kl, real64)
      dt_min = min(dt_min, dt_per_launch)
    end do

    ! ---- 3. Read prices back, accuracy & sanity check on host ----
    call read_and_score(kind, kl)
  end subroutine run_one

  ! Dispatch one kernel launch of the chosen arithmetic type.  Pulled
  ! into its own routine so calibration and timing share the exact same
  ! launch path.
  subroutine dispatch_kernel(kind)
    integer, intent(in) :: kind
    select case (kind)
    case (0)
      call bs_kernel_fp64()
    case (1)
      call bs_kernel_fp32mp2()
    end select
  end subroutine dispatch_kernel

  ! Copy prices back to host, compute max/avg relative error vs the
  ! host fp64 reference.  fp32mp2 prices are converted to fp64 here for
  ! the comparison — outside the timed window, so the conversion cost
  ! does not contaminate the throughput numbers.
  subroutine read_and_score(kind, kl)
    integer, intent(in) :: kind, kl
    integer      :: ii, n_used
    real(real64) :: pi, re, max_re, sum_re, mean_p

    !$acc wait(0)
    select case (kind)
    case (0)
      !$acc update self(prices_fp64)
    case (1)
      !$acc update self(prices_ff)
    end select

    max_re = 0.0_real64
    sum_re = 0.0_real64
    n_used = 0
    mean_p = 0.0_real64
    do ii = 1, N
      if (kind == 0) then
        pi = prices_fp64(ii)
      else
        ! `fpmp_api` exposes a host-callable `fpmp_to_double` so
        ! we can use it directly here.  Outside the timed window, so
        ! conversion cost does not contaminate the throughput numbers.
        pi = fpmp_to_double(prices_ff(ii))
      end if
      mean_p = mean_p + pi
      if (abs(ref(ii)) >= 1.0e-12_real64) then
        re     = abs(pi - ref(ii)) / abs(ref(ii))
        if (re > max_re) max_re = re
        sum_re = sum_re + re
        n_used = n_used + 1
      end if
    end do
    mean_p = mean_p / real(N, real64)

    mean_arr  (kind + 1) = mean_p
    max_re_arr(kind + 1) = max_re
    avg_re_arr(kind + 1) = sum_re / real(max(1, n_used), real64)
    dt_arr    (kind + 1) = dt_min
    kl_arr    (kind + 1) = kl
  end subroutine read_and_score

  subroutine print_table()
    call print_row('real(real64) ', 0)
    call print_row('type(fp32mp2)', 1)
  end subroutine print_table

  ! Same column layout as fortran_bs_device.cuf so the two tables stack
  ! cleanly when both targets are run.
  subroutine print_row(label, kind)
    character(len=*), intent(in) :: label
    integer,          intent(in) :: kind
    real(real64) :: mopt_per_s, speedup
    mopt_per_s = real(N, real64) / max(dt_arr(kind + 1), 1.0e-12_real64) * 1.0e-6_real64
    speedup    = dt_arr(1) / max(dt_arr(kind + 1), 1.0e-12_real64)
    print '(2X,A,2X,ES11.3,2X,ES11.3,2X,ES10.3,2X,ES10.3,2X,F10.3,2X,I8)', &
      label, max_re_arr(kind + 1), avg_re_arr(kind + 1), mopt_per_s, &
      dt_arr(kind + 1), speedup, kl_arr(kind + 1)
  end subroutine print_row

end program fortran_bs_acc
